import datetime
import enum
import time
import typing

from homeassistant import const as hac
from homeassistant.core import HassJob, callback
from homeassistant.helpers import event
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .. import const as pmc, controller
from ..binary_sensor import BinarySensor
from ..helpers import validation as hv
from ..sensor import RestoreSensor

if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State


class CycleMode(enum.StrEnum):
    NONE = enum.auto()
    YEARLY = enum.auto()
    MONTHLY = enum.auto()
    WEEKLY = enum.auto()
    DAILY = enum.auto()
    HOURLY = enum.auto()


class ControllerConfig(pmc.BaseConfig):
    pv_power_entity_id: str
    """The source entity_id of the pv power"""
    cycle_modes: list[CycleMode]
    """list of 'metering' sensors to configure"""
    integration_period: int
    """If set, calculates accumulation of energy independently of pv_power changes"""
    maximum_latency: int
    """If set, in case pv_power doesn't update in the period, stops accumulating pv_power
    since this might be indication of a sensor failure"""


class EntryConfig(ControllerConfig, pmc.EntityConfig):
    """TypedDict for ConfigEntry data"""


class PVEnergySensor(RestoreSensor):

    controller: "Controller"

    native_value: float

    __slots__ = (
        "cycle_mode",
        "_dt_last_reset",
        "_dt_next_reset",
        "_async_track_cycle_job",
        "_async_track_cycle_unsub",
    )

    def __init__(
        self,
        controller: "Controller",
        cycle_mode: CycleMode,
    ):
        RestoreSensor.__init__(
            self,
            controller,
            f"pv_energy_sensor_{cycle_mode}",
            device_class=self.DeviceClass.ENERGY,
            name=controller.config.get("name", "pv_energy_sensor")
            + ("" if cycle_mode == CycleMode.NONE else f" ({cycle_mode})"),
        )
        self.native_value = 0
        self.native_unit_of_measurement = hac.UnitOfEnergy.WATT_HOUR
        self.cycle_mode = cycle_mode

    async def async_added_to_hass(self):

        if self.cycle_mode == CycleMode.NONE:
            restored_sensor_data = await self.async_get_last_sensor_data()
            if restored_sensor_data:
                self.native_value = restored_sensor_data.native_value  # type: ignore

        else:
            now = dt_util.now()
            match self.cycle_mode:
                case CycleMode.YEARLY:
                    self._dt_last_reset = datetime.datetime(
                        year=now.year, month=1, day=1, tzinfo=now.tzinfo
                    )
                    self._dt_next_reset = datetime.datetime(
                        year=now.year + 1, month=1, day=1, tzinfo=now.tzinfo
                    )
                case CycleMode.MONTHLY:
                    self._dt_last_reset = datetime.datetime(
                        year=now.year, month=now.month, day=1, tzinfo=now.tzinfo
                    )
                    self._dt_next_reset = (
                        self._dt_last_reset + datetime.timedelta(days=32)
                    ).replace(day=1)
                case CycleMode.WEEKLY:
                    today = datetime.datetime(
                        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                    )
                    self._dt_last_reset = today - datetime.timedelta(
                        days=today.weekday()
                    )
                    self._dt_next_reset = self._dt_last_reset + datetime.timedelta(
                        weeks=1
                    )
                case CycleMode.DAILY:
                    self._dt_last_reset = datetime.datetime(
                        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                    )
                    self._dt_next_reset = self._dt_last_reset + datetime.timedelta(
                        days=1
                    )
                case CycleMode.HOURLY:
                    self._dt_last_reset = datetime.datetime(
                        year=now.year,
                        month=now.month,
                        day=now.day,
                        hour=now.hour,
                        tzinfo=now.tzinfo,
                    )
                    self._dt_next_reset = self._dt_last_reset + datetime.timedelta(
                        hours=1
                    )

            restored_state = await self.async_get_last_state()
            if restored_state:
                last_changed = restored_state.last_changed
                if last_changed.astimezone(now.tzinfo) > self._dt_last_reset:
                    restored_sensor_data = await self.async_get_last_sensor_data()
                    if restored_sensor_data:
                        self.native_value = restored_sensor_data.native_value  # type: ignore

            self._async_track_cycle_job = HassJob(
                self._async_track_cycle, f"_async_track_cycle({self.cycle_mode})"
            )
            self._async_track_cycle_unsub = event.async_track_point_in_utc_time(self.hass, self._async_track_cycle_job, self._dt_next_reset)  # type: ignore
            self.log(
                self.DEBUG,
                "Scheduled first cycle at: %s",
                self._dt_next_reset.isoformat(),
            )

        await super().async_added_to_hass()
        self.controller.pv_energy_sensors.add(self)

    async def async_will_remove_from_hass(self):
        self.controller.pv_energy_sensors.remove(self)

        if self.cycle_mode != CycleMode.NONE:
            self._async_track_cycle_unsub()

        return await super().async_will_remove_from_hass()

    def accumulate(self, energy_wh: float):
        self.native_value += energy_wh
        self._async_write_ha_state()

    async def _async_track_cycle(self, _dt: "datetime.datetime"):

        self._dt_last_reset = now = dt_util.now()
        match self.cycle_mode:
            case CycleMode.YEARLY:
                self._dt_next_reset = datetime.datetime(
                    year=now.year + 1, month=1, day=1, tzinfo=now.tzinfo
                )
            case CycleMode.MONTHLY:
                month = now.month
                if month < 12:
                    month += 1
                    year = now.year
                else:
                    month = 1
                    year = now.year + 1
                self._dt_next_reset = datetime.datetime(
                    year=year, month=month, day=1, tzinfo=now.tzinfo
                )
            case CycleMode.WEEKLY:
                today = datetime.datetime(
                    year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                )
                self._dt_next_reset = today + datetime.timedelta(
                    days=7 - today.weekday()
                )
            case CycleMode.DAILY:
                self._dt_next_reset = datetime.datetime(
                    year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                ) + datetime.timedelta(days=1)
            case CycleMode.HOURLY:
                self._dt_next_reset = datetime.datetime(
                    year=now.year,
                    month=now.month,
                    day=now.day,
                    hour=now.hour,
                    tzinfo=now.tzinfo,
                ) + datetime.timedelta(hours=1)

        self._async_track_cycle_unsub = event.async_track_point_in_utc_time(self.hass, self._async_track_cycle_job, self._dt_next_reset)  # type: ignore
        self.log(
            self.DEBUG, "Scheduled next cycle at: %s", self._dt_next_reset.isoformat()
        )
        # TODO: adjust fractions of accumulated energy? shouldn't be really needed
        self.native_value = 0
        self._async_write_ha_state()


class Controller(controller.Controller[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_CALCULATOR

    PLATFORMS = {PVEnergySensor.PLATFORM}

    pv_energy_sensors: set[PVEnergySensor]

    __slots__ = (
        "pv_energy_sensors",
        "_integration_callback_unsub",
        "_maximum_latency_callback_unsub",
        "maximum_latency_alarm_binary_sensor",
        "_pv_power",
        "_pv_power_epoch",
        "_pv_power_tracking_unsub",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict) -> dict:
        return hv.entity_schema(user_input, name="PV Energy") | {
            hv.required("cycle_modes", user_input, CycleMode.NONE): hv.select_selector(
                options=list(CycleMode), multiple=True
            ),
            hv.required(pmc.CONF_PV_POWER_ENTITY_ID, user_input): hv.sensor_selector(
                device_class="power"
            ),
            hv.required(
                pmc.CONF_INTEGRATION_PERIOD, user_input, 5
            ): hv.time_period_selector(),
            hv.required(
                pmc.CONF_MAXIMUM_LATENCY, user_input, 300
            ): hv.time_period_selector(),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)

        self.pv_energy_sensors = set()

        for cycle_mode in self.config["cycle_modes"]:
            PVEnergySensor(self, cycle_mode)

        self._integration_callback_unsub = None
        self._maximum_latency_callback_unsub = None

        self._pv_power = self._get_pv_power_from_state(
            self.hass.states.get(self.config["pv_power_entity_id"])
        )
        self._pv_power_epoch = time.monotonic()

        self._pv_power_tracking_unsub = event.async_track_state_change_event(
            self.hass,
            self.config["pv_power_entity_id"],
            self._pv_power_tracking_callback,
        )
        self._integration_callback_unsub = (
            self.schedule_callback(
                self.config["integration_period"], self._integration_callback
            )
            if self.config["integration_period"]
            else None
        )
        maximum_latency = self.config["maximum_latency"]
        if maximum_latency:
            self._maximum_latency_callback_unsub = self.schedule_callback(
                maximum_latency, self._maximum_latency_callback
            )
            self.maximum_latency_alarm_binary_sensor = BinarySensor(
                self,
                "maximum_latency_alarm",
                device_class=BinarySensor.DeviceClass.PROBLEM,
                is_on=False,
            )
        else:
            self._maximum_latency_callback_unsub = None

    async def async_shutdown(self):
        if await super().async_shutdown():
            self._pv_power_tracking_unsub()
            if self._integration_callback_unsub:
                self._integration_callback_unsub.cancel()
                self._integration_callback_unsub = None
            if self._maximum_latency_callback_unsub:
                self._maximum_latency_callback_unsub.cancel()
                self._maximum_latency_callback_unsub = None
                self.maximum_latency_alarm_binary_sensor: BinarySensor = None # type:ignore
            return True
        return False

    @callback
    def _pv_power_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        now = time.monotonic()
        pv_power = self._get_pv_power_from_state(event.data.get("new_state"))

        try:
            # TODO: trapezoidal rule might be unneeded (or even dangerous) if pv_power
            # updates are not 'subsampled' with respect to the real pv power. In fact
            # a 'left' sample integration might be more appropriate. However, the eventual
            # internal 'integration_period' sampling might totally invalidate the
            # trapezoidal algorithm and just work as a 'left' rectangle integration.
            energy_wh = (
                (self._pv_power + pv_power) * (now - self._pv_power_epoch) / 7200  # type: ignore
            )
            for sensor in self.pv_energy_sensors:
                sensor.accumulate(energy_wh)

        except:  # in case any pv_power is None i.e. not valid...
            pass

        self._pv_power = pv_power
        self._pv_power_epoch = now

        if self._maximum_latency_callback_unsub:
            # retrigger maximum_latency
            self._maximum_latency_callback_unsub.cancel()
            self._maximum_latency_callback_unsub = self.schedule_callback(
                self.config["maximum_latency"], self._maximum_latency_callback
            )
            self.maximum_latency_alarm_binary_sensor.update(False)

    @callback
    def _integration_callback(self):
        """Called on a timer (if 'integration_period' is set) to accumulate energy in-between
        pv_power changes. In general this shouldn't be needed provided pv_power refreshes frequently
        since the accumulation is also done in _pv_power_tracking_callback"""
        self._integration_callback_unsub = self.schedule_callback(
            self.config["integration_period"], self._integration_callback
        )
        if self._pv_power is None:
            return
        now = time.monotonic()
        try:
            energy_wh = self._pv_power * (now - self._pv_power_epoch) / 3600
            for sensor in self.pv_energy_sensors:
                sensor.accumulate(energy_wh)
        except:
            pass
        self._pv_power_epoch = now

    @callback
    def _maximum_latency_callback(self):
        """Called when we don't have pv_power updates over 'maximum_latency' and might
        be regarded as a warning/error in data collection. We thus reset accumulating.
        """
        self._maximum_latency_callback_unsub = self.schedule_callback(
            self.config["maximum_latency"], self._maximum_latency_callback
        )
        self._pv_power = None
        self.maximum_latency_alarm_binary_sensor.update(True)

    def _get_pv_power_from_state(self, pv_power_state: "State | None"):
        if pv_power_state:
            unit = pv_power_state.attributes.get(self.hac.ATTR_UNIT_OF_MEASUREMENT)
            try:
                return PowerConverter.convert(
                    float(pv_power_state.state),
                    unit,
                    self.hac.UnitOfPower.WATT,
                )
            except Exception as e:
                self.log_exception(
                    self.WARNING,
                    e,
                    "Invalid state for entity %s: %s [%s] when converting to [W]",
                    pv_power_state.entity_id,
                    pv_power_state.state,
                    unit,
                )

        return None
