from datetime import datetime, timedelta
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


class ControllerConfig(typing.TypedDict):
    power_entity_id: str
    """The source entity_id of the pv power"""
    cycle_modes: list[CycleMode]
    """list of 'metering' sensors to configure"""
    integration_period_seconds: int
    """If set, calculates accumulation of energy independently of pv_power changes"""
    maximum_latency_seconds: int
    """If set, in case pv_power doesn't update in the period, stops accumulating pv_power
    since this might be indication of a sensor failure"""


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""


class EnergySensor(RestoreSensor):

    controller: "Controller"

    native_value: float

    __slots__ = (
        "cycle_mode",
        "last_reset",  # HA property
        "last_reset_dt",
        "last_reset_ts",  # UTC timestamp of last reset
        "next_reset_dt",
        "next_reset_ts",  # UTC timestamp of next reset
        "_async_track_cycle_job",
        "_async_track_cycle_unsub",
    )

    def __init__(
        self,
        controller: "Controller",
        cycle_mode: CycleMode,
    ):
        self.cycle_mode: typing.Final = cycle_mode
        self.last_reset = None
        RestoreSensor.__init__(
            self,
            controller,
            f"pv_energy_sensor_{cycle_mode}",
            device_class=self.DeviceClass.ENERGY,
            state_class=self.StateClass.TOTAL,
            name=controller.config.get("name", "pv_energy_sensor")
            + ("" if cycle_mode == CycleMode.NONE else f" ({cycle_mode})"),
            native_value=0,
            native_unit_of_measurement=hac.UnitOfEnergy.WATT_HOUR,
        )

    async def async_added_to_hass(self):

        if self.cycle_mode == CycleMode.NONE:
            restored_sensor_data = await self.async_get_last_sensor_data()
            if restored_sensor_data:
                self.native_value = restored_sensor_data.native_value  # type: ignore

        else:
            self._compute_cycle(False)

            restored_state = await self.async_get_last_state()
            if restored_state:
                if (
                    self.last_reset_ts
                    < restored_state.last_changed_timestamp
                    < self.next_reset_ts
                ):
                    restored_sensor_data = await self.async_get_last_sensor_data()
                    if restored_sensor_data:
                        self.native_value = restored_sensor_data.native_value  # type: ignore

            self._async_track_cycle_job = HassJob(
                self._async_track_cycle, f"_async_track_cycle({self.cycle_mode})"
            )
            self._async_track_cycle_unsub = event.async_track_point_in_utc_time(self.hass, self._async_track_cycle_job, self.next_reset_dt)  # type: ignore

        await super().async_added_to_hass()
        self.controller.energy_sensors.add(self)

    async def async_will_remove_from_hass(self):
        self.controller.energy_sensors.remove(self)

        if self.cycle_mode != CycleMode.NONE:
            self._async_track_cycle_unsub()

        return await super().async_will_remove_from_hass()

    def accumulate(self, energy_wh: float):
        if self.cycle_mode == CycleMode.NONE:
            self.native_value += energy_wh
        else:
            now_ts = time.time()
            if now_ts >= self.next_reset_ts:
                self._async_track_cycle_unsub()
                self._compute_cycle(True)
                self.native_value = energy_wh
            else:
                self.native_value += energy_wh

        self._async_write_ha_state()

    async def _async_track_cycle(self, _dt: datetime):
        self._compute_cycle(True)
        # TODO: adjust fractions of accumulated energy? shouldn't be really needed
        self.native_value = 0
        self._async_write_ha_state()

    def _compute_cycle(self, reschedule: bool):
        """'now' is local time."""
        now = dt_util.now()
        match self.cycle_mode:
            case CycleMode.YEARLY:
                self.last_reset_dt = datetime(
                    year=now.year, month=1, day=1, tzinfo=now.tzinfo
                )
                self.next_reset_dt = datetime(
                    year=now.year + 1, month=1, day=1, tzinfo=now.tzinfo
                )
            case CycleMode.MONTHLY:
                self.last_reset_dt = datetime(
                    year=now.year, month=now.month, day=1, tzinfo=now.tzinfo
                )
                self.next_reset_dt = (self.last_reset_dt + timedelta(days=32)).replace(
                    day=1
                )
            case CycleMode.WEEKLY:
                today = datetime(
                    year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                )
                self.last_reset_dt = today - timedelta(days=today.weekday())
                self.next_reset_dt = self.last_reset_dt + timedelta(weeks=1)
            case CycleMode.DAILY:
                self.last_reset_dt = datetime(
                    year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                )
                self.next_reset_dt = self.last_reset_dt + timedelta(days=1)
            case CycleMode.HOURLY:
                self.last_reset_dt = datetime(
                    year=now.year,
                    month=now.month,
                    day=now.day,
                    hour=now.hour,
                    tzinfo=now.tzinfo,
                )
                self.next_reset_dt = self.last_reset_dt + timedelta(hours=1)

        self.last_reset_dt = self.last_reset_dt.astimezone(dt_util.UTC)
        self.last_reset_ts = self.last_reset_dt.timestamp()
        self.next_reset_dt = self.next_reset_dt.astimezone(dt_util.UTC)
        self.next_reset_ts = self.next_reset_dt.timestamp()

        self.last_reset = self.last_reset_dt
        if reschedule:
            self._async_track_cycle_unsub = event.async_track_point_in_utc_time(self.hass, self._async_track_cycle_job, self.next_reset_dt)  # type: ignore
            self.log(
                self.DEBUG,
                "Scheduled next cycle at: %s",
                self.next_reset_dt.isoformat(),
            )


class Controller(controller.Controller[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.ENERGY_CALCULATOR

    PLATFORMS = {EnergySensor.PLATFORM}

    energy_sensors: set[EnergySensor]

    __slots__ = (
        "energy_sensors",
        "maximum_latency_alarm_binary_sensor",
        "_power",
        "_power_epoch",
        "_integration_callback_unsub",
        "_maximum_latency_callback_unsub",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict) -> dict:
        return hv.entity_schema(user_input, name="PV Energy") | {
            hv.required("power_entity_id", user_input): hv.sensor_selector(
                device_class=EnergySensor.DeviceClass.POWER
            ),
            hv.required(
                "cycle_modes", user_input, [CycleMode.NONE]
            ): hv.select_selector(options=list(CycleMode), multiple=True),
            hv.required(
                "integration_period_seconds", user_input, 5
            ): hv.time_period_selector(),
            hv.optional(
                "maximum_latency_seconds", user_input, 300
            ): hv.time_period_selector(),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)

        self.energy_sensors = set()

        for cycle_mode in self.config["cycle_modes"]:
            EnergySensor(self, cycle_mode)

        self._power = self._get_power_from_state(
            self.hass.states.get(self.config["power_entity_id"])
        )
        self._power_epoch = time.monotonic()

        self.track_state(self.config["power_entity_id"], self._power_tracking_callback)

        self._integration_callback_unsub = (
            self.schedule_callback(
                self.config["integration_period_seconds"], self._integration_callback
            )
            if self.config["integration_period_seconds"]
            else None
        )
        maximum_latency = self.config.get("maximum_latency_seconds")
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
        await super().async_shutdown()
        if self._integration_callback_unsub:
            self._integration_callback_unsub.cancel()
            self._integration_callback_unsub = None
        if self._maximum_latency_callback_unsub:
            self._maximum_latency_callback_unsub.cancel()
            self._maximum_latency_callback_unsub = None
            self.maximum_latency_alarm_binary_sensor: BinarySensor = None  # type:ignore

    @callback
    def _power_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        now = time.monotonic()
        power = self._get_power_from_state(event.data.get("new_state"))

        try:
            # TODO: trapezoidal rule might be unneeded (or even dangerous) if pv_power
            # updates are not 'subsampled' with respect to the real pv power. In fact
            # a 'left' sample integration might be more appropriate. However, the eventual
            # internal 'integration_period' sampling might totally invalidate the
            # trapezoidal algorithm and just work as a 'left' rectangle integration.
            energy_wh = (
                (self._power + power) * (now - self._power_epoch) / 7200  # type: ignore
            )
            for sensor in self.energy_sensors:
                sensor.accumulate(energy_wh)

        except:  # in case any power is None i.e. not valid...
            pass

        self._power = power
        self._power_epoch = now

        if self._maximum_latency_callback_unsub:
            # retrigger maximum_latency
            self._maximum_latency_callback_unsub.cancel()
            self._maximum_latency_callback_unsub = self.schedule_callback(
                self.config["maximum_latency_seconds"], self._maximum_latency_callback
            )
            self.maximum_latency_alarm_binary_sensor.update(False)

    @callback
    def _integration_callback(self):
        """Called on a timer (if 'integration_period' is set) to accumulate energy in-between
        pv_power changes. In general this shouldn't be needed provided pv_power refreshes frequently
        since the accumulation is also done in _pv_power_tracking_callback"""
        self._integration_callback_unsub = self.schedule_callback(
            self.config["integration_period_seconds"], self._integration_callback
        )
        if self._power is None:
            return
        now = time.monotonic()
        try:
            energy_wh = self._power * (now - self._power_epoch) / 3600
            for sensor in self.energy_sensors:
                sensor.accumulate(energy_wh)
        except:
            pass
        self._power_epoch = now

    @callback
    def _maximum_latency_callback(self):
        """Called when we don't have pv_power updates over 'maximum_latency' and might
        be regarded as a warning/error in data collection. We thus reset accumulating.
        """
        self._maximum_latency_callback_unsub = self.schedule_callback(
            self.config["maximum_latency_seconds"], self._maximum_latency_callback
        )
        self._power = None
        self.maximum_latency_alarm_binary_sensor.update(True)

    def _get_power_from_state(self, power_state: "State | None"):
        if power_state:
            try:
                return PowerConverter.convert(
                    float(power_state.state),
                    power_state.attributes["unit_of_measurement"],
                    self.hac.UnitOfPower.WATT,
                )
            except Exception as e:
                self.log_exception(
                    self.WARNING,
                    e,
                    "Invalid state for entity %s: %s when converting to [W]",
                    power_state.entity_id,
                    power_state.state,
                )

        return None
