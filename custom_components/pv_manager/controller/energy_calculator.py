from datetime import datetime, timedelta
import enum
import time
import typing

from homeassistant import const as hac
from homeassistant.core import HassJob, callback
from homeassistant.helpers import event
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .. import const as pmc, controller, sensor
from ..binary_sensor import BinarySensor
from ..helpers import validation as hv

if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State


class EnergySensor(sensor.EnergySensor):

    controller: "Controller"

    _attr_parent_attr = None

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.controller.energy_sensors.add(self)

    async def async_will_remove_from_hass(self):
        self.controller.energy_sensors.remove(self)
        return await super().async_will_remove_from_hass()


class ControllerConfig(typing.TypedDict):
    power_entity_id: str
    """The source entity_id of the pv power"""
    cycle_modes: list[EnergySensor.CycleMode]
    """list of 'metering' sensors to configure"""
    integration_period_seconds: int
    """If set, calculates accumulation of energy independently of pv_power changes"""
    maximum_latency_seconds: int
    """If set, in case pv_power doesn't update in the period, stops accumulating pv_power
    since this might be indication of a sensor failure"""


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""


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
        return hv.entity_schema(user_input, name="Energy") | {
            hv.required("power_entity_id", user_input): hv.sensor_selector(
                device_class=EnergySensor.DeviceClass.POWER
            ),
            hv.required(
                "cycle_modes", user_input, [EnergySensor.CycleMode.NONE]
            ): hv.select_selector(options=list(EnergySensor.CycleMode), multiple=True),
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
            EnergySensor(
                self, "pv_energy_sensor", cycle_mode, name=self.config.get("name")
            )

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
            self.maximum_latency_alarm_binary_sensor.update_safe(False)

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
        self.maximum_latency_alarm_binary_sensor.update_safe(True)

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
