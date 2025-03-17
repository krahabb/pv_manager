"""
Controller for creating an entity simulating pv power based off sun tracking
"""

import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.helpers.event import async_track_state_change_event

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..sensor import Sensor

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.event import EventStateChangedData


class Config(pmc.SensorConfig, pmc.BaseConfig):
    """ConfigEntry data"""

    peak_power: float
    """The peak power of the pv system"""
    simulate_weather: bool


class PVPowerSimulatorSensor(Sensor):

    native_value: float | None

    __slots__ = (
        "peak_power",
        "simulate_weather",
        "_sun_tracking_unsub",
    )

    def __init__(self, controller: "Controller"):
        Sensor.__init__(self,
            controller,
            pmc.ConfigEntryType.PV_POWER_SIMULATOR,
            device_class=self.DeviceClass.POWER,
        )
        self.native_value = None
        helpers.apply_config(self, controller.config_entry.data, Config)

    async def async_added_to_hass(self):
        self._update_pv_power(self.hass.states.get("sun.sun"))
        self._sun_tracking_unsub = async_track_state_change_event(
            self.hass, "sun.sun", self._sun_tracking_callback
        )
        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self):
        self._sun_tracking_unsub()
        await super().async_will_remove_from_hass()

    @callback
    def _sun_tracking_callback(self, event: "Event[EventStateChangedData]"):
        self._update_pv_power(event.data.get("new_state"))
        self.async_write_ha_state()

    def _update_pv_power(self, sun_state: "State | None"):
        if sun_state:
            elevation = sun_state.attributes.get("elevation")
            if elevation and elevation > 0:
                self.native_value = self.peak_power * elevation / 90
            else:
                self.native_value = 0
        else:
            self.native_value = None


class Controller(controller.Controller):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_POWER_SIMULATOR

    __slots__ = ()

    @staticmethod
    def get_config_entry_schema(user_input: dict) -> dict:
        return hv.sensor_schema(
            user_input,
            name="PV power",
            native_unit_of_measurement=hac.UnitOfPower.WATT,
        ) | {
            hv.required(pmc.CONF_PEAK_POWER, user_input, 1000): int,
            hv.required(pmc.CONF_SIMULATE_WEATHER, user_input, True): bool,
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)
        PVPowerSimulatorSensor(self)
