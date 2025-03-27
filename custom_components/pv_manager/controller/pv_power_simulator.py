"""
Controller for creating an entity simulating pv power based off sun tracking
"""

import random
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.helpers import event
from homeassistant.util.unit_conversion import DistanceConverter, TemperatureConverter

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..sensor import Sensor

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.event import EventStateChangedData


class EntryConfig(pmc.SensorConfig, pmc.EntryConfig):
    """ConfigEntry data"""

    peak_power: float
    """The peak power of the pv system"""
    weather_entity_id: typing.NotRequired[str]
    """The weather entity used to 'modulate' pv power"""


class PVPowerSimulatorSensor(Sensor):

    native_value: float | None

    __slots__ = (
        "peak_power",
        "weather_entity_id",
        "_sun_tracking_unsub",
        "_weather_tracking_unsub",
        "_weather_state",
        "_weather_cloud_coverage",
        "_weather_temperature",
        "_weather_visibility",
    )

    def __init__(self, controller: "Controller"):
        Sensor.__init__(
            self,
            controller,
            pmc.ConfigEntryType.PV_POWER_SIMULATOR,
            device_class=self.DeviceClass.POWER,
        )
        self.weather_entity_id = ""
        self._weather_state = None
        helpers.apply_config(self, controller.config_entry.data, EntryConfig)

    async def async_added_to_hass(self):
        if self.weather_entity_id:
            self._update_weather(self.hass.states.get(self.weather_entity_id))
            self._weather_tracking_unsub = event.async_track_state_change_event(
                self.hass,
                self.weather_entity_id,
                self._weather_tracking_callback,
            )
        else:
            self._weather_tracking_unsub = None
        self._update_pv_power(self.hass.states.get("sun.sun"))
        self._sun_tracking_unsub = event.async_track_state_change_event(
            self.hass, "sun.sun", self._sun_tracking_callback
        )

        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self):
        self._sun_tracking_unsub()
        if self._weather_tracking_unsub:
            self._weather_tracking_unsub()
            self._weather_tracking_unsub = None
        await super().async_will_remove_from_hass()

    @callback
    def _sun_tracking_callback(self, event: "Event[EventStateChangedData]"):
        self._update_pv_power(event.data.get("new_state"))
        self.async_write_ha_state()

    @callback
    def _weather_tracking_callback(self, event: "Event[EventStateChangedData]"):
        self._update_weather(event.data.get("new_state"))

    def _update_pv_power(self, sun_state: "State | None"):
        if sun_state:
            elevation = sun_state.attributes.get("elevation")
            if elevation and elevation > 0:
                power = self.peak_power * elevation / 90
                if self._weather_state:

                    if self._weather_visibility is not None:
                        # assume greater visibility gives less diffraction so better power output
                        if self._weather_visibility > 1:
                            visibility_gain = (
                                self._weather_visibility - 0.5
                            ) / self._weather_visibility
                        else:
                            # almost foggy?
                            visibility_gain = self._weather_visibility * 0.5
                    else:
                        visibility_gain = 1
                    if self._weather_cloud_coverage is not None:
                        # assume 100% cloud_coverage would reduce pv yield to 20%
                        cloud_attenuation = (self._weather_cloud_coverage / 100) * 0.8
                        cloud_coverage_gain = 1 - cloud_attenuation
                    else:
                        cloud_coverage_gain = 1

                    match self._weather_state:
                        case "partlycloudy":
                            # when partlycloudy we have random shadowing from clouds
                            cloud_coverage_gain *= random.randint(8, 13) * 0.1
                        case "pouring":
                            cloud_coverage_gain *= 0.5
                        case "snowy":
                            cloud_coverage_gain *= 0.5
                    gain = visibility_gain * cloud_coverage_gain
                    if gain < 0.10:
                        power *= 0.10
                    elif gain < 1:
                        power *= gain

                self.native_value = power
            else:
                self.native_value = 0
        else:
            self.native_value = None

    def _update_weather(self, state: "State | None"):
        if state:
            self._weather_state = state.state
            attributes = state.attributes
            self._weather_cloud_coverage = attributes.get("cloud_coverage")
            self._weather_temperature = TemperatureConverter.convert(
                float(attributes["temperature"]),
                attributes["temperature_unit"],
                hac.UnitOfTemperature.CELSIUS,
            )
            if "visibility" in attributes:
                self._weather_visibility = DistanceConverter.convert(
                    attributes["visibility"],
                    attributes["visibility_unit"],
                    hac.UnitOfLength.KILOMETERS,
                )
            else:
                self._weather_visibility = None
        else:
            self._weather_state = None


class Controller(controller.Controller[EntryConfig]):
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
            hv.required("peak_power", user_input, 1000): int,
            hv.optional("weather_entity_id", user_input): hv.weather_selector(),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)
        PVPowerSimulatorSensor(self)
