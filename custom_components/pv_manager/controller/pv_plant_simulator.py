"""
Controller for creating an entity simulating pv power based off sun tracking
"""

import math
import random
import time
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv, event
from homeassistant.util.unit_conversion import DistanceConverter, TemperatureConverter

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..sensor import Sensor

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.event import EventStateChangedData

    from ..sensor import SensorArgs


class EntryConfig(pmc.SensorConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""

    peak_power: float
    """The peak power of the pv system"""
    weather_entity_id: typing.NotRequired[str]
    """The weather entity used to 'modulate' pv power"""

    battery_voltage: typing.NotRequired[int]
    battery_capacity: typing.NotRequired[int]

    consumption_baseload_power_w: typing.NotRequired[float]
    consumption_daily_extra_power_w: typing.NotRequired[float]
    consumption_daily_fill_factor: typing.NotRequired[float]


class Controller(controller.Controller[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_PLANT_SIMULATOR

    __slots__ = (
        "peak_power",
        "weather_entity_id",
        "battery_voltage",
        "battery_capacity",
        "consumption_baseload_power_w",
        "consumption_daily_extra_power_w",
        "consumption_daily_fill_factor",
        "pv_power_simulator_sensor",
        "battery_voltage_sensor",
        "battery_current_sensor",
        "battery_charge_sensor",
        "_power_last_update_ts",
        "_weather_state",
        "_weather_cloud_coverage",
        "_weather_temperature",
        "_weather_visibility",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict):
        return hv.sensor_schema(
            user_input,
            name="PV simulator",
            native_unit_of_measurement=hac.UnitOfPower.WATT,
        ) | {
            hv.required("peak_power", user_input, 1000): int,
            hv.optional("weather_entity_id", user_input): hv.weather_selector(),
            hv.optional("battery_voltage", user_input, 48): cv.positive_int,
            hv.optional("battery_capacity", user_input, 100): cv.positive_int,
            hv.optional(
                "consumption_baseload_power_w", user_input, 100
            ): hv.positive_number_selector(unit_of_measurement=hac.UnitOfPower.WATT),
            hv.optional(
                "consumption_daily_extra_power_w", user_input, 500
            ): hv.positive_number_selector(unit_of_measurement=hac.UnitOfPower.WATT),
            hv.optional(
                "consumption_daily_fill_factor", user_input, 0.2
            ): hv.positive_number_selector(max=1, step=0.1),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)
        self.peak_power = self.config["peak_power"]
        self.weather_entity_id = self.config.get("weather_entity_id")

        self._power_last_update_ts = None
        self._weather_state = None

        self.pv_power_simulator_sensor = Sensor(
            self,
            "pv_power_simulator",
            device_class=Sensor.DeviceClass.POWER,
            name=self.config["name"],
            native_unit_of_measurement=self.config["native_unit_of_measurement"],
        )

        self.battery_voltage = self.config.get("battery_voltage", 0)
        self.battery_capacity = self.config.get("battery_capacity", 0)
        if self.battery_voltage:
            self.battery_voltage_sensor = Sensor(
                self,
                "battery_voltage",
                device_class=Sensor.DeviceClass.VOLTAGE,
                name="Battery voltage",
                native_unit_of_measurement=hac.UnitOfElectricPotential.VOLT,
                native_value=self.battery_voltage,
            )
            self.battery_current_sensor = Sensor(
                self,
                "battery_current",
                device_class=Sensor.DeviceClass.CURRENT,
                name="Battery current",
                native_unit_of_measurement=hac.UnitOfElectricCurrent.AMPERE,
            )
            self.battery_charge_sensor = Sensor(
                self,
                "battery_charge",
                state_class=Sensor.StateClass.MEASUREMENT,
                name="Battery charge",
                native_unit_of_measurement="Ah",
                native_value=self.battery_capacity / 2,
            )

        self.consumption_baseload_power_w = self.config.get(
            "consumption_baseload_power_w", 0
        )
        self.consumption_daily_extra_power_w = self.config.get(
            "consumption_daily_extra_power_w", 0
        )
        self.consumption_daily_fill_factor = self.config.get(
            "consumption_daily_fill_factor", 0.2
        )
        self.consumption_sensor = Sensor(
            self,
            "consumption",
            device_class=Sensor.DeviceClass.POWER,
            name="Consumption",
            native_unit_of_measurement=hac.UnitOfPower.WATT,
            state_class=Sensor.StateClass.MEASUREMENT,
        )

    async def async_init(self):
        if self.weather_entity_id:
            self.track_state(self.weather_entity_id, self._weather_tracking_callback)
            self._update_weather(self.hass.states.get(self.weather_entity_id))
        self.track_state("sun.sun", self._sun_tracking_callback)
        self._update_power(self.hass.states.get("sun.sun"))
        return await super().async_init()

    async def async_shutdown(self):
        await super().async_shutdown()
        self.pv_power_simulator_sensor: Sensor = None  # type: ignore
        self.battery_voltage_sensor: Sensor = None  # type: ignore
        self.battery_current_sensor: Sensor = None  # type: ignore
        self.battery_charge_sensor: Sensor = None  # type: ignore
        self.consumption_sensor: Sensor = None  # type: ignore
        self._power_last_update_ts = None

    @callback
    def _sun_tracking_callback(self, event: "Event[EventStateChangedData]"):
        self._update_power(event.data.get("new_state"))

    @callback
    def _weather_tracking_callback(self, event: "Event[EventStateChangedData]"):
        self._update_weather(event.data.get("new_state"))

    def _update_power(self, sun_state: "State | None"):

        try:
            elevation = sun_state.attributes["elevation"]  # type: ignore
            if elevation > -5:  # roughly dusk
                # it is very hard to model the transition night/day since when the sun is low
                # and starts to rise/set the sun energy is very low even if the elevation is relatively high
                # with respect to the plant slope. Here we take a simple approach with
                # slope almost horizontal
                slope = 85
                pv_power = self.peak_power * math.cos(
                    (slope - elevation) * math.pi / 180
                )
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
                        pv_power *= 0.10
                    elif gain < 1:
                        pv_power *= gain

                consumption_power = self.consumption_baseload_power_w
                p1 = random.randint(1, 100) / 100
                a = random.randint(1, 100) / 100
                if a < (self.consumption_daily_fill_factor / p1):
                    consumption_power += self.consumption_daily_extra_power_w * p1
            else:  # night time
                pv_power = 0
                consumption_power = self.consumption_baseload_power_w

            self.pv_power_simulator_sensor.update(pv_power)
            self.consumption_sensor.update(consumption_power)
            if self.battery_voltage:
                battery_power = consumption_power - pv_power
                battery_current = battery_power / self.battery_voltage
                # assume a voltage drop of 5% of the battery voltage at 1C
                battery_resistance = self.battery_voltage / (
                    20 * (self.battery_capacity or 100)
                )
                battery_voltage = (
                    self.battery_voltage - battery_resistance * battery_current
                )
                battery_current = battery_power / battery_voltage
                self.battery_voltage_sensor.update(battery_voltage)
                self.battery_current_sensor.update(battery_current)
                now_ts = time.monotonic()
                if self._power_last_update_ts:
                    delta_t = now_ts - self._power_last_update_ts
                    delta_battery_charge = battery_current * delta_t / 3600
                    battery_charge: float = self.battery_charge_sensor.native_value or 0  # type: ignore
                    battery_charge -= delta_battery_charge
                    if battery_charge < 0:
                        battery_charge = 0
                    elif self.battery_capacity and (
                        battery_charge > self.battery_capacity
                    ):
                        battery_charge = self.battery_capacity
                    self.battery_charge_sensor.update(battery_charge)
                    # we should now derate the pv output should the battery be full...
                self._power_last_update_ts = now_ts

        except (KeyError, AttributeError):
            self._power_last_update_ts = None
            self.pv_power_simulator_sensor.update(None)
            self.consumption_sensor.update(None)
            if self.battery_voltage:
                self.battery_voltage_sensor.update(self.battery_voltage)
                self.battery_current_sensor.update(0)

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
