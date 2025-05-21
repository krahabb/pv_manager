"""
Controller for creating an entity simulating pv power based off sun tracking
"""

import math
import random
import time
import typing

from astral import sun
from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.helpers import config_validation as cv, sun as sun_helpers
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import DistanceConverter, TemperatureConverter

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..manager import Manager
from ..sensor import BatteryChargeSensor, PowerSensor, Sensor

if typing.TYPE_CHECKING:
    from typing import NotRequired, Unpack

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, EventStateChangedData, State

    class EntryConfig(pmc.SensorConfig, controller.Controller.Config):
        """TypedDict for ConfigEntry data"""

        peak_power: float
        """The peak power of the pv system"""
        weather_entity_id: NotRequired[str]
        """The weather entity used to 'modulate' pv power"""

        battery_voltage: NotRequired[int]
        battery_capacity: NotRequired[int]

        consumption_baseload_power_w: NotRequired[float]
        consumption_daily_extra_power_w: NotRequired[float]
        consumption_daily_fill_factor: NotRequired[float]

        inverter_zeroload_power_w: NotRequired[float]
        inverter_efficiency: NotRequired[float]


class Controller(controller.Controller["EntryConfig"]):
    """Base controller class for managing ConfigEntry behavior."""

    if typing.TYPE_CHECKING:

        class Config(EntryConfig):
            pass

    TYPE = pmc.ConfigEntryType.PV_PLANT_SIMULATOR

    SAMPLING_PERIOD: float = 5

    __slots__ = (
        "peak_power",
        "weather_entity_id",
        "battery_voltage",
        "battery_capacity",
        "consumption_baseload_power_w",
        "consumption_daily_extra_power_w",
        "consumption_daily_fill_factor",
        "inverter_zeroload_power",
        "inverter_efficiency",
        "astral_observer",
        "pv_power_simulator_sensor",
        "battery_voltage_sensor",
        "battery_current_sensor",
        "battery_charge_sensor",
        "consumption_sensor",
        "inverter_losses_sensor",
        "_inverter_on",
        "_weather_state",
        "_weather_cloud_coverage",
        "_weather_temperature",
        "_weather_visibility",
    )

    @staticmethod
    def get_config_entry_schema(config: "Config | None") -> pmc.ConfigSchema:
        if not config:
            config = {
                "name": "PV simulator",
                "native_unit_of_measurement": hac.UnitOfPower.WATT,
                "peak_power": 1000,
                "battery_voltage": 48,
                "battery_capacity": 100,
                "consumption_baseload_power_w": 100,
                "consumption_daily_extra_power_w": 500,
                "consumption_daily_fill_factor": 0.2,
                "inverter_zeroload_power_w": 20,
                "inverter_efficiency": 0.9,
            }
        return hv.sensor_schema(config, hac.UnitOfPower) | {
            hv.req_config("peak_power", config): int,
            hv.opt_config("weather_entity_id", config): hv.weather_entity_selector(),
            hv.opt_config("battery_voltage", config): cv.positive_int,
            hv.opt_config("battery_capacity", config): cv.positive_int,
            hv.opt_config(
                "consumption_baseload_power_w", config
            ): hv.positive_number_selector(unit_of_measurement=hac.UnitOfPower.WATT),
            hv.opt_config(
                "consumption_daily_extra_power_w", config
            ): hv.positive_number_selector(unit_of_measurement=hac.UnitOfPower.WATT),
            hv.opt_config(
                "consumption_daily_fill_factor", config
            ): hv.positive_number_selector(max=1, step=0.1),
            hv.opt_config(
                "inverter_zeroload_power_w", config
            ): hv.positive_number_selector(unit_of_measurement=hac.UnitOfPower.WATT),
            hv.opt_config("inverter_efficiency", config): hv.positive_number_selector(
                max=1, step=0.01
            ),
        }

    def __init__(self, config_entry: "ConfigEntry"):
        super().__init__(config_entry)

        location, elevation = sun_helpers.get_astral_location(Manager.hass)
        self.astral_observer = sun.Observer(
            location.latitude, location.longitude, elevation
        )

        config = self.config
        self.peak_power = config["peak_power"]
        self.weather_entity_id = config.get("weather_entity_id")

        self._weather_state = None
        self._inverter_on = True

        self.pv_power_simulator_sensor = Sensor(
            self,
            "pv_power_simulator",
            device_class=Sensor.DeviceClass.POWER,
            name=config["name"],
            native_unit_of_measurement=config["native_unit_of_measurement"],
        )

        self.battery_voltage = config.get("battery_voltage", 0)
        self.battery_capacity = config.get("battery_capacity", 0)
        self.battery_voltage_sensor: Sensor | None = None
        self.battery_current_sensor: Sensor | None = None
        self.battery_charge_sensor: BatteryChargeSensor | None = None
        if self.battery_voltage:
            Sensor(
                self,
                "battery_voltage",
                device_class=Sensor.DeviceClass.VOLTAGE,
                name="Battery voltage",
                native_unit_of_measurement=hac.UnitOfElectricPotential.VOLT,
                native_value=self.battery_voltage,
                parent_attr=Sensor.ParentAttr.DYNAMIC,
            )
            Sensor(
                self,
                "battery_current",
                device_class=Sensor.DeviceClass.CURRENT,
                name="Battery current",
                native_unit_of_measurement=hac.UnitOfElectricCurrent.AMPERE,
                parent_attr=Sensor.ParentAttr.DYNAMIC,
            )
            if self.battery_capacity:
                self._inverter_on = False  # let the capacity check start the inverter
                BatteryChargeSensor(
                    self,
                    "battery_charge",
                    name="Battery charge",
                    capacity=self.battery_capacity,
                    native_value=self.battery_capacity / 2,
                    parent_attr=Sensor.ParentAttr.DYNAMIC,
                )

        self.consumption_baseload_power_w = config.get(
            "consumption_baseload_power_w", 0
        )
        self.consumption_daily_extra_power_w = config.get(
            "consumption_daily_extra_power_w", 0
        )
        self.consumption_daily_fill_factor = config.get(
            "consumption_daily_fill_factor", 0
        )
        self.consumption_sensor = PowerSensor(
            self,
            "consumption",
            name="Consumption",
        )

        self.inverter_zeroload_power = config.get("inverter_zeroload_power_w", 0)
        self.inverter_efficiency = config.get("inverter_efficiency", 1)
        self.inverter_losses_sensor = PowerSensor(
            self,
            "inverter_losses",
            name="Inverter losses",
        )

    async def async_setup(self):
        if self.weather_entity_id:
            self.track_state(self.weather_entity_id, self._weather_update)

        self.track_timer(self.SAMPLING_PERIOD, self._timer_callback)
        await super().async_setup()
        self._timer_callback()

    @callback
    def _timer_callback(self):
        sun_zenith, sun_azimuth = sun.zenith_and_azimuth(
            self.astral_observer,
            dt_util.now(),
        )
        elevation = 90 - sun_zenith
        daytime = elevation > -5  # roughly dusk
        if daytime:
            # it is very hard to model the transition night/day since when the sun is low
            # and starts to rise/set the sun energy is very low even if the elevation is relatively high
            # with respect to the plant slope. Here we take a simple approach with
            # slope almost horizontal
            slope = 85
            pv_power = (
                self.peak_power
                * math.cos((slope - elevation) * math.pi / 180)
                * random.randint(98, 102)
                / 100
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

        else:  # night time
            pv_power = 0

        if self._inverter_on:
            consumption_power = (
                self.consumption_baseload_power_w * random.randint(90, 110) / 100
            )
            if daytime:
                p1 = random.randint(1, 100) / 100
                a = random.randint(1, 100) / 100
                if a < (self.consumption_daily_fill_factor / p1):
                    consumption_power += self.consumption_daily_extra_power_w * p1
            total_consumption_power = (
                consumption_power / self.inverter_efficiency
                + self.inverter_zeroload_power
            )
        else:
            consumption_power = 0
            total_consumption_power = 0

        if self.battery_voltage:
            # assume a voltage drop of 5% of the battery voltage at 1C
            battery_resistance = self.battery_voltage / (
                20 * (self.battery_capacity or 100)
            )
            battery_power = total_consumption_power - pv_power
            battery_current = battery_power / self.battery_voltage
            battery_voltage = (
                self.battery_voltage - battery_resistance * battery_current
            )
            battery_current = battery_power / battery_voltage
            if self.battery_charge_sensor:
                battery_charge = self.battery_charge_sensor.update_current(
                    battery_current
                )
                if battery_charge == 0:
                    self._inverter_on = False
                elif battery_charge == self.battery_capacity:
                    pv_power = total_consumption_power
                    battery_current = 0
                elif not self._inverter_on:
                    if battery_charge > (self.battery_capacity * 0.1):
                        self._inverter_on = True
            else:
                self._inverter_on = True
            if self.battery_voltage_sensor:
                self.battery_voltage_sensor.update(round(battery_voltage, 2))
            if self.battery_current_sensor:
                self.battery_current_sensor.update(round(battery_current, 2))

        self.pv_power_simulator_sensor.update_safe(round(pv_power, 2))
        self.consumption_sensor.update_safe(round(consumption_power, 2))
        self.inverter_losses_sensor.update_safe(
            round(total_consumption_power - consumption_power, 2)
        )

    @callback
    def _weather_update(self, event: "Event[EventStateChangedData] | Controller.Event"):
        if state := event.data["new_state"]:
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
