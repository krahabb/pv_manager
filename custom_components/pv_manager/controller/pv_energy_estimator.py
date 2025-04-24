"""
Controller for pv energy production estimation
"""

import datetime as dt
import enum
import typing

import astral
import astral.sun
from homeassistant import const as hac
from homeassistant.components import weather
from homeassistant.components.recorder import history
from homeassistant.core import callback
from homeassistant.helpers import sun as sun_helpers
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    DistanceConverter,
    TemperatureConverter,
)

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..sensor import DiagnosticSensor, Sensor
from .common.estimator_pvenergy_heuristic import (
    Estimator_PVEnergy_Heuristic,
    TimeSpanEnergyModel,
    WeatherSample,
)

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State

    from ..helpers.entity import EntityArgs


class ControllerConfig(controller.EnergyEstimatorControllerConfig):
    weather_entity_id: typing.NotRequired[str]
    """The entity used for weather forecast in the system"""


class EntryConfig(ControllerConfig, pmc.EntityConfig):
    """TypedDict for ConfigEntry data"""


class DiagnosticSensorsEnum(enum.StrEnum):
    observed_ratio = enum.auto()
    weather_cloud_constant = enum.auto()


class Controller(controller.EnergyEstimatorController[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    estimator: Estimator_PVEnergy_Heuristic

    __slots__ = (
        # configuration
        "weather_entity_id",
        # state
        "weather_state",
    )

    # interface: EnergyEstimatorController
    @staticmethod
    def get_config_entry_schema(config: EntryConfig | None) -> pmc.ConfigSchema:
        _config = config or {
            "name": "PV energy estimation",
        }
        return (
            hv.entity_schema(_config)
            | {
                hv.opt_config("weather_entity_id", _config): hv.weather_selector(),
            }
            | controller.EnergyEstimatorController.get_config_entry_schema(config)
        )

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):

        location, elevation = sun_helpers.get_astral_location(hass)

        super().__init__(
            hass,
            config_entry,
            Estimator_PVEnergy_Heuristic,
            astral_observer=astral.sun.Observer(
                location.latitude, location.longitude, elevation
            ),
        )

        self.weather_entity_id = self.config.get("weather_entity_id")
        self.weather_state = None

    async def async_init(self):
        await super().async_init()
        if self.weather_entity_id:
            await self.async_track_state_update(
                self.weather_entity_id, self._async_weather_update
            )

    def _create_diagnostic_entities(self):
        sensors = self.entities[Sensor.PLATFORM]
        for diagnostic_sensor_enum in DiagnosticSensorsEnum:
            if diagnostic_sensor_enum not in sensors:
                DiagnosticSensor(self, diagnostic_sensor_enum)

    # interface: EnergyEstimatorController
    def _update_estimate(self, estimator: Estimator_PVEnergy_Heuristic):

        sensors = self.entities[Sensor.PLATFORM]

        if DiagnosticSensorsEnum.observed_ratio in sensors:
            sensors[DiagnosticSensorsEnum.observed_ratio].update_safe(
                estimator.observed_ratio
            )
        if DiagnosticSensorsEnum.weather_cloud_constant in sensors:
            sensors[DiagnosticSensorsEnum.weather_cloud_constant].update_safe(
                TimeSpanEnergyModel.Wc
            )

        super()._update_estimate(estimator)

    def _restore_history(self, history_start_time: dt.datetime):
        if self._restore_history_exit:
            return

        if self.weather_entity_id:
            weather_states = history.state_changes_during_period(
                self.hass,
                history_start_time,
                None,
                self.weather_entity_id,
                no_attributes=False,
            )
            for weather_state in weather_states[self.weather_entity_id]:
                if self._restore_history_exit:
                    return
                try:
                    self.estimator.add_weather(
                        Controller._weather_from_state(weather_state)
                    )
                except:
                    pass

        super()._restore_history(history_start_time)

    # interface: self
    async def _async_weather_update(self, weather_state: "State | None"):
        self.weather_state = weather_state
        try:
            if weather_state:
                self.estimator.add_weather(
                    Controller._weather_from_state(weather_state)
                )

                forecasts: list[WeatherSample] = []
                try:
                    response = await self.hass.services.async_call(
                        "weather",
                        "get_forecasts",
                        service_data={
                            "type": "hourly",
                            "entity_id": self.weather_entity_id,
                        },
                        blocking=True,
                        return_response=True,
                    )
                    forecasts = [
                        self._weather_from_forecast(f)
                        for f in response[self.weather_entity_id][  # type:ignore
                            "forecast"
                        ]
                    ]

                except Exception as e:
                    self.log_exception(
                        self.DEBUG, e, "requesting hourly weather forecasts"
                    )

                try:
                    response = await self.hass.services.async_call(
                        "weather",
                        "get_forecasts",
                        service_data={
                            "type": "daily",
                            "entity_id": self.weather_entity_id,
                        },
                        blocking=True,
                        return_response=True,
                    )
                    daily_weather_forecasts = [
                        self._weather_from_forecast(f)
                        for f in response[self.weather_entity_id][  # type:ignore
                            "forecast"
                        ]
                    ]
                    if daily_weather_forecasts:
                        if forecasts:
                            # We're adding daily forecasts at the end of our (eventual) hourly forecasts
                            # When doing so, we take special care as to not overlap the end of the hourly
                            # list with the beginning of the daily list
                            last_hourly_forecast = forecasts[-1]
                            last_hourly_forecast_end_ts = (
                                last_hourly_forecast.time_ts + 3600
                            )
                            index = 0
                            for daily_forecast in daily_weather_forecasts:

                                if daily_forecast.time_ts < last_hourly_forecast_end_ts:
                                    index += 1
                                    continue

                                if (
                                    daily_forecast.time_ts > last_hourly_forecast_end_ts
                                ) and index:
                                    # this is not the first daily so we add an 'interpolation' between
                                    # the end of the hourly list with the beginning of the daily one
                                    daily_forecast_prev = daily_weather_forecasts[
                                        index - 1
                                    ]
                                    daily_forecast_prev.time_ts = (
                                        last_hourly_forecast_end_ts
                                    )
                                    daily_forecast_prev.time = (
                                        helpers.datetime_from_epoch(
                                            last_hourly_forecast_end_ts
                                        )
                                    )
                                    forecasts.append(daily_forecast_prev)

                                forecasts += daily_weather_forecasts[index:]
                                break

                        else:
                            forecasts = daily_weather_forecasts

                except Exception as e:
                    self.log_exception(
                        self.DEBUG, e, "requesting daily weather forecasts"
                    )

                self.estimator.set_weather_forecasts(forecasts)

        except Exception as e:
            self.log_exception(self.DEBUG, e, "_async_update_weather")

    _WEATHER_CONDITION_TO_CLOUD: typing.Final[dict[str | None, float | None]] = {
        None: None,
        weather.ATTR_CONDITION_CLEAR_NIGHT: 0,
        weather.ATTR_CONDITION_CLOUDY: 100,
        weather.ATTR_CONDITION_EXCEPTIONAL: 80,
        weather.ATTR_CONDITION_FOG: 80,
        weather.ATTR_CONDITION_HAIL: 80,
        weather.ATTR_CONDITION_LIGHTNING: 70,
        weather.ATTR_CONDITION_LIGHTNING_RAINY: 70,
        weather.ATTR_CONDITION_PARTLYCLOUDY: 50,
        weather.ATTR_CONDITION_POURING: 80,
        weather.ATTR_CONDITION_RAINY: 60,
        weather.ATTR_CONDITION_SNOWY: 100,
        weather.ATTR_CONDITION_SNOWY_RAINY: 100,
        weather.ATTR_CONDITION_SUNNY: 0,
        weather.ATTR_CONDITION_WINDY: 0,
        weather.ATTR_CONDITION_WINDY_VARIANT: 0,
    }

    @staticmethod
    def _weather_from_state(weather_state: "State"):
        attributes = weather_state.attributes

        condition = weather_state.state
        if "cloud_coverage" in attributes:
            cloud_coverage = attributes["cloud_coverage"]
        else:
            cloud_coverage = Controller._WEATHER_CONDITION_TO_CLOUD.get(condition)

        if "visibility" in attributes:
            visibility = DistanceConverter.convert(
                attributes["visibility"],
                attributes["visibility_unit"],
                hac.UnitOfLength.KILOMETERS,
            )
        else:
            visibility = None

        return WeatherSample(
            time=weather_state.last_updated,
            time_ts=weather_state.last_updated_timestamp,
            condition=condition,
            temperature=TemperatureConverter.convert(
                float(attributes["temperature"]),
                attributes["temperature_unit"],
                hac.UnitOfTemperature.CELSIUS,
            ),
            cloud_coverage=cloud_coverage,
            visibility=visibility,
        )

    def _weather_from_forecast(self, forecast: dict):
        assert self.weather_state

        time = dt_util.as_utc(dt_util.parse_datetime(forecast["datetime"]))  # type: ignore
        time_ts = time.timestamp()

        weather_attributes = self.weather_state.attributes

        if "temperature" in forecast:
            temperature = TemperatureConverter.convert(
                float(forecast["temperature"]),
                weather_attributes["temperature_unit"],
                hac.UnitOfTemperature.CELSIUS,
            )
        else:
            temperature = None

        condition = forecast.get("condition")
        if "cloud_coverage" in forecast:
            cloud_coverage = forecast["cloud_coverage"]
        else:
            cloud_coverage = self._WEATHER_CONDITION_TO_CLOUD.get(condition)

        return WeatherSample(
            time=time,
            time_ts=time_ts,
            condition=condition,
            temperature=temperature,
            cloud_coverage=cloud_coverage,
            visibility=None,
        )
