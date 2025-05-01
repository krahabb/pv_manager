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
from homeassistant.helpers import sun as sun_helpers
from homeassistant.util import dt as dt_util

from .. import const as pmc, helpers
from ..controller import EnergyEstimatorController, EnergyEstimatorControllerConfig
from ..helpers import validation as hv
from ..sensor import DiagnosticSensor, EstimatorDiagnosticSensor
from .common.estimator_pvenergy import WEATHER_MODELS, WeatherSample
from .common.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Final

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State

    from ..helpers.entity import EntityArgs
    from .common.estimator import Estimator


class ControllerConfig(EnergyEstimatorControllerConfig):
    weather_entity_id: typing.NotRequired[str]
    """The entity used for weather forecast in the system"""
    weather_model: typing.NotRequired[str]


class EntryConfig(ControllerConfig, pmc.EntityConfig):
    """TypedDict for ConfigEntry data"""


# TODO: create a global generalization for diagnostic sensors linked to estimator
class ObservedRatioDiagnosticSensor(EstimatorDiagnosticSensor):

    def __init__(self, controller: "Controller", id: str):
        super().__init__(controller, id, controller.estimator)

    def on_estimator_update(self, estimator: HeuristicPVEnergyEstimator):
        self.update_safe(estimator.observed_ratio)


class WeatherModelDiagnosticSensor(EstimatorDiagnosticSensor):

    __slots__ = ("weather_param_index",)

    def __init__(self, controller: "Controller", id: str, index: int):
        self.weather_param_index = index
        super().__init__(controller, id, controller.estimator)

    def on_estimator_update(self, estimator: HeuristicPVEnergyEstimator):
        self.update_safe(estimator.weather_model.get_param(self.weather_param_index))


class DiagnosticDescr:

    id: "Final[str]"
    init: "Final[Callable[[Controller], DiagnosticSensor]]"
    value: "Final[Callable[[Controller], Any]]"

    __slots__ = (
        "id",
        "init",
        "value",
    )

    def __init__(
        self,
        id: str,
        init_func: "Callable[[Controller], DiagnosticSensor]",
        value_func: "Callable[[Controller], Any]" = lambda c: None,
    ):
        self.id = id
        self.init = init_func
        self.value = value_func

    @staticmethod
    def Sensor(id: str, value_func: "Callable[[Controller], Any]"):
        return DiagnosticDescr(
            id,
            lambda c: DiagnosticSensor(c, id, native_value=value_func(c)),
            value_func,
        )

    @staticmethod
    def EstimatorSensor(id: str, estimator_update_func: "Callable[[Estimator], Any]"):
        return DiagnosticDescr(
            id,
            lambda c: EstimatorDiagnosticSensor(
                c, id, c.estimator, estimator_update_func=estimator_update_func
            ),
            lambda c: estimator_update_func(c.estimator),
        )


DIAGNOSTIC_DESCR = {
    "observed_ratio": DiagnosticDescr(
        "observed_ratio", lambda c: ObservedRatioDiagnosticSensor(c, "observed_ratio")
    ),
    "weather_cloud_constant_0": DiagnosticDescr(
        "weather_cloud_constant_0",
        lambda c: WeatherModelDiagnosticSensor(c, "weather_cloud_constant_0", 0),
    ),
    "weather_cloud_constant_1": DiagnosticDescr(
        "weather_cloud_constant_1",
        lambda c: WeatherModelDiagnosticSensor(c, "weather_cloud_constant_1", 1),
    ),
}


class Controller(EnergyEstimatorController[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    estimator: "HeuristicPVEnergyEstimator"

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
            "weather_model": "simple",
        }
        return (
            hv.entity_schema(_config)
            | {
                hv.opt_config("weather_entity_id", _config): hv.weather_selector(),
                hv.opt_config("weather_model", _config): hv.select_selector(
                    options=[
                        model_name for model_name in WEATHER_MODELS.keys() if model_name
                    ],
                ),
            }
            | EnergyEstimatorController.get_config_entry_schema(config)
        )

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        location, elevation = sun_helpers.get_astral_location(hass)
        super().__init__(
            hass,
            config_entry,
            HeuristicPVEnergyEstimator,
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
        diagnostic_entities = self.diagnostic_entities
        for d_e_d in DIAGNOSTIC_DESCR.values():
            if d_e_d.id not in diagnostic_entities:
                d_e_d.init(self)

    # interface: EnergyEstimatorController
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
        weather.ATTR_CONDITION_CLOUDY: 1,
        weather.ATTR_CONDITION_EXCEPTIONAL: 0.8,
        weather.ATTR_CONDITION_FOG: 0.8,
        weather.ATTR_CONDITION_HAIL: 0.8,
        weather.ATTR_CONDITION_LIGHTNING: 0.7,
        weather.ATTR_CONDITION_LIGHTNING_RAINY: 0.7,
        weather.ATTR_CONDITION_PARTLYCLOUDY: 0.5,
        weather.ATTR_CONDITION_POURING: 0.8,
        weather.ATTR_CONDITION_RAINY: 0.6,
        weather.ATTR_CONDITION_SNOWY: 1,
        weather.ATTR_CONDITION_SNOWY_RAINY: 1,
        weather.ATTR_CONDITION_SUNNY: 0,
        weather.ATTR_CONDITION_WINDY: 0,
        weather.ATTR_CONDITION_WINDY_VARIANT: 0,
    }

    @staticmethod
    def _weather_from_state(weather_state: "State"):
        attributes = weather_state.attributes

        condition = weather_state.state
        if "cloud_coverage" in attributes:
            cloud_coverage = attributes["cloud_coverage"] / 100
        else:
            cloud_coverage = Controller._WEATHER_CONDITION_TO_CLOUD.get(condition)

        return WeatherSample(
            time=weather_state.last_updated,
            time_ts=weather_state.last_updated_timestamp,
            condition=condition,
            cloud_coverage=cloud_coverage,
        )

    def _weather_from_forecast(self, forecast: dict):
        time = dt_util.as_utc(dt_util.parse_datetime(forecast["datetime"]))  # type: ignore
        time_ts = time.timestamp()

        condition = forecast.get("condition")
        if "cloud_coverage" in forecast:
            cloud_coverage = forecast["cloud_coverage"] / 100
        else:
            cloud_coverage = Controller._WEATHER_CONDITION_TO_CLOUD.get(condition)

        return WeatherSample(
            time=time,
            time_ts=time_ts,
            condition=condition,
            cloud_coverage=cloud_coverage,
        )
