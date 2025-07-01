from collections import deque
from dataclasses import dataclass
import datetime as dt
from time import time
import typing

from astral import sun
from homeassistant.components import weather
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.helpers import sun as sun_helpers
from homeassistant.util import dt as dt_util

from ..helpers import datetime_from_epoch, validation as hv
from ..helpers.dataattr import DataAttr, DataAttrClass, DataAttrParam
from ..helpers.manager import Manager
from .estimator_energy import EnergyObserverEstimator

if typing.TYPE_CHECKING:
    from typing import Final, NotRequired, Unpack

    from homeassistant.components.energy.types import SolarForecastType
    from homeassistant.core import Event, EventStateChangedData, State

    from .. import const as pmc
    from ..helpers.history import CompressedState

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


@dataclass(slots=True, eq=False)
class WeatherSample:
    time: dt.datetime
    time_ts: float
    condition: str | None
    cloud_coverage: float | None  #  0 -> 1 (100% cloud coverage)
    next: "WeatherSample | None" = None

    @staticmethod
    def from_state(state: "State"):
        attributes = state.attributes
        condition = state.state
        if "cloud_coverage" in attributes:
            cloud_coverage = attributes["cloud_coverage"] / 100
        else:
            cloud_coverage = _WEATHER_CONDITION_TO_CLOUD.get(condition)

        return WeatherSample(
            time=state.last_updated,
            time_ts=state.last_updated_timestamp,
            condition=condition,
            cloud_coverage=cloud_coverage,
        )

    @staticmethod
    def from_compressed_state(state: "CompressedState"):
        attributes = state["a"]
        condition = state["s"]
        if "cloud_coverage" in attributes:
            cloud_coverage = attributes["cloud_coverage"] / 100
        else:
            cloud_coverage = _WEATHER_CONDITION_TO_CLOUD.get(condition)
        time_ts = state["lu"]
        return WeatherSample(
            time=datetime_from_epoch(time_ts),
            time_ts=time_ts,
            condition=condition,
            cloud_coverage=cloud_coverage,
        )

    @staticmethod
    def from_forecast(forecast: dict):
        time = dt_util.as_utc(dt_util.parse_datetime(forecast["datetime"]))  # type: ignore
        condition = forecast.get("condition")
        if "cloud_coverage" in forecast:
            cloud_coverage = forecast["cloud_coverage"] / 100
        else:
            cloud_coverage = _WEATHER_CONDITION_TO_CLOUD.get(condition)

        return WeatherSample(
            time=time,
            time_ts=time.timestamp(),
            condition=condition,
            cloud_coverage=cloud_coverage,
        )

    def as_dict(self):
        return {
            "time": self.time.isoformat(),
            "condition": self.condition,
            "cloud_coverage": self.cloud_coverage,
        }


class WeatherModel(DataAttrClass):
    # TODO: inherit from dataattr class to automatically provide as_dict behavior
    """Base (abstract) class modeling the influence of weather on PV production."""

    def get_param(self, *args):
        """Generic inspector for model parameters (state)"""
        return None

    def get_energy_estimate(self, energy: float, weather: WeatherSample):
        return energy

    def update_estimate(self, energy: float, weather: WeatherSample, expected: float):
        pass


class SimpleWeatherModel(WeatherModel):
    """Basic 1st order based on clouds with naive gradient descent."""

    if typing.TYPE_CHECKING:
        Wc_max: Final
        Wc_min: Final
        Wc_lr: Final

    Wc_max = 0.8  # maximum 80% pv power derate when 100% clouds
    Wc_min = 0.4  # minimum 40% pv power derate when 100% clouds
    Wc_lr = 0.00000005  # 'learning rate'

    Wc: DataAttr[float] = (Wc_max + Wc_min) / 2

    @typing.override
    def get_param(self, *args):
        return self.Wc

    @typing.override
    def get_energy_estimate(self, energy_max: float, weather: WeatherSample):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            return energy_max * (1 - self.Wc * cloud_coverage)
        return energy_max

    @typing.override
    def update_estimate(
        self, energy_max: float, weather: WeatherSample, expected: float
    ):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            error = expected - energy_max * (1 - self.Wc * cloud_coverage)
            self.Wc -= error * cloud_coverage * self.Wc_lr
            if self.Wc > self.Wc_max:
                self.Wc = self.Wc_max
            elif self.Wc < self.Wc_min:
                self.Wc = self.Wc_min


class CubicWeatherModel(WeatherModel):
    """Basic 3rd order estimator based on clouds with naive gradient descent."""

    """
    The reason for an higher order model is that the liner one keeps oscillating and
    that's likely because there's really no linear (enough) relationship between
    sun irradiance and cloud coverage

    2nd order is not convex enough in my opinion so we're testing right 3rd order
    """

    if typing.TYPE_CHECKING:
        Wc_max: Final
        Wc_min: Final
        Wc_lr: Final

    Wc_max = 0.8  # maximum 80% pv power derate when 100% clouds
    Wc_min = 0.4  # minimum 40% pv power derate when 100% clouds
    Wc_lr = 0.0000000005  # 'learning rate'

    Wc1: DataAttr[float] = 0
    Wc3: DataAttr[float] = (Wc_max + Wc_min) / 2

    @typing.override
    def get_param(self, n, *args):
        return self.Wc3 if n else self.Wc1

    @typing.override
    def get_energy_estimate(self, energy_max: float, weather: WeatherSample):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            return energy_max * (
                1
                - self.Wc1 * cloud_coverage
                - self.Wc3 * cloud_coverage * cloud_coverage * cloud_coverage
            )
        return energy_max

    @typing.override
    def update_estimate(
        self, energy_max: float, weather: WeatherSample, expected: float
    ):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            cloud_coverage_2 = cloud_coverage * cloud_coverage
            cloud_coverage_3 = cloud_coverage_2 * cloud_coverage
            estimated = energy_max * (
                1 - self.Wc1 * cloud_coverage - self.Wc3 * cloud_coverage_3
            )
            error = expected - estimated
            dWc1 = error * cloud_coverage * self.Wc_lr
            self.Wc1 -= dWc1
            self.Wc3 -= dWc1 * cloud_coverage_2
            # constraining the Wc vector is rather complex...
            # We'll take a 'convenient' approach by actually limiting the highest order
            # and eventually cap the lower ones in a simpler way
            if self.Wc3 > self.Wc_max:
                self.Wc3 = self.Wc_max
            elif self.Wc3 < self.Wc_min:
                self.Wc3 = self.Wc_min
            if self.Wc1 < 0:
                self.Wc1 = 0
            else:
                wc_max = self.Wc_max - self.Wc3
                if self.Wc1 > wc_max:
                    self.Wc1 = wc_max


class PVEnergyEstimator(EnergyObserverEstimator):
    """
    Base class for estimator implementations based off different approaches.
    Beside the current HeuristicEstimator we should think about using neural networks for implementation.
    At the time, lacking any real specialization, the generalization of this class is pretty basic and likely unstable.
    """

    class Sample(EnergyObserverEstimator.Sample):
        """PV energy/power history data extraction. This sample is used to build energy production
        in a time window (1 hour by design) by querying either a PV power sensor or a PV energy sensor.
        Building from PV power should be preferrable due to the 'failable' nature of energy accumulation.
        """

        SUN_NOT_SET = -360

        weather: DataAttr[WeatherSample | None]
        sun_azimuth: DataAttr[float] = SUN_NOT_SET
        """Position of the sun (at mid sample interval)"""
        sun_zenith: DataAttr[float] = SUN_NOT_SET
        """Position of the sun (at mid sample interval)"""

        def __init__(self, time_ts: float, estimator: "PVEnergyEstimator", /):
            EnergyObserverEstimator.Sample.__init__(self, time_ts, estimator)
            self.weather = estimator.get_weather_at(time_ts)

    class Forecast(EnergyObserverEstimator.Forecast):

        weather: DataAttr[WeatherSample | None, DataAttrParam.hide] = None

    if typing.TYPE_CHECKING:

        class Config(EnergyObserverEstimator.Config):
            weather_entity_id: NotRequired[str]
            weather_model: NotRequired[str]

        class Args(EnergyObserverEstimator.Args):
            config: "PVEnergyEstimator.Config"

        config: Config  # (override base typehint)
        forecasts: Final[list[Forecast]]  # type: ignore (override base typehint)
        _forecasts_recycle: Final[list[Forecast]]  # type: ignore override (override base typehint)

        weather_entity_id: str
        weather_model: Final[WeatherModel]
        weather_history: Final[deque[WeatherSample]]
        weather_forecasts: list[WeatherSample]

        _solar_forecast: SolarForecastType

    DEFAULT_NAME = "PV energy estimator"

    WEATHER_MODELS: typing.Final[dict[str | None, type[WeatherModel]]] = {
        None: WeatherModel,
        "simple": SimpleWeatherModel,
        "cubic": CubicWeatherModel,
    }

    _SLOTS_ = (
        "astral_observer",
        "weather_entity_id",
        "weather_model",
        "weather_history",
        "weather_forecasts",
        "_solar_forecast",
        "_noon_ts",
        "_sunrise_ts",
        "_sunset_ts",
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None", /) -> "pmc.ConfigSchema":
        _config = config or {"weather_model": "simple"}
        return super().get_config_schema(config) | {
            hv.opt_config("weather_entity_id", _config): hv.weather_entity_selector(),
            hv.opt_config("weather_model", _config): cls.weather_model_selector(),
        }

    @staticmethod
    def weather_model_selector():
        return hv.select_selector(
            options=[
                model_name
                for model_name in PVEnergyEstimator.WEATHER_MODELS.keys()
                if model_name
            ],
        )

    @staticmethod
    def weather_model_build(model: str | None):
        try:
            return PVEnergyEstimator.WEATHER_MODELS[model]()
        except KeyError:
            return WeatherModel()

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):
        location, elevation = sun_helpers.get_astral_location(Manager.hass)
        self.astral_observer = sun.Observer(
            location.latitude, location.longitude, elevation
        )
        config = kwargs["config"]
        self.weather_entity_id = config.get("weather_entity_id", "")
        self.weather_model = self.__class__.weather_model_build(
            config.get("weather_model", None)
        )
        self.weather_history = deque()
        self.weather_forecasts = []
        self._noon_ts: int = 0
        self._sunrise_ts: int = 0
        self._sunset_ts: int = 0
        super().__init__(id, **kwargs)

    # interface: Estimator
    async def async_start(self):
        await super().async_start()
        if self.weather_entity_id:
            self.track_state(
                self.weather_entity_id,
                self._async_weather_update,
                PVEnergyEstimator.HassJobType.Coroutinefunction,
            )

    def as_diagnostic_dict(self):
        return super().as_diagnostic_dict() | {
            "weather_entity_id": self.weather_entity_id,
            "weather_model": self.weather_model.as_formatted_dict(),
            "weather_history": list(self.weather_history),
            "weather_forecasts": self.weather_forecasts,
        }

    def as_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().as_state_dict() | {
            "weather_model": self.weather_model.as_formatted_dict(),
            "weather": self.get_weather_at(self.estimation_time_ts),
        }

    def _observed_energy_daystart(self, time_ts: int):
        try:
            del self._solar_forecast
        except AttributeError:
            pass
        return super()._observed_energy_daystart(time_ts)

    @typing.override
    def _history_entities(self) -> "EnergyObserverEstimator.HistoryEntitiesDesc":
        return super()._history_entities() | {
            self.weather_entity_id: self._history_process_weather_entity_id
        }

    def _history_process_weather_entity_id(self, state: "CompressedState", callback, /):
        self.add_weather(WeatherSample.from_compressed_state(state))

    # interface: self
    def add_weather(self, weather: WeatherSample):
        self.weather_history.append(weather)
        try:
            weather_min_ts = weather.time_ts - self.history_duration_ts
            # check if we can discard it since the next is old enough
            while self.weather_history[1].time_ts <= weather_min_ts:
                self.weather_history.popleft()
        except IndexError:
            pass

    def get_weather_at(self, time_ts: float):
        weather_prev = None
        for weather in self.weather_history:
            if weather.time_ts > time_ts:
                break
            weather_prev = weather
        return weather_prev

    def set_weather_forecasts(self, weather_forecasts: list[WeatherSample]):
        self.weather_forecasts = weather_forecasts
        # Setup internal linked-list used to optimize sequential scan of forecasts
        # Assume WeatherHistory(s) are initialized with next = None
        if len(weather_forecasts) > 1:
            prev = weather_forecasts[0]
            for weather in weather_forecasts:
                prev.next = weather
                prev = weather
        self.update_estimate()

    def get_weather_forecast_at(self, time_ts: float):
        try:
            weather_prev = self.weather_forecasts[0]
            for weather in self.weather_forecasts:
                if weather.time_ts > time_ts:
                    break
                weather_prev = weather

            return weather_prev
        except IndexError:
            # no forecasts available
            try:
                # go with 'current' weather
                return self.weather_history[-1]
            except IndexError:
                # we should almost always have one (or not?)
                return None

    async def _async_weather_update(
        self, event: "Event[EventStateChangedData] | PVEnergyEstimator.Event"
    ):
        try:
            self.add_weather(WeatherSample.from_state(event.data["new_state"]))  # type: ignore
            forecasts: list[WeatherSample] = []
            try:
                response = await Manager.hass.services.async_call(
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
                    WeatherSample.from_forecast(f)
                    for f in response[self.weather_entity_id][  # type:ignore
                        "forecast"
                    ]
                ]

            except Exception as e:
                self.log_exception(self.DEBUG, e, "requesting hourly weather forecasts")

            try:
                response = await Manager.hass.services.async_call(
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
                    WeatherSample.from_forecast(f)
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
                                daily_forecast_prev = daily_weather_forecasts[index - 1]
                                daily_forecast_prev.time_ts = (
                                    last_hourly_forecast_end_ts
                                )
                                daily_forecast_prev.time = datetime_from_epoch(
                                    last_hourly_forecast_end_ts
                                )
                                forecasts.append(daily_forecast_prev)

                            forecasts += daily_weather_forecasts[index:]
                            break

                    else:
                        forecasts = daily_weather_forecasts

            except Exception as e:
                self.log_exception(self.DEBUG, e, "requesting daily weather forecasts")

            if forecasts:
                self.set_weather_forecasts(forecasts)

        except Exception as e:
            self.log_exception(self.WARNING, e, "_async_weather_update")

    def get_solar_forecast(self) -> "SolarForecastType":
        """Returns the forecasts array for HA energy integration"""
        try:
            wh_hours = self._solar_forecast["wh_hours"]
            # wh_hours is cached for the day since we want to 'preserve' past forecasts
            # for the HA energy dashboard. Our get_estimated_energy in fact, when invoked on the past
            # would return a different estimate since that contains updated data from the past itself
            # This way, when HA requests to refresh the forecasts would get 'stable values for past hours
        except AttributeError:
            # on demand
            wh_hours = {}
            self._solar_forecast = {"wh_hours": wh_hours}

        # with current implementation we can only estimate forward time
        now = dt_util.now().replace(minute=0, second=0, microsecond=0) + dt.timedelta(
            hours=1
        )
        time_ts = int(now.astimezone(dt_util.UTC).timestamp())
        for i in range(48):
            time_next_ts = time_ts + 3600
            wh_hours[datetime_from_epoch(time_ts, dt_util.UTC).isoformat()] = (
                self.get_estimated_energy(time_ts, time_next_ts)
            )
            time_ts = time_next_ts

        return self._solar_forecast
