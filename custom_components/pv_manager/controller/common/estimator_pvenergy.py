from collections import deque
import dataclasses
import datetime as dt
import typing

from astral import sun

from .. import helpers
from .estimator import Estimator, Observation, ObservedEnergy

if typing.TYPE_CHECKING:
    from .estimator import EstimatorConfig


@dataclasses.dataclass(slots=True)
class WeatherSample:
    time: dt.datetime
    time_ts: float
    condition: str | None
    temperature: float | None  # °C
    cloud_coverage: float | None  # %
    visibility: float | None  # km
    next: "WeatherSample | None" = None

    def as_dict(self):
        return {
            "time": self.time.isoformat(),
            "condition": self.condition,
            "temperature": self.temperature,
            "cloud_coverage": self.cloud_coverage,
            "visibility": self.visibility,
        }


@dataclasses.dataclass(slots=True)
class ObservedPVEnergy(ObservedEnergy):
    """PV energy/power history data extraction. This sample is used to build energy production
    in a time window (1 hour by design) by querying either a PV power sensor or a PV energy sensor.
    Building from PV power should be preferrable due to the 'failable' nature of energy accumulation.
    """

    weather: WeatherSample | None

    sun_azimuth: float
    """Position of the sun (at mid sample interval)"""
    sun_zenith: float
    """Position of the sun (at mid sample interval)"""

    SUN_NOT_SET = -360

    def __init__(
        self,
        observation: Observation,
        sampling_interval_ts: int,
        weather: WeatherSample | None,
    ):
        ObservedEnergy.__init__(self, observation, sampling_interval_ts)
        self.weather = weather
        self.sun_azimuth = self.sun_zenith = self.SUN_NOT_SET

    @property
    def cloud_coverage(self):
        return self.weather.cloud_coverage if self.weather else None


class Estimator_PVEnergy(Estimator):
    """
    Base class for estimator implementations based off different approaches.
    Beside the current HeuristicEstimator we should think about using neural networks for implementation.
    At the time, lacking any real specialization, the generalization of this class is pretty basic and likely unstable.
    """

    __slots__ = (
        "astral_observer",
        "weather_history",
        "weather_forecasts",
        "_noon_ts",
        "_sunrise_ts",
        "_sunset_ts",
    )

    def __init__(
        self,
        *,
        astral_observer: "sun.Observer",
        tzinfo: dt.tzinfo,
        **kwargs: "typing.Unpack[EstimatorConfig]",
    ):
        Estimator.__init__(
            self,
            tzinfo=tzinfo,
            **kwargs,
        )
        self.astral_observer = astral_observer
        self.weather_history: typing.Final[deque[WeatherSample]] = deque()
        self.weather_forecasts: list[WeatherSample] = []
        self._noon_ts: int = 0
        self._sunrise_ts: int = 0
        self._sunset_ts: int = 0

    # interface: Estimator
    def as_dict(self):
        return super().as_dict() | {
            "weather_history": list(self.weather_history),
            "weather_forecasts": self.weather_forecasts,
        }

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "weather": self.get_weather_at(self.observed_time_ts),
        }

    def _observed_energy_new(self, observation: Observation):
        return ObservedPVEnergy(
            observation,
            self.sampling_interval_ts,
            self.get_weather_at(observation.time_ts),
        )

    """
    def _observed_energy_daystart(self, time_ts: int):
        Estimator._observed_energy_daystart(self, time_ts)
        self._noon_ts = self._today_local_ts + 43200
        # TODO: check this..we assume that the GMT date is the same as local at noon
        today = helpers.datetime_from_epoch(self._noon_ts)
        noon = sun.noon(self.astral_observer, today)
        self._sunrise_ts = int(sun.time_of_transit(self.astral_observer, today, 90.0 + sun.SUN_APPARENT_RADIUS, sun.SunDirection.RISING).timestamp())
        self._sunset_ts = int(sun.time_of_transit(self.astral_observer, today, 90.0 + sun.SUN_APPARENT_RADIUS, sun.SunDirection.SETTING).timestamp())
    """

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
