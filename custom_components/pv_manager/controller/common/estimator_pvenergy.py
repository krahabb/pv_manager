from collections import deque
from dataclasses import dataclass
import datetime as dt
import typing

from astral import sun

from .estimator import Estimator, EstimatorConfig, ObservedEnergy

if typing.TYPE_CHECKING:
    pass


@dataclass(slots=True)
class WeatherSample:
    time: dt.datetime
    time_ts: float
    condition: str | None
    cloud_coverage: float | None  #  0 -> 1 (100% cloud coverage)
    next: "WeatherSample | None" = None

    def as_dict(self):
        return {
            "time": self.time.isoformat(),
            "condition": self.condition,
            "cloud_coverage": self.cloud_coverage,
        }


@dataclass(slots=True)
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
        time_ts: int,
        sampling_interval_ts: int,
        weather: WeatherSample | None,
    ):
        ObservedEnergy.__init__(self, time_ts, sampling_interval_ts)
        self.weather = weather
        self.sun_azimuth = self.sun_zenith = self.SUN_NOT_SET


class WeatherModel:
    """Base (abstract) class modeling the influence of weather on PV production."""

    __slots__ = ()

    @staticmethod
    def build(model: str | None):
        try:
            return WEATHER_MODELS[model]()
        except KeyError:
            return WeatherModel()

    def as_dict(self):
        return {}

    def get_param(self, *args):
        """Generic inspector for model parameters (state)"""
        return None

    def get_energy_estimate(self, energy: float, weather: WeatherSample):
        return energy

    def update_estimate(self, energy: float, weather: WeatherSample, expected: float):
        pass


class SimpleWeatherModel(WeatherModel):
    """Basic 1st order based on clouds with naive gradient descent."""

    Wc: float
    Wc_max: typing.Final = 0.8  # maximum 80% pv power derate when 100% clouds
    Wc_min: typing.Final = 0.4  # minimum 40% pv power derate when 100% clouds
    Wc_lr: typing.Final = 0.00000005  # 'learning rate'

    __slots__ = ("Wc",)

    def __init__(self):
        self.Wc = (self.Wc_max + self.Wc_min) / 2

    def as_dict(self):
        return {
            "cloud_weight": self.Wc,
        }

    def get_param(self, *args):
        return self.Wc

    def get_energy_estimate(self, energy_max: float, weather: WeatherSample):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            return energy_max * (1 - self.Wc * cloud_coverage)
        return energy_max

    def update_estimate(
        self, energy_max: float, weather: WeatherSample, expected: float
    ):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            error = expected - self.get_energy_estimate(energy_max, weather)
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
    Wc1: float
    Wc3: float
    Wc_max: typing.Final = 0.8  # maximum 80% pv power derate when 100% clouds
    Wc_min: typing.Final = 0.4  # minimum 40% pv power derate when 100% clouds
    Wc_lr: typing.Final = 0.0000000005  # 'learning rate'

    __slots__ = (
        "Wc1",
        "Wc3",
    )

    def __init__(self):
        self.Wc1 = 0
        self.Wc3 = (self.Wc_max + self.Wc_min) / 2

    def as_dict(self):
        return {
            "cloud_weight_1": self.Wc1,
            "cloud_weight_3": self.Wc3,
        }

    def get_param(self, n, *args):
        return self.Wc3 if n else self.Wc1

    def get_energy_estimate(self, energy_max: float, weather: WeatherSample):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage:
            return energy_max * (
                1
                - self.Wc1 * cloud_coverage
                - self.Wc3 * cloud_coverage * cloud_coverage * cloud_coverage
            )
        return energy_max

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


WEATHER_MODELS: dict[str | None, type[WeatherModel]] = {
    None: WeatherModel,
    "simple": SimpleWeatherModel,
    "cubic": CubicWeatherModel,
}


class Estimator_PVEnergyConfig(EstimatorConfig):
    weather_model: typing.NotRequired[str]


class Estimator_PVEnergy(Estimator):
    """
    Base class for estimator implementations based off different approaches.
    Beside the current HeuristicEstimator we should think about using neural networks for implementation.
    At the time, lacking any real specialization, the generalization of this class is pretty basic and likely unstable.
    """

    weather_model: typing.Final[WeatherModel]
    weather_history: typing.Final[deque[WeatherSample]]
    weather_forecasts: list[WeatherSample]

    __slots__ = (
        "astral_observer",
        "weather_model",
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
        **kwargs: "typing.Unpack[Estimator_PVEnergyConfig]",
    ):
        self.astral_observer = astral_observer
        self.weather_model = WeatherModel.build(kwargs.pop("weather_model", None))
        self.weather_history = deque()
        self.weather_forecasts = []
        self._noon_ts: int = 0
        self._sunrise_ts: int = 0
        self._sunset_ts: int = 0
        Estimator.__init__(
            self,
            tzinfo=tzinfo,
            **kwargs,
        )

    # interface: Estimator
    def as_dict(self):
        return super().as_dict() | {
            "weather_model": self.weather_model.as_dict(),
            "weather_history": list(self.weather_history),
            "weather_forecasts": self.weather_forecasts,
        }

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "weather_model": self.weather_model.as_dict(),
            "weather": self.get_weather_at(self.observed_time_ts),
        }

    def _observed_energy_new(self, time_ts: int):
        return ObservedPVEnergy(
            time_ts,
            self.sampling_interval_ts,
            self.get_weather_at(time_ts),
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
