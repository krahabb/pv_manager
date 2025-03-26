from collections import deque
import dataclasses
import datetime as dt
import typing

from .. import helpers

if typing.TYPE_CHECKING:
    pass


@dataclasses.dataclass(slots=True)
class Observation:
    time_ts: float
    value: float

@dataclasses.dataclass(slots=True)
class WeatherHistory:
    time: dt.datetime
    time_ts: float
    temperature: float  # °C
    cloud_coverage: float | None  # %
    visibility: float | None  # km


@dataclasses.dataclass(slots=True)
class ObservationHistory:
    """PV energy/power history data extraction. This sample is used to build energy production
    in a time window (1 hour by design) by querying either a PV power sensor or a PV energy sensor.
    Building from PV power should be preferrable due to the 'failable' nature of energy accumulation.
    """

    time: dt.datetime
    """The sample time start"""

    time_ts: int
    time_next_ts: int

    energy: float
    """The effective accumulated energy considering interpolation at the (time) limits"""

    samples: int
    """Number of samples in the time window (could be seen as a quality indicator of sampling)"""

    weather: WeatherHistory | None

    sun_azimuth: float
    """Position of the sun (at mid sample interval)"""
    sun_zenith: float
    """Position of the sun (at mid sample interval)"""

    SUN_NOT_SET = -360


    def __init__(self, observation: Observation, sampling_interval_ts: int, weather: WeatherHistory | None):
        time_ts = int(observation.time_ts)
        time_ts -= (time_ts % sampling_interval_ts)
        self.time = helpers.datetime_from_epoch(time_ts)
        self.time_ts = time_ts
        self.time_next_ts = time_ts + sampling_interval_ts
        self.energy = 0
        self.samples = 1
        self.weather = weather
        self.sun_azimuth = self.sun_zenith = self.SUN_NOT_SET


class Estimator:
    """
    Base class for estimator implementations based off different approaches.
    Beside the current HeuristicEstimator we should think about using neural networks for implementation.
    At the time, lacking any real specialization, the generalization of this class is pretty basic and likely unstable.
    """

    __slots__ = (
        "sampling_interval_ts",
        "history_duration_ts",
        "observation_duration_ts",
        "maximum_latency_ts",
        "local_offset_ts",
        "observed_samples",
        "weather_samples",
        "history_sample_curr",
        "observation_prev",
        "_today_local_ts",
        "_tomorrow_local_ts",
        "today_energy",
        "forecast_today_energy",
        "tomorrow_forecast_energy",
    )

    def __init__(
        self,
        *,
        sampling_interval_ts: int,
        history_duration_ts: float,
        observation_duration_ts: float,
        maximum_latency_ts: float,
        local_offset_ts: float,
    ):
        assert (
            sampling_interval_ts % 300
        ) == 0, "sampling_interval must be a multiple of 5 minutes"
        self.sampling_interval_ts: typing.Final = sampling_interval_ts
        self.history_duration_ts: typing.Final = history_duration_ts
        self.observation_duration_ts: typing.Final = observation_duration_ts
        self.maximum_latency_ts: typing.Final = maximum_latency_ts
        # offset of local time with respect to UTC for solar day alignment
        self.local_offset_ts: typing.Final = local_offset_ts

        self.observed_samples: typing.Final[deque[ObservationHistory]] = deque()
        self.weather_samples: typing.Final[deque[WeatherHistory]] = deque()
        # do not define here..we're relying on AttributeError for proper initialization
        # self.history_sample_curr = None
        # self.observation_prev = None

        self._today_local_ts = 0
        self._tomorrow_local_ts = 0
        self.today_energy = 0
        self.forecast_today_energy = None
        self.tomorrow_forecast_energy = None

    def get_observed_energy(self) -> tuple[float, float, float]:
        """compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        observed_energy = 0
        for sample in self.observed_samples:
            observed_energy += sample.energy

        return (
            observed_energy,
            self.observed_samples[0].time_ts,
            self.observed_samples[-1].time_next_ts,
        )


    def _history_sample_add(self, history_sample: ObservationHistory):
        pass

    def _new_energy_from_observation(
        self, observation: Observation, delta_time_ts: float
    ):
        # virual
        return 0

    def _new_interpolate_samples(
        self,
        observation_curr: Observation,
        observation_prev: Observation,
        delta_time_ts: float,
        sample_curr: ObservationHistory,
        sample_prev: ObservationHistory,
    ):
        """
        observation_curr: new (current) observation
        observation_prev: previous observation
        delta_time_ts: time between current and previous observation
        sample_prev: history sample ending (to be integrated with energy across history samples boundary)
        sample_curr: new (current) history sample (to be integrated with energy at the boundary)
        """
        # virtual
        return

    def add_observation(self, observation: Observation) -> bool:

        try:
            if observation.time_ts < self.history_sample_curr.time_next_ts:
                delta_time_ts = observation.time_ts - self.observation_prev.time_ts
                if delta_time_ts < self.maximum_latency_ts:
                    self.history_sample_curr.energy += (
                        self._new_energy_from_observation(observation, delta_time_ts)
                    )
                    self.history_sample_curr.samples += 1

                self.observation_prev = observation
                return False
            else:
                history_sample_prev = self.history_sample_curr
                self.history_sample_curr = ObservationHistory(observation, self.sampling_interval_ts, self.get_weather_at(observation.time_ts))
                if self.history_sample_curr.time_ts == history_sample_prev.time_next_ts:
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    delta_time_ts = observation.time_ts - self.observation_prev.time_ts
                    if delta_time_ts < self.maximum_latency_ts:
                        self._new_interpolate_samples(
                            observation,
                            self.observation_prev,
                            delta_time_ts,
                            self.history_sample_curr,
                            history_sample_prev,
                        )

                self.observation_prev = observation
                self.observed_samples.append(history_sample_prev)

                if history_sample_prev.time_ts >= self._tomorrow_local_ts:
                    self._today_local_ts = (
                        history_sample_prev.time_ts
                        - (history_sample_prev.time_ts % 86400)
                        + self.local_offset_ts
                    )
                    self._tomorrow_local_ts = self._today_local_ts + 86400
                    self.today_energy = 0

                self.today_energy += history_sample_prev.energy

                try:
                    observation_min_ts = (
                        history_sample_prev.time_next_ts - self.observation_duration_ts
                    )
                    # check if we can discard it since the next is old enough
                    while self.observed_samples[1].time_ts < observation_min_ts:
                        # We need to update the model with incoming observations but we
                        # don't want this to affect 'current' estimation.
                        # Since estimation is based against old observations up to
                        # old_observation.time_ts we should be safe enough adding the
                        # discarded here since they're now out of the estimation 'observation' window
                        self._history_sample_add(self.observed_samples.popleft())
                except IndexError:
                    # at start when observed_samples is empty
                    return False

                return True

        except AttributeError as e:
            if e.name == "history_sample_curr":
                # expected right at the first call..use this to initialize the state
                # and avoid needless checks on subsequent calls
                self.history_sample_curr = ObservationHistory(observation, self.sampling_interval_ts, self.get_weather_at(observation.time_ts))
                self.observation_prev = observation
                return False
            else:
                raise e

    def add_weather(self, weather: WeatherHistory):
        self.weather_samples.append(weather)

        try:
            weather_min_ts = weather.time_ts - self.history_duration_ts
            # check if we can discard it since the next is old enough
            while self.weather_samples[1].time_ts <= weather_min_ts:
                self.weather_samples.popleft()
        except IndexError:
            pass

    def get_weather_at(self, time_ts: float):

        weather_prev = None
        for weather in self.weather_samples:
            if weather.time_ts > time_ts:
                break
            weather_prev = weather

        return weather_prev


class EnergyObserver(Estimator if typing.TYPE_CHECKING else object):
    """Mixin class to add to the actual Estimator in order to process energy input observations."""

    def _new_energy_from_observation(
        self, observation: Observation, delta_time_ts: float
    ):
        if observation.value >= self.observation_prev.value:
            return observation.value - self.observation_prev.value
        else:
            # assume an energy reset
            return observation.value

    def _new_interpolate_samples(
        self,
        observation_curr: Observation,
        observation_prev: Observation,
        delta_time_ts: float,
        sample_curr: ObservationHistory,
        sample_prev: ObservationHistory,
    ):
        """
        observation_curr: new (current) observation
        observation_prev: previous observation
        delta_time_ts: time between current and previous observation
        sample_prev: history sample ending (to be integrated with energy across history samples boundary)
        sample_curr: new (current) history sample (to be integrated with energy at the boundary)
        """
        delta_energy = observation_curr.value - observation_prev.value
        if delta_energy > 0:
            # The next sample starts with more energy than previous so we interpolate both
            power_avg = delta_energy / delta_time_ts
            sample_prev.energy += power_avg * (
                sample_prev.time_next_ts - observation_prev.time_ts
            )
            sample_curr.energy += power_avg * (
                observation_curr.time_ts - sample_curr.time_ts
            )



class PowerObserver(Estimator if typing.TYPE_CHECKING else object):
    """Mixin class to add to the actual Estimator in order to process power input observations."""

    def _new_energy_from_observation(
        self, observation: Observation, delta_time_ts: float
    ):
        return (self.observation_prev.value + observation.value) * delta_time_ts / 7200

    def _new_interpolate_samples(
        self,
        observation_curr: Observation,
        observation_prev: Observation,
        delta_time_ts: float,
        sample_curr: ObservationHistory,
        sample_prev: ObservationHistory,
    ):
        """
        observation_curr: new (current) observation
        observation_prev: previous observation
        delta_time_ts: time between current and previous observation
        sample_prev: history sample ending (to be integrated with energy across history samples boundary)
        sample_curr: new (current) history sample (to be integrated with energy at the boundary)
        """
        prev_delta_time_ts = sample_prev.time_next_ts - observation_prev.time_ts
        prev_power_next = (
            observation_prev.value
            + ((observation_curr.value - observation_prev.value) * prev_delta_time_ts)
            / delta_time_ts
        )
        sample_prev.energy += (
            (observation_prev.value + prev_power_next) * prev_delta_time_ts / 7200
        )
        next_delta_time_ts = observation_curr.time_ts - sample_curr.time_ts
        sample_curr.energy += (
            (prev_power_next + observation_curr.value) * next_delta_time_ts / 7200
        )
