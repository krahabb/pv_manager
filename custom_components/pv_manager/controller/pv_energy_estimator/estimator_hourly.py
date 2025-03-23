"""
PV energy estimator based on hourly energy measurement
"""

from collections import deque
import dataclasses
import datetime as dt
import time
import typing

import astral
import astral.sun

from ...helpers import datetime_from_epoch
from .estimator import Estimator, Observation, ObservationHistory, WeatherHistory

if typing.TYPE_CHECKING:
    pass


@dataclasses.dataclass(slots=True)
class EnergyModel:
    samples: list[ObservationHistory]

    sample_max: ObservationHistory

    def __init__(self, sample: ObservationHistory):
        self.samples = [sample]
        self.sample_max = sample

    def add_sample(self, sample: ObservationHistory):
        self.samples.append(sample)
        if sample.energy > self.sample_max.energy:
            self.sample_max = sample

    def pop_sample(self, sample: ObservationHistory):
        try:
            self.samples.remove(sample)
            if sample == self.sample_max:
                if self.samples:
                    self._recalc()
        except ValueError:
            # sample not in list
            pass
        return len(self.samples) == 0

    def _recalc(self):
        self.sample_max = self.samples[0]
        for sample in self.samples:
            if sample.energy > self.sample_max.energy:
                self.sample_max = sample


@dataclasses.dataclass
class HourlyEnergyEstimation:
    energy: float

    def __init__(self):
        self.energy = 0


class HourlyEstimator(Estimator):

    history_samples: deque[ObservationHistory]
    weather_samples: deque[WeatherHistory]
    model: dict[int, EnergyModel]
    estimations: list[HourlyEnergyEstimation]

    __slots__ = (
        "history_duration_ts",
        "history_sampling_interval_ts",
        "history_samples_per_day",
        "astral_observer",
        "history_samples",
        "weather_samples",
        "model",
        "estimations",
    )

    def __init__(
        self,
        history_duration_ts: float,
        history_sampling_interval_ts: int,
        observation_duration_ts: float,
        maximum_latency_ts: float,
        astral_observer: "astral.sun.Observer",
    ):
        self.history_duration_ts: typing.Final = history_duration_ts
        assert (history_sampling_interval_ts % 300) == 0  # multiples of 5 minutes
        self.history_sampling_interval_ts: typing.Final = history_sampling_interval_ts
        self.history_samples_per_day = int(86400 / history_sampling_interval_ts)

        self.astral_observer = astral_observer

        self.history_samples: typing.Final = deque()
        self.weather_samples: typing.Final = deque()

        self.estimations: typing.Final = [HourlyEnergyEstimation() for _t in range(24)]
        self.model: typing.Final = {}
        super().__init__(
            observation_duration_ts=observation_duration_ts,
            maximum_latency_ts=maximum_latency_ts,
        )

    def process_observation(self, observation: "Observation") -> bool:
        """Process a new sample trying to update the forecast of energy production."""

        if not super().process_observation(observation):
            return False

        observed_energy, observed_begin_ts, observed_end_ts = self.get_observed_energy()
        # we now have observed_energy generated during observed_duration

        estimated_observed_energy, missing = self._get_estimated_energy_max(
            observed_begin_ts, observed_end_ts
        )
        if missing or (estimated_observed_energy <= 0):
            # no energy in our model at observation time
            return False

        ratio = observed_energy / estimated_observed_energy

        estimation_time_begin_ts = observation.time_ts
        for _t in range(len(self.estimations)):
            estimation_time_end_ts = estimation_time_begin_ts + 3600
            estimated_energy, missing = self._get_estimated_energy_max(
                estimation_time_begin_ts, estimation_time_end_ts
            )
            self.estimations[_t].energy = estimated_energy * ratio

            estimation_time_begin_ts = estimation_time_end_ts

        return True

    def add_weather(self, weather: WeatherHistory):
        self.weather_samples.append(weather)

        weather_min_ts = weather.time_ts - self.history_duration_ts

        if self.weather_samples[0].time_ts > weather_min_ts:
            return

        # check if we can discard it since the next is old enough
        while self.weather_samples[1].time_ts <= weather_min_ts:
            self.weather_samples.popleft()

    def _get_estimated_energy_max(self, time_begin_ts: float, time_end_ts: float):

        energy = 0
        missing = False
        time_ts = int(time_begin_ts)
        model_time_ts = time_ts - (time_ts % self.history_sampling_interval_ts)

        while time_begin_ts < time_end_ts:

            model_time_next_ts = model_time_ts + self.history_sampling_interval_ts
            try:
                model = self.model[model_time_ts % 86400]
                if time_end_ts < model_time_next_ts:
                    energy += model.sample_max.energy * (time_end_ts - time_begin_ts)
                else:
                    energy += model.sample_max.energy * (
                        model_time_next_ts - time_begin_ts
                    )
            except KeyError:
                # no energy in model
                missing = True

            time_begin_ts = model_time_ts = model_time_next_ts

        return energy / self.history_sampling_interval_ts, missing

    def _history_sample_create(self, observation: Observation) -> ObservationHistory:
        time_ts = int(observation.time_ts)
        sample_time_ts = time_ts - (time_ts % self.history_sampling_interval_ts)
        return ObservationHistory(
            time=datetime_from_epoch(sample_time_ts),
            time_ts=sample_time_ts,
            duration_ts=self.history_sampling_interval_ts,
        )

    def _history_sample_add(self, history_sample: ObservationHistory):
        self.history_samples.append(history_sample)
        if history_sample.energy:
            sample_mid_time_ts = (
                history_sample.time_ts + history_sample.time_next_ts
            ) / 2
            sample_mid_time = datetime_from_epoch(sample_mid_time_ts)
            history_sample.sun_zenith, history_sample.sun_azimuth = (
                astral.sun.zenith_and_azimuth(self.astral_observer, sample_mid_time)
            )
            model_time_ts = history_sample.time_ts % 86400
            try:
                self.model[model_time_ts].add_sample(history_sample)
            except KeyError:
                self.model[model_time_ts] = EnergyModel(history_sample)

        if self.history_samples[0].time_ts < (
            history_sample.time_ts - self.history_duration_ts
        ):
            discarded_sample = self.history_samples.popleft()
            if discarded_sample.energy:
                model_time_ts = discarded_sample.time_ts % 86400
                if self.model[model_time_ts].pop_sample(discarded_sample):
                    self.model.pop(model_time_ts)
