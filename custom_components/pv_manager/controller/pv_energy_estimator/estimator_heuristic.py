"""
PV energy estimator based on hourly energy measurement
"""

from collections import deque
import dataclasses
import typing

import astral
import astral.sun

from ...helpers import datetime_from_epoch
from .estimator import Estimator, ObservationHistory

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


class HeuristicEstimator(Estimator):
    """
    Proof-of-concept of an estimator model based on some heuristics:

    The main concept states that the pv production is almost the same in the same time window
    between subsequent days (should the weather be ok i.e. sunny)
    This way seasonal variations over months due to shadows or anything are not being taken care of
    since they're 'already in the data'

    Also, the estimation is adjusted on the fly by observing current energy production over an
    'observation time window' and comparing it against its estimation. The ratio will be used to adjust
    the forecasts for the time ahead.

    Weather could also be considered to provide more stable long term estimations (not implemented atm).

    """

    history_samples: deque[ObservationHistory]
    model: dict[int, EnergyModel]

    __slots__ = (
        "astral_observer",
        "history_samples",
        "model",
        "current_energy_ratio",
    )

    def __init__(
        self,
        *,
        history_duration_ts: float,
        sampling_interval_ts: int,
        observation_duration_ts: float,
        maximum_latency_ts: float,
        astral_observer: "astral.sun.Observer",
    ):
        super().__init__(
            sampling_interval_ts=sampling_interval_ts,
            history_duration_ts=history_duration_ts,
            observation_duration_ts=observation_duration_ts,
            maximum_latency_ts=maximum_latency_ts,
            local_offset_ts=astral_observer.longitude * 4 * 60,
        )

        self.astral_observer = astral_observer

        self.history_samples: typing.Final = deque()
        self.model: typing.Final = {}
        self.current_energy_ratio = 1

    def update_estimate(self):
        """Process a new sample trying to update the forecast of energy production."""

        observed_energy, observed_begin_ts, observed_end_ts = self.get_observed_energy()
        # we now have observed_energy generated during observed_duration

        estimated_observed_energy, missing = self._get_estimated_energy_max(
            observed_begin_ts, observed_end_ts
        )
        if missing or (estimated_observed_energy <= 0):
            # no energy in our model at observation time
            self.current_energy_ratio = 1
        else:
            # using recent observed energy to 'modulate' prediction (key part of this heuristic estimator)
            self.current_energy_ratio = observed_energy / estimated_observed_energy

        self.today_forecast_energy = self.today_energy + self.get_estimated_energy(observed_end_ts, self._tomorrow_local_ts)

    def get_estimated_energy(self, time_begin_ts: float, time_end_ts: float) -> float:
        return self._get_estimated_energy_max(time_begin_ts, time_end_ts)[0] * self.current_energy_ratio

    def _get_estimated_energy_max(self, time_begin_ts: float, time_end_ts: float):

        energy = 0
        missing = False
        time_ts = int(time_begin_ts)
        model_time_ts = time_ts - (time_ts % self.sampling_interval_ts)

        while time_begin_ts < time_end_ts:

            model_time_next_ts = model_time_ts + self.sampling_interval_ts
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

        return energy / self.sampling_interval_ts, missing

    def _history_sample_add(self, history_sample: ObservationHistory):
        self.history_samples.append(history_sample)

        if history_sample.energy:

            history_sample.sun_zenith, history_sample.sun_azimuth = (
                astral.sun.zenith_and_azimuth(
                    self.astral_observer,
                    datetime_from_epoch(
                        (history_sample.time_ts + history_sample.time_next_ts) / 2
                    ),
                )
            )

            sample_time_of_day_ts = history_sample.time_ts % 86400
            try:
                self.model[sample_time_of_day_ts].add_sample(history_sample)
            except KeyError:
                self.model[sample_time_of_day_ts] = EnergyModel(history_sample)

        if self.history_samples[0].time_ts < (
            history_sample.time_ts - self.history_duration_ts
        ):
            discarded_sample = self.history_samples.popleft()
            if discarded_sample.energy:
                sample_time_of_day_ts = discarded_sample.time_ts % 86400
                if self.model[sample_time_of_day_ts].pop_sample(discarded_sample):
                    self.model.pop(sample_time_of_day_ts)
