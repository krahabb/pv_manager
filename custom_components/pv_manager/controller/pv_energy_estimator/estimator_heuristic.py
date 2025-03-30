"""
PV energy estimator based on hourly energy measurement
"""

from collections import deque
import dataclasses
import typing

import astral
import astral.sun

from ...helpers import datetime_from_epoch
from .estimator import Estimator, ObservationHistory, WeatherHistory

if typing.TYPE_CHECKING:
    pass


class TimeSpanEnergyModel:
    samples: list[ObservationHistory]

    sample_max: ObservationHistory

    energy_max: float

    # initial value means 100% cloud_coverage would reduce output by 80%
    Wc: typing.ClassVar = 0.008
    Wc_max: typing.ClassVar = 0.008

    # 'learning rate'
    energy_lr: typing.ClassVar = 0.2
    Wc_lr: typing.ClassVar = 0.0000005

    """
    Trying to implement a very basic linear model where

    energy = energy_max * (1 - Wc * cloud_coverage)

    energy_max is (almost) a constant and physically represent the maximum expected yield for the time span
    Wc is the (unknown) coefficient modeling the effect of clouds

    also:
    - energy_max is 'local' to this instance of samples - i.e. is dependant on the time sample or in other words,
    it depends on sun azimuth, zenith which are more or less stable when relating to this 'EnergyModel' instance.
    It will vary though from day to day as the season goes..
    - Wc is global for the whole model (of EnergyModels)
    """

    def __init__(self, sample: ObservationHistory):
        self.samples = [sample]
        self.sample_max = sample
        cloud_coverage = sample.cloud_coverage
        if cloud_coverage is not None:
            self.energy_max = sample.energy / (
                1 - TimeSpanEnergyModel.Wc * cloud_coverage
            )
        else:
            self.energy_max = sample.energy

    def add_sample(self, sample: ObservationHistory):
        self.samples.append(sample)
        if sample.energy > self.sample_max.energy:
            self.sample_max = sample

        # only estimate params if weather data are available
        if sample.weather:
            energy_estimate = self.get_energy_estimate(sample.weather)
            error = energy_estimate - sample.energy

            cloud_coverage = sample.cloud_coverage
            if cloud_coverage:
                TimeSpanEnergyModel.Wc += (
                    error
                    * (cloud_coverage / self.energy_max)
                    * TimeSpanEnergyModel.Wc_lr
                )
                if TimeSpanEnergyModel.Wc > TimeSpanEnergyModel.Wc_max:
                    TimeSpanEnergyModel.Wc = TimeSpanEnergyModel.Wc_max
                elif TimeSpanEnergyModel.Wc < 0.0:
                    TimeSpanEnergyModel.Wc = 0.0

            self.energy_max -= error * TimeSpanEnergyModel.energy_lr

        if self.energy_max < self.sample_max.energy:
            self.energy_max = self.sample_max.energy

    def get_energy_estimate(self, weather: WeatherHistory):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage is not None:
            return self.energy_max * (1 - TimeSpanEnergyModel.Wc * cloud_coverage)
        return self.energy_max

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

        # TEMP: test with and without weather
        if self.sample_max.weather is None:
            self.energy_max = self.sample_max.energy

    def as_dict(self):
        return {
            "samples": self.samples,
            "sample_max": self.sample_max,
            "energy_max": self.energy_max,
        }


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
    model: dict[int, TimeSpanEnergyModel]

    __slots__ = (
        "astral_observer",
        "history_samples",
        "model",
        "observed_ratio",
        "observed_ratio_ts",
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
        self.observed_ratio = 1
        self.observed_ratio_ts = 0

    def update_estimate(self):
        """Process a new sample trying to update the forecast of energy production."""

        observed_energy, observed_begin_ts, observed_end_ts = self.get_observed_energy()
        # we now have observed_energy generated during observed_duration
        estimated_observed_energy_max, missing = self._get_estimated_energy_max(
            observed_begin_ts, observed_end_ts
        )
        if missing or (estimated_observed_energy_max <= 0):
            # no energy in our model at observation time
            self.observed_ratio = 1
        else:
            # using recent observed energy to 'modulate' prediction (key part of this heuristic estimator)
            self.observed_ratio = observed_energy / estimated_observed_energy_max
        self.observed_ratio_ts = observed_end_ts

        self.today_forecast_energy = self.today_energy + self.get_estimated_energy(
            observed_end_ts, self._tomorrow_local_ts
        )
        self.tomorrow_forecast_energy = self.get_estimated_energy(
            self._tomorrow_local_ts, self._tomorrow_local_ts + 86400
        )

    def get_estimated_energy(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ) -> float:

        weather_forecast = self.get_weather_forecast_at(time_begin_ts)
        if weather_forecast:
            weather_forecast_next = weather_forecast.next
            energy = 0

            time_begin_ts = int(time_begin_ts)
            time_end_ts = int(time_end_ts)
            model_time_ts = time_begin_ts - (time_begin_ts % self.sampling_interval_ts)

            # we 'blend' in recent 'ratio' of energy production with respect to max energy
            # end 'dumb' weather based estimations on the long term. It is hard to say how much
            # this blending should last but the heuristic tells us the, despite weather forecasts,
            # energy production in the recent future should more or less follow the same
            # pattern as in recent observation (i.e. self.observed_ratio) and then as time passes
            # this heuristic could have no sense (say a big cloud will come in in an hour or so...)
            # How much this 'observed_ratio' should stand depends on the weather though with
            # partly_cloudy or variable weather in general being the more erratic.
            observed_ratio = self.observed_ratio
            weight_or = 1  # if time_begin_ts == self.observed_ratio_ts
            weight_or_decay = self.sampling_interval_ts / (
                3600 * 4
            )  # fixed 4 hours decay
            if time_begin_ts > self.observed_ratio_ts:
                weight_or -= (
                    (time_begin_ts - self.observed_ratio_ts) / self.sampling_interval_ts
                ) * weight_or_decay
                if weight_or < weight_or_decay:
                    weight_or = 0

            while time_begin_ts < time_end_ts:
                model_time_next_ts = model_time_ts + self.sampling_interval_ts
                try:
                    model = self.model[model_time_ts % 86400]
                    if weight_or:
                        model_energy = (
                            model.energy_max * observed_ratio * weight_or
                            + model.get_energy_estimate(weather_forecast)
                            * (1 - weight_or)
                        )
                        if weight_or > weight_or_decay:
                            weight_or -= weight_or_decay
                        else:
                            weight_or = 0
                    else:  # save some calc when not blending anymore
                        model_energy = model.get_energy_estimate(weather_forecast)
                    if time_end_ts < model_time_next_ts:
                        energy += model_energy * (time_end_ts - time_begin_ts)
                        break
                    else:
                        energy += model_energy * (model_time_next_ts - time_begin_ts)
                except KeyError:
                    # no energy in model
                    pass

                time_begin_ts = model_time_ts = model_time_next_ts
                if weather_forecast_next:
                    if weather_forecast_next.time_ts <= time_begin_ts:
                        if weather_forecast_next.condition != weather_forecast.condition:
                            # weather condition changing so we immediately 'drop' the
                            # short term energy adjustment (observed_ratio)
                            weight_or = 0
                        weather_forecast = weather_forecast_next
                        weather_forecast_next = weather_forecast.next

            return energy / self.sampling_interval_ts
        else:
            return (
                self._get_estimated_energy_max(time_begin_ts, time_end_ts)[0]
                * self.observed_ratio
            )

    def _get_estimated_energy_max(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):
        """Computes the 'maximum' expected energy in the time window."""
        energy = 0
        missing = False
        time_begin_ts = int(time_begin_ts)
        time_end_ts = int(time_end_ts)
        model_time_ts = time_begin_ts - (time_begin_ts % self.sampling_interval_ts)

        while time_begin_ts < time_end_ts:
            model_time_next_ts = model_time_ts + self.sampling_interval_ts
            try:
                model = self.model[model_time_ts % 86400]
                if time_end_ts < model_time_next_ts:
                    energy += model.energy_max * (time_end_ts - time_begin_ts)
                    break
                else:
                    energy += model.energy_max * (model_time_next_ts - time_begin_ts)
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
                self.model[sample_time_of_day_ts] = TimeSpanEnergyModel(history_sample)

        if self.history_samples[0].time_ts < (
            history_sample.time_ts - self.history_duration_ts
        ):
            discarded_sample = self.history_samples.popleft()
            if discarded_sample.energy:
                sample_time_of_day_ts = discarded_sample.time_ts % 86400
                if self.model[sample_time_of_day_ts].pop_sample(discarded_sample):
                    self.model.pop(sample_time_of_day_ts)
