from collections import deque
import typing

from ..helpers.dataattr import DataAttr
from .estimator_energy import EnergyObserverEstimator

if typing.TYPE_CHECKING:
    from typing import Final, Unpack


class EnergyModel:
    """Consumption estimator focusing on base load and average energy consumption."""

    if typing.TYPE_CHECKING:

        type Sample = "HeuristicConsumptionEstimator.Sample"

        samples: list[Sample]
        energy_min: float  # base load in this time frame
        energy_avg: float  # average energy in this time frame

    __slots__ = (
        "samples",
        "energy_min",
        "energy_avg",
    )

    def __init__(self, sample: "Sample"):
        self.samples = [sample]
        self.energy_min = sample.energy
        self.energy_avg = sample.energy

    def add_sample(self, sample: "Sample"):
        self.samples.append(sample)
        if self.energy_min > sample.energy:
            self.energy_min = sample.energy
        self.energy_avg = self.energy_avg * 0.5 + sample.energy * 0.5

    def pop_sample(self, sample: "Sample"):
        self.samples.remove(sample)
        if self.samples:
            self.energy_min = self.energy_avg = self.samples[0].energy
            for sample in self.samples:
                if self.energy_min > sample.energy:
                    self.energy_min = sample.energy
                self.energy_avg = self.energy_avg * 0.5 + sample.energy * 0.5
            return False
        else:
            # This model will be discarded
            return True

    def as_dict(self):
        return {
            "samples": self.samples,
            "energy_min": self.energy_min,
            "energy_avg": self.energy_avg,
        }


class HeuristicConsumptionEstimator(EnergyObserverEstimator):
    """
    Proof-of-concept of an estimator model based on some heuristics:

    """

    if typing.TYPE_CHECKING:

        class Sample(EnergyObserverEstimator.Sample):
            pass

        class Args(EnergyObserverEstimator.Args):
            pass

        history_samples: Final[deque[Sample]]
        model: Final[dict[int, EnergyModel]]

    DEFAULT_NAME = "Consumption estimator"

    observed_ratio: DataAttr[float] = 1

    _SLOTS_ = (
        "history_samples",
        "model",
    )

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):
        self.history_samples = deque()
        self.model = {}
        super().__init__(id, **kwargs)

    # interface: Estimator
    def as_diagnostic_dict(self):
        return super().as_diagnostic_dict() | {"model": self.model}

    @typing.override
    def update_estimate(self):
        """Process a new sample trying to update the forecast of energy production."""

        """
        We use recent observations to adjust the forecast implementing a kind of short-term linear auto-regressive model.
        The observed_ratio is a good indication of near future energy production.

        Here we're taking an approach similar to PV energy heuristic but simplified.
        """

        sum_energy_max = 0
        sum_observed_weighted = 0
        model = self.model
        try:
            for observed_energy in self.observed_samples:
                _model = model[observed_energy.time_begin_ts % 86400]
                sum_energy_max += _model.energy_avg
                sum_observed_weighted += observed_energy.energy - _model.energy_avg
            self.observed_ratio = 1 + (sum_observed_weighted / sum_energy_max)
        except (KeyError, ZeroDivisionError):
            # no data or invalid
            self.observed_ratio = 1

        super().update_estimate()

    @typing.override
    def _ensure_forecasts(self, count: int, /):
        estimation_time_ts = self.estimation_time_ts
        sampling_interval_ts = self.sampling_interval_ts
        observed_ratio = self.observed_ratio
        forecasts = self.forecasts
        _forecasts_recycle = self._forecasts_recycle
        model = self.model

        time_ts = estimation_time_ts + len(forecasts) * sampling_interval_ts
        time_end_ts = estimation_time_ts + count * sampling_interval_ts
        # We 'blend' in recent 'ratio' of energy production with respect to avg energy.
        # It is hard to say how much
        weight_or_decay = sampling_interval_ts / (3600 * 2)  # fixed 2 hours decay
        weight_or = 1 - (len(forecasts) * weight_or_decay)
        if weight_or < weight_or_decay:
            weight_or = 0

        while time_ts < time_end_ts:
            time_next_ts = time_ts + sampling_interval_ts
            try:
                _f = _forecasts_recycle.pop()
                _f.__init__(time_ts, time_next_ts)
            except IndexError:
                _f = self.__class__.Forecast(time_ts, time_next_ts)

            try:
                _model = model[time_ts % 86400]
                if weight_or:
                    _f.energy = _model.energy_avg * (
                        observed_ratio * weight_or + (1 - weight_or)
                    )
                    if weight_or > weight_or_decay:
                        weight_or -= weight_or_decay
                    else:
                        weight_or = 0
                else:  # save some calc when not blending anymore
                    _f.energy = _model.energy_avg

                _f.energy_min = _model.energy_min
                _f.energy_max = _model.energy_avg
            except KeyError:
                # no energy in model
                pass

            forecasts.append(_f)
            time_ts = time_next_ts

    @typing.override
    def _observed_energy_history_add(self, history_sample: "Sample"):

        if history_sample.energy:
            self.history_samples.append(history_sample)

            try:
                model = self.model[history_sample.time_begin_ts % 86400]
                model.add_sample(history_sample)
            except KeyError as e:
                self.model[history_sample.time_begin_ts % 86400] = model = EnergyModel(
                    history_sample
                )

        # flush history
        history_min_ts = history_sample.time_begin_ts - self.history_duration_ts
        try:
            while self.history_samples[0].time_begin_ts < history_min_ts:
                discarded_sample = self.history_samples.popleft()
                sample_time_of_day_ts = discarded_sample.time_begin_ts % 86400
                model = self.model[sample_time_of_day_ts]
                if model.pop_sample(discarded_sample):
                    del self.model[sample_time_of_day_ts]
        except IndexError:
            # history empty
            pass
