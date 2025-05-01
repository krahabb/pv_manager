from collections import deque
import datetime as dt
import typing

from .estimator import Estimator, ObservedEnergy

if typing.TYPE_CHECKING:
    from .estimator import EstimatorConfig


class EnergyModel:
    """Consumption estimator focusing on base load and average energy consumption."""

    samples: list[ObservedEnergy]

    energy_min: float  # base load in this time frame
    energy_avg: float  # average energy in this time frame
    __slots__ = (
        "samples",
        "energy_min",
        "energy_avg",
    )

    def __init__(self, sample: ObservedEnergy):
        self.samples = [sample]
        self.energy_min = sample.energy
        self.energy_avg = sample.energy

    def add_sample(self, sample: ObservedEnergy):
        self.samples.append(sample)
        if self.energy_min > sample.energy:
            self.energy_min = sample.energy
        self.energy_avg = self.energy_avg * 0.5 + sample.energy * 0.5

    def pop_sample(self, sample: ObservedEnergy):
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


class HeuristicConsumptionEstimator(Estimator):
    """
    Proof-of-concept of an estimator model based on some heuristics:

    """

    history_samples: typing.Final[deque[ObservedEnergy]]
    model: typing.Final[dict[int, EnergyModel]]

    __slots__ = (
        "history_samples",
        "model",
        "observed_ratio",
    )

    def __init__(
        self,
        *,
        tzinfo: "dt.tzinfo",
        **kwargs: "typing.Unpack[EstimatorConfig]",
    ):
        Estimator.__init__(
            self,
            tzinfo=tzinfo,
            **kwargs,
        )
        self.history_samples = deque()
        self.model = {}
        self.observed_ratio: float = 1

    # interface: Estimator
    def as_dict(self):
        return super().as_dict() | {"model": self.model}

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "observed_ratio": self.observed_ratio,
        }

    def update_estimate(self):
        """Process a new sample trying to update the forecast of energy production."""

        """
        We use recent observations to adjust the forecast implementing a kind of short-term linear auto-regressive model.
        The observed_ratio is a good indication of near future energy production.

        Here we're taking an approach similar to PV energy heuristic but simplified.
        """

        sum_energy_max = 0
        sum_observed_weighted = 0
        try:
            for observed_energy in self.observed_samples:
                model = self.model[observed_energy.time_ts % 86400]
                sum_energy_max += model.energy_avg
                sum_observed_weighted += observed_energy.energy - model.energy_avg
            self.observed_ratio = 1 + (sum_observed_weighted / sum_energy_max)
        except (KeyError, ZeroDivisionError):
            # no data or invalid
            self.observed_ratio = 1

        if self.on_update_estimate:
            self.on_update_estimate(self)

    def get_estimated_energy(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):

        energy = 0

        time_begin_ts = int(time_begin_ts)
        time_end_ts = int(time_end_ts)
        model_time_ts = time_begin_ts - (time_begin_ts % self.sampling_interval_ts)

        # We 'blend' in recent 'ratio' of energy production with respect to avg energy.
        # It is hard to say how much
        observed_ratio = self.observed_ratio
        weight_or = 1  # if time_begin_ts == self.observed_ratio_ts
        weight_or_decay = self.sampling_interval_ts / (3600 * 2)  # fixed 2 hours decay
        if time_begin_ts > self.observed_time_ts:
            weight_or -= (
                (time_begin_ts - self.observed_time_ts) / self.sampling_interval_ts
            ) * weight_or_decay
            if weight_or < weight_or_decay:
                weight_or = 0

        while time_begin_ts < time_end_ts:
            model_time_next_ts = model_time_ts + self.sampling_interval_ts
            try:
                model = self.model[model_time_ts % 86400]
                if weight_or:
                    model_energy = model.energy_avg * (
                        observed_ratio * weight_or + (1 - weight_or)
                    )
                    if weight_or > weight_or_decay:
                        weight_or -= weight_or_decay
                    else:
                        weight_or = 0
                else:  # save some calc when not blending anymore
                    model_energy = model.energy_avg
                if time_end_ts < model_time_next_ts:
                    energy += model_energy * (time_end_ts - time_begin_ts)
                    break
                else:
                    energy += model_energy * (model_time_next_ts - time_begin_ts)
            except KeyError:
                # no energy in model
                pass

            time_begin_ts = model_time_ts = model_time_next_ts

        return energy / self.sampling_interval_ts

    def get_estimated_energy_max(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):
        """Computes the 'maximum' expected energy in the time window."""
        energy = 0
        time_begin_ts = int(time_begin_ts)
        time_end_ts = int(time_end_ts)
        model_time_ts = time_begin_ts - (time_begin_ts % self.sampling_interval_ts)

        while time_begin_ts < time_end_ts:
            model_time_next_ts = model_time_ts + self.sampling_interval_ts
            try:
                model = self.model[model_time_ts % 86400]
                if time_end_ts < model_time_next_ts:
                    energy += model.energy_avg * (time_end_ts - time_begin_ts)
                    break
                else:
                    energy += model.energy_avg * (model_time_next_ts - time_begin_ts)
            except KeyError:
                # no energy in model
                pass

            time_begin_ts = model_time_ts = model_time_next_ts

        return energy / self.sampling_interval_ts

    def get_estimated_energy_min(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):
        """Computes the 'maximum' expected energy in the time window."""
        energy = 0
        time_begin_ts = int(time_begin_ts)
        time_end_ts = int(time_end_ts)
        model_time_ts = time_begin_ts - (time_begin_ts % self.sampling_interval_ts)

        while time_begin_ts < time_end_ts:
            model_time_next_ts = model_time_ts + self.sampling_interval_ts
            try:
                model = self.model[model_time_ts % 86400]
                if time_end_ts < model_time_next_ts:
                    energy += model.energy_min * (time_end_ts - time_begin_ts)
                    break
                else:
                    energy += model.energy_min * (model_time_next_ts - time_begin_ts)
            except KeyError:
                # no energy in model
                pass

            time_begin_ts = model_time_ts = model_time_next_ts

        return energy / self.sampling_interval_ts

    def _observed_energy_history_add(self, history_sample: ObservedEnergy):

        if history_sample.energy:
            self.history_samples.append(history_sample)

            try:
                model = self.model[history_sample.time_ts % 86400]
                model.add_sample(history_sample)
            except KeyError as e:
                self.model[history_sample.time_ts % 86400] = model = EnergyModel(
                    history_sample
                )

        # flush history
        history_min_ts = history_sample.time_ts - self.history_duration_ts
        try:
            while self.history_samples[0].time_ts < history_min_ts:
                discarded_sample = self.history_samples.popleft()
                sample_time_of_day_ts = discarded_sample.time_ts % 86400
                model = self.model[sample_time_of_day_ts]
                if model.pop_sample(discarded_sample):
                    self.model.pop(sample_time_of_day_ts)
        except IndexError:
            # history empty
            pass
