"""
PV energy estimator based on hourly energy measurement
"""

from collections import deque
import typing

import astral
import astral.sun

from ...helpers import datetime_from_epoch
from .estimator_pvenergy import Estimator_PVEnergy, ObservedPVEnergy, WeatherSample

if typing.TYPE_CHECKING:
    import datetime as dt
    from .estimator import EstimatorConfig


class TimeSpanEnergyModel:
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

    samples: list[ObservedPVEnergy]

    sample_max: ObservedPVEnergy

    energy_max: float

    # initial value means 100% cloud_coverage would reduce output by 60%
    Wc: typing.ClassVar = 0.006
    Wc_max: typing.ClassVar = 0.008 # maximum 80% pv power derate when 100% clouds
    Wc_min: typing.ClassVar = 0.004 # minimum 40% pv power derate when 100% clouds
    Wc_lr: typing.ClassVar = 0.00000005  # 'learning rate'

    __slots__ = (
        "samples",
        "sample_max",
        "energy_max",
    )

    def __init__(self, sample: ObservedPVEnergy):
        self.samples = [sample]
        self.sample_max = sample
        self.energy_max = sample.energy

    def as_dict(self):
        return {
            "samples": self.samples,
            "sample_max": self.sample_max,
            "energy_max": self.energy_max,
        }

    def add_sample(self, sample: ObservedPVEnergy):
        self.samples.append(sample)
        if sample.energy > self.sample_max.energy:
            self.sample_max = sample
        if sample.energy > self.energy_max:
            self.energy_max = sample.energy

        # only estimate params if weather data are available
        if sample.weather:
            energy_estimate = self.get_energy_estimate(sample.weather)
            error = energy_estimate - sample.energy
            cloud_coverage = sample.cloud_coverage
            if cloud_coverage:
                TimeSpanEnergyModel.Wc += (
                    error * cloud_coverage * TimeSpanEnergyModel.Wc_lr
                )
                if TimeSpanEnergyModel.Wc > TimeSpanEnergyModel.Wc_max:
                    TimeSpanEnergyModel.Wc = TimeSpanEnergyModel.Wc_max
                elif TimeSpanEnergyModel.Wc < TimeSpanEnergyModel.Wc_min:
                    TimeSpanEnergyModel.Wc = TimeSpanEnergyModel.Wc_min

    def get_energy_estimate(self, weather: WeatherSample):
        cloud_coverage = weather.cloud_coverage
        if cloud_coverage is not None:
            return self.energy_max * (1 - TimeSpanEnergyModel.Wc * cloud_coverage)
        return self.energy_max

    def pop_sample(self, sample: ObservedPVEnergy):
        self.samples.remove(sample)
        if self.samples:
            # Energy max is slowly varying and linked to sunny days when yield is at maximum.
            # When we discard history we might be left with only cloudy days with lower yields than
            # the potential maximum so we just derate the energy_max (assuming over the history lenght
            # the maximum yield will only eventually decay so fast)
            self.energy_max *= 0.95
            self.sample_max = self.samples[0]
            for sample in self.samples:
                if self.sample_max.energy < sample.energy:
                    self.sample_max = sample
                if self.energy_max < sample.energy:
                    self.energy_max = sample.energy
            return False
        else:
            # This model will be discarded
            return True


class Estimator_PVEnergy_Heuristic(Estimator_PVEnergy):
    """
    Proof-of-concept of an estimator model based on some heuristics:

    The main concept states that the pv production is almost the same in the same time window
    between subsequent days (should the weather be ok i.e. sunny)
    This way seasonal variations over months due to shadows or anything are not being taken care of
    since they're 'already in the data'

    Also, the estimation is adjusted on the fly by observing current energy production over an
    'observation time window' and comparing it against its estimation. The ratio will be used to adjust
    the forecasts for the time ahead.
    """

    history_samples: deque[ObservedPVEnergy]
    model: dict[int, TimeSpanEnergyModel]

    __slots__ = (
        "history_samples",
        "model",
        "observed_ratio",
        "_model_energy_max",
    )

    def __init__(
        self,
        *,
        astral_observer: "astral.sun.Observer",
        tzinfo: "dt.tzinfo",
        **kwargs: "typing.Unpack[EstimatorConfig]",
    ):
        Estimator_PVEnergy.__init__(
            self,
            astral_observer=astral_observer,
            tzinfo=tzinfo,
            **kwargs,
        )
        self.history_samples: typing.Final = deque()
        self.model: typing.Final = {}
        self.observed_ratio: float = 1

        self._model_energy_max: float = 0
        """
        _model_energy_max contains the maximum energy produced in a sampling_interval_ts during the day
        so it represents the 'peak' of the discrete function represented by 'model' and, depending
        on plant orientation it should more or less happen at noon in the model
        """

    # interface: Estimator_PVEnergy
    def as_dict(self):
        return super().as_dict() | {"model": self.model}

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "model_energy_max": self._model_energy_max,
            "observed_ratio": self.observed_ratio,
            "cloud_weight": TimeSpanEnergyModel.Wc,
        }

    def update_estimate(self):
        """Process a new sample trying to update the forecast of energy production."""

        """
        We use recent observations to adjust the forecast implementing a kind of short-term linear auto-regressive model.
        The observed_ratio is a good indication of near future energy production
        but is also very 'unstable' at the edges of the energy model (i.e. sunrise/sunset)
        so we have to derate this parameter accordingly. The idea is to have this ratio
        converging to 1 at sunrise/sunset i.e. when energy is a fraction of _model_energy_max
        while 'maxing out' when observing ratio at energy levels close to _model_energy_max
        The underlying formula extraction is a bit complicated but it involves weighting
        the d_ratio (i.e. d_ratio = ratio - 1) against the fraction of energy model at time of observation
        with respect to _model_energy_max.

        Without weighting we would have:
        - observed_ratio = sum(observed_energy) / sum(energy_max in model at observation time)

        this diverges too much from the 'ideal' value of 1 so we refactor computations based on d_ratio:
        - d_ratio = observed_ratio - 1
        - d_ratio = sum(observed_energy - energy_max) / sum(energy_max)
        and we try to scale this down at sunrise/sunset (weighting on energy_max)
        - d_ratio = sum((observed_energy - energy_max) * (energy_max / _model_energy_max)) / sum(energy_max)

        """

        sum_energy_max = 0
        sum_observed_weighted = 0
        try:
            for observed_energy in self.observed_samples:
                model = self.model[observed_energy.time_ts % 86400]
                sum_energy_max += model.energy_max
                sum_observed_weighted += (observed_energy.energy - model.energy_max) * (
                    model.energy_max / self._model_energy_max
                )
            self.observed_ratio = 1 + (sum_observed_weighted / sum_energy_max)
        except (KeyError, ZeroDivisionError):
            # no data or invalid
            self.observed_ratio = 1

        if self.on_update_estimate:
            self.on_update_estimate(self)

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
                        if (
                            weather_forecast_next.condition
                            != weather_forecast.condition
                        ):
                            # weather condition changing so we immediately 'drop' the
                            # short term energy adjustment (observed_ratio)
                            weight_or = 0
                        weather_forecast = weather_forecast_next
                        weather_forecast_next = weather_forecast.next

            return energy / self.sampling_interval_ts
        else:
            return (
                self.get_estimated_energy_max(time_begin_ts, time_end_ts)
                * self.observed_ratio
            )

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
                    energy += model.energy_max * (time_end_ts - time_begin_ts)
                    break
                else:
                    energy += model.energy_max * (model_time_next_ts - time_begin_ts)
            except KeyError:
                # no energy in model
                pass

            time_begin_ts = model_time_ts = model_time_next_ts

        return energy / self.sampling_interval_ts

    def get_estimated_energy_min(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):
        return 0

    def _observed_energy_history_add(self, history_sample: ObservedPVEnergy):

        if history_sample.energy:
            # Our model only contains data when energy is being produced leaving the model 'empty'
            # for time with no production
            self.history_samples.append(history_sample)

            history_sample.sun_zenith, history_sample.sun_azimuth = (
                astral.sun.zenith_and_azimuth(
                    self.astral_observer,
                    datetime_from_epoch(
                        (history_sample.time_ts + history_sample.time_next_ts) / 2
                    ),
                )
            )

            try:
                model = self.model[history_sample.time_ts % 86400]
                model.add_sample(history_sample)
            except KeyError as e:
                self.model[history_sample.time_ts % 86400] = model = (
                    TimeSpanEnergyModel(history_sample)
                )
            if self._model_energy_max < model.energy_max:
                self._model_energy_max = model.energy_max

        # flush history
        recalc_energy_max = False
        history_min_ts = history_sample.time_ts - self.history_duration_ts
        try:
            while self.history_samples[0].time_ts < history_min_ts:
                discarded_sample = self.history_samples.popleft()
                sample_time_of_day_ts = discarded_sample.time_ts % 86400
                model = self.model[sample_time_of_day_ts]
                if model.energy_max == self._model_energy_max:
                    recalc_energy_max = True
                if model.pop_sample(discarded_sample):
                    self.model.pop(sample_time_of_day_ts)
        except IndexError:
            # history empty
            pass

        if recalc_energy_max:
            self._model_energy_max = 0
            for model in self.model.values():
                if self._model_energy_max < model.energy_max:
                    self._model_energy_max = model.energy_max
