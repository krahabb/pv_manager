"""
PV energy estimator based on hourly energy measurement
"""

from collections import deque
import typing

from astral import sun

from ..helpers import datetime_from_epoch
from .estimator_pvenergy import PVEnergyEstimator, WeatherModel

if typing.TYPE_CHECKING:
    from typing import Final, Unpack


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

    if typing.TYPE_CHECKING:
        type Sample = "HeuristicPVEnergyEstimator.Sample"
        weather_model: Final[WeatherModel]
        samples: list[Sample]

        sample_max: Sample

        energy_max: float

    __slots__ = (
        "weather_model",
        "samples",
        "sample_max",
        "energy_max",
    )

    def __init__(self, weather_model: WeatherModel, sample: "Sample"):
        self.weather_model = weather_model
        self.samples = [sample]
        self.sample_max = sample
        self.energy_max = sample.energy

    def as_dict(self):
        return {
            "samples": self.samples,
            "sample_max": self.sample_max,
            "energy_max": self.energy_max,
        }

    def add_sample(self, sample: "Sample"):
        self.samples.append(sample)
        if sample.energy > self.sample_max.energy:
            self.sample_max = sample
        if sample.energy > self.energy_max:
            self.energy_max = sample.energy
        # only estimate params if weather data are available
        if sample.weather:
            self.weather_model.update_estimate(
                self.energy_max, sample.weather, sample.energy
            )

    def pop_sample(self, sample: "Sample"):
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


class HeuristicPVEnergyEstimator(PVEnergyEstimator):
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

    if typing.TYPE_CHECKING:

        class Config(PVEnergyEstimator.Config):
            pass

        class Args(PVEnergyEstimator.Args):
            pass

        type Sample = PVEnergyEstimator.Sample

        history_samples: Final[deque[Sample]]
        energy_model: Final[dict[int, TimeSpanEnergyModel]]

    _SLOTS_ = (
        "history_samples",
        "energy_model",
        "observed_ratio",
        "_model_energy_max",
    )

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):
        self.history_samples = deque()
        self.energy_model = {}
        self.observed_ratio: float = 1
        self._model_energy_max: float = 0
        """
        _model_energy_max contains the maximum energy produced in a sampling_interval_ts during the day
        so it represents the 'peak' of the discrete function represented by 'model' and, depending
        on plant orientation it should more or less happen at noon in the model
        """
        super().__init__(id, **kwargs)

    # interface: PVEnergyEstimator
    def as_dict(self):
        return super().as_dict() | {
            "energy_model": self.energy_model,
        }

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "history_samples": len(self.history_samples),
            "model_energy_max": self._model_energy_max,
            "observed_ratio": self.observed_ratio,
        }

    @typing.override
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
        model = self.energy_model
        _model_energy_max = self._model_energy_max
        try:
            for observed_energy in self.observed_samples:
                _model = model[observed_energy.time_begin_ts % 86400]
                sum_energy_max += _model.energy_max
                sum_observed_weighted += (observed_energy.energy - _model.energy_max) * (
                    _model.energy_max / _model_energy_max
                )
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
        model = self.energy_model

        time_ts = estimation_time_ts + len(forecasts) * sampling_interval_ts
        time_end_ts = estimation_time_ts + count * sampling_interval_ts

        weather = self.get_weather_forecast_at(time_ts)
        # We 'blend' in recent 'ratio' of energy production with respect to avg energy.
        # It is hard to say how much
        if weather:
            weather_next = weather.next
            weight_or_decay = sampling_interval_ts / (3600 * 4)  # fixed 4 hours decay
            weight_or = 1 - (len(forecasts) * weight_or_decay)
            if weight_or < weight_or_decay:
                weight_or = 0

        while time_ts < time_end_ts:
            time_next_ts = time_ts + sampling_interval_ts
            try:
                _forecast = _forecasts_recycle.pop()
                _forecast.__init__(time_ts, time_next_ts)
            except IndexError:
                _forecast = self.__class__.Forecast(time_ts, time_next_ts)
            _forecast.weather = weather
            try:
                _model = model[time_ts % 86400]
                if weather:
                    if weight_or:
                        _forecast.energy = (
                            _model.energy_max * observed_ratio * weight_or
                            + self.weather_model.get_energy_estimate(
                                _model.energy_max, weather
                            )
                            * (1 - weight_or)
                        )
                        if weight_or > weight_or_decay:
                            weight_or -= weight_or_decay
                        else:
                            weight_or = 0
                    else:  # save some calc when not blending anymore
                        _forecast.energy = self.weather_model.get_energy_estimate(
                            _model.energy_max, weather
                        )
                    if weather_next and (weather_next.time_ts <= time_next_ts):
                        if weather_next.condition != weather.condition:
                            # weather condition changing so we immediately 'drop' the
                            # short term energy adjustment (observed_ratio)
                            weight_or = 0
                        weather = weather_next
                        weather_next = weather.next
                else:
                    _forecast.energy = _model.energy_max * observed_ratio

                _forecast.energy_min = (
                    _model.energy_max * 0.15
                )  # TODO: better heuristic for min energy ?
                _forecast.energy_max = _model.energy_max
            except KeyError:
                # no energy in model
                pass

            forecasts.append(_forecast)
            time_ts = time_next_ts

    @typing.override
    def _observed_energy_history_add(self, sample: "Sample"):

        if sample.energy:
            # Our model only contains data when energy is being produced leaving the model 'empty'
            # for time with no production
            self.history_samples.append(sample)

            sample.sun_zenith, sample.sun_azimuth = sun.zenith_and_azimuth(
                self.astral_observer,
                datetime_from_epoch((sample.time_begin_ts + sample.time_end_ts) / 2),
            )

            try:
                model = self.energy_model[sample.time_begin_ts % 86400]
                model.add_sample(sample)
            except KeyError as e:
                self.energy_model[sample.time_begin_ts % 86400] = model = (
                    TimeSpanEnergyModel(self.weather_model, sample)
                )
            if self._model_energy_max < model.energy_max:
                self._model_energy_max = model.energy_max

        # flush history
        recalc_energy_max = False
        history_min_ts = sample.time_begin_ts - self.history_duration_ts
        try:
            while self.history_samples[0].time_begin_ts < history_min_ts:
                discarded_sample = self.history_samples.popleft()
                sample_time_of_day_ts = discarded_sample.time_begin_ts % 86400
                model = self.energy_model[sample_time_of_day_ts]
                if model.energy_max == self._model_energy_max:
                    recalc_energy_max = True
                if model.pop_sample(discarded_sample):
                    del self.energy_model[sample_time_of_day_ts]
        except IndexError:
            # history empty
            pass

        if recalc_energy_max:
            self._model_energy_max = 0
            for model in self.energy_model.values():
                if self._model_energy_max < model.energy_max:
                    self._model_energy_max = model.energy_max
