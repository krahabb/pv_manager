"""
Base estimator model common to all types of estimators
"""

import abc
from collections import deque
import dataclasses
from datetime import UTC, datetime, timedelta, tzinfo
from time import time
import typing

from homeassistant import const as hac
from homeassistant.components.recorder import get_instance as recorder_instance, history
from homeassistant.core import HassJob, HassJobType, callback
from homeassistant.helpers.json import save_json
from homeassistant.util import dt as dt_util

from . import Estimator, SignalEnergyProcessor
from .. import const as pmc
from ..helpers import datetime_from_epoch
from ..manager import Manager

if typing.TYPE_CHECKING:
    from typing import Final, Unpack


SAMPLING_INTERVAL_MODULO = 300  # 5 minutes


class EnergyEstimator(Estimator):

    if typing.TYPE_CHECKING:

        class Config(Estimator.Config):
            sampling_interval_minutes: int
            """Time resolution of model data"""

        class Args(Estimator.Args):
            config: "EnergyEstimator.Config"

        config: Config
        sampling_interval_ts: Final[int]
        tz: Final[tzinfo]
        today_ts: int
        tomorrow_ts: int
        today_energy: float

    OPS_DECAY: typing.Final = 0.9
    """Decay factor for the average number of observations per sample."""

    _SLOTS_ = (
        "sampling_interval_ts",
        "tzinfo",
        "observations_per_sample_avg",
        "today_ts",  # UTC time of local midnight (start of today)
        "tomorrow_ts",  # UTC time of local midnight tomorrow (start of tomorrow)
        "today_energy",  # effective energy measured today
    )

    def __init__(self, id, **kwargs):
        super().__init__(id, **kwargs)
        config = self.config
        self.sampling_interval_ts = (
            (int(config.get("sampling_interval_minutes", 0)) * 60)
            // SAMPLING_INTERVAL_MODULO
        ) * SAMPLING_INTERVAL_MODULO or SAMPLING_INTERVAL_MODULO
        self.tz = dt_util.get_default_time_zone()
        self.observations_per_sample_avg = 0
        self.today_ts = 0
        self.tomorrow_ts = 0
        self.today_energy = 0

    @typing.override
    def as_dict(self):
        return super().as_dict() | {
            "sampling_interval_minutes": self.sampling_interval_ts / 60,
            "tz_info": str(self.tz),
        }

    @typing.override
    def get_state_dict(self):
        return super().get_state_dict() | {
            "today": datetime_from_epoch(self.today_ts).isoformat(),
            "tomorrow": datetime_from_epoch(self.tomorrow_ts).isoformat(),
            "observations_per_sample_avg": self.observations_per_sample_avg,
            "today_measured": self.today_energy,
            "today_forecast_min": self.get_estimated_energy_min(
                self.today_ts, self.tomorrow_ts
            ),
            "today_forecast_max": self.get_estimated_energy_max(
                self.today_ts, self.tomorrow_ts
            ),
        }

    @abc.abstractmethod
    def get_estimated_energy(self, time_begin_ts: int, time_end_ts: int, /) -> float:
        """
        Returns the estimated energy in the (forward) time interval at current estimator state.
        """
        return 0

    @abc.abstractmethod
    def get_estimated_energy_max(self, time_begin_ts: int, time_end_ts: int, /):
        """
        Returns the estimated maximum energy (the 'capacity' of the plant) in the (forward) time
        interval at current estimator state.
        """
        return 0

    @abc.abstractmethod
    def get_estimated_energy_min(self, time_begin_ts: int, time_end_ts: int, /):
        """
        Returns the estimated minimum energy in the (forward) time
        interval at current estimator state.
        """
        return 0

    def _observed_energy_daystart(self, time_ts: int, /):
        """Called when starting a new day in observations."""
        time_local = datetime_from_epoch(time_ts, self.tz)
        today_local = datetime(
            time_local.year, time_local.month, time_local.day, tzinfo=self.tz
        )
        tomorrow_local = today_local + timedelta(days=1)
        self.today_ts = int(today_local.astimezone(UTC).timestamp())
        self.tomorrow_ts = int(tomorrow_local.astimezone(UTC).timestamp())
        self.today_energy = 0


class SignalEnergyEstimator(EnergyEstimator, SignalEnergyProcessor):
    """
    Base class for all (energy) estimators.
    """

    @dataclasses.dataclass(slots=True)
    class Sample:
        time: "Final[datetime]"
        """The sample time start"""

        time_begin_ts: "Final[int]"
        time_end_ts: "Final[int]"

        energy: float
        """The effective accumulated energy considering interpolation at the (time) limits"""
        samples: int
        """Number of samples in the time window (could be seen as a quality indicator of sampling)"""

        def __init__(self, time_ts: float, estimator: "SignalEnergyEstimator", /):
            time_ts = int(time_ts)
            time_ts -= time_ts % estimator.sampling_interval_ts
            self.time = datetime_from_epoch(time_ts)
            self.time_begin_ts = time_ts
            self.time_end_ts = time_ts + estimator.sampling_interval_ts
            self.energy = 0
            self.samples = 0

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimator.Config, SignalEnergyProcessor.Config):
            observation_duration_minutes: int  # minutes
            """The time window for calculating current energy production from incoming energy observation."""
            history_duration_days: int
            """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""

        class Args(EnergyEstimator.Args, SignalEnergyProcessor.Args):
            config: "SignalEnergyEstimator.Config"

        config: Config
        source_entity_id: str
        observed_samples: Final[deque[Sample]]
        _sample_curr: Sample

    DEFAULT_NAME = "Energy estimator"

    _SLOTS_ = (
        # configuration
        "history_duration_ts",
        "observation_duration_ts",
        # state
        "observed_samples",
        "_sample_curr",
        "_restore_history_exit",
        "_sampling_interval_unsub",
    )

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):
        super().__init__(id, **kwargs)
        config = self.config
        self.history_duration_ts: typing.Final = int(
            config.get("history_duration_days", 0) * 86400
        )
        self.observation_duration_ts: typing.Final = int(
            config.get("observation_duration_minutes", 20) * 60
        )
        self.observed_samples = deque()
        self._sampling_interval_unsub = None

    @typing.override
    async def async_start(self):
        if self.history_duration_ts and self.source_entity_id:
            self._restore_history_exit = False
            history_begin_ts = time() - self.history_duration_ts
            self._sample_curr = self.__class__.Sample(history_begin_ts, self)
            await recorder_instance(Manager.hass).async_add_executor_job(
                self._restore_history,
                datetime_from_epoch(history_begin_ts),
            )
        else:
            self._sample_curr = self.__class__.Sample(time(), self)

        self.update_estimate()
        await super().async_start()
        # triggers the sampling callback
        self._sampling_interval_callback()

    @typing.override
    def shutdown(self):
        self._restore_history_exit = True
        if self._sampling_interval_unsub:
            self._sampling_interval_unsub.cancel()
            self._sampling_interval_unsub = None
        return super().shutdown()

    @typing.override
    def as_dict(self):
        return super().as_dict() | {
            "observation_duration_minutes": self.observation_duration_ts / 60,
            "history_duration_days": self.history_duration_ts / 86400,
        }

    @typing.override
    def process(self, input: float | None, time_ts: float) -> float | None:
        """
        Add a new Observation to the model. Observations are collected to build an Observation window
        to get an average of recent observations used for 'recent time' auto-regressive estimation.
        When an ObservedSample exits the observation window it is added to the history to update the model.
        For simplicity and performance reasons observations are required to be either [J] for energy or
        [W] for power.
        This method returns the 'energy' accumulation of the last observed sample
        """
        sample_curr = self._check_sample_curr(time_ts)
        energy = SignalEnergyProcessor.process(self, input, time_ts)
        if energy is not None:
            sample_curr.energy += energy
            sample_curr.samples += 1
        return energy

    def get_observed_energy(self) -> tuple[float, float, float]:
        """compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        try:
            observed_energy = 0
            for sample in self.observed_samples:
                observed_energy += sample.energy

            return (
                observed_energy,
                self.observed_samples[0].time_begin_ts,
                self.observed_samples[-1].time_end_ts,
            )
        except IndexError:
            # no observations though
            return (0, 0, 0)

    def _observed_energy_history_add(self, sample: Sample):
        """Called when a sample exits the observation window and enters history.
        This should be overriden when an inherited energy estimator wants to store energy data (history)
        for it's model."""
        pass

    def _check_sample_curr(self, time_ts: float, /) -> Sample:
        """This is called either by _sampling_interval_callback or by the signal processing pipe before
        updating current sample statistics.
        Checks if sample_curr is finished and eventually calls for
        processing (send to history/update model)."""
        sample_curr = self._sample_curr
        if sample_curr.time_begin_ts <= time_ts < sample_curr.time_end_ts:
            return sample_curr
        else:
            # TODO: this could be the right point to call for interpolation till
            # the end of the current sample by calling self.update(sample_curr.time_end_ts - 0.001)
            # we must be careful about re-entrance though since this is likely to be called
            # in the self.process pipe
            return self._process_sample_curr(sample_curr, time_ts)

    def _process_sample_curr(self, sample_curr: Sample, time_ts: float, /) -> Sample:
        """Use this callback to flush/store samples or so and update estimates.
        This will also return the newly created sample_curr in sync with time_ts."""
        self.observed_samples.append(sample_curr)
        self._sample_curr = sample_next = self.__class__.Sample(time_ts, self)
        time_begin_next_ts = sample_next.time_begin_ts
        if sample_curr.samples:
            self.today_energy += sample_curr.energy
            self.observations_per_sample_avg = (
                self.observations_per_sample_avg * EnergyEstimator.OPS_DECAY
                + sample_curr.samples * (1 - EnergyEstimator.OPS_DECAY)
            )
            if sample_curr.time_end_ts != time_begin_next_ts:
                # samples are not consecutive: reset energy accumulation
                # to avoid going to the moon. We call the base to avoid re-entering here
                self.reset()
        else:
            # no observations in previous sample: better reset our accumulator
            self.reset()

        self.estimation_time_ts = time_begin_next_ts
        if time_begin_next_ts >= self.tomorrow_ts:
            # new day
            self._observed_energy_daystart(time_begin_next_ts)

        try:
            observation_min_ts = time_begin_next_ts - self.observation_duration_ts
            # check if we can discard it since the next is old enough
            while self.observed_samples[1].time_begin_ts < observation_min_ts:
                # We need to update the model with incoming observations but we
                # don't want this to affect 'current' estimation.
                # Since estimation is based against old observations up to
                # old_observation.time_ts we should be safe enough adding the
                # discarded here since they're now out of the estimation 'observation' window
                self._observed_energy_history_add(self.observed_samples.popleft())
        except IndexError:
            # at start when observed_samples is empty
            pass

        if self._update_listeners:
            # this is used as a possible optimization when initially loading history samples
            # where we don't want to update_estimate inline. Once a listener is installed then
            # we proceed to keeping the estimate updated
            self.update_estimate()

        return sample_next

    @callback
    def _sampling_interval_callback(self):
        """Use this callback to flush/store samples or so and update estimates."""
        time_ts = time()
        time_curr = self._sample_curr.time_end_ts
        time_next = self._check_sample_curr(time_ts).time_end_ts
        self._sampling_interval_unsub = Manager.schedule(
            time_next - time_ts, self._sampling_interval_callback
        )
        if self.isEnabledFor(self.DEBUG):
            self.log(
                self.DEBUG,
                "sampling: curr= '%s' next= '%s'",
                datetime_from_epoch(time_curr),
                datetime_from_epoch(time_next),
            )

    def _restore_history(self, history_start_time: datetime):
        """This runs in an executor."""
        if self._restore_history_exit:
            return

        source_entity_states = history.state_changes_during_period(
            Manager.hass,
            history_start_time,
            None,
            self.source_entity_id,
            no_attributes=False,
        )

        if not source_entity_states:
            self.log(
                self.WARNING,
                "Loading history for entity '%s' did not return any data. Is the entity correct?",
                self.source_entity_id,
            )
            return

        restore_start_ts = time()
        restore_states_failed = 0
        for state in source_entity_states[self.source_entity_id]:
            if self._restore_history_exit:
                return
            try:
                self.process(
                    float(state.state)
                    * self.input_convert[state.attributes["unit_of_measurement"]],
                    state.last_updated_timestamp,
                )  # type: ignore
                continue
            except AttributeError as e:
                if e.name == "input_convert":
                    exception = self.configure_source(
                        state, state.last_updated_timestamp
                    )
                    if not exception:
                        continue
                else:
                    exception = e
            except Exception as e:
                exception = e

            restore_states_failed += 1
            # this is expected and silently managed when state == None or 'unknown'
            self.process(None, state.last_updated_timestamp)
            if state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(
                    self.WARNING,
                    exception,
                    "_restore_history (state:%s)",
                    state,
                    timeout=300,  # short timeout since history loading should not last that long
                )

        if self.isEnabledFor(self.DEBUG):
            self.log(
                self.DEBUG,
                "Restored history for entity '%s' in %s sec (states = %d failed = %d)",
                self.source_entity_id,
                round(time() - restore_start_ts, 2),
                len(source_entity_states[self.source_entity_id]),
                restore_states_failed,
            )

        if pmc.DEBUG:
            filepath = pmc.DEBUG.get_debug_output_filename(
                Manager.hass,
                f"model_{self.source_entity_id}_{self.__class__.__name__.lower()}.json",
            )
            save_json(filepath, self.as_dict())


class EnergyBalanceEstimator(EnergyEstimator):
    """
    This class computes the balance (surplus vs deficit) between a production estimate
    and a consumption estimate (where production is likely pv energy and consumption is the load).
    This acts as a simpler base for a Battery estimator where the storage must also be taken care of.
    It could nevertheless be used in an 'on grid' system to forecast the excess/deficit of self
    production to eventually schedule load usage to maximize self consumption
    """

    class Forecast:

        time_ts: int
        production: float
        consumption: float

        __slots__ = (
            "time_ts",
            "production",
            "consumption",
        )

        def __init__(self):
            self.time_ts = 0
            self.production = 0
            self.consumption = 0

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimator.Config):
            forecast_duration_hours: int

        class Args(EnergyEstimator.Args):
            config: "EnergyBalanceEstimator.Config"

        config: Config
        production_estimator: SignalEnergyEstimator | None
        consumption_estimator: SignalEnergyEstimator | None

        forecasts: list[Forecast]

    _SLOTS_ = (
        "forecast_duration_ts",
        "production_estimator",
        "consumption_estimator",
        "forecasts",
        "_production_estimator_unsub",
        "_consumption_estimator_unsub",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        config = self.config
        self.forecast_duration_ts = (
            int(
                ((config.get("forecast_duration_hours") or 1) * 3600)
                // self.sampling_interval_ts
            )
            * self.sampling_interval_ts
        )
        self.production_estimator = None
        self.consumption_estimator = None
        self.forecasts = [
            self.__class__.Forecast()
            for i in range(self.forecast_duration_ts // self.sampling_interval_ts)
        ]
        self._production_estimator_unsub = None
        self._consumption_estimator_unsub = None

    def shutdown(self):
        if self._production_estimator_unsub:
            self._production_estimator_unsub()
            self._production_estimator_unsub = None
        if self._consumption_estimator_unsub:
            self._consumption_estimator_unsub()
            self._consumption_estimator_unsub = None
        self.production_estimator = None
        self.consumption_estimator = None
        return super().shutdown()

    def connect_production(self, estimator: "SignalEnergyEstimator"):
        if self._production_estimator_unsub:
            self._production_estimator_unsub()
        self.production_estimator = estimator
        self._production_estimator_unsub = estimator.listen_update(
            self._production_estimator_update
        )

    def connect_consumption(self, estimator: "SignalEnergyEstimator"):
        if self._consumption_estimator_unsub:
            self._consumption_estimator_unsub()
        self.consumption_estimator = estimator
        self._consumption_estimator_unsub = estimator.listen_update(
            self._consumption_estimator_update
        )

    @typing.override
    def update_estimate(self):

        # this code is just a sketch..battery estimator should be more sophisticated
        production_estimator = self.production_estimator
        consumption_estimator = self.consumption_estimator

        self.today_energy = 0
        if production_estimator:
            self.today_energy = production_estimator.today_energy
            if self.estimation_time_ts < production_estimator.estimation_time_ts:
                self.estimation_time_ts = production_estimator.estimation_time_ts
                if self.today_ts < production_estimator.today_ts:
                    self.today_ts = production_estimator.today_ts
                    self.tomorrow_ts = production_estimator.tomorrow_ts

        if consumption_estimator:
            self.today_energy -= consumption_estimator.today_energy
            if self.estimation_time_ts < consumption_estimator.estimation_time_ts:
                self.estimation_time_ts = consumption_estimator.estimation_time_ts
                if self.today_ts < consumption_estimator.today_ts:
                    self.today_ts = consumption_estimator.today_ts
                    self.tomorrow_ts = consumption_estimator.tomorrow_ts

        forecast_ts = self.estimation_time_ts
        sampling_interval_ts = self.sampling_interval_ts
        for forecast in self.forecasts:
            forecast_next_ts = forecast_ts + sampling_interval_ts
            forecast.time_ts = forecast_ts
            if production_estimator:
                forecast.production = production_estimator.get_estimated_energy(
                    forecast_ts, forecast_next_ts
                )
            else:
                forecast.production = 0
            if consumption_estimator:
                forecast.consumption = consumption_estimator.get_estimated_energy(
                    forecast_ts, forecast_next_ts
                )
            else:
                forecast.consumption = 0
            forecast_ts = forecast_next_ts

        for listener in self._update_listeners:
            listener(self)

    @typing.override
    def get_estimated_energy(self, time_begin_ts: int, time_end_ts: int) -> float:
        """
        Returns the estimated energy in the (forward) time interval at current estimator state.
        """
        energy = 0
        sampling_interval_ts = self.sampling_interval_ts
        for forecast in self.forecasts:
            if forecast.time_ts >= time_end_ts:
                break

            forecast_next_ts = forecast.time_ts + sampling_interval_ts
            if time_begin_ts >= forecast_next_ts:
                continue

            if time_end_ts <= forecast_next_ts:
                energy += (forecast.production - forecast.consumption) * (
                    time_end_ts - time_begin_ts
                )
                break
            else:
                energy += (forecast.production - forecast.consumption) * (
                    forecast_next_ts - time_begin_ts
                )

            time_begin_ts = forecast_next_ts

        return energy / sampling_interval_ts

    @typing.override
    def get_estimated_energy_max(self, time_begin_ts: int, time_end_ts: int):
        """
        Returns the estimated maximum energy (the 'capacity' of the plant) in the (forward) time
        interval at current estimator state.
        """
        # TODO
        return self.get_estimated_energy(time_begin_ts, time_end_ts)

    @typing.override
    def get_estimated_energy_min(self, time_begin_ts: int, time_end_ts: int):
        """
        Returns the estimated minimum energy in the (forward) time
        interval at current estimator state.
        """
        # TODO
        return self.get_estimated_energy(time_begin_ts, time_end_ts)

    def _production_estimator_update(self, estimator: "Estimator"):
        self.update_estimate()

    def _consumption_estimator_update(self, estimator: "Estimator"):
        self.update_estimate()
