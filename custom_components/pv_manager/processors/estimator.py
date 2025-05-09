"""
Base estimator model common to all types of estimators
"""

import abc
from collections import deque
import dataclasses
import datetime as dt
import typing

from homeassistant.components.recorder import get_instance as recorder_instance, history
from homeassistant.helpers.json import save_json
from homeassistant.util import dt as dt_util

from . import TIME_TS, BaseEnergyProcessor, BaseProcessor
from .. import const as pmc
from ..helpers import datetime_from_epoch
from ..manager import Manager

if typing.TYPE_CHECKING:
    from typing import Unpack


SAMPLING_INTERVAL_MODULO = 300  # 5 minutes


class Estimator[_input_t](BaseProcessor[_input_t]):

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config):
            pass

        class Args(BaseProcessor.Args):
            pass

    UPDATE_LISTENER_TYPE = typing.Callable[["Estimator"], None]
    _update_listeners: typing.Final[set[UPDATE_LISTENER_TYPE]]

    _SLOTS_ = ("_update_listeners",)

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        self._update_listeners = set()
        super().__init__(id, **kwargs)

    @typing.override
    def shutdown(self):
        self._update_listeners.clear()
        super().shutdown()

    # interface: self
    def listen_update(self, callback_func: UPDATE_LISTENER_TYPE):
        self._update_listeners.add(callback_func)

        def _unsub():
            try:
                self._update_listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    @abc.abstractmethod
    def update_estimate(self):
        for listener in self._update_listeners:
            listener(self)


@dataclasses.dataclass(slots=True)
class ObservedEnergy:
    time: dt.datetime
    """The sample time start"""

    time_ts: int
    time_next_ts: int

    energy: float
    """The effective accumulated energy considering interpolation at the (time) limits"""

    samples: int
    """Number of samples in the time window (could be seen as a quality indicator of sampling)"""

    def __init__(
        self,
        time_ts: int,
        sampling_interval_ts: int,
    ):
        time_ts -= time_ts % sampling_interval_ts
        self.time = datetime_from_epoch(time_ts)
        self.time_ts = time_ts
        self.time_next_ts = time_ts + sampling_interval_ts
        self.energy = 0
        self.samples = 0


class EnergyEstimator(Estimator[float], BaseEnergyProcessor):
    """
    Base class for all (energy) estimators.
    """

    if typing.TYPE_CHECKING:

        class Config(Estimator.Config, BaseEnergyProcessor.Config):
            sampling_interval_minutes: int
            """Time resolution of model data"""
            observation_duration_minutes: int  # minutes
            """The time window for calculating current energy production from incoming energy observation."""
            history_duration_days: int
            """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""

        class Args(Estimator.Args, BaseEnergyProcessor.Args):
            config: "EnergyEstimator.Config"

    OPS_DECAY: typing.Final = 0.9
    """Decay factor for the average number of observations per sample."""

    source_entity_id: str
    tzinfo: dt.tzinfo

    """Contains warnings and any other useful operating condition of the estimator."""
    observed_samples: typing.Final[deque[ObservedEnergy]]
    observed_time_ts: int
    today_ts: int
    tomorrow_ts: int
    today_energy: float
    _observed_sample_curr: ObservedEnergy

    _SLOTS_ = (
        # configuration
        "sampling_interval_ts",
        "history_duration_ts",
        "observation_duration_ts",
        "tzinfo",
        # state
        "observed_samples",
        "observed_time_ts",  # time of most recent observed sample
        "observations_per_sample_avg",
        "today_ts",  # UTC time of local midnight (start of today)
        "tomorrow_ts",  # UTC time of local midnight tomorrow (start of tomorrow)
        "today_energy",  # energy accumulated today
        "_observed_sample_curr",
        "_restore_history_exit",
    )

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):
        config = kwargs["config"]
        self.sampling_interval_ts: typing.Final = (
            (
                (config.get("ampling_interval_minutes", 0) * 60)
                // SAMPLING_INTERVAL_MODULO
            )
            * SAMPLING_INTERVAL_MODULO
        ) or SAMPLING_INTERVAL_MODULO
        self.history_duration_ts: typing.Final = (
            config.get("history_duration_days", 0) * 86400
        )
        self.observation_duration_ts: typing.Final = (
            config.get("observation_duration_minutes", 20) * 60
        )
        self.tzinfo = dt_util.get_default_time_zone()
        self.observed_samples = deque()
        self.observed_time_ts = 0
        self.observations_per_sample_avg = 0
        self.today_ts = 0
        self.tomorrow_ts = 0
        self.today_energy = 0
        # do not define here..we're relying on AttributeError for proper initialization
        # self._history_sample_curr = None
        super().__init__(id, **kwargs)

    @typing.override
    async def async_start(self):
        if self.history_duration_ts:
            self._restore_history_exit = False
            await recorder_instance(Manager.hass).async_add_executor_job(
                self._restore_history,
                datetime_from_epoch(TIME_TS() - self.history_duration_ts),
            )

        self.update_estimate()
        await super().async_start()

    @typing.override
    def shutdown(self):
        self._restore_history_exit = True
        return super().shutdown()

    @typing.override
    def as_dict(self):
        return super().as_dict() | {
            "tz_info": str(self.tzinfo),
            "sampling_interval_minutes": self.sampling_interval_ts / 60,
            "observation_duration_minutes": self.observation_duration_ts / 60,
            "history_duration_days": self.history_duration_ts / 86400,
        }

    @typing.override
    def get_state_dict(self):
        return super().get_state_dict() | {
            "today": datetime_from_epoch(self.today_ts).isoformat(),
            # "tomorrow_ts": estimator._tomorrow_local_ts,
            "tomorrow": datetime_from_epoch(self.tomorrow_ts).isoformat(),
            # "observed_time_ts": estimator.observed_time_ts,
            "observed_time": datetime_from_epoch(self.observed_time_ts).isoformat(),
            "observations_per_sample_avg": self.observations_per_sample_avg,
            "today_energy": self.today_energy,
            "today_energy_max": self.get_estimated_energy_max(
                self.today_ts, self.tomorrow_ts
            ),
            "today_energy_min": self.get_estimated_energy_min(
                self.today_ts, self.tomorrow_ts
            ),
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
        try:
            sample_curr = self._observed_sample_curr
            prev_time_ts = self.time_ts
            energy = BaseEnergyProcessor.process(self, input, time_ts)

            if sample_curr.time_ts < time_ts < sample_curr.time_next_ts:
                if energy is not None:
                    sample_curr.energy += energy
                    sample_curr.samples += 1
                return energy

            sample_prev = sample_curr
            self._observed_sample_curr = sample_curr = self._observed_energy_new(
                int(time_ts)
            )
            if sample_curr.time_ts == sample_prev.time_next_ts:
                # previous and next samples in history are contiguous in time so we try
                # to interpolate energy accumulation in between
                if energy is not None:
                    power = energy / (time_ts - prev_time_ts)
                    sample_prev.energy += power * (
                        sample_prev.time_next_ts - prev_time_ts
                    )
                    sample_curr.energy += power * (time_ts - sample_curr.time_ts)
                    # for simplicity we consider interpolation as adding a full 1 sample
                    sample_prev.samples += 1
                    sample_curr.samples += 1

            self.observations_per_sample_avg = (
                self.observations_per_sample_avg * EnergyEstimator.OPS_DECAY
                + sample_prev.samples * (1 - EnergyEstimator.OPS_DECAY)
            )
            self.observed_samples.append(sample_prev)
            self.observed_time_ts = sample_prev.time_next_ts

            if sample_prev.time_ts >= self.tomorrow_ts:
                # new day
                self._observed_energy_daystart(sample_prev.time_ts)

            self.today_energy += sample_prev.energy

            try:
                observation_min_ts = (
                    sample_prev.time_next_ts - self.observation_duration_ts
                )
                # check if we can discard it since the next is old enough
                while self.observed_samples[1].time_ts < observation_min_ts:
                    # We need to update the model with incoming observations but we
                    # don't want this to affect 'current' estimation.
                    # Since estimation is based against old observations up to
                    # old_observation.time_ts we should be safe enough adding the
                    # discarded here since they're now out of the estimation 'observation' window
                    self._observed_energy_history_add(self.observed_samples.popleft())
            except IndexError:
                # at start when observed_samples is empty
                return energy

            if self._update_listeners:
                # this is used as a possible optimization when initially loading history samples
                # where we don't want to update_estimate inline. Once a listener is installed then
                # we proceed to keeping the estimate updated
                self.update_estimate()

            return energy

        except AttributeError as error:
            if error.name == "_observed_sample_curr":
                # expected right at the first call..use this to initialize the state
                # and avoid needless checks on subsequent calls
                self._observed_sample_curr = self._observed_energy_new(int(time_ts))
                self.time_ts = time_ts
                self.input = input
                return None
            else:
                raise error

    def get_observed_energy(self) -> tuple[float, float, float]:
        """compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        try:
            observed_energy = 0
            for sample in self.observed_samples:
                observed_energy += sample.energy

            return (
                observed_energy,
                self.observed_samples[0].time_ts,
                self.observed_samples[-1].time_next_ts,
            )
        except IndexError:
            # no observations though
            return (0, 0, 0)

    @abc.abstractmethod
    def get_estimated_energy(self, time_begin_ts: float, time_end_ts: float) -> float:
        """
        Returns the estimated energy in the (forward) time interval at current estimator state.
        """
        return 0

    @abc.abstractmethod
    def get_estimated_energy_max(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):
        """
        Returns the estimated maximum energy (the 'capacity' of the plant) in the (forward) time
        interval at current estimator state.
        """
        return 0

    @abc.abstractmethod
    def get_estimated_energy_min(
        self, time_begin_ts: float | int, time_end_ts: float | int
    ):
        """
        Returns the estimated minimum energy in the (forward) time
        interval at current estimator state.
        """
        return 0

    def _observed_energy_new(self, time_ts: int):
        """Called when starting data collection for a new ObservedEnergy sample."""
        return ObservedEnergy(time_ts, self.sampling_interval_ts)

    def _observed_energy_daystart(self, time_ts: int):
        """Called when starting a new day in observations."""
        time_local = datetime_from_epoch(time_ts, self.tzinfo)
        today_local = dt.datetime(
            time_local.year, time_local.month, time_local.day, tzinfo=self.tzinfo
        )
        tomorrow_local = today_local + dt.timedelta(days=1)
        self.today_ts = int(today_local.astimezone(dt.UTC).timestamp())
        self.tomorrow_ts = int(tomorrow_local.astimezone(dt.UTC).timestamp())
        # previous day alignment was based off location with:
        # local_offset_ts=int(longitude * 4 * 60),
        # self.today_ts = (
        #    time_ts - (time_ts % 86400) - self.local_offset_ts # sun hour without considering local declination for simplicity
        # )
        # self.tomorrow_ts = self._today_local_ts + 86400
        self.today_energy = 0

    @abc.abstractmethod
    def _observed_energy_history_add(self, history_sample: ObservedEnergy):
        """Called when an ObservedEnergy sample exits the observation window and enters history."""
        pass

    def _restore_history(self, history_start_time: dt.datetime):
        """This runs in an executor."""
        if self._restore_history_exit:
            return

        observed_entity_states = history.state_changes_during_period(
            Manager.hass,
            history_start_time,
            None,
            self.source_entity_id,
            no_attributes=False,
        )

        if not observed_entity_states:
            self.log(
                self.WARNING,
                "Loading history for entity '%s' did not return any data. Is the entity correct?",
                self.source_entity_id,
            )
            return

        for state in observed_entity_states[self.source_entity_id]:
            if self._restore_history_exit:
                return
            try:
                self.process(
                    self._state_convert(
                        float(state.state),
                        state.attributes["unit_of_measurement"],
                        self.input_unit,
                    ),
                    state.last_updated_timestamp,
                )
            except:
                # in case the state doesn't represent a proper value
                # just discard it
                pass

        if pmc.DEBUG:
            filepath = pmc.DEBUG.get_debug_output_filename(
                Manager.hass,
                f"model_{self.source_entity_id}_{self.__class__.__name__.lower()}.json",
            )
            save_json(filepath, self.as_dict())
