"""
Base estimator model common to all types of estimators
"""

import abc
from collections import deque
import dataclasses
import datetime as dt
import typing

from ...helpers import datetime_from_epoch

if typing.TYPE_CHECKING:
    pass


@dataclasses.dataclass(slots=True)
class Observation:
    time_ts: float
    value: float


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
        observation: Observation,
        sampling_interval_ts: int,
    ):
        time_ts = int(observation.time_ts)
        time_ts -= time_ts % sampling_interval_ts
        self.time = datetime_from_epoch(time_ts)
        self.time_ts = time_ts
        self.time_next_ts = time_ts + sampling_interval_ts
        self.energy = 0
        self.samples = 1


class EstimatorConfig(typing.TypedDict):
    sampling_interval_minutes: int
    """Time resolution of model data"""
    observation_duration_minutes: int  # minutes
    """The time window for calculating current energy production from incoming energy observation."""
    history_duration_days: int
    """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""
    maximum_latency_minutes: int
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""


class Estimator(abc.ABC):
    """
    Base class for all (energy) estimators.
    """

    tzinfo: dt.tzinfo
    on_update_estimate: typing.Callable | None

    observed_time_ts: int
    today_ts: int
    tomorrow_ts: int
    today_energy: float
    _observed_sample_curr: ObservedEnergy
    _observation_prev: Observation

    __slots__ = (
        # configuration
        "sampling_interval_ts",
        "history_duration_ts",
        "observation_duration_ts",
        "maximum_latency_ts",
        "tzinfo",
        "on_update_estimate",
        # state
        "observed_samples",
        "observed_time_ts",  # time of most recent observed sample
        "today_ts",  # UTC time of local midnight (start of today)
        "tomorrow_ts",  # UTC time of local midnight tomorrow (start of tomorrow)
        "today_energy",  # energy accumulated today
        "_observed_sample_curr",
        "_observation_prev",
    )

    def __init__(
        self,
        *,
        tzinfo: dt.tzinfo,
        **kwargs: typing.Unpack[EstimatorConfig],
    ):
        sampling_interval_ts = kwargs.get("sampling_interval_minutes", 5) * 60
        assert (
            sampling_interval_ts % 300
        ) == 0, "sampling_interval must be a multiple of 5 minutes"
        self.sampling_interval_ts: typing.Final = sampling_interval_ts
        self.history_duration_ts: typing.Final = (
            kwargs.get("history_duration_days", 7) * 86400
        )
        self.observation_duration_ts: typing.Final = (
            kwargs.get("observation_duration_minutes", 20) * 60
        )
        self.maximum_latency_ts: typing.Final = (
            kwargs.get("maximum_latency_minutes", 5) * 60
        )
        self.tzinfo = tzinfo
        self.on_update_estimate = None
        self.observed_samples: typing.Final[deque[ObservedEnergy]] = deque()
        self.observed_time_ts = 0
        self.today_ts = 0
        self.tomorrow_ts = 0
        self.today_energy = 0
        # do not define here..we're relying on AttributeError for proper initialization
        # self._history_sample_curr = None
        # self._observation_prev = None

    def as_dict(self):
        """Returns the state info of the estimator as a dictionary."""
        return {
            "sampling_interval_minutes": self.sampling_interval_ts / 60,
            "observation_duration_minutes": self.observation_duration_ts / 60,
            "history_duration_days": self.history_duration_ts / 86400,
            "maximum_latency_minutes": self.maximum_latency_ts / 60,
        }

    def add_observation(self, observation: Observation) -> bool:
        """
        Add a new Observation to the model. Observations are collected to build an Observation window
        to get an average of recent observations used for 'recent time' auto-regressive estimation.
        When an ObservedSample exits the observation window it is added to the history to update the model.
        Estimation is based both on local observations and on the 'long term' history model.
        """
        try:
            if observation.time_ts < self._observed_sample_curr.time_next_ts:
                delta_time_ts = observation.time_ts - self._observation_prev.time_ts
                if delta_time_ts < self.maximum_latency_ts:
                    self._observed_sample_curr.energy += self._observer_energy_compute(
                        observation, delta_time_ts
                    )
                    self._observed_sample_curr.samples += 1

                self._observation_prev = observation
                return False
            else:
                history_sample_prev = self._observed_sample_curr
                self._observed_sample_curr = self._observed_energy_new(observation)
                if (
                    self._observed_sample_curr.time_ts
                    == history_sample_prev.time_next_ts
                ):
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    delta_time_ts = observation.time_ts - self._observation_prev.time_ts
                    if delta_time_ts < self.maximum_latency_ts:
                        self._observer_energy_interpolate(
                            observation,
                            self._observation_prev,
                            delta_time_ts,
                            self._observed_sample_curr,
                            history_sample_prev,
                        )

                self._observation_prev = observation
                self.observed_samples.append(history_sample_prev)
                self.observed_time_ts = history_sample_prev.time_next_ts

                if history_sample_prev.time_ts >= self.tomorrow_ts:
                    # new day
                    self._observed_energy_daystart(history_sample_prev.time_ts)

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
                        self._observed_energy_history_add(
                            self.observed_samples.popleft()
                        )
                except IndexError:
                    # at start when observed_samples is empty
                    return False

                if self.on_update_estimate:
                    # this is used as a possible optimization when initially loading history samples
                    # where we don't want to update_estimate inline. Once a listener is installed then
                    # we proceed to keeping the estimate updated
                    self.update_estimate()

                return True

        except AttributeError as e:
            if e.name == "_observed_sample_curr":
                # expected right at the first call..use this to initialize the state
                # and avoid needless checks on subsequent calls
                self._observed_sample_curr = self._observed_energy_new(observation)
                self._observation_prev = observation
                return False
            else:
                raise e

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
    def update_estimate(self):
        if self.on_update_estimate:
            self.on_update_estimate()

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


    def _observed_energy_new(self, observation: Observation):
        """Called when starting data collection for a new ObservedEnergy sample."""
        return ObservedEnergy(observation, self.sampling_interval_ts)

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

    @abc.abstractmethod
    def _observer_energy_compute(self, observation: Observation, delta_time_ts: float):
        """To be implemented in the Observer mixin."""
        return 0

    @abc.abstractmethod
    def _observer_energy_interpolate(
        self,
        observation_curr: Observation,
        observation_prev: Observation,
        delta_time_ts: float,
        sample_curr: ObservedEnergy,
        sample_prev: ObservedEnergy,
    ):
        """
        observation_curr: new (current) observation
        observation_prev: previous observation
        delta_time_ts: time between current and previous observation
        sample_prev: history sample ending (to be integrated with energy across history samples boundary)
        sample_curr: new (current) history sample (to be integrated with energy at the boundary)

        To be implemented in the Observer mixin.
        """
        return


class EnergyObserver(Estimator if typing.TYPE_CHECKING else object):
    """Mixin class to add to the actual Estimator in order to process energy input observations."""

    def _observer_energy_compute(self, observation: Observation, delta_time_ts: float):
        if observation.value >= self._observation_prev.value:
            return observation.value - self._observation_prev.value
        else:
            # assume an energy reset
            return 0

    def _observer_energy_interpolate(
        self,
        observation_curr: Observation,
        observation_prev: Observation,
        delta_time_ts: float,
        sample_curr: ObservedEnergy,
        sample_prev: ObservedEnergy,
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

    def _observer_energy_compute(self, observation: Observation, delta_time_ts: float):
        return (self._observation_prev.value + observation.value) * delta_time_ts / 7200

    def _observer_energy_interpolate(
        self,
        observation_curr: Observation,
        observation_prev: Observation,
        delta_time_ts: float,
        sample_curr: ObservedEnergy,
        sample_prev: ObservedEnergy,
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
