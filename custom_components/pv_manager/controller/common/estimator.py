"""
Base estimator model common to all types of estimators
"""

import abc
from collections import deque
import dataclasses
import datetime as dt
import enum
import typing

from ...helpers import datetime_from_epoch

if typing.TYPE_CHECKING:
    pass


SAFE_MAXIMUM_POWER_INFINITE = 1e6

@dataclasses.dataclass(slots=True)
class Observation:
    time_ts: float
    value: float
    """[W] if power observation - [J] if energy observation"""
    is_energy: bool


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
        self.samples = 0


class EstimatorWarning(enum.StrEnum):
    maximum_latency = enum.auto()
    out_of_range = enum.auto()


class EstimatorConfig(typing.TypedDict):
    sampling_interval_minutes: int
    """Time resolution of model data"""
    observation_duration_minutes: int  # minutes
    """The time window for calculating current energy production from incoming energy observation."""
    history_duration_days: int
    """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""
    maximum_latency_minutes: int
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""
    safe_maximum_power_w: typing.NotRequired[float]
    """Maximum power expected at the input used to filter out outliers from observations. If not set disables the chcek."""


class Estimator(abc.ABC):
    """
    Base class for all (energy) estimators.
    """

    OPS_DECAY: typing.Final = 0.9
    """Decay factor for the average number of observations per sample."""

    tzinfo: dt.tzinfo
    on_update_estimate: typing.Callable[["Estimator"], None] | None
    on_state_change: typing.Callable[["Estimator"], None] | None

    state: set
    """Contains warnings and any other useful operating condition of the estimator."""
    observed_samples: typing.Final[deque[ObservedEnergy]]
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
        "safe_maximum_power",
        "tzinfo",
        "on_update_estimate",
        "on_state_change",
        # state
        "state",
        "observed_samples",
        "observed_time_ts",  # time of most recent observed sample
        "observations_per_sample_avg",
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
        sampling_interval_ts = kwargs.get("sampling_interval_minutes", 10) * 60
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
        self.safe_maximum_power: typing.Final = kwargs.get("safe_maximum_power_w", SAFE_MAXIMUM_POWER_INFINITE)
        self.tzinfo = tzinfo
        self.on_update_estimate = None
        self.on_state_change = None
        self.state = set()
        self.observed_samples = deque()
        self.observed_time_ts = 0
        self.observations_per_sample_avg = 0
        self.today_ts = 0
        self.tomorrow_ts = 0
        self.today_energy = 0
        # do not define here..we're relying on AttributeError for proper initialization
        # self._history_sample_curr = None
        # self._observation_prev = None

    def as_dict(self):
        """Returns the full state info of the estimator as a dictionary.
        Used for serialization to debug logs or so."""
        return {
            "sampling_interval_minutes": self.sampling_interval_ts / 60,
            "observation_duration_minutes": self.observation_duration_ts / 60,
            "history_duration_days": self.history_duration_ts / 86400,
            "maximum_latency_minutes": self.maximum_latency_ts / 60,
            "safe_maximum_power": self.safe_maximum_power,
        }

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return {
            "state": list(self.state),
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

    def add_observation(self, observation: Observation) -> bool:
        """
        Add a new Observation to the model. Observations are collected to build an Observation window
        to get an average of recent observations used for 'recent time' auto-regressive estimation.
        When an ObservedSample exits the observation window it is added to the history to update the model.
        For simplicity and performance reasons observations are required to be either [J] for energy or
        [W] for power
        """
        try:

            sample_curr = self._observed_sample_curr
            observation_prev = self._observation_prev
            self._observation_prev = observation
            _notify_state = None
            d_ts = observation.time_ts - observation_prev.time_ts
            if 0 < d_ts < self.maximum_latency_ts:
                if EstimatorWarning.maximum_latency in self.state:
                    self.state.remove(EstimatorWarning.maximum_latency)
                    _notify_state = self.on_state_change

                if observation.is_energy:
                    energy = observation.value - observation_prev.value
                    if 0 <= energy < self.safe_maximum_power * d_ts:
                        good = True
                    else:
                        # assume an energy reset or out of range
                        good = False
                else:
                    if 0 <= observation.value < self.safe_maximum_power:
                        good = True
                        # power left rect integration
                        energy = observation_prev.value * d_ts
                    else:
                        # discard the out of range observation
                        good = False
                        observation.value = 0

                if good:
                    if EstimatorWarning.out_of_range in self.state:
                        self.state.remove(EstimatorWarning.out_of_range)
                        _notify_state = self.on_state_change
                else:
                    if EstimatorWarning.out_of_range not in self.state:
                        self.state.add(EstimatorWarning.out_of_range)
                        _notify_state = self.on_state_change
            else:
                good = None
                if EstimatorWarning.maximum_latency not in self.state:
                    self.state.add(EstimatorWarning.maximum_latency)
                    _notify_state = self.on_state_change

            if _notify_state:
                _notify_state(self)

            if sample_curr.time_ts < observation.time_ts < sample_curr.time_next_ts:
                if good:
                    sample_curr.energy += energy
                    sample_curr.samples += 1
                return False
            else:
                sample_prev = sample_curr
                self._observed_sample_curr = sample_curr = self._observed_energy_new(
                    observation
                )
                if sample_curr.time_ts == sample_prev.time_next_ts:
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    if good:
                        power = energy / d_ts
                        sample_prev.energy += power * (
                            sample_prev.time_next_ts - observation_prev.time_ts
                        )
                        sample_curr.energy += power * (
                            observation.time_ts - sample_curr.time_ts
                        )
                        # for simplicity we consider interpolation as adding a full 1 sample
                        sample_prev.samples += 1
                        sample_curr.samples += 1

                self.observations_per_sample_avg = (
                    self.observations_per_sample_avg * Estimator.OPS_DECAY
                    + sample_prev.samples * (1 - Estimator.OPS_DECAY)
                )
                sample_prev.energy /= 3600  # from now on energy is [WH]
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

        except AttributeError as error:
            if error.name == "_observed_sample_curr":
                # expected right at the first call..use this to initialize the state
                # and avoid needless checks on subsequent calls
                self._observed_sample_curr = self._observed_energy_new(observation)
                self._observation_prev = observation
                if not observation.is_energy:
                    if 0 <= observation.value < self.safe_maximum_power:
                        pass
                    else:
                        observation.value = 0
                        self.state.add(EstimatorWarning.out_of_range)
                        if self.on_state_change:
                            self.on_state_change(self)
                return False
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
    def update_estimate(self):
        if self.on_update_estimate:
            self.on_update_estimate(self)

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
