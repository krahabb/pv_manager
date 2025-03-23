from collections import deque
import dataclasses
import datetime as dt
import typing

if typing.TYPE_CHECKING:
    pass


@dataclasses.dataclass(slots=True)
class Observation:
    time_ts: float
    value: float


@dataclasses.dataclass(slots=True)
class ObservationHistory:
    """PV energy/power history data extraction. This sample is used to build energy production
    in a time window (1 hour by design) by querying either a PV power sensor or a PV energy sensor.
    Building from PV power should be preferrable due to the 'failable' nature of energy accumulation.
    """

    time: dt.datetime
    """The sample time start"""

    time_ts: int
    time_next_ts: int

    energy: float
    """The effective accumulated energy considering interpolation at the (time) limits"""

    samples: int
    """Number of samples in the time window (could be seen as a quality indicator of sampling)"""

    sun_azimuth: float
    """Position of the sun (at mid sample interval)"""
    sun_zenith: float
    """Position of the sun (at mid sample interval)"""

    SUN_NOT_SET = -360

    def __init__(self, time: dt.datetime, time_ts: int, duration_ts: int):
        self.time = time
        self.time_ts = time_ts
        self.time_next_ts = time_ts + duration_ts
        self.energy = 0
        self.samples = 1
        self.sun_azimuth = self.sun_zenith = self.SUN_NOT_SET


@dataclasses.dataclass(slots=True)
class WeatherHistory:
    time: dt.datetime
    time_ts: float
    temperature: float  # °C
    cloud_coverage: float | None  # %
    visibility: float | None  # km


class Estimator:

    observations: deque["Observation"]

    __slots__ = (
        "observation_duration_ts",
        "maximum_latency_ts",
        "observations",
        "history_sample_curr",
        "observation_prev",
    )

    def __init__(self, *, observation_duration_ts: float, maximum_latency_ts: float):
        self.observations = deque()
        self.observation_duration_ts = observation_duration_ts
        self.maximum_latency_ts = maximum_latency_ts
        self.history_sample_curr = None
        self.observation_prev = None

    def add_observation(self, observation: "Observation"):
        """Virtual method: add a new observation to the model."""
        pass

    def process_observation(self, observation: "Observation") -> bool:
        """Process a new sample trying to update the forecast of energy production."""

        self.observations.append(observation)
        observation_min_ts = observation.time_ts - self.observation_duration_ts

        if self.observations[0].time_ts > observation_min_ts:
            # TODO: warning, not enough sampling here...
            return False
        # check if we can discard it since the next is old enough
        while self.observations[1].time_ts <= observation_min_ts:
            # We need to update the model with incoming observations but we
            # don't want this to affect 'current' estimation.
            # Since estimation is based against old observations up to
            # old_observation.time_ts we should be safe enough adding the
            # discarded here since they're now out of the estimation 'observation' window
            self.add_observation(self.observations.popleft())

        return True

    def get_observed_energy(self) -> tuple[float, float, float]:
        """Virtual method: compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        return (0, 0, 0)

    def _history_sample_create(self, observation: Observation) -> ObservationHistory:
        return None  # type: ignore

    def _history_sample_add(self, history_sample: ObservationHistory):
        pass


class EnergyObserver(Estimator if typing.TYPE_CHECKING else object):
    """Mixin class to add to the actual Estimator in order to process energy input observations."""

    def add_observation(self, observation: Observation):

        if not self.history_sample_curr:
            # first observation entering the model
            self.history_sample_curr = self._history_sample_create(observation)
        else:
            delta_time_ts = observation.time_ts - self.observation_prev.time_ts
            if observation.time_ts < self.history_sample_curr.time_next_ts:
                if delta_time_ts < self.maximum_latency_ts:
                    if observation.value >= self.observation_prev.value:
                        self.history_sample_curr.energy += (
                            observation.value - self.observation_prev.value
                        )
                    else:
                        # assume an energy reset
                        self.history_sample_curr.energy += observation.value
                self.history_sample_curr.samples += 1
            else:
                history_sample_prev = self.history_sample_curr
                self.history_sample_curr = self._history_sample_create(observation)
                if self.history_sample_curr.time_ts == history_sample_prev.time_next_ts:
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    if (delta_time_ts < self.maximum_latency_ts) and (
                        observation.value > self.observation_prev.value
                    ):
                        delta_energy = observation.value - self.observation_prev.value
                        # The next sample starts with more energy than previous so we interpolate both
                        history_sample_prev.energy += (
                            delta_energy
                            * (
                                history_sample_prev.time_next_ts
                                - self.observation_prev.time_ts
                            )
                        ) / delta_time_ts
                        self.history_sample_curr.energy += (
                            delta_energy
                            * (observation.time_ts - self.history_sample_curr.time_ts)
                        ) / delta_time_ts

                self._history_sample_add(history_sample_prev)

        self.observation_prev = observation

    def get_observed_energy(self) -> tuple[float, float, float]:
        """Virtual method: compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        observed_energy = 0
        observation_prev = None
        for observation in self.observations:
            if observation_prev:
                if observation.value > observation_prev.value:
                    observed_energy += observation.value - observation_prev.value
                else:
                    # detected energy reset
                    observed_energy += observation.value
            observation_prev = observation

        return (
            observed_energy,
            self.observations[0].time_ts,
            self.observations[-1].time_ts,
        )


class PowerObserver(Estimator if typing.TYPE_CHECKING else object):
    """Mixin class to add to the actual Estimator in order to process power input observations."""

    def add_observation(self, observation: Observation):

        if not self.history_sample_curr:
            # first observation entering the model
            self.history_sample_curr = self._history_sample_create(observation)
        else:
            delta_time_ts = observation.time_ts - self.observation_prev.time_ts
            if observation.time_ts < self.history_sample_curr.time_next_ts:
                if delta_time_ts < self.maximum_latency_ts:
                    self.history_sample_curr.energy += (
                        (self.observation_prev.value + observation.value)
                        * delta_time_ts
                        / 7200
                    )
                self.history_sample_curr.samples += 1
            else:
                history_sample_prev = self.history_sample_curr
                self.history_sample_curr = self._history_sample_create(observation)
                if self.history_sample_curr.time_ts == history_sample_prev.time_next_ts:
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    if delta_time_ts < self.maximum_latency_ts:
                        prev_delta_time_ts = (
                            history_sample_prev.time_next_ts
                            - self.observation_prev.time_ts
                        )
                        prev_power_next = (
                            self.observation_prev.value
                            + (
                                (observation.value - self.observation_prev.value)
                                * prev_delta_time_ts
                            )
                            / delta_time_ts
                        )
                        history_sample_prev.energy += (
                            (self.observation_prev.value + prev_power_next)
                            * prev_delta_time_ts
                            / 7200
                        )
                        next_delta_time_ts = (
                            observation.time_ts - self.history_sample_curr.time_ts
                        )
                        self.history_sample_curr.energy += (
                            (prev_power_next + observation.value)
                            * next_delta_time_ts
                            / 7200
                        )

                self._history_sample_add(history_sample_prev)

        self.observation_prev = observation

    def get_observed_energy(self) -> tuple[float, float, float]:
        """Virtual method: compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        observed_energy = 0
        observation_prev = None
        for observation in self.observations:
            if observation_prev:
                observed_energy += (
                    (observation_prev.value + observation.value)
                    * (observation.time_ts - observation_prev.time_ts)
                    / 7200
                )
            observation_prev = observation

        return (
            observed_energy,
            self.observations[0].time_ts,
            self.observations[-1].time_ts,
        )
