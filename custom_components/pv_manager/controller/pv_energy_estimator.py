"""
Controller for creating an entity simulating pv power based off sun tracking
"""

from collections import deque
import dataclasses
import datetime as dt
import time
import typing

import astral
import astral.sun
from homeassistant import const as hac
from homeassistant.components.recorder import get_instance as recorder_instance, history
from homeassistant.core import callback
from homeassistant.helpers import event, json, sun as sun_helpers
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter, PowerConverter

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..sensor import Sensor

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State


@dataclasses.dataclass
class Observation:
    time: dt.datetime
    time_ts: float
    value: float


@dataclasses.dataclass(slots=True)
class EnergyHistorySample:
    """PV energy/power history data extraction. This sample is used to build energy production
    in a time window (1 hour by design) by querying either a PV power sensor or a PV energy sensor.
    Building from PV power should be preferrable due to the 'failable' nature of energy accumulation.
    """

    time: dt.datetime
    """The 'hourly aligned' sample time (i.e. the begin of the hour time window)"""
    time_ts: float
    time_next: dt.datetime
    """The end of the hour time window"""
    time_next_ts: float

    observation_begin: Observation
    """First state update time during the [time..time_next] window"""
    observation_end: Observation
    """Last state update time during the [time..time_next] window"""

    energy: float
    """The effective accumulated energy considering interpolation at the (time) limits"""

    samples: int
    """Number of samples in the time window (could be seen as a quality indicator of sampling)"""

    sun_azimuth: float
    """Position of the sun (at mid sample interval)"""
    sun_zenith: float
    """Position of the sun (at mid sample interval)"""

    SUN_NOT_SET = -360

    def __init__(self, observation: Observation):
        self.time = observation.time.replace(minute=0, second=0, microsecond=0)
        self.time_ts = self.time.timestamp()
        self.time_next = self.time + Estimator.SAMPLING_INTERVAL_DT
        self.time_next_ts = self.time_next.timestamp()
        self.observation_begin = observation
        self.observation_end = observation
        self.energy = 0
        self.samples = 1
        self.sun_azimuth = self.sun_zenith = self.SUN_NOT_SET


@dataclasses.dataclass(slots=True)
class EnergyModel:
    samples: list[EnergyHistorySample]

    energy_average: float
    energy_max: float
    energy_min: float

    def __init__(self):
        self.samples = []
        self.energy_average = 0
        self.energy_max = 0
        self.energy_min = 0

    def add_sample(self, sample: EnergyHistorySample):

        self.samples.append(sample)
        self._recalc()

    def pop_sample(self, sample: EnergyHistorySample):
        try:
            self.samples.remove(sample)
            if self.samples:
                self._recalc()
            else:
                self.energy_average = 0
                self.energy_max = 0
                self.energy_min = 0
        except ValueError:
            # sample not in list
            pass

    def _recalc(self):
        self.energy_average = 0
        self.energy_max = self.energy_min = self.samples[0].energy
        for sample in self.samples:
            self.energy_average += sample.energy
            if sample.energy > self.energy_max:
                self.energy_max = sample.energy
            elif sample.energy < self.energy_min:
                self.energy_min = sample.energy
        self.energy_average /= len(self.samples)


@dataclasses.dataclass
class HourlyEnergyEstimation:
    energy: float

    def __init__(self):
        self.energy = 0


class Estimator:

    SAMPLING_INTERVAL_DT = dt.timedelta(hours=1)
    SAMPLING_INTERVAL_TS = 3600
    DAY_SAMPLES = int(86400 / SAMPLING_INTERVAL_TS)

    history_samples: deque[EnergyHistorySample]
    model: dict[int, EnergyModel]
    estimations: list[HourlyEnergyEstimation]
    observations: deque[Observation]

    __slots__ = (
        "history_duration_ts",
        "observation_duration_ts",
        "maximum_latency_ts",
        "astral_observer",
        "history_samples",
        "history_sample_curr",
        "model",
        "estimations",
        "observations",
    )

    def __init__(
        self,
        history_duration_ts: float,
        observation_duration_ts: float,
        maximum_latency_ts: float,
        astral_observer: "astral.sun.Observer",
    ):
        self.history_duration_ts: typing.Final = history_duration_ts
        assert observation_duration_ts < self.SAMPLING_INTERVAL_TS
        self.observation_duration_ts: typing.Final = observation_duration_ts
        self.maximum_latency_ts = maximum_latency_ts
        self.astral_observer = astral_observer

        self.history_samples: typing.Final = deque()
        self.history_sample_curr = None
        self.estimations: typing.Final = [
            HourlyEnergyEstimation() for _t in range(self.DAY_SAMPLES)
        ]
        self.model: typing.Final = {_t: EnergyModel() for _t in range(self.DAY_SAMPLES)}
        self.observations: typing.Final = deque()

    def add_sample(self, observation: Observation):
        """Virtual method: add a new observation to the model."""
        pass

    def process_observation(self, observation: Observation) -> bool:
        """Process a new sample trying to update the forecast of energy production."""

        self.observations.append(observation)
        observation_min_ts = observation.time_ts - self.observation_duration_ts

        # start from oldest observation
        observation_begin = self.observations[0]
        if observation_begin.time_ts > observation_min_ts:
            # TODO: warning, not enough sampling here...
            return False

        # check if we can discard it since the next is old enough
        while self.observations[1].time_ts < observation_min_ts:
            self.observations.popleft()
            # We need to update the model with incoming observations but we
            # don't want this to affect 'current' estimation.
            # Since estimation is based against old observations up to
            # old_observation.time_ts we should be safe enough adding the
            # discarded here since they're now out of the estimation 'observation' window
            self.add_sample(observation_begin)
            observation_begin = self.observations[0]

        observed_energy, observed_begin_ts, observed_end_ts = self.get_observed_energy()
        # we now have observed_energy generated during observed_duration

        estimated_observed_energy = self._get_estimated_energy_max(
            observed_begin_ts, observed_end_ts
        )
        if estimated_observed_energy <= 0:
            # no energy in our model at observation time
            return False

        ratio = observed_energy / estimated_observed_energy

        estimation_time_begin_ts = observation.time_ts
        for _t in range(len(self.estimations)):
            estimation_time_end_ts = (
                estimation_time_begin_ts + self.SAMPLING_INTERVAL_TS
            )
            self.estimations[_t].energy = (
                self._get_estimated_energy_max(
                    estimation_time_begin_ts, estimation_time_end_ts
                )
                * ratio
            )
            estimation_time_begin_ts = estimation_time_end_ts

        return True

    def get_observed_energy(self) -> tuple[float, float, float]:
        """Virtual method: compute the energy stored in the 'observations'.
        Returns: (energy, observation_begin_ts, observation_end_ts)"""
        return (0, 0, 0)

    def _get_estimated_energy_max(self, time_begin_ts: float, time_end_ts: float):
        time_begin = helpers.datetime_from_epoch(time_begin_ts)

        energy = 0
        hour = time_begin.hour
        model_time_ts = time_begin_ts - time_begin.minute * 60 - time_begin.second

        while time_begin_ts < time_end_ts:

            model = self.model[hour]
            model_time_next_ts = model_time_ts + self.SAMPLING_INTERVAL_TS
            if time_end_ts < model_time_next_ts:
                energy += model.energy_max * (time_end_ts - time_begin_ts)
                break
            else:
                energy += model.energy_max * (model_time_next_ts - time_begin_ts)
                hour = hour + 1 if hour < 23 else 0
                time_begin_ts = model_time_ts = model_time_next_ts

        return energy / self.SAMPLING_INTERVAL_TS

    def _history_sample_add(self, history_sample: EnergyHistorySample):
        self.history_samples.append(history_sample)
        if history_sample.energy:
            sample_mid_time_ts = (history_sample.time_ts + history_sample.time_next_ts) / 2
            sample_mid_time = helpers.datetime_from_epoch(sample_mid_time_ts)
            history_sample.sun_zenith, history_sample.sun_azimuth = astral.sun.zenith_and_azimuth(self.astral_observer, sample_mid_time)

            self.model[history_sample.time.hour].add_sample(history_sample)

        if self.history_samples[0].time_ts < (
            history_sample.time_ts - self.history_duration_ts
        ):
            discarded_sample = self.history_samples.popleft()
            self.model[discarded_sample.time.hour].pop_sample(discarded_sample)


class EstimatorEnergyObserver(Estimator):

    def add_sample(self, observation: Observation):

        if not self.history_sample_curr:
            # first observation entering the model
            self.history_sample_curr = EnergyHistorySample(observation)
        else:
            observation_prev = self.history_sample_curr.observation_end
            delta_time_ts = observation.time_ts - observation_prev.time_ts
            if observation.time_ts < self.history_sample_curr.time_next_ts:
                if delta_time_ts < self.maximum_latency_ts:
                    if observation.value >= observation_prev.value:
                        self.history_sample_curr.energy += (
                            observation.value - observation_prev.value
                        )
                    else:
                        # assume an energy reset
                        self.history_sample_curr.energy += observation.value
                self.history_sample_curr.observation_end = observation
                self.history_sample_curr.samples += 1
            else:
                history_sample_prev = self.history_sample_curr
                self.history_sample_curr = EnergyHistorySample(observation)
                if self.history_sample_curr.time_ts == history_sample_prev.time_next_ts:
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    if (delta_time_ts < self.maximum_latency_ts) and (
                        observation.value > observation_prev.value
                    ):
                        delta_energy = observation.value - observation_prev.value
                        # The next sample starts with more energy than previous so we interpolate both
                        history_sample_prev.energy += (
                            delta_energy
                            * (
                                history_sample_prev.time_next_ts
                                - observation_prev.time_ts
                            )
                        ) / delta_time_ts
                        self.history_sample_curr.energy += (
                            delta_energy
                            * (observation.time_ts - self.history_sample_curr.time_ts)
                        ) / delta_time_ts

                self._history_sample_add(history_sample_prev)

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


class EstimatorPowerObserver(Estimator):

    def add_sample(self, observation: Observation):

        if not self.history_sample_curr:
            # first observation entering the model
            self.history_sample_curr = EnergyHistorySample(observation)
        else:
            observation_prev = self.history_sample_curr.observation_end
            delta_time_ts = observation.time_ts - observation_prev.time_ts
            if observation.time_ts < self.history_sample_curr.time_next_ts:
                if delta_time_ts < self.maximum_latency_ts:
                    self.history_sample_curr.energy += (
                        (observation_prev.value + observation.value)
                        * delta_time_ts
                        / 7200
                    )
                self.history_sample_curr.observation_end = observation
                self.history_sample_curr.samples += 1
            else:
                history_sample_prev = self.history_sample_curr
                self.history_sample_curr = EnergyHistorySample(observation)
                if self.history_sample_curr.time_ts == history_sample_prev.time_next_ts:
                    # previous and next samples in history are contiguous in time so we try
                    # to interpolate energy accumulation in between
                    if delta_time_ts < self.maximum_latency_ts:
                        prev_delta_time_ts = (
                            history_sample_prev.time_next_ts - observation_prev.time_ts
                        )
                        prev_power_next = (
                            observation_prev.value
                            + (
                                (observation.value - observation_prev.value)
                                * prev_delta_time_ts
                            )
                            / delta_time_ts
                        )
                        history_sample_prev.energy += (
                            (observation_prev.value + prev_power_next)
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


class ControllerConfig(typing.TypedDict):
    pv_energy_entity_id: typing.NotRequired[str]
    """The source entity_id of the pv energy of the system"""
    pv_power_entity_id: typing.NotRequired[str]
    """The source entity_id of the pv power of the system"""
    observation_duration_minutes: int  # minutes
    """The time duration of the incoming energy readings observation."""
    refresh_period_minutes: int

    maximum_latency_minutes: int
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""
    history_duration_days: int
    """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.BaseConfig):
    """TypedDict for ConfigEntry data"""


class Controller(controller.Controller[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    __slots__ = (
        # configuration
        "pv_source_entity_id",
        "refresh_period_ts",
        # state
        "estimator",
        "pv_energy_estimator_sensor",
        "_sun_offset",
        "_state_convert_func",
        "_state_convert_unit",
        "_pv_entity_tracking_unsub",
        "_estimate_callback_unsub",
        "_restore_history_task",
        "_restore_history_exit",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict) -> dict:
        return hv.entity_schema(
            user_input,
            name="PV energy estimation",
        ) | {
            hv.optional("pv_energy_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.ENERGY
            ),
            hv.optional("pv_power_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.required(
                "observation_duration_minutes", user_input, 20
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.required(
                "refresh_period_minutes", user_input, 5
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.required(
                "maximum_latency_minutes", user_input, 1
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.required(
                "history_duration_days", user_input, 14
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.DAYS, max=30),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)

        self.refresh_period_ts = self.config.get("refresh_period_minutes", 5) * 60
        # TODO: get sun_offset from local tz
        self._sun_offset = dt.timedelta(hours=1)

        if "pv_power_entity_id" in self.config:
            estimator_class = EstimatorPowerObserver
            self._state_convert_func = PowerConverter.convert
            self._state_convert_unit = hac.UnitOfPower.WATT
            self.pv_source_entity_id = self.config["pv_power_entity_id"]
        elif "pv_energy_entity_id" in self.config:
            estimator_class = EstimatorEnergyObserver
            self._state_convert_func = EnergyConverter.convert
            self._state_convert_unit = hac.UnitOfEnergy.WATT_HOUR
            self.pv_source_entity_id = self.config["pv_energy_entity_id"]
        else:
            raise Exception(
                "missing either 'pv_power_entity_id' or 'pv_energy_entity_id' in config"
            )

        location, elevation = sun_helpers.get_astral_location(hass)
        self.estimator = estimator_class(
            history_duration_ts=self.config.get("history_duration_days", 7) * 86400,
            observation_duration_ts=self.config.get("observation_duration_minutes", 20)
            * 60,
            maximum_latency_ts=self.config.get("maximum_latency_minutes", 1) * 60,
            astral_observer=astral.sun.Observer(location.latitude, location.longitude, elevation),
        )

        self.pv_energy_estimator_sensor = Sensor(
            self,
            pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR,
            device_class=Sensor.DeviceClass.ENERGY,
            state_class=None,
            name=self.config["name"],
            native_unit_of_measurement=hac.UnitOfEnergy.WATT_HOUR,
        )

        for _t in range(self.estimator.DAY_SAMPLES):
            Sensor(
                self,
                f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}",
                device_class=Sensor.DeviceClass.ENERGY,
                state_class=None,
                name=f"{self.config["name"]} ({_t+1:02})",
                native_unit_of_measurement=hac.UnitOfEnergy.WATT_HOUR,
            )

    async def async_init(self):
        self._restore_history_exit = False
        self._restore_history_task = recorder_instance(
            self.hass
        ).async_add_executor_job(self._restore_history)
        await self._restore_history_task
        self._restore_history_task = None

        self._pv_entity_tracking_unsub = event.async_track_state_change_event(
            self.hass,
            self.pv_source_entity_id,
            self._pv_entity_tracking_callback,
        )

        self._estimate_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )

        self._update_estimate(self.hass.states.get(self.pv_source_entity_id))
        
        await super().async_init()

    async def async_shutdown(self):
        if await super().async_shutdown():
            if self._pv_entity_tracking_unsub:
                self._pv_entity_tracking_unsub()
                self._pv_entity_tracking_unsub = None
            if self._estimate_callback_unsub:
                self._estimate_callback_unsub.cancel()
                self._estimate_callback_unsub = None
            if self._restore_history_task:
                if not self._restore_history_task.done():
                    self._restore_history_exit = True
                    await self._restore_history_task
                self._restore_history_task = None
            self.pv_energy_estimator_sensor: Sensor = None  # type: ignore
            return True
        return False

    @callback
    def _pv_entity_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        self._update_estimate(event.data.get("new_state"))

    async def _async_refresh_callback(self):
        self._estimate_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )
        self._update_estimate(self.hass.states.get(self.pv_source_entity_id))

    def _update_estimate(self, tracked_state: "State | None"):
        if tracked_state:
            try:
                now_ts = time.time()
                if self.estimator.process_observation(
                    Observation(
                        dt_util.utc_from_timestamp(now_ts),
                        now_ts,
                        self._state_convert_func(
                            float(tracked_state.state),
                            tracked_state.attributes["unit_of_measurement"],
                            self._state_convert_unit,
                        ),
                    )
                ):

                    accumulate_daily = True
                    accumulated_energy = 0
                    daily_energy = 0
                    for _t in range(self.estimator.DAY_SAMPLES):
                        _energy = self.estimator.estimations[_t].energy
                        accumulated_energy += _energy
                        if accumulate_daily:
                            # this is just a dumb trick to compute total energy estimated production until sunset
                            if _energy > 0:
                                daily_energy += _energy
                            else:
                                accumulate_daily = False
                        sensor: Sensor = self.entities[Sensor.PLATFORM][
                            f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}"
                        ]  # type:ignore
                        sensor.update(accumulated_energy)

                    self.pv_energy_estimator_sensor.update(daily_energy)
                    return

            except Exception as e:
                self.log_exception(self.DEBUG, e, "updating estimate")

        self.pv_energy_estimator_sensor.update(None)
        for _t in range(self.estimator.DAY_SAMPLES):
            sensor: Sensor = self.entities[Sensor.PLATFORM][
                f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}"
            ]  # type:ignore
            sensor.update(None)

    def _restore_history(self):
        if self._restore_history_exit:
            return
        now_ts = time.time()
        observation_start_ts = now_ts - self.estimator.observation_duration_ts
        start_time = helpers.datetime_from_epoch(now_ts - self.estimator.history_duration_ts)
        states = history.state_changes_during_period(
            self.hass, start_time, None, self.pv_source_entity_id, no_attributes=False
        )

        for state in states[self.pv_source_entity_id]:
            if self._restore_history_exit:
                return
            try:
                state_ts = state.last_updated_timestamp
                if state_ts < observation_start_ts:
                    # bypass observation/estimation steps since these are too old
                    self.estimator.add_sample(
                        Observation(
                            state.last_updated,
                            state_ts,
                            self._state_convert_func(
                                float(state.state),
                                state.attributes["unit_of_measurement"],
                                self._state_convert_unit,
                            ),
                        )
                    )
                else:
                    # start populating observations and estimates
                    # so the estimator is ready when HA restarts during the day
                    self.estimator.process_observation(
                        Observation(
                            state.last_updated,
                            state_ts,
                            self._state_convert_func(
                                float(state.state),
                                state.attributes["unit_of_measurement"],
                                self._state_convert_unit,
                            ),
                        )
                    )
            except:
                # in case the state doesn't represent a proper value
                # just discard it
                pass

        if helpers.DEBUG:
            filepath = helpers.DEBUG.get_debug_output_filename(
                self.hass, f"model_{self.pv_source_entity_id}.json"
            )
            dump_model = {
                "samples": list(self.estimator.history_samples),
                "model": self.estimator.model,
            }
            json.save_json(filepath, dump_model)
