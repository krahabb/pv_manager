"""
Base estimator model common to all types of estimators
"""

import abc
from collections import deque
from datetime import datetime
from time import time
import typing

from homeassistant import const as hac
from homeassistant.components.recorder import history
from homeassistant.core import callback
from homeassistant.helpers.json import save_json
from homeassistant.util import dt as dt_util

from . import Estimator, SignalEnergyProcessor
from .. import const as pmc
from ..helpers import datetime_from_epoch, history as hh, validation as hv
from ..helpers.dataattr import DataAttr, DataAttrClass, DataAttrParam, timestamp_i
from ..helpers.manager import Manager

if typing.TYPE_CHECKING:
    from datetime import tzinfo
    from typing import (
        Any,
        Callable,
        ClassVar,
        Final,
        Iterable,
        Mapping,
        NotRequired,
        Self,
        TypedDict,
        Unpack,
    )

    from homeassistant.core import CALLBACK_TYPE


class EnergyEstimator(Estimator):
    """
    Base class for all (energy) estimators.
    """

    class Sample(DataAttrClass):
        time: DataAttr["datetime"]
        """The sample time start"""
        time_begin_ts: DataAttr[int]
        time_end_ts: DataAttr[int]
        energy: DataAttr[float] = 0
        """The effective accumulated energy considering interpolation at the (time) limits"""
        samples: DataAttr[int] = 0
        """Number of samples in the time window (could be seen as a quality indicator of sampling)"""

        def __init__(self, time_ts: float, estimator: "EnergyEstimator", /):
            DataAttrClass.__init__(self)
            time_ts = int(time_ts)
            time_ts -= time_ts % estimator.sampling_interval_ts
            estimator.estimation_time_ts = time_ts
            # expecting time_ts going forward or backward
            if not (estimator.today_ts <= time_ts < estimator.tomorrow_ts):
                # new day
                estimator._observed_energy_daystart(time_ts)
            self.time = datetime_from_epoch(time_ts)
            self.time_begin_ts = time_ts
            self.time_end_ts = time_ts + estimator.sampling_interval_ts

    class Forecast(DataAttrClass):
        """Cached forecast sample calculated once per 'update_estimate'.
        forecasts are built on demand and cached in the forecasts container
        for reuse in calculations. They're invalidated whenever estimation_time updates.
        """

        time_begin_ts: DataAttr[timestamp_i]
        time_end_ts: DataAttr[timestamp_i]
        energy: DataAttr[float] = 0
        energy_min: DataAttr[float] = 0
        energy_max: DataAttr[float] = 0

        def __init__(self, time_begin_ts: int, time_end_ts: int, /):
            DataAttrClass.__init__(self)
            self.time_begin_ts = time_begin_ts
            self.time_end_ts = time_end_ts

        def add(self, forecast: "Self", /):
            self.energy += forecast.energy
            self.energy_min += forecast.energy_min
            self.energy_max += forecast.energy_max

        def addmul(self, forecast: "Self", ratio: float, /):
            self.energy += forecast.energy * ratio
            self.energy_min += forecast.energy_min * ratio
            self.energy_max += forecast.energy_max * ratio

        def as_formatted_dict(self, tz: "tzinfo", /):
            result = super().as_formatted_dict() | {
                "time_begin": datetime_from_epoch(self.time_begin_ts, tz).isoformat(),
                "time_end": datetime_from_epoch(self.time_end_ts, tz).isoformat(),
            }
            return result

    if typing.TYPE_CHECKING:

        class Config(Estimator.Config):
            sampling_interval_minutes: int
            """Time resolution of model data"""

        class Args(Estimator.Args):
            config: "EnergyEstimator.Config"

        SAMPLING_INTERVAL_MODULO: Final
        OPS_DECAY: Final

        config: Config
        sampling_interval_ts: Final[int]
        tz: Final[tzinfo]
        forecasts: Final[list[Forecast]]
        _forecasts_recycle: Final[list[Forecast]]
        _sample_curr: Sample
        _sampling_interval_unsub: CALLBACK_TYPE | None

    DEFAULT_NAME = "Energy estimator"
    SAMPLING_INTERVAL_MODULO = 300  # 5 minutes
    OPS_DECAY = 0.9
    """Decay factor for the average number of observations per sample."""

    today_ts: DataAttr[timestamp_i, DataAttrParam.stored] = 0
    """UTC time of local midnight (start of today)."""
    tomorrow_ts: DataAttr[timestamp_i, DataAttrParam.stored] = 0
    """UTC time of local midnight tomorrow (start of tomorrow)."""
    today_energy: DataAttr[float, DataAttrParam.stored] = 0
    """effective energy measured today"""
    observations_per_sample_avg: DataAttr[float] = 0

    _SLOTS_ = (
        "sampling_interval_ts",
        "tzinfo",
        "forecasts",
        "_forecasts_recycle",
        "_sample_curr",
        "_sampling_interval_unsub",
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None") -> pmc.ConfigSchema:
        return super().get_config_schema(config) | {
            hv.req_config(
                "sampling_interval_minutes",
                config or {"sampling_interval_minutes": 10},
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
        }

    def __init__(self, id, **kwargs):
        super().__init__(id, **kwargs)
        config = self.config
        self.sampling_interval_ts = (
            (
                (int(config.get("sampling_interval_minutes", 0)) * 60)
                // EnergyEstimator.SAMPLING_INTERVAL_MODULO
            )
            * EnergyEstimator.SAMPLING_INTERVAL_MODULO
            or EnergyEstimator.SAMPLING_INTERVAL_MODULO
        )
        self.tz = dt_util.get_default_time_zone()
        self.forecasts = []
        self._forecasts_recycle = []
        self._sample_curr = self.__class__.Sample(time(), self)
        self._sampling_interval_unsub = None

    async def async_start(self):
        await super().async_start()
        # triggers the sampling callback
        self._sampling_interval_callback()

    @typing.override
    def shutdown(self):
        if self._sampling_interval_unsub:
            self._sampling_interval_unsub()
            self._sampling_interval_unsub = None
        super().shutdown()

    @typing.override
    def as_diagnostic_dict(self):
        return super().as_diagnostic_dict() | {
            "sampling_interval_minutes": self.sampling_interval_ts / 60,
            "tz_info": str(self.tz),
        }

    @typing.override
    def as_state_dict(self):
        return {
            "forecast": self.get_forecast(
                self.estimation_time_ts, self.tomorrow_ts
            ).as_formatted_dict(self.tz),
        } | super().as_state_dict()

    def process_energy(self, energy: float, time_ts: float):
        sample_curr = self._check_sample_curr(time_ts)
        sample_curr.energy += energy
        sample_curr.samples += 1

    @typing.override
    def update_estimate(self):
        self._forecasts_recycle.extend(self.forecasts)
        self.forecasts.clear()
        for listener in self._update_listeners:
            listener(self)

    def get_forecast(self, time_begin_ts: int, time_end_ts: int, /) -> Forecast:
        if time_begin_ts == time_end_ts:
            return self.__class__.Forecast(time_begin_ts, time_end_ts)

        e_t_ts = self.estimation_time_ts
        assert (
            e_t_ts <= time_begin_ts < time_end_ts
        ), f"Invalid forecast range (estimation_ts:{e_t_ts} begin_ts:{time_begin_ts} end_ts:{time_end_ts})"
        s_i_ts = self.sampling_interval_ts
        index = (time_begin_ts - e_t_ts) // s_i_ts
        _dt = time_end_ts - e_t_ts
        index_end = _dt // s_i_ts if (_dt % s_i_ts) else (_dt // s_i_ts) - 1

        # ensure the forecasts are up to time_end_ts
        forecasts = self.forecasts
        try:
            f_end = forecasts[index_end]
        except IndexError:
            # on demand build
            self._ensure_forecasts(time_end_ts)
            f_end = forecasts[index_end]

        forecast = self.__class__.Forecast(time_begin_ts, time_end_ts)

        if index == index_end:
            forecast.addmul(f_end, (time_end_ts - time_begin_ts) / s_i_ts)
            return forecast

        f_i = forecasts[index]
        forecast.addmul(f_i, (f_i.time_end_ts - time_begin_ts) / s_i_ts)

        for index in range(index + 1, index_end):
            forecast.add(forecasts[index])

        forecast.addmul(f_end, (time_end_ts - f_end.time_begin_ts) / s_i_ts)

        return forecast

    def get_estimated_energy(self, time_begin_ts: int, time_end_ts: int, /) -> float:
        """
        Returns the estimated energy in the (forward) time interval at current estimator state.
        """
        if time_begin_ts == time_end_ts:
            return 0

        e_t_ts = self.estimation_time_ts
        assert (
            e_t_ts <= time_begin_ts < time_end_ts
        ), f"Invalid forecast range (estimation_ts:{e_t_ts} begin_ts:{time_begin_ts} end_ts:{time_end_ts})"
        s_i_ts = self.sampling_interval_ts
        index = (time_begin_ts - e_t_ts) // s_i_ts
        _dt = time_end_ts - e_t_ts
        index_end = _dt // s_i_ts if (_dt % s_i_ts) else (_dt // s_i_ts) - 1

        # ensure the forecasts are up to time_end_ts
        forecasts = self.forecasts
        try:
            f_end = forecasts[index_end]
        except IndexError:
            # on demand build
            self._ensure_forecasts(time_end_ts)
            f_end = forecasts[index_end]

        if index == index_end:
            return f_end.energy * (time_end_ts - time_begin_ts) / s_i_ts

        f_i = forecasts[index]
        energy = f_i.energy * (f_i.time_end_ts - time_begin_ts) / s_i_ts

        for index in range(index + 1, index_end):
            energy += forecasts[index].energy

        energy += f_end.energy * (time_end_ts - f_end.time_begin_ts) / s_i_ts

        return energy

    @callback
    def _sampling_interval_callback(self):
        """Use this callback to flush/store samples or so and update estimates."""
        time_ts = time()
        time_curr = self._sample_curr.time_end_ts
        time_next = self._check_sample_curr(time_ts).time_end_ts
        self._sampling_interval_unsub = Manager.schedule_at_epoch(
            time_next, self._sampling_interval_callback
        )
        if self.isEnabledFor(self.DEBUG):
            self.log(
                self.DEBUG,
                "sampling: curr= '%s' next= '%s'",
                datetime_from_epoch(time_curr),
                datetime_from_epoch(time_next),
            )

    def _observed_energy_daystart(self, time_ts: int, /):
        """Called when starting a new day in observations."""
        ds = Manager.get_daystart(time_ts, self.tz)
        self.today_ts = ds.today_ts
        self.tomorrow_ts = ds.tomorrow_ts
        self.today_energy = 0

    def _check_sample_curr(self, time_ts: float, /) -> Sample:
        """This is called either by _sampling_interval_callback or by the signal processing pipe before
        updating current sample statistics. Checks if sample_curr is finished and eventually
        calls _process_sample_curr processing (send to history/update model)."""
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
        """Called when data collection (in Sample) over sampling_period ends. After 'consuming'
        the data this should return a fresh sample for the new time window."""
        sample_curr.__init__(
            time_ts, self
        )  # stub implementation: recycle current Sample
        return sample_curr

    @abc.abstractmethod
    def _ensure_forecasts(self, time_end_ts: int, /):
        """Build the forecasts cache up to time_end_ts. This must be overriden with
        the proper implementation depending on the model."""


class EnergyObserverEstimator(EnergyEstimator):

    if typing.TYPE_CHECKING:

        class Sample(EnergyEstimator.Sample):
            pass

        type HistoryProcessT = Callable[
            [hh.CompressedState, SignalEnergyProcessor.EnergyListenerT], Any
        ]
        type HistoryEntitiesDesc = dict[str, HistoryProcessT]

        class Config(EnergyEstimator.Config):
            observation_duration_minutes: int
            """The time window for calculating current energy production from incoming energy observation."""
            history_duration_days: int
            """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""

        class Args(EnergyEstimator.Args):
            config: "EnergyObserverEstimator.Config"

        config: Config
        history_duration_ts: Final[int]
        observation_duration_ts: Final[int]
        observed_samples: Final[deque[Sample]]

        _restore_history_task_ts: float
        """Current timestamp of restore hystory Task start."""

    _SLOTS_ = (
        # configuration
        "history_duration_ts",
        "observation_duration_ts",
        # state
        "observed_samples",
        "_restore_history_task_ts",  # TODO: use task cancellation to terminate history processing
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None") -> pmc.ConfigSchema:
        _config = config or {
            "observation_duration_minutes": 20,
            "history_duration_days": 7,
        }
        return super().get_config_schema(config) | {
            hv.req_config(
                "observation_duration_minutes", _config
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.req_config("history_duration_days", _config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.DAYS, max=30
            ),
        }

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):
        super().__init__(id, **kwargs)
        config = self.config
        self.history_duration_ts = int(config.get("history_duration_days", 0) * 86400)
        self.observation_duration_ts = int(
            config.get("observation_duration_minutes", 20) * 60
        )
        self.observed_samples = deque()

    @typing.override
    async def async_start(self):
        if self.history_duration_ts:
            self._restore_history_task_ts = time()
            self._sample_curr.__init__(
                self._restore_history_task_ts - self.history_duration_ts, self
            )
            await history.get_instance(Manager.hass).async_add_executor_job(
                self._restore_history
            )
        self.update_estimate()
        await super().async_start()

    @typing.override
    def shutdown(self):
        self._restore_history_task_ts = 0  # force history task termination
        super().shutdown()

    @typing.override
    def as_diagnostic_dict(self):
        return super().as_diagnostic_dict() | {
            "observation_duration_minutes": self.observation_duration_ts / 60,
            "history_duration_days": self.history_duration_ts / 86400,
        }

    # interface: EnergyEstimator
    def _process_sample_curr(
        self, sample_curr: "Sample", time_ts: float, /
    ) -> "Sample":
        """Use this callback to flush/store samples or so and update estimates.
        This will also return the newly created sample_curr in sync with time_ts."""
        self.observed_samples.append(sample_curr)

        if sample_curr.samples:
            time_end_curr_ts = sample_curr.time_end_ts
            self.today_energy += sample_curr.energy
            self.observations_per_sample_avg = (
                self.observations_per_sample_avg * EnergyEstimator.OPS_DECAY
                + sample_curr.samples * (1 - EnergyEstimator.OPS_DECAY)
            )
        else:
            # no observations in previous sample: better reset our accumulator
            time_end_curr_ts = 0

        self._sample_curr = sample_next = self.__class__.Sample(time_ts, self)
        if time_end_curr_ts != sample_next.time_begin_ts:
            # when samples are not consecutive
            self._reset_energy_accumulation()

        try:
            observation_min_ts = self.estimation_time_ts - self.observation_duration_ts
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

    # interface: self
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

    def _observed_energy_history_add(self, sample: "Sample"):
        """Called when a sample exits the observation window and enters history.
        This should be overriden when an inherited energy estimator wants to store energy data (history)
        for it's model."""
        pass

    def _reset_energy_accumulation(self):
        """Called when a new Sample entering the pipe is not time-consecutive with the previous.
        This is a signal we have to eventually reset energy accumulation in the source processing
        machinery. (See SignalEnergyEstimator for an example)"""
        pass

    def _history_entities(self, /) -> "HistoryEntitiesDesc":
        return {}

    class _HistoryIterator:

        if typing.TYPE_CHECKING:
            process: Final["EnergyObserverEstimator.HistoryProcessT"]
            iter: Final

        __slots__ = (
            "process",
            "iter",
            "state",
            "time_ts",
        )

        def __init__(
            self,
            process: "EnergyObserverEstimator.HistoryProcessT",
            states: "list[hh.CompressedState]",
        ):
            self.process = process
            self.iter = iter(states)
            try:
                self.state = next(self.iter)
                self.time_ts = self.state["lu"]
            except StopIteration:
                self.time_ts = pmc.TIMESTAMP_MAX

    def _restore_history(self):

        try:
            history_entities = self._history_entities()
            entity_ids = list(history_entities.keys())

            source_entity_states: "dict[str, list[hh.CompressedState]]" = hh.get_significant_states(  # type: ignore
                Manager.hass,
                entity_ids=entity_ids,
                start_time_ts=self._sample_curr.time_begin_ts,
                significant_changes_only=False,
                minimal_response=False,
                no_attributes=False,
                compressed_state_format=True,
            )

            # Scan the entities states lists to find the next state update event in time ordering
            entity_iterator_class = EnergyObserverEstimator._HistoryIterator
            state_iterators = [
                entity_iterator_class(history_entities[entity_id], states)
                for entity_id, states in source_entity_states.items()
            ]
            fake_iterator = entity_iterator_class(lambda s, c: None, [])
            while True:
                if not self._restore_history_task_ts:
                    # Shutting down
                    return

                iterator = fake_iterator
                for _iterator in state_iterators:
                    if _iterator.time_ts < iterator.time_ts:
                        iterator = _iterator

                if iterator is fake_iterator:
                    break

                iterator.process(iterator.state, self.process_energy)
                try:
                    iterator.state = next(iterator.iter)
                    iterator.time_ts = iterator.state["lu"]
                except StopIteration:
                    state_iterators.remove(iterator)

            if self.isEnabledFor(self.DEBUG):
                self.log(
                    self.DEBUG,
                    "Restored history for {%s} in %s sec",
                    str(
                        {
                            _entity_id: len(_states)
                            for _entity_id, _states in source_entity_states.items()
                        }
                    ),
                    round(time() - self._restore_history_task_ts, 2),
                )

        except Exception as e:
            self.log_exception(self.WARNING, e, "_restore_history")

        self._restore_history_task_ts = 0

        if pmc.DEBUG:
            filepath = pmc.DEBUG.get_debug_output_filename(
                Manager.hass,
                # TODO: create a meaningful file name
                f"model_{self.id}_{self.__class__.__name__.lower()}.json",
            )
            save_json(filepath, self.as_diagnostic_dict())


class SignalEnergyEstimator(EnergyObserverEstimator, SignalEnergyProcessor):
    """
    EnergyObserverEstimator joined to a SignalEnergyProcessor as the source of sampled energy.
    """

    if typing.TYPE_CHECKING:

        class Sample(EnergyObserverEstimator.Sample):
            pass

        class Config(EnergyObserverEstimator.Config, SignalEnergyProcessor.Config):
            pass

        class Args(EnergyObserverEstimator.Args, SignalEnergyProcessor.Args):
            config: "SignalEnergyEstimator.Config"

        config: Config
        source_entity_id: str

    @typing.override
    async def async_start(self):
        self.listen_energy(self.process_energy)
        await super().async_start()

    @typing.override
    def disconnect(self, time_ts):
        super().disconnect(time_ts)
        self._check_sample_curr(time_ts)

    # interface: EnergyObserverEstimator
    def _reset_energy_accumulation(self):
        """Called when a new Sample entering the pipe is not time-consecutive with the previous.
        This is a signal we have to eventually reset energy accumulation in the source processing
        machinery. (See SignalEnergyEstimator for an example)"""
        self.reset()

    @typing.override
    def _history_entities(self) -> "EnergyObserverEstimator.HistoryEntitiesDesc":
        return {self.source_entity_id: self.history_process}


class EnergyBalanceEstimator(EnergyEstimator):
    """
    This class computes the balance (surplus vs deficit) between a production estimate
    and a consumption estimate (where production is likely pv energy and consumption is the load).
    This acts as a simpler base for a Battery estimator where the storage must also be taken care of.
    It could nevertheless be used in an 'on grid' system to forecast the excess/deficit of self
    production to eventually schedule load usage to maximize self consumption
    """

    class Forecast(EnergyEstimator.Forecast):

        production: DataAttr[float] = 0
        production_min: DataAttr[float] = 0
        production_max: DataAttr[float] = 0
        consumption: DataAttr[float] = 0
        consumption_min: DataAttr[float] = 0
        consumption_max: DataAttr[float] = 0

        def add(self, forecast: "Self", /):
            EnergyEstimator.Forecast.add(self, forecast)
            self.production += forecast.production
            self.production_min += forecast.production_min
            self.production_max += forecast.production_max
            self.consumption += forecast.consumption
            self.consumption_min += forecast.consumption_min
            self.consumption_max += forecast.consumption_max

        def addmul(self, forecast: "Self", ratio: float, /):
            EnergyEstimator.Forecast.addmul(self, forecast, ratio)
            self.production += forecast.production * ratio
            self.production_min += forecast.production_min * ratio
            self.production_max += forecast.production_max * ratio
            self.consumption += forecast.consumption * ratio
            self.consumption_min += forecast.consumption_min * ratio
            self.consumption_max += forecast.consumption_max * ratio

    class _FakeEstimator(EnergyEstimator):
        """An 'empty' implementation to be placed in production and consumption estimators members
        at start so that we can avoid checks when any of those is not binded to an actual estimator.
        This class should mock an estimator which doesn't estimate anything."""

        _empty_forecast = EnergyEstimator.Forecast(0, 0)

        def get_forecast(self, time_begin_ts: int, time_end_ts: int, /):
            self._empty_forecast.__init__(time_begin_ts, time_end_ts)
            return self._empty_forecast

        def get_estimated_energy(self, time_begin_ts: int, time_end_ts: int, /):
            return 0

        def _check_sample_curr(self, time_ts: float, /):
            self._sample_curr.__init__(time_ts, self)
            return self._sample_curr

        def _ensure_forecasts(self, time_end_ts: int, /):
            pass

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimator.Config):
            forecast_duration_hours: int

        class Args(EnergyEstimator.Args):
            config: "EnergyBalanceEstimator.Config"
            production_estimator: NotRequired[EnergyEstimator]
            consumption_estimator: NotRequired[EnergyEstimator]

        _FAKE_ESTIMATOR: ClassVar

        config: Config  # (override base typehint)
        forecasts: Final[list[Forecast]]  # type: ignore (override base typehint)
        _forecasts_recycle: Final[list[Forecast]]  # type: ignore override (override base typehint)

        production_estimator: EnergyEstimator
        consumption_estimator: EnergyEstimator

        def get_forecast(self, time_begin_ts: int, time_end_ts: int, /) -> Forecast:  # type: ignore
            pass

    _FAKE_ESTIMATOR = _FakeEstimator("", config={})

    _SLOTS_ = (
        "production_estimator",
        "consumption_estimator",
        "_production_estimator_unsub",
        "_consumption_estimator_unsub",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        self.production_estimator = kwargs.pop(
            "production_estimator", self.__class__._FAKE_ESTIMATOR
        )
        self.consumption_estimator = kwargs.pop(
            "consumption_estimator", self.__class__._FAKE_ESTIMATOR
        )
        self._production_estimator_unsub = None
        self._consumption_estimator_unsub = None
        super().__init__(id, **kwargs)

    def shutdown(self):
        if self._production_estimator_unsub:
            self._production_estimator_unsub()
            self._production_estimator_unsub = None
        if self._consumption_estimator_unsub:
            self._consumption_estimator_unsub()
            self._consumption_estimator_unsub = None
        self.production_estimator = None  # type: ignore
        self.consumption_estimator = None  # type: ignore
        return super().shutdown()

    def connect_production(self, estimator: "EnergyEstimator"):
        if self._production_estimator_unsub:
            self._production_estimator_unsub()
        self.production_estimator = estimator
        self._production_estimator_unsub = estimator.listen_update(
            self._production_estimator_update
        )

    def connect_consumption(self, estimator: "EnergyEstimator"):
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
        self.today_energy = production_estimator.today_energy
        if self.estimation_time_ts < production_estimator.estimation_time_ts:
            self.estimation_time_ts = production_estimator.estimation_time_ts
            if self.today_ts < production_estimator.today_ts:
                self.today_ts = production_estimator.today_ts
                self.tomorrow_ts = production_estimator.tomorrow_ts

        consumption_estimator = self.consumption_estimator
        self.today_energy -= consumption_estimator.today_energy
        if self.estimation_time_ts < consumption_estimator.estimation_time_ts:
            self.estimation_time_ts = consumption_estimator.estimation_time_ts
            if self.today_ts < consumption_estimator.today_ts:
                self.today_ts = consumption_estimator.today_ts
                self.tomorrow_ts = consumption_estimator.tomorrow_ts

        super().update_estimate()

    @typing.override
    def _ensure_forecasts(self, time_end_ts: int, /):
        estimation_time_ts = self.estimation_time_ts
        sampling_interval_ts = self.sampling_interval_ts
        forecasts = self.forecasts
        _forecasts_recycle = self._forecasts_recycle
        production_estimator = self.production_estimator
        production_estimator._ensure_forecasts(time_end_ts)
        consumption_estimator = self.consumption_estimator
        consumption_estimator._ensure_forecasts(time_end_ts)

        time_ts = estimation_time_ts + len(forecasts) * sampling_interval_ts
        while time_ts < time_end_ts:
            time_next_ts = time_ts + sampling_interval_ts
            try:
                _f = _forecasts_recycle.pop()
                _f.__init__(time_ts, time_next_ts)
            except IndexError:
                _f = self.__class__.Forecast(time_ts, time_next_ts)

            _f_p = production_estimator.get_forecast(time_ts, time_next_ts)
            _f.production = _f_p.energy
            _f.production_min = _f_p.energy_min
            _f.production_max = _f_p.energy_max

            _f_c = consumption_estimator.get_forecast(time_ts, time_next_ts)
            _f.consumption = _f_c.energy
            _f.consumption_min = _f_c.energy_min
            _f.consumption_max = _f_c.energy_max

            _f.energy = _f.production - _f.consumption
            _f.energy_max = _f.production_max - _f.consumption_min
            _f.energy_min = _f.production_min - _f.consumption_max

            forecasts.append(_f)
            time_ts = time_next_ts

    def _production_estimator_update(self, estimator: "Estimator"):
        self.update_estimate()

    def _consumption_estimator_update(self, estimator: "Estimator"):
        self.update_estimate()
