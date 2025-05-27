"""
The component global api.
"""

from collections import deque
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass
import enum
from threading import Lock
from time import time
import typing

from homeassistant import const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import callback
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    event,
)
from homeassistant.util import dt as dt_util

from . import const as pmc
from .helpers import Loggable, datetime_from_epoch

if typing.TYPE_CHECKING:
    from asyncio.events import TimerHandle
    from datetime import tzinfo
    from logging import Logger
    from typing import Any, Callable, Coroutine, Final, Iterable

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import (
        HassJob,
        HomeAssistant,
    )

    from .controller import (
        Controller,
        EnergyEstimatorController,
    )


class MeteringCycle(Loggable):
    """This class acts as a broker for dispatching cycling resets to MeteringEntity(s).
    It does so by installing a single utc time event callback per CycleMode and dispatching
    the reset to all the registered entities for the same CycleMode.
    Since the loop callbacks might be delayed for any reason it can also cooperates with
    MeteringEntity to asynchronously trigger the reset should we detect the cycle end
    time has passed.
    """

    class Mode(enum.StrEnum):
        """Cycle modes for metering sensors."""

        TOTAL = enum.auto()
        YEARLY = enum.auto()
        MONTHLY = enum.auto()
        WEEKLY = enum.auto()
        DAILY = enum.auto()
        HOURLY = enum.auto()

    class Sink(Loggable if typing.TYPE_CHECKING else object):
        """Mixin style class for metering sensors"""

        cycle_mode: "Final[MeteringCycle.Mode]"

        def _reset_cycle(self, metering_cycle: "MeteringCycle", /):
            pass

    if typing.TYPE_CHECKING:
        cycle_mode: Final[Mode]
        is_total: Final[bool]
        _sinks: set[Sink]

    __slots__ = (
        "cycle_mode",
        "is_total",
        "last_reset_dt",
        "last_reset_ts",  # UTC timestamp of last reset
        "next_reset_dt",
        "next_reset_ts",  # UTC timestamp of next reset
        "_sinks",
        "_reset_cycle_unsub",
    )

    def __init__(self, cycle_mode: Mode, /):
        Loggable.__init__(self, cycle_mode, logger=Manager)
        self.cycle_mode = cycle_mode
        self._sinks = set()
        self._reset_cycle_unsub = None
        if cycle_mode is MeteringCycle.Mode.TOTAL:
            self.is_total = True
            self.last_reset_dt = None
            self.last_reset_ts = 0
            self.next_reset_dt = None
            self.next_reset_ts = 2147483647
        else:
            self.is_total = False
            self.update(None)

    def update(self, time_ts: float | None, /):
        """This is actually called with None only by the loop timer callback or the constructor
        so we know the TimerHandle needs not to be cancelled. When called with a timestamp (float)
        it means the cycle reset/update is 'async' so we'll check for cancellation of the eventually
        active callback."""

        if time_ts:
            if self._reset_cycle_unsub:
                self._reset_cycle_unsub.cancel()
                self._reset_cycle_unsub = None
        else:
            time_ts = time()
        dt_utc = datetime_from_epoch(time_ts)

        if self.cycle_mode is MeteringCycle.Mode.HOURLY:
            # fast track in UTC
            # this also avoids possible issues at DST transitions
            self.last_reset_dt = datetime(
                year=dt_utc.year,
                month=dt_utc.month,
                day=dt_utc.day,
                hour=dt_utc.hour,
                tzinfo=dt_utc.tzinfo,
            )
            self.next_reset_dt = self.last_reset_dt + timedelta(hours=1)

        else:
            now = dt_utc.astimezone(dt_util.get_default_time_zone())
            match self.cycle_mode:
                case MeteringCycle.Mode.YEARLY:
                    last_reset_dt = datetime(
                        year=now.year, month=1, day=1, tzinfo=now.tzinfo
                    )
                    next_reset_dt = datetime(
                        year=now.year + 1, month=1, day=1, tzinfo=now.tzinfo
                    )
                case MeteringCycle.Mode.MONTHLY:
                    last_reset_dt = datetime(
                        year=now.year, month=now.month, day=1, tzinfo=now.tzinfo
                    )
                    next_reset_dt = (last_reset_dt + timedelta(days=32)).replace(day=1)
                case MeteringCycle.Mode.WEEKLY:
                    today = datetime(
                        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                    )
                    last_reset_dt = today - timedelta(days=today.weekday())
                    next_reset_dt = last_reset_dt + timedelta(weeks=1)
                case MeteringCycle.Mode.DAILY:
                    last_reset_dt = datetime(
                        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                    )
                    next_reset_dt = last_reset_dt + timedelta(days=1)
            self.last_reset_dt = last_reset_dt.astimezone(dt_util.UTC)
            self.next_reset_dt = next_reset_dt.astimezone(dt_util.UTC)

        self.last_reset_ts = self.last_reset_dt.timestamp()
        self.next_reset_ts = self.next_reset_dt.timestamp()

        for sink in self._sinks:
            sink._reset_cycle(self)

        self._reset_cycle_unsub = Manager.schedule(
            self.next_reset_ts - time_ts, self.update, None
        )
        self.log(
            Loggable.WARNING, "Scheduled metering cycle at: %s", self.next_reset_dt
        )

    def unregister(self, sink: Sink, /):
        self.log(Loggable.WARNING, "Unregistering synk: %s", sink.logtag)
        self._sinks.remove(sink)
        if self._reset_cycle_unsub and not self._sinks:
            self._reset_cycle_unsub.cancel()
            self._reset_cycle_unsub = None
            self.log(
                Loggable.WARNING, "Cancelled metering cycle at: %s", self.next_reset_dt
            )


class ManagerClass(Loggable):
    """
    Singleton global manager/helper class
    """

    @dataclass(slots=True)
    class _DayStart:
        today_ts: "Final[int]"
        tomorrow_ts: "Final[int]"

        def __init__(self, time_ts: float, tz: "tzinfo", /):
            time = datetime_from_epoch(time_ts, tz)
            today = datetime(time.year, time.month, time.day, tzinfo=tz)
            tomorrow = today + timedelta(days=1)
            self.today_ts = int(today.astimezone(UTC).timestamp())
            self.tomorrow_ts = int(tomorrow.astimezone(UTC).timestamp())

    if typing.TYPE_CHECKING:
        logger: Logger
        hass: "Final[HomeAssistant]"
        device_registry: "Final[dr.DeviceRegistry]"
        entity_registry: "Final[er.EntityRegistry]"
        _call_later: "Final[Callable[[float, Callable, ], TimerHandle]]"
        _metering_cycles: Final[dict[MeteringCycle.Mode, MeteringCycle]]
        _daystart_cache: Final[dict[tzinfo, deque["ManagerClass._DayStart"]]]
        _lock: Lock

    __slots__ = (
        "hass",
        "device_registry",
        "entity_registry",
        "_call_later",
        "_metering_cycles",
        "_daystart_cache",
        "_lock",
    )

    @staticmethod
    def get(hass: "HomeAssistant", /) -> "ManagerClass":
        """Safe binder call"""
        try:
            return hass.data[pmc.DOMAIN]
        except KeyError:
            Manager.hass = hass
            Manager.device_registry = dr.async_get(hass)
            Manager.entity_registry = er.async_get(hass)
            Manager._call_later = hass.loop.call_later
            hass.data[pmc.DOMAIN] = Manager

            async def _async_unload(_event) -> None:
                del hass.data[pmc.DOMAIN]
                Manager.hass = None  # type: ignore
                Manager.device_registry = None  # type: ignore
                Manager.entity_registry = None  # type: ignore
                Manager._call_later = lambda *args: Manager.log(Manager.DEBUG, "call_later called while shutting down (args:%s)", args)  # type: ignore

            hass.bus.async_listen_once(hac.EVENT_HOMEASSISTANT_STOP, _async_unload)

            return Manager

    def __init__(self, /):
        super().__init__(None)
        self._metering_cycles = {}
        self._daystart_cache = {}
        self._lock = Lock()

    # interface: Loggable
    def log(self, level: int, msg: str, /, *args, **kwargs):
        if (logger := self.logger).isEnabledFor(level):
            logger._log(level, msg, args, **kwargs)

    # interface: self
    @callback
    def async_create_task[_R](
        self,
        target: "Coroutine[Any, Any, _R]",
        name: str,
        /,
        eager_start: bool = True,
    ):
        return self.hass.async_create_task(target, f"PVManager({name})", eager_start)

    def schedule_async(
        self, delay: float, target: "Callable[..., Coroutine]", /, *args
    ):
        @callback
        def _callback(*_args):
            self.async_create_task(target(*_args), "._callback")

        return self._call_later(delay, _callback, *args)

    def schedule(self, delay: float, target: "Callable", /, *args):
        return self._call_later(delay, target, *args)

    def schedule_at(
        self,
        dt: "datetime",
        target: "HassJob[[datetime], Coroutine[Any, Any, None] | None] | Callable[[datetime], Coroutine[Any, Any, None] | None]",
        /,
    ):
        return event.async_track_point_in_utc_time(self.hass, target, dt)

    def schedule_at_epoch(self, epoch: float, target: "Callable", /, *args):
        # fragile api: this doesn't really care about the loop drifting or so.
        # schedule_at is better at doing the job (relying on the HA api for that) but bulkier.
        # Either way, we're concerned with what happens when the loop restarts after a power suspend.
        return self._call_later(epoch - time(), target, *args)

    def register_metering_synk(self, sink: MeteringCycle.Sink, /):
        time_ts = time()
        try:
            metering_cycle = self._metering_cycles[sink.cycle_mode]
            if time_ts >= metering_cycle.next_reset_ts:
                if metering_cycle.is_total:
                    # Houston we have a problem
                    metering_cycle.log(Loggable.CRITICAL, "Time overflow")
                else:
                    metering_cycle.update(time_ts)
        except KeyError:
            # ensure we pass in a 'true' CycleMode
            metering_cycle = MeteringCycle(MeteringCycle.Mode(sink.cycle_mode))
            self._metering_cycles[sink.cycle_mode] = metering_cycle

        metering_cycle.log(Loggable.WARNING, "Registering sink: %s", sink.logtag)
        metering_cycle._sinks.add(sink)
        if not (metering_cycle.is_total or metering_cycle._reset_cycle_unsub):
            metering_cycle._reset_cycle_unsub = Manager.schedule(
                metering_cycle.next_reset_ts - time_ts, metering_cycle.update, None
            )
            metering_cycle.log(
                Loggable.WARNING, "Scheduled cycle at: %s", metering_cycle.next_reset_dt
            )
        return metering_cycle

    def get_daystart(self, time_ts: float, tz: "tzinfo", /):
        with self._lock:
            # locking here is required since this could be called in the context
            # of recorder executor threads when history is loaded.
            try:
                ds_queue = self._daystart_cache[tz]
                index = 0
                for ds in ds_queue:
                    if time_ts < ds.today_ts:
                        ds = ManagerClass._DayStart(time_ts, tz)
                        ds_queue.insert(index, ds)
                        # this branch would skip dequeue clipping: we hope
                        # data are not always loaded going back in time
                        return ds
                    if time_ts < ds.tomorrow_ts:
                        return ds
                    index += 1
                # set a reasonable limit for the cache:
                # the deque need to grow when loading histories and we should preserve
                # the calculations so that (at start) when multiple load history
                # are carried on, we can reuse the data in cache. Since our
                # histories should be more or less capped at 14 days we set 15
                # as max length for the dequeue
                while len(ds_queue) > 14:
                    ds_queue.popleft()

            except KeyError:
                ds_queue = deque()
                self._daystart_cache[tz] = ds_queue

            ds = ManagerClass._DayStart(time_ts, tz)
            ds_queue.append(ds)
            return ds

    def lookup_estimator_controller(
        self, entity_id: str, entry_type: pmc.ConfigEntryType | None = None
    ):
        """Given an entity_id, looks through config_entries if any estimator exists for that
        entity."""
        if typing.TYPE_CHECKING:
            config: EnergyEstimatorController.Config
            config_entry: ConfigEntry[
                EnergyEstimatorController[EnergyEstimatorController.Config]
            ]
        estimator_entry_types = (
            pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR,
            pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR,
        )
        for config_entry in self.hass.config_entries.async_entries(domain=pmc.DOMAIN):
            _entry_type = pmc.ConfigEntryType.get_from_entry(config_entry)
            if _entry_type in estimator_entry_types:
                config = config_entry.data  # type: ignore
                if (config.get("source_entity_id") == entity_id) and (
                    (not entry_type) or (entry_type is _entry_type)
                ):
                    return config_entry, (
                        config_entry.runtime_data
                        if config_entry.state == ConfigEntryState.LOADED
                        else None
                    )


Manager: ManagerClass = ManagerClass()
