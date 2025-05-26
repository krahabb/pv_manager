"""
Helper class for metering sensors like EnergySensor.
"""

from datetime import datetime, timedelta
import enum
from time import time
import typing

from homeassistant.util import dt as dt_util

from ..helpers import Loggable, datetime_from_epoch
from ..manager import Manager

if typing.TYPE_CHECKING:
    from typing import Final


class CycleMode(enum.StrEnum):
    """Cycle modes for metering sensors."""

    TOTAL = enum.auto()
    YEARLY = enum.auto()
    MONTHLY = enum.auto()
    WEEKLY = enum.auto()
    DAILY = enum.auto()
    HOURLY = enum.auto()


class MeteringEntity(Loggable if typing.TYPE_CHECKING else object):
    """Mixin style class for metering sensors"""

    cycle_mode: "Final[CycleMode]"

    def _reset_cycle(self, metering_cycle: "MeteringCycle"):
        pass


class MeteringCycle(Loggable):
    """This class acts as a broker for dispatching cycling resets to MeteringEntity(s).
    It does so by installing a single utc time event callback per CycleMode and dispatching
    the reset to all the registered entities for the same CycleMode.
    Since the loop callbacks might be delayed for any reason it can also cooperates with
    MeteringEntity to asynchronously trigger the reset should we detect the cycle end
    time has passed.
    """

    if typing.TYPE_CHECKING:
        cycle_mode: Final[CycleMode]
        is_total: Final[bool]
        _entities: set[MeteringEntity]
        _CYCLES: Final[dict[CycleMode, "MeteringCycle"]]

    _CYCLES = {}

    __slots__ = (
        "cycle_mode",
        "is_total",
        "last_reset_dt",
        "last_reset_ts",  # UTC timestamp of last reset
        "next_reset_dt",
        "next_reset_ts",  # UTC timestamp of next reset
        "_entities",
        "_reset_cycle_unsub",
    )

    @staticmethod
    def register(entity: MeteringEntity):
        time_ts = time()
        try:
            metering_cycle = MeteringCycle._CYCLES[entity.cycle_mode]
            if time_ts >= metering_cycle.next_reset_ts:
                if metering_cycle.is_total:
                    # Houston we have a problem
                    metering_cycle.log(Loggable.CRITICAL, "Time overflow")
                else:
                    metering_cycle.update(time_ts)
        except KeyError:
            # ensure we pass in a 'true' CycleMode
            metering_cycle = MeteringCycle(CycleMode(entity.cycle_mode), time_ts)

        metering_cycle.log(Loggable.WARNING, "Registering entity: %s", entity.logtag)
        metering_cycle._entities.add(entity)
        if not (metering_cycle.is_total or metering_cycle._reset_cycle_unsub):
            metering_cycle._reset_cycle_unsub = Manager.schedule(
                metering_cycle.next_reset_ts - time_ts, metering_cycle.update, None
            )
            metering_cycle.log(
                Loggable.WARNING, "Scheduled cycle at: %s", metering_cycle.next_reset_dt
            )
        return metering_cycle

    def __init__(self, cycle_mode: CycleMode, time_ts: float):
        Loggable.__init__(self, cycle_mode)
        self.cycle_mode = cycle_mode
        self._entities = set()
        self._reset_cycle_unsub = None
        if cycle_mode is CycleMode.TOTAL:
            self.is_total = True
            self.last_reset_dt = None
            self.last_reset_ts = 0
            self.next_reset_dt = None
            self.next_reset_ts = 2147483647
        else:
            self.is_total = False
            self.update(None)
        MeteringCycle._CYCLES[cycle_mode] = self

    def update(self, time_ts: float | None):
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

        if self.cycle_mode is CycleMode.HOURLY:
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
                case CycleMode.YEARLY:
                    last_reset_dt = datetime(
                        year=now.year, month=1, day=1, tzinfo=now.tzinfo
                    )
                    next_reset_dt = datetime(
                        year=now.year + 1, month=1, day=1, tzinfo=now.tzinfo
                    )
                case CycleMode.MONTHLY:
                    last_reset_dt = datetime(
                        year=now.year, month=now.month, day=1, tzinfo=now.tzinfo
                    )
                    next_reset_dt = (last_reset_dt + timedelta(days=32)).replace(day=1)
                case CycleMode.WEEKLY:
                    today = datetime(
                        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                    )
                    last_reset_dt = today - timedelta(days=today.weekday())
                    next_reset_dt = last_reset_dt + timedelta(weeks=1)
                case CycleMode.DAILY:
                    last_reset_dt = datetime(
                        year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                    )
                    next_reset_dt = last_reset_dt + timedelta(days=1)
            self.last_reset_dt = last_reset_dt.astimezone(dt_util.UTC)
            self.next_reset_dt = next_reset_dt.astimezone(dt_util.UTC)

        self.last_reset_ts = self.last_reset_dt.timestamp()
        self.next_reset_ts = self.next_reset_dt.timestamp()

        for entity in self._entities:
            entity._reset_cycle(self)

        self._reset_cycle_unsub = Manager.schedule(
            self.next_reset_ts - time_ts, self.update, None
        )
        self.log(Loggable.WARNING, "Scheduled cycle at: %s", self.next_reset_dt)

    def unregister(self, entity: MeteringEntity):
        self.log(Loggable.WARNING, "Unregistering entity: %s", entity.logtag)
        self._entities.remove(entity)
        if self._reset_cycle_unsub and not self._entities:
            self._reset_cycle_unsub.cancel()
            self._reset_cycle_unsub = None
            self.log(Loggable.WARNING, "Cancelled cycle at: %s", self.next_reset_dt)
