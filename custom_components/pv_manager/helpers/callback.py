from time import time
import typing

from homeassistant import const as hac
from homeassistant.core import HassJobType, callback
from homeassistant.helpers.event import async_track_state_change_event

from . import Loggable
from .. import const as pmc
from ..manager import Manager

if typing.TYPE_CHECKING:
    from asyncio.events import TimerHandle
    from datetime import datetime
    from typing import Any, Callable, Coroutine, Final, Iterable

    from homeassistant.core import (
        CALLBACK_TYPE,
        Event,
        EventStateChangedData,
        HassJob,
        HomeAssistant,
        State,
    )


class CallbackTracker(Loggable):

    HassJobType = HassJobType

    _callbacks: dict[str | float, "CALLBACK_TYPE"]
    # This attribute is only created 'on demand'

    _SLOTS_ = ("_callbacks",)

    def shutdown(self):
        try:
            for _callback in self._callbacks.values():
                _callback()
        except AttributeError:
            pass
        super().shutdown()

    @callback
    def async_create_task[_R](
        self,
        target: "Coroutine[Any, Any, _R]",
        name: str,
        eager_start: bool = True,
    ):
        return Manager.hass.async_create_task(target, f"PVManager({name})", eager_start)

    def track_callback(self, key: str, callback: "CALLBACK_TYPE"):
        try:
            self._callbacks[key] = callback
        except AttributeError:
            self._callbacks = {key: callback}

    def track_timer(
        self,
        delay: float,
        action: "Callable",
        job_type: HassJobType = HassJobType.Callback,
    ):
        try:
            _callbacks = self._callbacks
        except AttributeError:
            self._callbacks = _callbacks = {}

        # we're using the delay value to mark this remove callback.
        # eventually
        _call_later = Manager._call_later

        def _target():
            _handle = _call_later(delay, _target)
            _callbacks[delay] = lambda: _handle.cancel()
            if job_type is HassJobType.Callback:
                action()
            elif job_type is HassJobType.Coroutinefunction:
                self.async_create_task(
                    action(),
                    f"track_timer({delay})",
                )

        handle = _call_later(delay, _target)
        _callbacks[delay] = lambda: handle.cancel()

    @typing.final
    class Event:
        """Mocks an Event-like object for usage in initial updates of our track_state api.
        This is based on the fact that we're not using all of the event attributes but
        just those set here."""

        if typing.TYPE_CHECKING:

            class DataType(typing.TypedDict):
                new_state: State

        data: "DataType"
        time_fired_timestamp: float

        __slots__ = (
            "data",
            "time_fired_timestamp",
        )

        def __init__(self, state: "State"):
            self.time_fired_timestamp = time()
            self.data = {"new_state": state}

    def track_state(
        self,
        entity_id: str,
        action: "Callable[[Event[EventStateChangedData] | CallbackTracker.Event], Any]",
        job_type: HassJobType = HassJobType.Callback,
        update: bool = True,
    ):
        """Track a state change for the given entity_id."""
        _callback = async_track_state_change_event(
            Manager.hass, entity_id, action, job_type
        )
        try:
            self._callbacks[entity_id] = _callback
        except AttributeError:
            self._callbacks = {entity_id: _callback}

        if update:
            state = Manager.hass.states.get(entity_id)
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                if job_type is HassJobType.Callback:
                    action(CallbackTracker.Event(state))
                elif job_type is HassJobType.Coroutinefunction:
                    self.async_create_task(
                        action(CallbackTracker.Event(state)),
                        f"track_state({entity_id})",
                    )
