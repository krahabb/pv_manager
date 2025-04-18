"""
The component global api.
"""

import enum
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    event,
    json,
)

from . import const as pmc
from .helpers import Loggable

if typing.TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable, Coroutine, Final

    from homeassistant.core import HassJob, HomeAssistant


class ManagerClass(Loggable):
    """
    Singleton global manager/helper class
    """

    hass: "Final[HomeAssistant]"
    device_registry: "Final[dr.DeviceRegistry]"
    entity_registry: "Final[er.EntityRegistry]"

    __slots__ = (
        "hass",
        "device_registry",
        "entity_registry",
    )

    @staticmethod
    def get(hass: "HomeAssistant") -> "ManagerClass":
        """Safe binder call"""
        try:
            return hass.data[pmc.DOMAIN]
        except KeyError:
            Manager.hass = hass
            Manager.device_registry = dr.async_get(hass)
            Manager.entity_registry = er.async_get(hass)
            hass.data[pmc.DOMAIN] = Manager

            async def _async_unload(_event) -> None:
                Manager.hass = None  # type: ignore
                Manager.device_registry = None  # type: ignore
                Manager.entity_registry = None  # type: ignore
                hass.data.pop(pmc.DOMAIN)

            hass.bus.async_listen_once(hac.EVENT_HOMEASSISTANT_STOP, _async_unload)

            return Manager

    def __init__(self):
        super().__init__(None)

    # interface: Loggable
    def configure_logger(self):
        self.logtag = "Manager"

    # interface: self
    @callback
    def async_create_task[_R](
        self,
        target: "Coroutine[Any, Any, _R]",
        name: str,
        eager_start: bool = True,
    ):
        return self.hass.async_create_task(target, f"PVManager({name})", eager_start)

    def schedule_async(self, delay: float, target: "Callable[..., Coroutine]", *args):
        @callback
        def _callback(_target, *_args):
            self.async_create_task(_target(*_args), "._callback")

        return self.hass.loop.call_later(delay, _callback, target, *args)

    def schedule(self, delay: float, target: "Callable", *args):
        return self.hass.loop.call_later(delay, target, *args)

    def schedule_at(
        self,
        dt: "datetime",
        target: "HassJob[[datetime], Coroutine[Any, Any, None] | None] | Callable[[datetime], Coroutine[Any, Any, None] | None]",
    ):
        return event.async_track_point_in_utc_time(self.hass, target, dt)


Manager = ManagerClass()
