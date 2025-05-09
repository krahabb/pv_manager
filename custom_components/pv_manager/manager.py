"""
The component global api.
"""

import typing

from homeassistant import const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import callback
from homeassistant.helpers import (
    device_registry as dr,
    entity_registry as er,
    event,
)

from . import const as pmc
from .helpers import Loggable

if typing.TYPE_CHECKING:
    from logging import Logger
    from asyncio.events import TimerHandle
    from datetime import datetime
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


class ManagerClass(Loggable):
    """
    Singleton global manager/helper class
    """

    logger: "Logger"
    hass: "Final[HomeAssistant]"
    device_registry: "Final[dr.DeviceRegistry]"
    entity_registry: "Final[er.EntityRegistry]"
    _call_later: "Final[Callable[[float, Callable, ], TimerHandle]]"

    __slots__ = (
        "hass",
        "device_registry",
        "entity_registry",
        "_call_later",
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
            Manager._call_later = hass.loop.call_later
            hass.data[pmc.DOMAIN] = Manager

            async def _async_unload(_event) -> None:
                del hass.data[pmc.DOMAIN]
                Manager.hass = None  # type: ignore
                Manager.device_registry = None  # type: ignore
                Manager.entity_registry = None  # type: ignore
                Manager._call_later = None  # type: ignore

            hass.bus.async_listen_once(hac.EVENT_HOMEASSISTANT_STOP, _async_unload)

            return Manager

    def __init__(self):
        super().__init__(None)

    # interface: Loggable
    def log(self, level: int, msg: str, *args, **kwargs):
        if (logger := self.logger).isEnabledFor(level):
            logger._log(level, msg, args, **kwargs)

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
        def _callback(*_args):
            self.async_create_task(target(*_args), "._callback")

        return self._call_later(delay, _callback, *args)

    def schedule(self, delay: float, target: "Callable", *args):
        return self._call_later(delay, target, *args)

    def schedule_at(
        self,
        dt: "datetime",
        target: "HassJob[[datetime], Coroutine[Any, Any, None] | None] | Callable[[datetime], Coroutine[Any, Any, None] | None]",
    ):
        return event.async_track_point_in_utc_time(self.hass, target, dt)

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
