import typing

from homeassistant import const as hac
from homeassistant.core import callback

from ... import const as pmc
from ...helpers import Loggable
from ...helpers.callback import CallbackTracker
from ...manager import Manager

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        ClassVar,
        Coroutine,
        Final,
        NotRequired,
        TypedDict,
        Unpack,
    )

    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.device_registry import DeviceInfo
    from homeassistant.helpers.event import EventStateChangedData

    from .. import Controller


class Device(CallbackTracker, Loggable):

    if typing.TYPE_CHECKING:

        class Config(TypedDict):
            pass

        class Args(Loggable.Args):
            controller: Controller
            config: NotRequired["Device.Config"]
            name: NotRequired[str]
            model: NotRequired[str]

    controller: "Final[Controller]"
    device_info: "Final[DeviceInfo]"

    __slots__ = (
        "controller",
        "device_info",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        self.controller = controller = kwargs["controller"]
        entry_id = controller.config_entry.entry_id
        if controller is self:
            self.device_info = {"identifiers": {(pmc.DOMAIN, entry_id)}}
            via_device = None
        else:
            self.device_info = {"identifiers": {(pmc.DOMAIN, f"{entry_id}.{id}")}}
            via_device = controller.device_info["identifiers"]  # type: ignore
        Manager.device_registry.async_get_or_create(
            config_entry_id=entry_id,
            name=kwargs.get("name", controller.config_entry.title),
            model=kwargs.pop("model", controller.TYPE),
            via_device=via_device,
            **self.device_info,  # type: ignore
        )
        if "logger" not in kwargs:
            kwargs["logger"] = controller
        super().__init__(id, **kwargs)
        controller.devices[id] = self

    async def async_start(self):
        pass

    def shutdown(self):
        super().shutdown()
        self.controller = None # type: ignore
