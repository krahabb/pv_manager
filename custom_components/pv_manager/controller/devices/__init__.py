import typing

from ... import const as pmc
from ...binary_sensor import ProcessorWarningBinarySensor
from ...helpers import Loggable
from ...helpers.callback import CallbackTracker
from ...manager import Manager
from ...processors import BaseProcessor, SignalEnergyProcessor

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
    """Base concrete class for any device."""

    if typing.TYPE_CHECKING:

        class Config(TypedDict):
            pass

        class Args(Loggable.Args):
            controller: Controller
            config: NotRequired["Device.Config"]
            name: NotRequired[str]
            model: NotRequired[str]

    DEFAULT_NAME: "ClassVar[str]" = ""

    controller: "Final[Controller]"
    unique_id: "Final[str]"
    device_info: "Final[DeviceInfo]"

    __slots__ = (
        "controller",
        "unique_id",
        "device_info",
    )

    @classmethod
    def get_config_schema(cls, config: pmc.ConfigMapping | None) -> "pmc.ConfigSchema":
        return {}

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        self.controller = controller = kwargs["controller"]
        entry_id = controller.config_entry.entry_id
        if controller is self:
            self.unique_id = entry_id
            via_device = None
        else:
            self.unique_id = f"{entry_id}.{id}"
            via_device = next(iter(controller.device_info["identifiers"]))  # type: ignore
        self.device_info = {"identifiers": {(pmc.DOMAIN, self.unique_id)}}
        Manager.device_registry.async_get_or_create(
            config_entry_id=entry_id,
            name=kwargs.get(
                "name", self.__class__.DEFAULT_NAME or controller.config_entry.title
            ),
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
        del self.controller.devices[self.id]
        super().shutdown()
        # self.controller = None  # type: ignore


class ProcessorDevice(BaseProcessor, Device):
    """Common class for any device representing any type of Processor."""

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config, Device.Config):
            pass

        class Args(BaseProcessor.Args, Device.Args):
            config: "ProcessorDevice.Config"

    def __init__(self, id, **kwargs: "Unpack[Args]"):

        super().__init__(id, **kwargs)

        for warning in self.warnings:
            ProcessorWarningBinarySensor(self, f"{warning.id}_warning", warning)


class SignalEnergyProcessorDevice(SignalEnergyProcessor, ProcessorDevice):
    """Device class featuring a functional signal energy processor.
    This class is then able to connect to a source entity and measure its energy."""

    if typing.TYPE_CHECKING:

        class Config(SignalEnergyProcessor.Config, Device.Config):
            pass

        class Args(SignalEnergyProcessor.Args, Device.Args):
            config: "SignalEnergyProcessorDevice.Config"

        def __init__(self, id, **kwargs: "Unpack[Args]"):

            super().__init__(id, **kwargs)
