import typing

from ... import const as pmc
from ...binary_sensor import ProcessorWarningBinarySensor
from ...helpers import Loggable, validation as hv
from ...helpers.callback import CallbackTracker
from ...manager import Manager
from ...processors import BaseProcessor, SignalEnergyProcessor
from ...sensor import EnergySensor

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        ClassVar,
        Coroutine,
        Final,
        Iterable,
        NotRequired,
        Self,
        TypedDict,
        Unpack,
    )

    from homeassistant.helpers.device_registry import DeviceInfo

    from .. import Controller, EntryData
    from ...processors import EnergyBroadcast


class Device(CallbackTracker, Loggable):
    """Base concrete class for any device."""

    if typing.TYPE_CHECKING:

        class Config(TypedDict):
            pass

        class Args(Loggable.Args):
            controller: Controller
            config_subentry_id: NotRequired[str | None]
            config: NotRequired["Device.Config"]
            name: NotRequired[str]
            model: NotRequired[str]

        DEFAULT_NAME: ClassVar[str]
        controller: Final[Controller]
        config_subentry_id: Final[str | None]
        unique_id: Final[str]
        device_info: Final[DeviceInfo]

    DEFAULT_NAME = ""
    __slots__ = (
        "controller",
        "config_subentry_id",
        "unique_id",
        "device_info",
    )

    @classmethod
    def get_config_schema(cls, config: pmc.ConfigMapping | None, /) -> pmc.ConfigSchema:
        return {}

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):
        self.controller = controller = kwargs["controller"]
        self.config_subentry_id = kwargs.pop("config_subentry_id", None)
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
            config_subentry_id=self.config_subentry_id,
            name=kwargs.pop(
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

    async def update_entry(self, entry_data: "EntryData", /):
        """Entry point called when the config (sub)entry changes."""
        pass


class ProcessorDevice(BaseProcessor, Device):
    """Common class for any device representing any type of Processor."""

    # Inherit ProcessorDevice from BaseProcessor first so that in general we give
    # mro precedence to BaseProcessor hierarchy methods

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config, Device.Config):
            pass

        class Args(BaseProcessor.Args, Device.Args):
            config: "ProcessorDevice.Config"

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):

        super().__init__(id, **kwargs)

        for warning in self.warnings:
            ProcessorWarningBinarySensor(self, f"{warning.id}_warning", warning)


class SignalEnergyProcessorDevice(ProcessorDevice, SignalEnergyProcessor):
    """Device class featuring a functional signal energy processor.
    This class is then able to connect to a source entity and measure its energy."""

    if typing.TYPE_CHECKING:

        class Config(
            SignalEnergyProcessor.Config, ProcessorDevice.Config, pmc.EntityConfig
        ):
            cycle_modes: NotRequired[list[EnergySensor.CycleMode]]
            """list of 'metering' sensors to configure"""

        class Args(SignalEnergyProcessor.Args, Device.Args):
            config: "SignalEnergyProcessorDevice.Config"

        config: Config

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None", /) -> pmc.ConfigSchema:
        _config = config or {
            "name": cls.DEFAULT_NAME,
            "cycle_modes": [],
        }
        return (
            hv.entity_schema(_config)
            | super().get_config_schema(config)
            | {
                hv.opt_config("cycle_modes", _config): hv.cycle_modes_selector(),
            }
        )

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self._create_energy_sensors(
            "energy_sensor",
            self.config.get("name", self.__class__.DEFAULT_NAME),
            self,
            self.config.get("cycle_modes", ()),
        )

    @typing.override
    async def update_entry(self, entry_data: "EntryData[Config]", /):
        self.config = config = entry_data.config
        await self._update_energy_sensors(
            "energy_sensor",
            "cycle_modes",
            config.get("name", self.__class__.DEFAULT_NAME),
            self,
            entry_data,
        )

    def _create_energy_sensors(
        self,
        entity_id: str,
        name: str,
        energy_dispatcher: "EnergyBroadcast",
        cycle_modes: "Iterable[EnergySensor.CycleMode]",
        /,
    ):
        for cycle_mode in cycle_modes:
            EnergySensor(
                self,
                entity_id,
                cycle_mode,
                energy_dispatcher,
                name=name,
            )

    async def _update_energy_sensors(
        self,
        entity_id: str,
        config_key: str,
        name: str,
        energy_dispatcher: "EnergyBroadcast",
        entry_data: "EntryData[Config]",
        /,
    ):
        cycle_modes_new = set(self.config.get(config_key, ()))
        for energy_sensor in tuple(entry_data.entities.values()):
            if isinstance(energy_sensor, EnergySensor) and (
                energy_sensor.energy_dispatcher is energy_dispatcher
            ):
                try:
                    cycle_modes_new.remove(energy_sensor.cycle_mode)  # type: ignore
                    # cycle_mode still present: update
                    energy_sensor.update_name(energy_sensor.formatted_name(name))
                except KeyError:
                    # cycle_mode removed from updated config
                    await energy_sensor.async_shutdown(True)
        # leftovers are those newly added cycle_mode(s)
        self._create_energy_sensors(entity_id, name, energy_dispatcher, cycle_modes_new)
