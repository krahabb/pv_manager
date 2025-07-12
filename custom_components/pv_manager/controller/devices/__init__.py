from enum import IntEnum
import typing

from ... import const as pmc
from ...binary_sensor import ProcessorWarningBinarySensor
from ...helpers import Loggable, validation as hv
from ...helpers.callback import CallbackTracker
from ...helpers.manager import Manager
from ...processors import BaseProcessor, EnergyBroadcast, SignalEnergyProcessor
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


class Device(CallbackTracker, Loggable):
    """Base concrete class for any device."""

    class Priority(IntEnum):
        """Used to determine the initialization order of devices in the controller.async_setup method."""

        HIGH = 10
        STANDARD = 100
        LOW = 1000

    if typing.TYPE_CHECKING:

        class Config(TypedDict):
            pass

        class Args(Loggable.Args):
            controller: Controller
            priority: NotRequired[int]
            config_subentry_id: NotRequired[str | None]
            config: NotRequired["Device.Config"]
            name: NotRequired[str]
            model: NotRequired[str]

        DEFAULT_NAME: ClassVar[str]
        controller: Final[Controller]
        priority: int
        config_subentry_id: Final[str | None]
        unique_id: Final[str]
        device_info: Final[DeviceInfo]
        name: Final[str]

    DEFAULT_NAME = ""
    __slots__ = (
        "controller",
        "priority",
        "config_subentry_id",
        "unique_id",
        "device_info",
        "name",
    )

    @classmethod
    def get_config_schema(cls, config: pmc.ConfigMapping | None, /) -> pmc.ConfigSchema:
        return {}

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):
        self.controller = controller = kwargs["controller"]
        self.priority = kwargs.get("priority") or Device.Priority.STANDARD
        self.config_subentry_id = kwargs.pop("config_subentry_id", None)
        config_entry_id = controller.config_entry.entry_id
        if controller is self:
            self.unique_id = config_entry_id
            via_device = None
        else:
            self.unique_id = f"{config_entry_id}.{id}"
            via_device = next(iter(controller.device_info["identifiers"]))  # type: ignore
        self.device_info = {"identifiers": {(pmc.DOMAIN, self.unique_id)}}
        self.name = (
            kwargs.pop("name")
            if "name" in kwargs
            else (kwargs["config"].get("name") if "config" in kwargs else None)
            or self.__class__.DEFAULT_NAME
            or (
                controller.config_entry.title
                if self.config_subentry_id is None
                else controller.config_entry.subentries[self.config_subentry_id].title
            )
        )

        Manager.device_registry.async_get_or_create(
            config_entry_id=config_entry_id,
            config_subentry_id=self.config_subentry_id,
            name=self.name,
            model=kwargs.pop("model", controller.TYPE),
            via_device=via_device,
            **self.device_info,  # type: ignore
        )
        if "logger" not in kwargs:
            kwargs["logger"] = controller
        super().__init__(id, **kwargs)
        controller.devices.append(self)

    async def async_start(self):
        pass

    def shutdown(self):
        self.controller.devices.remove(self)
        super().shutdown()
        # self.controller = None  # type: ignore

    async def async_update_entry(self, entry_data: "EntryData", /):
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


class EnergyMeterDevice(ProcessorDevice, EnergyBroadcast):
    """Device class (mainly used as a virtual base) featuring a set of metering sensors linked to self."""

    if typing.TYPE_CHECKING:

        class Config(ProcessorDevice.Config, pmc.EntityConfig):
            cycle_modes: NotRequired[list[EnergySensor.CycleMode]]
            """list of 'metering' sensors to configure"""

        class Args(ProcessorDevice.Args):
            config: "EnergyMeterDevice.Config"

        config: Config

    @classmethod
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
    async def async_update_entry(self, entry_data: "EntryData[Config]", /):
        self.config = config = entry_data.config
        await self._async_update_energy_sensors(
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
        energy_broadcast: EnergyBroadcast,
        cycle_modes: "Iterable[EnergySensor.CycleMode]",
        /,
    ):
        for cycle_mode in cycle_modes:
            EnergySensor(
                self,
                entity_id,
                cycle_mode,
                energy_broadcast,
                name=name,
            )

    async def _async_update_energy_sensors(
        self,
        entity_id: str,
        config_key: str,
        name: str,
        energy_broadcast: EnergyBroadcast,
        entry_data: "EntryData[Config]",
        /,
    ):
        cycle_modes_new = set(self.config.get(config_key, ()))
        for energy_sensor in tuple(entry_data.entities.values()):
            if isinstance(energy_sensor, EnergySensor) and (
                energy_sensor.energy_broadcast is energy_broadcast
            ):
                try:
                    cycle_modes_new.remove(energy_sensor.cycle_mode)  # type: ignore
                    # cycle_mode still present: update
                    energy_sensor.update_name(energy_sensor.formatted_name(name))
                except KeyError:
                    # cycle_mode removed from updated config
                    await energy_sensor.async_shutdown(True)
        # leftovers are those newly added cycle_mode(s)
        self._create_energy_sensors(entity_id, name, energy_broadcast, cycle_modes_new)


class SignalEnergyProcessorDevice(EnergyMeterDevice, SignalEnergyProcessor):
    """Device class featuring a functional signal energy processor with a set of
    configurable metering entities."""

    if typing.TYPE_CHECKING:

        class Config(SignalEnergyProcessor.Config, EnergyMeterDevice.Config):
            pass

        class Args(SignalEnergyProcessor.Args, EnergyMeterDevice.Args):
            config: "SignalEnergyProcessorDevice.Config"

        config: Config
