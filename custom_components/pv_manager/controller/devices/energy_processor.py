import datetime as dt
from time import time as TIME_TS
import typing

from homeassistant import const as hac

from custom_components.pv_manager.binary_sensor import ProcessorWarningBinarySensor

from . import Device
from ... import const as pmc
from ...processors import BaseEnergyProcessor, BaseProcessor

if typing.TYPE_CHECKING:
    from typing import Any, Callable, ClassVar, Coroutine, Final, Unpack

    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.device_registry import DeviceInfo
    from homeassistant.helpers.event import EventStateChangedData

    from .. import Controller




class EnergyProcessorDevice(
    BaseEnergyProcessor, Device
):
    if typing.TYPE_CHECKING:
        class Config(BaseEnergyProcessor.Config, Device.Config):
            pass

        class Args(BaseEnergyProcessor.Args, Device.Args):
            config: "EnergyProcessorDevice.Config"

    def __init__(self, id, **kwargs):

        super().__init__(id, **kwargs)

        for warning in self.warnings:
            ProcessorWarningBinarySensor(self, f"{warning.id}_warning", warning)
