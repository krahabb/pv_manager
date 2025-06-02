import typing

from .. import const as pmc, controller
from .devices import SignalEnergyProcessorDevice

if typing.TYPE_CHECKING:
    from typing import Unpack

    class EntryConfig(
        SignalEnergyProcessorDevice.Config,
        controller.Controller.Config,
    ):
        pass


class Controller(controller.Controller["EntryConfig"], SignalEnergyProcessorDevice):  # type: ignore
    """Energy calculator controller."""

    if typing.TYPE_CHECKING:
        Config = EntryConfig

    TYPE = pmc.ConfigEntryType.ENERGY_CALCULATOR
    DEFAULT_NAME = "Energy"

    PLATFORMS = {"sensor"}
