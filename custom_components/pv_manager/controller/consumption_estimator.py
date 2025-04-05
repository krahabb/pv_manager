import typing

from .common.estimator_consumption_heuristic import (
    Estimator_Consumption_Heuristic,
)
from .. import const as pmc, controller
from ..helpers import validation as hv


if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


class ControllerConfig(controller.EnergyEstimatorControllerConfig):
    pass


class EntryConfig(ControllerConfig, pmc.EntityConfig):
    pass


class Controller(controller.EnergyEstimatorController[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR

    __slots__ = ()

    # interface: EnergyEstimatorController
    @staticmethod
    def get_config_entry_schema(user_input: dict):

        return hv.entity_schema(
            user_input,
            name="Consumption estimation",
        ) | controller.EnergyEstimatorController.get_config_entry_schema(user_input)

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):

        super().__init__(
            hass,
            config_entry,
            Estimator_Consumption_Heuristic,
        )
