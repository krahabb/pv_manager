import typing

from .. import const as pmc, controller
from ..helpers import validation as hv
from .common.estimator_consumption_heuristic import (
    HeuristicConsumptionEstimator,
)

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
    def get_config_entry_schema(config: EntryConfig | None) -> pmc.ConfigSchema:
        return hv.entity_schema(
            config or {"name": "Consumption estimation"},
        ) | controller.EnergyEstimatorController.get_config_entry_schema(config)

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(
            hass,
            config_entry,
            HeuristicConsumptionEstimator,
        )
