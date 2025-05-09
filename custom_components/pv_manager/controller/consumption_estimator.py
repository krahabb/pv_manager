import typing

from .. import const as pmc
from ..controller import EnergyEstimatorController
from ..helpers import validation as hv
from ..processors.estimator_consumption_heuristic import (
    HeuristicConsumptionEstimator,
)

if typing.TYPE_CHECKING:
    from typing import Unpack

    from homeassistant.config_entries import ConfigEntry

class Controller(EnergyEstimatorController["Controller.Config"]):  # type: ignore
    """Base controller class for managing ConfigEntry behavior."""

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimatorController.Config):
            pass

    TYPE = pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR

    # interface: EnergyEstimatorController
    @staticmethod
    def get_config_entry_schema(config: "Config | None") -> pmc.ConfigSchema:
        return hv.entity_schema(
            config or {"name": "Consumption estimation"},
        ) | EnergyEstimatorController.get_config_entry_schema(config)

    def __init__(self, config_entry: "ConfigEntry"):
        super().__init__(config_entry, HeuristicConsumptionEstimator)
