import typing

from . import EnergyEstimatorController
from .. import const as pmc
from ..helpers import validation as hv
from ..processors.estimator_consumption_heuristic import (
    HeuristicConsumptionEstimator,
)
from .devices.estimator_processor import SignalEnergyEstimatorDevice

if typing.TYPE_CHECKING:
    from typing import Unpack


class Controller(EnergyEstimatorController["Controller.Config"], HeuristicConsumptionEstimator):  # type: ignore
    """Base controller class for consumption estimation."""

    if typing.TYPE_CHECKING:

        class Config(
            EnergyEstimatorController.Config, HeuristicConsumptionEstimator.Config
        ):
            pass

    TYPE = pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR

    @staticmethod
    def get_config_entry_schema(config: "Config | None") -> pmc.ConfigSchema:
        # TODO: fix the class hierarchy for config building
        return SignalEnergyEstimatorDevice.get_config_schema(config)
