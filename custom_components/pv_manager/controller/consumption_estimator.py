import typing

from .. import const as pmc
from ..controller import EnergyEstimatorController
from ..helpers import validation as hv
from ..processors.estimator_consumption_heuristic import (
    HeuristicConsumptionEstimator,
)

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
