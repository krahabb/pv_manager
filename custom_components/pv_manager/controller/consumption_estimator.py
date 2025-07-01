from . import EnergyEstimatorController
from .. import const as pmc
from ..processors.estimator_consumption_heuristic import HeuristicConsumptionEstimator
from ..processors.estimator_energy import SignalEnergyEstimator


class Controller(
    EnergyEstimatorController, HeuristicConsumptionEstimator, SignalEnergyEstimator
):
    """Base controller class for consumption estimation."""

    TYPE = pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR
