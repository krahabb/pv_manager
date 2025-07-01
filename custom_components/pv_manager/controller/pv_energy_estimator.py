"""
Controller for pv energy production estimation
"""

import typing

from . import EnergyEstimatorController
from .. import const as pmc
from ..processors.estimator_energy import SignalEnergyEstimator
from ..processors.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator
from ..sensor import DiagnosticSensor, EstimatorDiagnosticSensor

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Final

    from ..processors import Estimator
    from .devices.estimator_device import EnergyEstimatorDevice


# TODO: create a global generalization for diagnostic sensors linked to estimator
class ObservedRatioDiagnosticSensor(EstimatorDiagnosticSensor):

    def __init__(self, device: "EnergyEstimatorDevice", id: str):
        super().__init__(device, id, device)

    def on_estimator_update(self, estimator: HeuristicPVEnergyEstimator):
        self.update_safe(estimator.observed_ratio)


class WeatherModelDiagnosticSensor(EstimatorDiagnosticSensor):

    __slots__ = ("weather_param_index",)

    def __init__(self, device: "EnergyEstimatorDevice", id: str, index: int):
        self.weather_param_index = index
        super().__init__(device, id, device)

    def on_estimator_update(self, estimator: HeuristicPVEnergyEstimator):
        self.update_safe(estimator.weather_model.get_param(self.weather_param_index))


class DiagnosticDescr:

    if typing.TYPE_CHECKING:
        type InitT = Callable[[EnergyEstimatorDevice], DiagnosticSensor]
        type ValueT = Callable[[EnergyEstimatorDevice], Any]

        id: Final[str]
        init: Final[InitT]
        value: Final[ValueT]

    __slots__ = (
        "id",
        "init",
        "value",
    )

    def __init__(
        self,
        id: str,
        init_func: "InitT",
        value_func: "ValueT" = lambda c: None,
    ):
        self.id = id
        self.init = init_func
        self.value = value_func

    @staticmethod
    def Sensor(id: str, value_func: "ValueT"):
        return DiagnosticDescr(
            id,
            lambda device: DiagnosticSensor(
                device, id, native_value=value_func(device)
            ),
            value_func,
        )

    @staticmethod
    def EstimatorSensor(id: str, estimator_update_func: "Callable[[Estimator], Any]"):
        return DiagnosticDescr(
            id,
            lambda device: EstimatorDiagnosticSensor(
                device, id, device, estimator_update_func=estimator_update_func
            ),
            lambda device: estimator_update_func(device),
        )


DIAGNOSTIC_DESCR = {
    "observed_ratio": DiagnosticDescr(
        "observed_ratio",
        lambda device: ObservedRatioDiagnosticSensor(device, "observed_ratio"),
    ),
    "weather_cloud_constant_0": DiagnosticDescr(
        "weather_cloud_constant_0",
        lambda device: WeatherModelDiagnosticSensor(
            device, "weather_cloud_constant_0", 0
        ),
    ),
    "weather_cloud_constant_1": DiagnosticDescr(
        "weather_cloud_constant_1",
        lambda device: WeatherModelDiagnosticSensor(
            device, "weather_cloud_constant_1", 1
        ),
    ),
}


class Controller(
    EnergyEstimatorController, HeuristicPVEnergyEstimator, SignalEnergyEstimator
):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    @typing.override
    def _create_diagnostic_entities(self):
        diagnostic_entities = self.diagnostic_entities
        for d_e_d in DIAGNOSTIC_DESCR.values():
            if d_e_d.id not in diagnostic_entities:
                d_e_d.init(self)

    @typing.override
    def get_solar_forecast(self):
        return HeuristicPVEnergyEstimator.get_solar_forecast(self)
