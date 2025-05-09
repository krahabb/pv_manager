"""
Controller for pv energy production estimation
"""

import datetime as dt
import enum
import typing

from homeassistant import const as hac

from .. import const as pmc, helpers
from ..controller import EnergyEstimatorController
from ..helpers import validation as hv
from ..manager import Manager
from ..processors.estimator_pvenergy import WEATHER_MODELS
from ..processors.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator
from ..sensor import DiagnosticSensor, EstimatorDiagnosticSensor

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Final, Unpack

    from homeassistant.config_entries import ConfigEntry

    from ..helpers.entity import EntityArgs
    from ..processors.estimator import Estimator


# TODO: create a global generalization for diagnostic sensors linked to estimator
class ObservedRatioDiagnosticSensor(EstimatorDiagnosticSensor):

    def __init__(self, controller: "Controller", id: str):
        super().__init__(controller, id, controller.estimator)

    def on_estimator_update(self, estimator: HeuristicPVEnergyEstimator):
        self.update_safe(estimator.observed_ratio)


class WeatherModelDiagnosticSensor(EstimatorDiagnosticSensor):

    __slots__ = ("weather_param_index",)

    def __init__(self, controller: "Controller", id: str, index: int):
        self.weather_param_index = index
        super().__init__(controller, id, controller.estimator)

    def on_estimator_update(self, estimator: HeuristicPVEnergyEstimator):
        self.update_safe(estimator.weather_model.get_param(self.weather_param_index))


class DiagnosticDescr:

    id: "Final[str]"
    init: "Final[Callable[[Controller], DiagnosticSensor]]"
    value: "Final[Callable[[Controller], Any]]"

    __slots__ = (
        "id",
        "init",
        "value",
    )

    def __init__(
        self,
        id: str,
        init_func: "Callable[[Controller], DiagnosticSensor]",
        value_func: "Callable[[Controller], Any]" = lambda c: None,
    ):
        self.id = id
        self.init = init_func
        self.value = value_func

    @staticmethod
    def Sensor(id: str, value_func: "Callable[[Controller], Any]"):
        return DiagnosticDescr(
            id,
            lambda c: DiagnosticSensor(c, id, native_value=value_func(c)),
            value_func,
        )

    @staticmethod
    def EstimatorSensor(id: str, estimator_update_func: "Callable[[Estimator], Any]"):
        return DiagnosticDescr(
            id,
            lambda c: EstimatorDiagnosticSensor(
                c, id, c.estimator, estimator_update_func=estimator_update_func
            ),
            lambda c: estimator_update_func(c.estimator),
        )


DIAGNOSTIC_DESCR = {
    "observed_ratio": DiagnosticDescr(
        "observed_ratio", lambda c: ObservedRatioDiagnosticSensor(c, "observed_ratio")
    ),
    "weather_cloud_constant_0": DiagnosticDescr(
        "weather_cloud_constant_0",
        lambda c: WeatherModelDiagnosticSensor(c, "weather_cloud_constant_0", 0),
    ),
    "weather_cloud_constant_1": DiagnosticDescr(
        "weather_cloud_constant_1",
        lambda c: WeatherModelDiagnosticSensor(c, "weather_cloud_constant_1", 1),
    ),
}


class Controller(EnergyEstimatorController["Controller.Config"]):  # type: ignore
    """Base controller class for managing ConfigEntry behavior."""

    if typing.TYPE_CHECKING:

        class Config(
            HeuristicPVEnergyEstimator.Config,
            EnergyEstimatorController.Config,
            pmc.EntityConfig,
        ):
            pass

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    estimator: "HeuristicPVEnergyEstimator"

    # interface: EnergyEstimatorController
    @staticmethod
    def get_config_entry_schema(config: "Config | None") -> pmc.ConfigSchema:
        _config = config or {
            "name": "PV energy estimation",
            "weather_model": "simple",
        }
        return (
            hv.entity_schema(_config)
            | {
                hv.opt_config("weather_entity_id", _config): hv.weather_selector(),
                hv.opt_config("weather_model", _config): hv.select_selector(
                    options=[
                        model_name for model_name in WEATHER_MODELS.keys() if model_name
                    ],
                ),
            }
            | EnergyEstimatorController.get_config_entry_schema(config)
        )

    def __init__(self, config_entry: "ConfigEntry"):
        super().__init__(config_entry, HeuristicPVEnergyEstimator)

    def _create_diagnostic_entities(self):
        diagnostic_entities = self.diagnostic_entities
        for d_e_d in DIAGNOSTIC_DESCR.values():
            if d_e_d.id not in diagnostic_entities:
                d_e_d.init(self)
