import typing

from ..common import BaseProcessor
from .estimator import SAMPLING_INTERVAL_MODULO, Estimator

if typing.TYPE_CHECKING:
    from .estimator import EnergyEstimator


class BatteryChargeForecast:

    time_ts: float

    pv_energy: float
    consumption_energy: float

    __slots__ = (
        "time_ts",
        "pv_energy",
        "consumption_energy",
    )

    def __init__(self):
        self.time_ts = 0
        self.pv_energy = 0
        self.consumption_energy = 0


class BatteryEstimatorConfig(typing.TypedDict):
    sampling_interval_minutes: int
    """Time resolution of model data"""
    forecast_duration_hours: int


class BatteryEstimator(Estimator, BaseProcessor[float]):

    estimator_pv: "EnergyEstimator"
    estimator_consumption: "EnergyEstimator"

    forecasts: list[BatteryChargeForecast]

    __slots__ = (
        "sampling_interval_ts",
        "forecast_duration_ts",
        "forecasts",
    ) + Estimator._SLOTS_

    def __init__(self, id, **kwargs: typing.Unpack[BatteryEstimatorConfig]):
        BaseProcessor.__init__(self, id)
        Estimator.__init__(self)
        self.sampling_interval_ts: typing.Final = (
            (
                ((kwargs.get("sampling_interval_minutes") or 10) * 60)
                // SAMPLING_INTERVAL_MODULO
            )
            * SAMPLING_INTERVAL_MODULO
        ) or SAMPLING_INTERVAL_MODULO
        self.forecast_duration_ts = (
            ((kwargs.get("forecast_duration_hours") or 1) * 3600)
            // self.sampling_interval_ts
        ) * self.sampling_interval_ts

        self.forecasts = [
            BatteryChargeForecast()
            for i in range(self.forecast_duration_ts // self.sampling_interval_ts)
        ]

    @typing.override
    def shutdown(self):
        Estimator.shutdown(self)

    @typing.override
    def update_estimate(self):
        pass
