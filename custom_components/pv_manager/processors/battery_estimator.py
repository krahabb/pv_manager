import typing

from .estimator import SAMPLING_INTERVAL_MODULO, Estimator

if typing.TYPE_CHECKING:
    from typing import Unpack

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


class BatteryEstimator(Estimator[float]):

    if typing.TYPE_CHECKING:

        class Config(Estimator.Config):
            sampling_interval_minutes: int
            """Time resolution of model data"""
            forecast_duration_hours: int

        class Args(Estimator.Args):
            config: "BatteryEstimator.Config"

    estimator_pv: "EnergyEstimator"
    estimator_consumption: "EnergyEstimator"

    forecasts: list[BatteryChargeForecast]

    _SLOTS_ = (
        "sampling_interval_ts",
        "forecast_duration_ts",
        "forecasts",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        config = kwargs["config"]
        self.sampling_interval_ts: typing.Final = (
            (
                (config.get("sampling_interval_minutes", 0) * 60)
                // SAMPLING_INTERVAL_MODULO
            )
            * SAMPLING_INTERVAL_MODULO
        ) or SAMPLING_INTERVAL_MODULO
        self.forecast_duration_ts = (
            ((config.get("forecast_duration_hours") or 1) * 3600)
            // self.sampling_interval_ts
        ) * self.sampling_interval_ts

        self.forecasts = [
            BatteryChargeForecast()
            for i in range(self.forecast_duration_ts // self.sampling_interval_ts)
        ]
        super().__init__(id, **kwargs)

    @typing.override
    def update_estimate(self):
        pass # TODO
