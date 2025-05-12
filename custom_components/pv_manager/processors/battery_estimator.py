import typing

from .estimator import SAMPLING_INTERVAL_MODULO, EnergyBalanceEstimator

if typing.TYPE_CHECKING:
    from typing import Unpack

    from .estimator import SignalEnergyEstimator


class BatteryEstimator(EnergyBalanceEstimator):

    if typing.TYPE_CHECKING:

        class Config(EnergyBalanceEstimator.Config):
            pass

        class Args(EnergyBalanceEstimator.Args):
            config: "BatteryEstimator.Config"

    class Forecast(EnergyBalanceEstimator.Forecast):

        __slots__ = ()

    forecasts: list[Forecast]

    _SLOTS_ = ()

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
