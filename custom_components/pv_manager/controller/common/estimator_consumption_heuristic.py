from collections import deque
import dataclasses
import datetime as dt
import typing

from astral import sun

from .. import helpers
from .estimator import Estimator, Observation, ObservedEnergy

if typing.TYPE_CHECKING:
    from .estimator import EstimatorConfig



class Estimator_Consumption_Heuristic(Estimator):
    """
    Basic consumption estimator based off some heuristics.
    """

    __slots__ = (
    )

    def __init__(
        self,
        *,
        tzinfo: "dt.tzinfo",
        **kwargs: "typing.Unpack[EstimatorConfig]",
    ):
        Estimator.__init__(
            self,
            tzinfo=tzinfo,
            **kwargs,
        )

    # interface: Estimator
    def as_dict(self):
        return super().as_dict() | {
        }

    # interface: self
