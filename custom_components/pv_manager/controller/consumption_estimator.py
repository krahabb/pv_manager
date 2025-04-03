import datetime
import enum
import time
import typing

from homeassistant import const as hac
from homeassistant.core import HassJob, callback
from homeassistant.helpers import event
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .. import const as pmc, controller
from ..binary_sensor import BinarySensor
from ..helpers import validation as hv
from ..sensor import RestoreSensor

if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State


class ControllerConfig(controller.EnergyEstimatorControllerConfig):
    pass

class EntryConfig(ControllerConfig, pmc.EntityConfig):
    pass



class Controller(controller.Controller[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR



