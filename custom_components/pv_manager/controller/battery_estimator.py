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


class ControllerConfig(typing.TypedDict):
    battery_voltage_entity_id: str
    battery_current_entity_id: str
    battery_charge_entity_id: typing.NotRequired[str]

    battery_capacity: float


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""


class Controller(controller.Controller[EntryConfig]):
    """Battery charge estimator and utilities"""

    TYPE = pmc.ConfigEntryType.BATTERY_ESTIMATOR

    __slots__ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        # state
        "battery_voltage",
        "_battery_voltage_tracking_unsub",
        "battery_current",
        "_battery_current_tracking_unsub",
        "battery_charge",
        "_battery_charge_tracking_unsub",
    )

    def __init__(self, hass, config_entry):
        super().__init__(hass, config_entry)
        self.battery_voltage_entity_id = self.config["battery_voltage_entity_id"]
        self.battery_current_entity_id = self.config["battery_current_entity_id"]
        self.battery_charge_entity_id = self.config.get("battery_charge_entity_id")

        self._battery_voltage_tracking_unsub = None
        self._battery_current_tracking_unsub = None
        self._battery_charge_tracking_unsub = None

    async def async_init(self):

        self.track_state(
            self.battery_voltage_entity_id, self._battery_voltage_tracking_callback
        )

        return await super().async_init()

    async def async_shutdown(self):
        await super().async_shutdown()

    # interface: self
    @callback
    def _battery_voltage_tracking_callback(
        self, event: "Event[event.EventStateChangedData]"
    ):
        pass
