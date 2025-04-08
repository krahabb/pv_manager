import datetime
import enum
import time
import typing

from homeassistant import const as hac
from homeassistant.core import HassJob, callback
from homeassistant.helpers import event
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    ElectricCurrentConverter,
    ElectricPotentialConverter,
)

from .. import const as pmc, controller
from ..binary_sensor import BinarySensor
from ..helpers import validation as hv
from ..sensor import EnergySensor, Sensor, RestoreSensor

if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State

    from ..helpers.entity import EntityArgs


class BatteryChargeSensor(RestoreSensor):

    controller: "Controller"

    native_value: float

    _attr_icon = "mdi:battery"
    _attr_native_unit_of_measurement = "Ah"

    def __init__(self, controller: "Controller", **kwargs: "typing.Unpack[EntityArgs]"):
        super().__init__(
            controller,
            "battery_charge",
            native_value=0,
            **kwargs,
        )

    async def async_added_to_hass(self):
        restored_sensor_data = await self.async_get_last_sensor_data()
        if restored_sensor_data:
            try:
                self.native_value = float(restored_sensor_data.native_value)  # type: ignore
            except Exception as e:
                self.log_exception(self.DEBUG, e, "restoring sensor state")
        await super().async_added_to_hass()

    def accumulate(self, value: float):
        self.native_value += value
        self._async_write_ha_state()


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

    VOLTAGE_UNIT = hac.UnitOfElectricPotential.VOLT
    CURRENT_UNIT = hac.UnitOfElectricCurrent.AMPERE

    __slots__ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        # state
        "battery_voltage",
        "battery_current",
        "_battery_current_last_ts",
        "battery_charge",
        "battery_charge_estimate",
        "battery_power",
        "_time_last_ts",
        "battery_charge_sensor",
        "battery_energy_in_none_sensor",
        "battery_energy_out_none_sensor",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict) -> dict:
        return hv.entity_schema(user_input, name="Battery") | {
            hv.required("battery_voltage_entity_id", user_input): hv.sensor_selector(
                device_class=EnergySensor.DeviceClass.VOLTAGE
            ),
            hv.required("battery_current_entity_id", user_input): hv.sensor_selector(
                device_class=EnergySensor.DeviceClass.CURRENT
            ),
            hv.required(
                "battery_capacity", user_input, 100
            ): hv.positive_number_selector(unit_of_measurement="Ah"),
        }

    def __init__(self, hass, config_entry):
        super().__init__(hass, config_entry)
        self.battery_voltage_entity_id = self.config["battery_voltage_entity_id"]
        self.battery_current_entity_id = self.config["battery_current_entity_id"]
        self.battery_charge_entity_id = self.config.get("battery_charge_entity_id")

        self.battery_capacity = self.config["battery_capacity"]
        self.battery_voltage: float | None = None
        self.battery_current: float | None = None
        self._battery_current_last_ts: float | None = None
        self.battery_charge = None
        self.battery_charge_estimate = None
        self.battery_power: float | None = None
        self._time_last_ts: float | None = None

        self.battery_charge_sensor: BatteryChargeSensor | None = None
        self.battery_energy_in_none_sensor: EnergySensor | None = None
        self.battery_energy_out_none_sensor: EnergySensor | None = None

        # TODO: setup according to some sort of configuration
        BatteryChargeSensor(
            self,
            name=f"{self.config["name"]} charge",
            parent_attr=EnergySensor.ParentAttr.DYNAMIC,
        )
        EnergySensor(
            self,
            "battery_energy_in",
            EnergySensor.CycleMode.NONE,
            name=f"{self.config["name"]} energy in",
            parent_attr=EnergySensor.ParentAttr.DYNAMIC,
        )
        EnergySensor(
            self,
            "battery_energy_out",
            EnergySensor.CycleMode.NONE,
            name=f"{self.config["name"]} energy out",
            parent_attr=EnergySensor.ParentAttr.DYNAMIC,
        )

    async def async_init(self):

        self.track_state_update(
            self.battery_voltage_entity_id, self._battery_voltage_update
        )
        self.track_state_update(
            self.battery_current_entity_id, self._battery_current_update
        )
        if self.battery_charge_entity_id:
            self.track_state_update(
                self.battery_charge_entity_id, self._battery_charge_update
            )

        return await super().async_init()

    async def async_shutdown(self):
        await super().async_shutdown()

    # interface: self
    def _battery_voltage_update(self, state: "State | None"):
        try:
            self.battery_voltage = ElectricPotentialConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                self.VOLTAGE_UNIT,
            )

        except Exception as e:
            self.battery_voltage = None
            if state and state.state != hac.STATE_UNKNOWN:
                self.log_exception(self.WARNING, e, "_battery_voltage_update")

    def _battery_current_update(self, state: "State | None"):
        try:
            battery_current = ElectricCurrentConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                self.CURRENT_UNIT,
            )
        except Exception as e:
            self.battery_current = None
            self.battery_power = None
            if state and state.state != hac.STATE_UNKNOWN:
                self.log_exception(self.WARNING, e, "_battery_current_update")
            return

        now_ts = time.monotonic()
        try:
            # left rectangle integration
            # We assume 'generator convention' for current
            # i.e. positive current = discharging
            charge = self.battery_current * (self._battery_current_last_ts - now_ts) / 3600  # type: ignore
            if self.battery_charge_sensor:
                self.battery_charge_sensor.accumulate(charge)
        except Exception as e:
            # catch self.battery_current == None
            charge = 0

        self.battery_current = battery_current
        self._battery_current_last_ts = now_ts

        try:
            self.battery_power = self.battery_voltage * battery_current  # type: ignore
            energy = self.battery_voltage * charge  # type: ignore

            if energy >= 0:
                if self.battery_energy_in_none_sensor:
                    self.battery_energy_in_none_sensor.accumulate(energy)
            else:
                if self.battery_energy_out_none_sensor:
                    self.battery_energy_out_none_sensor.accumulate(-energy)

        except Exception as e:
            # catch self.battery_voltage == None
            self.battery_power = None

    def _battery_charge_update(self, state: "State | None"):
        pass
