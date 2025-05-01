from datetime import datetime, timedelta
import enum
import time
import typing

from homeassistant.components import sensor
from homeassistant.core import HassJob, callback
from homeassistant.helpers import event
from homeassistant.util import dt as dt_util

from . import const as pmc
from .helpers import entity as he
from .helpers.metering import CycleMode, MeteringCycle, MeteringEntity

if typing.TYPE_CHECKING:
    from typing import ClassVar, Final, NotRequired, Unpack

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller
    from .helpers.entity import EntityArgs

    SensorStateType = sensor.StateType | sensor.date | sensor.datetime | sensor.Decimal

    class SensorArgs(EntityArgs):
        device_class: NotRequired[sensor.SensorDeviceClass | None]
        state_class: NotRequired[sensor.SensorStateClass | None]
        native_value: NotRequired[SensorStateType]
        native_unit_of_measurement: NotRequired[str]


async def async_setup_entry(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry[Controller]",
    add_entities: "AddConfigEntryEntitiesCallback",
):
    await config_entry.runtime_data.async_setup_entry_platform(
        Sensor.PLATFORM, add_entities
    )


class Sensor(he.Entity, sensor.SensorEntity):

    PLATFORM = sensor.DOMAIN

    DeviceClass = sensor.SensorDeviceClass
    StateClass = sensor.SensorStateClass

    DEVICE_CLASS_TO_STATE_CLASS: dict[
        sensor.SensorDeviceClass | None, sensor.SensorStateClass | None
    ] = {
        None: StateClass.MEASUREMENT,  # generic numeric sensors
        DeviceClass.CURRENT: StateClass.MEASUREMENT,
        DeviceClass.ENUM: None,
        DeviceClass.ENERGY: StateClass.TOTAL_INCREASING,
        DeviceClass.POWER: StateClass.MEASUREMENT,
        DeviceClass.VOLTAGE: StateClass.MEASUREMENT,
    }

    _attr_device_class: "ClassVar[sensor.SensorDeviceClass | None]" = None
    _attr_native_value = None
    _attr_native_unit_of_measurement: "ClassVar[str | None]" = None

    __slots__ = (
        "device_class",
        "state_class",
        "native_value",
        "native_unit_of_measurement",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        **kwargs: "typing.Unpack[SensorArgs]",
    ):
        self.device_class = kwargs.pop("device_class", self._attr_device_class)
        self.native_value = kwargs.pop("native_value", self._attr_native_value)
        self.native_unit_of_measurement = kwargs.pop(
            "native_unit_of_measurement", self._attr_native_unit_of_measurement
        )
        if "state_class" in kwargs:
            self.state_class = kwargs.pop("state_class")
        else:
            self.state_class = self.DEVICE_CLASS_TO_STATE_CLASS.get(self.device_class)
        he.Entity.__init__(self, controller, id, **kwargs)

    def update(self, native_value: "SensorStateType"):
        if self.native_value != native_value:
            self.native_value = native_value
            self._async_write_ha_state()

    def update_safe(self, native_value: "SensorStateType"):
        if self.native_value != native_value:
            self.native_value = native_value
            if self.added_to_hass:
                self._async_write_ha_state()


class DiagnosticSensor(he.DiagnosticEntity, Sensor):
    pass


class EstimatorDiagnosticSensor(he.EstimatorEntity, DiagnosticSensor):

    __slots__ = he.EstimatorEntity._SLOTS_


class PowerSensor(Sensor):

    _attr_device_class = Sensor.DeviceClass.POWER
    _attr_native_unit_of_measurement = Sensor.hac.UnitOfPower.WATT

    def __init__(
        self,
        controller: "Controller",
        id: str,
        **kwargs: "Unpack[EntityArgs]",
    ):
        Sensor.__init__(
            self,
            controller,
            id,
            **kwargs,
        )


class EnergySensor(MeteringEntity, Sensor, he.RestoreEntity):

    CycleMode = CycleMode

    controller: "Controller"

    metering_cycle: "MeteringCycle"

    native_value: int
    _integral_value: float

    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_value = 0
    _attr_native_unit_of_measurement = Sensor.hac.UnitOfEnergy.WATT_HOUR

    __slots__ = (
        "cycle_mode",
        "metering_cycle",
        "last_reset",  # HA property
        "next_reset_ts",  # UTC timestamp of next reset
        "_integral_value",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        cycle_mode: CycleMode,
        **kwargs: "Unpack[EntityArgs]",
    ):
        self.cycle_mode = cycle_mode
        self.last_reset = None
        self._integral_value = 0

        if cycle_mode == CycleMode.TOTAL:
            self.accumulate = self._accumulate_total
        else:
            name = kwargs.pop("name") or id
            kwargs["name"] = self.formatted_name(name)

        Sensor.__init__(
            self,
            controller,
            f"{id}_{cycle_mode}",
            state_class=self.StateClass.TOTAL,
            **kwargs,
        )

    async def async_added_to_hass(self):

        self.metering_cycle = metering_cycle = MeteringCycle.register(self)
        self.next_reset_ts = metering_cycle.next_reset_ts
        self.last_reset = metering_cycle.last_reset_dt

        restored_data = self._async_get_restored_data()
        try:
            if (
                metering_cycle.last_reset_ts
                < restored_data.state.last_changed_timestamp  # type: ignore
                < metering_cycle.next_reset_ts
            ):
                extra_data = restored_data.extra_data.as_dict()  # type: ignore
                self._integral_value = extra_data["native_value"]
        except:
            pass

        self.native_value = int(self._integral_value)
        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self):
        self.metering_cycle.unregister(self)
        return await super().async_will_remove_from_hass()

    @property
    def extra_restore_state_data(self):
        return he.ExtraStoredDataDict({"native_value": self._integral_value})

    # interface: self
    def formatted_name(self, name: str):
        return f"{name} ({self.cycle_mode})"

    def accumulate(self, energy_wh: float, time_ts: float):
        # assert self.added_to_hass
        self._integral_value += energy_wh
        if time_ts >= self.next_reset_ts:
            # update done in _reset_cycle
            self.metering_cycle.update(time_ts)
            return

        _rounded = int(self._integral_value)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()

    def _accumulate_total(self, energy_wh: float, time_ts: float):
        """Custom 'accumulate' installed when cycle_mode == TOTAL"""
        # assert self.added_to_hass
        self._integral_value += energy_wh
        _rounded = int(self._integral_value)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()

    def _reset_cycle(self, metering_cycle: MeteringCycle):
        self.next_reset_ts = metering_cycle.next_reset_ts
        self.last_reset = metering_cycle.last_reset_dt
        self._integral_value -= self.native_value
        self.native_value = int(self._integral_value)
        self._async_write_ha_state()


class BatteryChargeSensor(Sensor, he.RestoreEntity):

    controller: "Controller"

    native_value: float
    charge: float

    _attr_icon = "mdi:battery"
    _attr_native_unit_of_measurement = "Ah"
    _attr_suggested_display_precision = 1

    __slots__ = (
        "capacity",
        "charge",
        "_current",
        "_current_ts",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        *,
        capacity: float,
        native_value: float = 0,
        **kwargs: "Unpack[EntityArgs]",
    ):
        self.capacity = capacity
        self.charge = native_value
        self._current = 0
        self._current_ts = None
        super().__init__(
            controller,
            id,
            native_value=native_value,
            **kwargs,
        )

    async def async_added_to_hass(self):
        restored_data = self._async_get_restored_data()
        try:
            extra_data = restored_data.extra_data.as_dict()  # type: ignore
            self.charge = extra_data["native_value"]
            self.native_value = round(self.charge, 1)
            self._current = 0
            self._current_ts = None
        except:
            pass
        await super().async_added_to_hass()

    @property
    def extra_restore_state_data(self):
        return he.ExtraStoredDataDict({"native_value": self.charge})

    def update(self, value: float):
        self.charge = value
        _rounded = round(self.charge, 1)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()

    def accumulate(self, value: float):
        self.update(self.charge + value)

    def update_current(self, current: float):
        now_ts = time.monotonic()
        if self._current_ts:
            charge = self.charge - (self._current * (now_ts - self._current_ts) / 3600)
            if charge < 0:
                charge = 0
                if current > 0:
                    current = 0
            elif charge > self.capacity:
                charge = self.capacity
                if current < 0:
                    current = 0
            self.update(charge)
        self._current = current
        self._current_ts = now_ts
        return self.charge
