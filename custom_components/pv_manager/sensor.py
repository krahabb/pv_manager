import time
import typing

from homeassistant.components import sensor

from . import const as pmc
from .helpers import entity as he
from .manager import Manager, MeteringCycle

if typing.TYPE_CHECKING:
    from typing import ClassVar, Final, NotRequired, Unpack

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller, Device
    from .processors import EnergyBroadcast

    SensorStateType = sensor.StateType | sensor.date | sensor.datetime | sensor.Decimal


async def async_setup_entry(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry[Controller]",
    add_entities: "AddConfigEntryEntitiesCallback",
):
    await config_entry.runtime_data.async_setup_entry_platform(
        Sensor.PLATFORM, add_entities
    )


class Sensor(he.Entity, sensor.SensorEntity):

    if typing.TYPE_CHECKING:

        class Args(he.Entity.Args):
            device_class: NotRequired[sensor.SensorDeviceClass | None]
            state_class: NotRequired[sensor.SensorStateClass | None]
            native_value: NotRequired[SensorStateType]
            native_unit_of_measurement: NotRequired[str]
            suggested_display_precision: NotRequired[int]

        _attr_device_class: ClassVar[sensor.SensorDeviceClass | None]
        _attr_native_unit_of_measurement: ClassVar[str | None]
        _attr_suggested_display_precision: ClassVar[int | None]

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

    _attr_device_class = None
    _attr_native_value = None
    _attr_native_unit_of_measurement = None
    _attr_suggested_display_precision = None

    __slots__ = (
        "device_class",
        "state_class",
        "native_value",
        "native_unit_of_measurement",
        "suggested_display_precision",
    )

    def __init__(self, device: "Device", id: str, /, **kwargs: "Unpack[Args]"):
        self.device_class = kwargs.pop("device_class", self._attr_device_class)
        self.native_value = kwargs.pop("native_value", self._attr_native_value)
        self.native_unit_of_measurement = kwargs.pop(
            "native_unit_of_measurement", self._attr_native_unit_of_measurement
        )
        self.suggested_display_precision = kwargs.pop(
            "suggested_display_precision", self._attr_suggested_display_precision
        )
        if "state_class" in kwargs:
            self.state_class = kwargs.pop("state_class")
        else:
            self.state_class = self.DEVICE_CLASS_TO_STATE_CLASS.get(self.device_class)
        super().__init__(device, id, **kwargs)

    @typing.override
    def update(self, native_value: "SensorStateType", /):
        if self.native_value != native_value:
            self.native_value = native_value
            self._async_write_ha_state()

    @typing.override
    def update_safe(self, native_value: "SensorStateType", /):
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
    _attr_suggested_display_precision = 0


class EnergySensor(MeteringCycle.Sink, Sensor, he.RestoreEntity):

    CycleMode = MeteringCycle.Mode

    if typing.TYPE_CHECKING:
        cycle_mode: Final[CycleMode]
        energy_dispatcher: Final[EnergyBroadcast]

        _metering_cycle: "MeteringCycle"

        native_value: int
        _integral_value: float

    _attr_parent_attr = None

    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_value = 0
    _attr_native_unit_of_measurement = Sensor.hac.UnitOfEnergy.WATT_HOUR

    __slots__ = (
        "cycle_mode",
        "energy_dispatcher",
        "last_reset",  # HA property
        "_integral_value",
        "_energy_dispatcher_unsub_",
        "_metering_cycle",
        "_next_reset_ts",  # UTC timestamp of next reset
    )

    def __init__(
        self,
        device: "Device",
        id: str,
        cycle_mode: CycleMode,
        energy_dispatcher: "EnergyBroadcast",
        /,
        **kwargs: "Unpack[he.Entity.Args]",
    ):
        self.cycle_mode = cycle_mode
        self.energy_dispatcher = energy_dispatcher
        self.last_reset = None
        self._integral_value = 0
        self._energy_dispatcher_unsub_ = None

        if cycle_mode == MeteringCycle.Mode.TOTAL:
            self.accumulate = self._accumulate_total
        else:
            name = kwargs.pop("name") or id
            kwargs["name"] = self.formatted_name(name)

        Sensor.__init__(
            self,
            device,
            f"{id}_{cycle_mode}",
            state_class=self.StateClass.TOTAL,
            **kwargs,
        )

    async def async_added_to_hass(self):
        self._metering_cycle = metering_cycle = Manager.register_metering_synk(self)
        self._next_reset_ts = metering_cycle.next_reset_ts
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
        self._energy_dispatcher_unsub_ = self.energy_dispatcher.listen_energy(
            self.accumulate
        )

    async def async_will_remove_from_hass(self):
        if self._energy_dispatcher_unsub_:
            self._energy_dispatcher_unsub_()
            self._energy_dispatcher_unsub_ = None
        self._metering_cycle.unregister(self)
        return await super().async_will_remove_from_hass()

    @property
    def extra_restore_state_data(self):
        return he.ExtraStoredDataDict({"native_value": self._integral_value})

    # interface: self
    def formatted_name(self, name: str, /):
        return f"{name} ({self.cycle_mode})"

    def accumulate(self, energy_wh: float, time_ts: float, /):
        # assert self.added_to_hass
        self._integral_value += energy_wh
        if time_ts >= self._next_reset_ts:
            # update done in _reset_cycle
            self._metering_cycle.update(time_ts)
            return

        _rounded = int(self._integral_value)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()

    def _accumulate_total(self, energy_wh: float, time_ts: float, /):
        """Custom 'accumulate' installed when cycle_mode == TOTAL"""
        # assert self.added_to_hass
        self._integral_value += energy_wh
        _rounded = int(self._integral_value)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()

    @typing.override
    def _reset_cycle(self, metering_cycle: MeteringCycle, /):
        self._next_reset_ts = metering_cycle.next_reset_ts
        self.last_reset = metering_cycle.last_reset_dt
        self._integral_value -= self.native_value
        self.native_value = int(self._integral_value)
        self._async_write_ha_state()


class BatteryChargeSensor(Sensor, he.RestoreEntity):

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
        device: "Device",
        id: str,
        /,
        *,
        capacity: float,
        native_value: float = 0,
        **kwargs: "Unpack[he.Entity.Args]",
    ):
        self.capacity = capacity
        self.charge = native_value
        self._current = 0
        self._current_ts = None
        super().__init__(
            device,
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

    @typing.override
    def update(self, value: float, /):
        self.charge = value
        _rounded = round(self.charge, 1)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()

    def accumulate(self, value: float, /):
        self.update(self.charge + value)

    def update_current(self, current: float, /):
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
