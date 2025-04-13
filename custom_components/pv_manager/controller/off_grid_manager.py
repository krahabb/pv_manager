from dataclasses import dataclass
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
    PowerConverter,
)

from .. import const as pmc, controller
from ..binary_sensor import BinarySensor
from ..helpers import entity as he, validation as hv
from ..sensor import EnergySensor, Sensor

if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.core import Event, HomeAssistant, State

    from ..helpers.entity import EntityArgs


VOLTAGE_UNIT = hac.UnitOfElectricPotential.VOLT
CURRENT_UNIT = hac.UnitOfElectricCurrent.AMPERE
POWER_UNIT = hac.UnitOfPower.WATT
ENERGY_UNIT = hac.UnitOfEnergy.WATT_HOUR

# define a common name so to eventually switch to time.monotonic for time integration
TIME_TS = time.time


class MeteringSource(enum.StrEnum):
    BATTERY = enum.auto()
    BATTERY_IN = enum.auto()
    BATTERY_OUT = enum.auto()
    LOAD = enum.auto()
    LOSSES = enum.auto()
    PV = enum.auto()


@dataclass(slots=True)
class BaseMeter:
    controller: "typing.Final[Controller]"
    metering_source: "typing.Final[MeteringSource]"
    energy_sensors: "typing.Final[set[EnergySensor]]"
    time_ts: float
    value: float
    energy: float

    def __init__(self, controller: "Controller", metering_source: MeteringSource):
        self.controller = controller
        self.metering_source = metering_source
        self.energy_sensors = set()
        self.time_ts = 0  # no need to call TIME_TS here since initial value = 0
        self.value = 0
        self.energy = 0
        controller.energy_meters[metering_source] = self

    def add(self, value: float, time_ts: float):
        pass

    def interpolate(self, time_ts: float):
        pass

    def track_state(self, state: "State | None"):
        pass


class PowerMeter(BaseMeter):

    def add(self, value: float, time_ts: float):
        # left rectangle integration
        if time_ts > self.time_ts:
            if self.energy_sensors:
                energy = self.value * (time_ts - self.time_ts) / 3600
                self.energy += energy
                for sensor in self.energy_sensors:
                    sensor.accumulate(energy)
            else:
                self.energy += self.value * (time_ts - self.time_ts) / 3600
        self.value = value
        self.time_ts = time_ts

    def interpolate(self, time_ts: float):
        self.add(self.value, time_ts)

    def track_state(self, state: "State | None"):
        try:
            self.add(
                PowerConverter.convert(
                    float(state.state),  # type: ignore
                    state.attributes["unit_of_measurement"],  # type: ignore
                    POWER_UNIT,
                ),
                state.last_updated_timestamp,  # type: ignore
            )  # type: ignore
        except Exception as e:
            # this is expected and silently managed when state == None or 'unknown'
            self.add(0, TIME_TS())
            if state and state.state != hac.STATE_UNKNOWN:
                self.controller.log_exception(
                    self.controller.WARNING,
                    e,
                    "PowerMeter.track_state (state:%s)",
                    state,
                )


class EnergyMeter(BaseMeter):

    def _add_internal(self, value: float, time_ts: float):
        self.energy += value
        for sensor in self.energy_sensors:
            sensor.accumulate(value)
        self.value = value
        self.time_ts = time_ts


class BatteryPowerMeter(PowerMeter):

    in_meter: EnergyMeter
    out_meter: EnergyMeter

    __slots__ = (
        "in_meter",
        "out_meter",
    )

    def __init__(self, controller):
        super().__init__(controller, MeteringSource.BATTERY)
        self.in_meter = EnergyMeter(controller, MeteringSource.BATTERY_IN)
        self.out_meter = EnergyMeter(controller, MeteringSource.BATTERY_OUT)

    def add(self, value: float, time_ts: float):
        # left rectangle integration
        if time_ts > self.time_ts:
            energy = self.value * (time_ts - self.time_ts) / 3600
            self.energy += energy
            for sensor in self.energy_sensors:
                sensor.accumulate(energy)
            if energy > 0:
                self.out_meter._add_internal(energy, time_ts)
            else:
                self.in_meter._add_internal(-energy, time_ts)

        self.value = value
        self.time_ts = time_ts


class BatteryChargeSensor(Sensor, he.RestoreEntity):

    controller: "Controller"

    native_value: float
    _integral_value: float

    _attr_icon = "mdi:battery"
    _attr_native_value = 0
    _attr_native_unit_of_measurement = "Ah"

    __slots__ = ("_integral_value",)

    def __init__(self, controller: "Controller", **kwargs: "typing.Unpack[EntityArgs]"):
        self._integral_value = 0
        super().__init__(
            controller,
            "battery_charge",
            **kwargs,
        )

    async def async_added_to_hass(self):
        restored_data = self._async_get_restored_data()
        try:
            extra_data = restored_data.extra_data.as_dict()  # type: ignore
            self._integral_value = extra_data["native_value"]
            self.native_value = round(self._integral_value)
        except:
            pass
        await super().async_added_to_hass()

    @property
    def extra_restore_state_data(self):
        return he.ExtraStoredDataDict({"native_value": self._integral_value})

    def accumulate(self, value: float):
        self._integral_value += value
        _rounded = round(self._integral_value)
        if self.native_value != _rounded:
            self.native_value = _rounded
            self._async_write_ha_state()


MANAGER_ENERGY_SENSOR_NAME = "Energy"  # default name for ManagerEnergySensor


class ManagerEnergySensorConfig(pmc.EntityConfig, pmc.SubentryConfig):
    """Configure additional energy (metering) sensors for the various energy sources."""

    metering_source: MeteringSource
    cycle_modes: list[EnergySensor.CycleMode]


class ManagerEnergySensor(EnergySensor):

    controller: "Controller"

    energy_meter: typing.Final[BaseMeter]

    _attr_parent_attr = None

    __slots__ = ("energy_meter",)

    def __init__(
        self,
        energy_meter: BaseMeter,
        name: str,
        config_subentry_id: str,
        cycle_mode: EnergySensor.CycleMode,
    ):
        self.energy_meter = energy_meter
        super().__init__(
            energy_meter.controller,
            f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{config_subentry_id}",
            cycle_mode,
            name=name,
            config_subentry_id=config_subentry_id,
        )

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.energy_meter.energy_sensors.add(self)

    async def async_will_remove_from_hass(self):
        self.energy_meter.energy_sensors.remove(self)
        return await super().async_will_remove_from_hass()


class ControllerConfig(typing.TypedDict):
    battery_voltage_entity_id: str
    battery_current_entity_id: str
    battery_charge_entity_id: typing.NotRequired[str]

    battery_capacity: float

    pv_power_entity_id: typing.NotRequired[str]
    load_power_entity_id: typing.NotRequired[str]

    sampling_interval_minutes: int
    """Time resolution of model data"""
    maximum_latency_minutes: int
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""


class Controller(controller.Controller[EntryConfig]):
    """Off-grid plant manager: a collection of integrated helpers for a basic off-grid system
    with PV BATTERY and LOAD."""

    TYPE = pmc.ConfigEntryType.OFF_GRID_MANAGER
    PLATFORMS = {Sensor.PLATFORM}

    __slots__ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        "pv_power_entity_id",
        "load_power_entity_id",
        "sampling_interval_ts",
        "maximum_latency_ts",
        # state
        "battery_voltage",
        "battery_current",
        "_battery_current_last_ts",
        "battery_charge",
        "battery_meter",
        "pv_meter",
        "load_meter",
        "energy_meters",
        "_sampling_callback_unsub",
        # entities
        "battery_charge_sensor",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict) -> dict:
        return hv.entity_schema(user_input, name="Battery") | {
            hv.required("battery_voltage_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.VOLTAGE
            ),
            hv.required("battery_current_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.CURRENT
            ),
            hv.required(
                "battery_capacity", user_input, 100
            ): hv.positive_number_selector(unit_of_measurement="Ah"),
            hv.optional("pv_power_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.optional("load_power_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.required(
                "sampling_interval_minutes", user_input, 5
            ): hv.time_period_selector(
                min=1, unit_of_measurement=hac.UnitOfTime.MINUTES
            ),
            hv.required(
                "maximum_latency_minutes", user_input, 1
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
        }

    @staticmethod
    def get_config_subentry_schema(subentry_type: str, user_input):
        match subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                return hv.entity_schema(user_input, name=MANAGER_ENERGY_SENSOR_NAME) | {
                    hv.required("metering_source", user_input): hv.select_selector(
                        options=list(MeteringSource)
                    ),
                    hv.required("cycle_modes", user_input): hv.select_selector(
                        options=list(EnergySensor.CycleMode), multiple=True
                    ),
                }

        return {}

    def __init__(self, hass, config_entry):
        super().__init__(hass, config_entry)
        self.battery_voltage_entity_id = self.config["battery_voltage_entity_id"]
        self.battery_current_entity_id = self.config["battery_current_entity_id"]
        self.battery_charge_entity_id = self.config.get("battery_charge_entity_id")

        self.battery_capacity = self.config["battery_capacity"]

        self.pv_power_entity_id = self.config.get("pv_power_entity_id")
        self.load_power_entity_id = self.config.get("load_power_entity_id")

        self.sampling_interval_ts = self.config.get("sampling_interval_minutes", 5) * 60
        self.maximum_latency_ts = self.config.get("maximum_latency_minutes", 0) * 60

        self.battery_voltage: float = 0
        self.battery_current: float = 0
        self._battery_current_last_ts: float = 0
        self.battery_charge: float = 0

        self.energy_meters: typing.Final[dict[MeteringSource, BaseMeter]] = {}
        self._sampling_callback_unsub = None

        self.battery_charge_sensor: BatteryChargeSensor | None = None

        # TODO: setup according to some sort of configuration
        BatteryChargeSensor(
            self,
            name=f"{self.config["name"]} charge",
            parent_attr=Sensor.ParentAttr.DYNAMIC,
        )

        self.battery_meter = BatteryPowerMeter(self)
        """ REMOVE
        self.battery_meter.energy_in_sensors.add(
            EnergySensor(
                self,
                "battery_energy_in",
                EnergySensor.CycleMode.NONE,
                name=f"{self.config["name"]} energy in",
                parent_attr=None,
            )
        )
        self.battery_meter.energy_out_sensors.add(
            EnergySensor(
                self,
                "battery_energy_out",
                EnergySensor.CycleMode.NONE,
                name=f"{self.config["name"]} energy out",
                parent_attr=None,
            )
        )
        """

        self.track_state_update(
            self.battery_voltage_entity_id, self._battery_voltage_callback
        )
        self.track_state_update(
            self.battery_current_entity_id, self._battery_current_callback
        )
        if self.battery_charge_entity_id:
            self.track_state_update(
                self.battery_charge_entity_id, self._battery_charge_callback
            )

        if self.pv_power_entity_id:
            self.pv_meter = PowerMeter(self, MeteringSource.PV)
            self.track_state_update(self.pv_power_entity_id, self.pv_meter.track_state)

        if self.load_power_entity_id:
            self.load_meter = PowerMeter(self, MeteringSource.LOAD)
            self.track_state_update(
                self.load_power_entity_id, self.load_meter.track_state
            )

        self._sampling_callback_unsub = self.schedule_callback(
            self.sampling_interval_ts, self._sampling_callback
        )

        for subentry_id, config_subentry in config_entry.subentries.items():
            match config_subentry.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                    subentry_data: ManagerEnergySensorConfig = config_subentry.data  # type: ignore
                    energy_meter = self.energy_meters.get(
                        subentry_data["metering_source"]
                    )
                    if energy_meter:
                        name = subentry_data["name"]
                        if name == MANAGER_ENERGY_SENSOR_NAME:
                            name = f"{energy_meter.metering_source} {name}"
                        for cycle_mode in subentry_data["cycle_modes"]:
                            ManagerEnergySensor(
                                energy_meter, name, subentry_id, cycle_mode
                            )

    async def async_init(self):
        return await super().async_init()

    async def async_shutdown(self):
        if self._sampling_callback_unsub:
            self._sampling_callback_unsub.cancel()
            self._sampling_callback_unsub = None
        self.energy_meters.clear()
        await super().async_shutdown()

    # interface: self
    def _battery_voltage_callback(self, state: "State | None"):
        try:
            self.battery_voltage = ElectricPotentialConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                VOLTAGE_UNIT,
            )
            self.battery_meter.add(
                self.battery_voltage * self.battery_current,
                state.last_updated_timestamp,  # type: ignore
            )
        except Exception as e:
            self.battery_voltage = 0
            self.battery_meter.add(0, TIME_TS())
            if state and state.state != hac.STATE_UNKNOWN:
                self.log_exception(
                    self.WARNING, e, "_battery_voltage_update(state:%s)", state
                )

    def _battery_current_callback(self, state: "State | None"):
        try:
            time_ts = state.last_updated_timestamp  # type: ignore
            battery_current = ElectricCurrentConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                CURRENT_UNIT,
            )
        except Exception as e:
            time_ts = TIME_TS()
            battery_current = 0
            if state and state.state != hac.STATE_UNKNOWN:
                self.log_exception(
                    self.WARNING, e, "_battery_current_update(state:%s)", state
                )

        # left rectangle integration
        # We assume 'generator convention' for current
        # i.e. positive current = discharging
        if time_ts > self._battery_current_last_ts:
            charge = (
                self.battery_current * (self._battery_current_last_ts - time_ts) / 3600
            )
            self.battery_charge += charge
            if self.battery_charge_sensor:
                self.battery_charge_sensor.accumulate(charge)

        self.battery_current = battery_current
        self._battery_current_last_ts = time_ts

        self.battery_meter.add(self.battery_current * self.battery_voltage, time_ts)

    def _battery_charge_callback(self, state: "State | None"):
        pass

    @callback
    def _sampling_callback(self):
        time_ts = TIME_TS()
        self._sampling_callback_unsub = self.schedule_callback(
            self.sampling_interval_ts, self._sampling_callback
        )
        for meter in self.energy_meters.values():
            meter.interpolate(time_ts)
