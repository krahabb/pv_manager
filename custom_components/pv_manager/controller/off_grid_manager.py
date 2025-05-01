from dataclasses import dataclass
import datetime
import enum
import time
import typing

from homeassistant import const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import callback
from homeassistant.helpers import storage
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    ElectricCurrentConverter,
    ElectricPotentialConverter,
    PowerConverter,
)

from .. import const as pmc, controller
from ..binary_sensor import BinarySensor
from ..helpers import entity as he, validation as hv
from ..sensor import BatteryChargeSensor, EnergySensor, PowerSensor, Sensor
from ._energy_meters import (
    BaseMeter,
    BatteryMeter,
    LoadMeter,
    LossesMeter,
    MeteringSource,
    MeterStoreType,
    PvMeter,
)

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, Unpack

    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.core import Event, HomeAssistant, State

    from ..controller import EntryData
    from ..helpers.entity import EntityArgs


MAXIMUM_LATENCY_INFINITE = 1e6
VOLTAGE_UNIT = hac.UnitOfElectricPotential.VOLT
CURRENT_UNIT = hac.UnitOfElectricCurrent.AMPERE
POWER_UNIT = hac.UnitOfPower.WATT
ENERGY_UNIT = hac.UnitOfEnergy.WATT_HOUR

# define a common name so to eventually switch to time.monotonic for time integration
TIME_TS = time.time


class ControllerStoreType(typing.TypedDict):
    time: str
    time_ts: float
    battery_charge_estimate: float
    battery_capacity_estimate: float

    battery: "NotRequired[MeterStoreType]"
    battery_in: "NotRequired[MeterStoreType]"
    battery_out: "NotRequired[MeterStoreType]"
    load: "NotRequired[MeterStoreType]"
    losses: "NotRequired[MeterStoreType]"
    pv: "NotRequired[MeterStoreType]"


class ControllerStore(storage.Store[ControllerStoreType]):
    VERSION = 1

    def __init__(self, hass: "HomeAssistant", entry_id: str):
        super().__init__(
            hass,
            self.VERSION,
            f"{pmc.DOMAIN}.{Controller.TYPE}.{entry_id}",
        )


MANAGER_ENERGY_SENSOR_NAME = "Energy"  # default name for ManagerEnergySensor


class ManagerEnergySensorConfig(pmc.EntityConfig, pmc.SubentryConfig):
    """Configure additional energy (metering) sensors for the various energy sources."""

    metering_source: MeteringSource
    cycle_modes: list[EnergySensor.CycleMode]


class ManagerEnergySensor(EnergySensor):

    controller: "Controller"

    energy_meter: "Final[BaseMeter]"

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
        # TODO: rename sensor id using energy_meter instead of subentry_id
        super().__init__(
            energy_meter.controller,
            f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{energy_meter.metering_source}",
            cycle_mode,
            name=name,
            config_subentry_id=config_subentry_id,
        )

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.energy_meter.energy_listeners.add(self.accumulate)

    async def async_will_remove_from_hass(self):
        self.energy_meter.energy_listeners.remove(self.accumulate)
        return await super().async_will_remove_from_hass()


class YieldSensorId(enum.StrEnum):
    system_yield = enum.auto()
    battery_yield = enum.auto()
    conversion_yield = enum.auto()
    conversion_yield_actual = enum.auto()


class YieldSensor(Sensor):

    _attr_parent_attr = Sensor.ParentAttr.DYNAMIC
    _attr_native_unit_of_measurement = "%"

    def __init__(
        self,
        controller: "Controller",
        id: YieldSensorId,
        config_subentry_id: str,
        name: str,
    ):
        super().__init__(
            controller,
            id,
            config_subentry_id=config_subentry_id,
            name=name,
        )


class ManagerYieldConfig(pmc.EntityConfig, pmc.SubentryConfig):
    """Configure measuring of pv plant losses through observations of input power/energy measures
    (pv, battery and load). Losses are computed so:

    battery = battery_out - battery_in
    losses + load = pv + battery

    should mainly consist in inverter losses + cabling (or any consumption
    not measured through load) assuming load is the energy measured at the output of the inverter.

    In the long term i.e. excluding battery storage:

    system efficiency = load / pv (should include losses + battery losses)
    conversion efficiency = load / (load + losses) (mainly inverter losses)
    battery efficiency = battery_out / battery_in (sampled at same battery charge)
    """

    cycle_modes: list[EnergySensor.CycleMode]
    """For energy losses sensor(s)"""

    sampling_interval_seconds: int

    # Yield sensors names/enabler
    system_yield: "NotRequired[str]"
    battery_yield: "NotRequired[str]"
    conversion_yield: "NotRequired[str]"
    conversion_yield_actual: "NotRequired[str]"


class ControllerConfig(typing.TypedDict):
    battery_voltage_entity_id: str
    battery_current_entity_id: str
    battery_charge_entity_id: "NotRequired[str]"

    battery_capacity: float

    pv_power_entity_id: "NotRequired[str]"
    load_power_entity_id: "NotRequired[str]"

    maximum_latency_minutes: int
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""


class Controller(controller.Controller[EntryConfig]):
    """Off-grid plant manager: a collection of integrated helpers for a basic off-grid system
    with PV BATTERY and LOAD."""

    TYPE = pmc.ConfigEntryType.OFF_GRID_MANAGER
    PLATFORMS = {Sensor.PLATFORM}

    STORE_SAVE_PERIOD = 3600

    energy_meters: "Final[dict[MeteringSource, BaseMeter]]"
    battery_meter: BatteryMeter
    battery_in_meter: BaseMeter
    battery_out_meter: BaseMeter
    pv_meter: PvMeter
    load_meter: LoadMeter
    losses_meter: LossesMeter | None

    battery_charge_sensor: BatteryChargeSensor | None
    losses_power_sensor: PowerSensor | None
    system_yield_sensor: Sensor | None
    battery_yield_sensor: Sensor | None
    conversion_yield_sensor: Sensor | None
    conversion_yield_actual_sensor: Sensor | None

    __slots__ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        "pv_power_entity_id",
        "load_power_entity_id",
        "maximum_latency_ts",
        # state
        "_store",
        "battery_voltage",
        "battery_current",
        "_battery_current_last_ts",
        "battery_charge",
        "battery_charge_estimate",
        "battery_capacity_estimate",
        "energy_meters",
        "battery_meter",
        "battery_in_meter",
        "battery_out_meter",
        "pv_meter",
        "load_meter",
        "losses_meter",
        # entities
        "battery_charge_sensor",
        "losses_power_sensor",
        "system_yield_sensor",
        "battery_yield_sensor",
        "conversion_yield_sensor",
        "conversion_yield_actual_sensor",
        # callbacks
        "_store_save_callback_unsub",
        "_final_write_unsub",
    )

    @staticmethod
    def get_config_entry_schema(config: EntryConfig | None) -> pmc.ConfigSchema:
        if not config:
            config = {
                "name": "Off grid Manager",
                "battery_voltage_entity_id": "",
                "battery_current_entity_id": "",
                "battery_capacity": 100,
                "maximum_latency_minutes": 1,
            }
        return hv.entity_schema(config) | {
            hv.req_config("battery_voltage_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.VOLTAGE
            ),
            hv.req_config("battery_current_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.CURRENT
            ),
            hv.req_config("battery_capacity", config): hv.positive_number_selector(
                unit_of_measurement="Ah"
            ),
            hv.opt_config("pv_power_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.opt_config("load_power_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.opt_config("maximum_latency_minutes", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.MINUTES
            ),
        }

    @staticmethod
    def get_config_subentry_schema(
        subentry_type: str, config: pmc.ConfigMapping | None
    ) -> pmc.ConfigSchema:
        match subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                if not config:
                    config = {
                        "name": MANAGER_ENERGY_SENSOR_NAME,
                    }
                    return hv.entity_schema(config) | {
                        hv.req_config("metering_source", config): hv.select_selector(
                            options=list(MeteringSource)
                        ),
                        hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
                    }
                else:
                    return hv.entity_schema(config) | {
                        hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
                    }

            case pmc.ConfigSubentryType.MANAGER_YIELD:
                if not config:
                    config = {
                        "name": "Losses",
                        "sampling_interval_seconds": 10,
                        "system_yield": "System yield",
                        "battery_yield": "Battery yield",
                        "conversion_yield": "Conversion yield",
                        "conversion_yield_actual": "Conversion yield (actual)",
                    }
                return (
                    hv.entity_schema(config)
                    | {
                        hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
                        hv.req_config(
                            "sampling_interval_seconds", config
                        ): hv.time_period_selector(),
                    }
                    | {
                        hv.opt_config(sensor_id.name, config): str
                        for sensor_id in YieldSensorId
                    }
                )

        return {}

    @staticmethod
    def get_config_subentry_unique_id(
        subentry_type: str, user_input: pmc.ConfigMapping
    ) -> str | None:
        match subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                return f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}.{user_input["metering_source"]}"
            case pmc.ConfigSubentryType.MANAGER_YIELD:
                return pmc.ConfigSubentryType.MANAGER_YIELD
        return None

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        config: EntryConfig = config_entry.data  # type: ignore
        self.battery_voltage_entity_id = config["battery_voltage_entity_id"]
        self.battery_current_entity_id = config["battery_current_entity_id"]
        self.battery_charge_entity_id = config.get("battery_charge_entity_id")
        self.battery_capacity = config.get("battery_capacity", 0)
        self.pv_power_entity_id = config.get("pv_power_entity_id")
        self.load_power_entity_id = config.get("load_power_entity_id")
        self.maximum_latency_ts = (
            config.get("maximum_latency_minutes", MAXIMUM_LATENCY_INFINITE) * 60
        )

        self._store = ControllerStore(hass, config_entry.entry_id)
        self.battery_voltage: float = 0
        self.battery_current: float = 0
        self._battery_current_last_ts: float = 0
        self.battery_charge: float = 0
        self.battery_charge_estimate: float = 0
        self.battery_capacity_estimate: float = self.battery_capacity

        self.energy_meters = {}
        self.battery_meter = BatteryMeter(self)
        self.pv_meter = PvMeter(self)
        self.load_meter = LoadMeter(self)
        self.losses_meter = None

        self.battery_charge_sensor: BatteryChargeSensor | None = None
        self._store_save_callback_unsub = None
        self._final_write_unsub = None

        super().__init__(hass, config_entry)

    async def async_init(self):

        if store_data := await self._store.async_load():

            for meter in self.energy_meters.values():
                with self.exception_warning(
                    "loading %s meter data", meter.metering_source.value
                ):
                    meter.load(store_data[meter.metering_source.value])
            with self.exception_warning("loading battery_charge_estimate"):
                self.battery_charge_estimate = store_data["battery_charge_estimate"]
            with self.exception_warning("loading battery_capacity_estimate"):
                self.battery_capacity_estimate = store_data["battery_capacity_estimate"]

        self._store_save_callback_unsub = self.schedule_async_callback(
            self.STORE_SAVE_PERIOD, self._async_store_save
        )
        self._final_write_unsub = self.hass.bus.async_listen_once(
            hac.EVENT_HOMEASSISTANT_FINAL_WRITE,
            self._async_store_save,
        )

        # TODO: setup according to some sort of configuration
        BatteryChargeSensor(
            self,
            "battery_charge",
            capacity=self.battery_capacity,
            name="Battery charge",
            parent_attr=Sensor.ParentAttr.DYNAMIC,
        )

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
            self.track_state_update(self.pv_power_entity_id, self.pv_meter.track_state)
        if self.load_power_entity_id:
            self.track_state_update(
                self.load_power_entity_id, self.load_meter.track_state
            )

        await super().async_init()
        # trigger now after adding entities to hass
        if self.losses_meter:
            self.losses_meter.start()

    async def async_shutdown(self):
        if self._final_write_unsub:
            self._final_write_unsub()
            self._final_write_unsub = None
        if self._store_save_callback_unsub:
            self._store_save_callback_unsub.cancel()
            self._store_save_callback_unsub = None
        if self.losses_meter:
            self.losses_meter.stop()

        await self._async_store_save(0)

        for energy_meter in tuple(self.energy_meters.values()):
            energy_meter.shutdown()
        assert not self.energy_meters

        await super().async_shutdown()

    def _subentry_add(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                sensor_config: ManagerEnergySensorConfig = entry_data.config  # type: ignore[assignment]
                energy_meter = self.energy_meters.get(sensor_config["metering_source"])
                if energy_meter:
                    name = sensor_config["name"]
                    if name == MANAGER_ENERGY_SENSOR_NAME:
                        name = f"{energy_meter.metering_source} {name}"
                    for cycle_mode in sensor_config["cycle_modes"]:
                        ManagerEnergySensor(energy_meter, name, subentry_id, cycle_mode)
            case pmc.ConfigSubentryType.MANAGER_YIELD:
                assert not self.losses_meter
                yield_config: ManagerYieldConfig = entry_data.config  # type: ignore
                self.losses_meter = LossesMeter(
                    self, yield_config["sampling_interval_seconds"]
                )
                name = yield_config["name"]
                PowerSensor(
                    self,
                    "losses_power",
                    config_subentry_id=subentry_id,
                    name=name,
                    parent_attr=PowerSensor.ParentAttr.DYNAMIC,
                )
                for cycle_mode in yield_config["cycle_modes"]:
                    ManagerEnergySensor(
                        self.losses_meter, name, subentry_id, cycle_mode
                    )
                for yield_sensor_id in YieldSensorId:
                    setattr(self, f"{yield_sensor_id}_sensor", None)
                    if name := yield_config.get(yield_sensor_id):
                        YieldSensor(
                            self,
                            yield_sensor_id,
                            config_subentry_id=subentry_id,
                            name=name,
                        )
                if self.config_entry.state == ConfigEntryState.LOADED:
                    self.losses_meter.start()

    async def _async_subentry_update(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                sensor_config: ManagerEnergySensorConfig = entry_data.config  # type: ignore[assignment]
                energy_meter = self.energy_meters.get(sensor_config["metering_source"])
                if energy_meter:
                    name = sensor_config["name"]
                    if name == MANAGER_ENERGY_SENSOR_NAME:
                        name = f"{energy_meter.metering_source} {name}"
                    cycle_modes_new = set(sensor_config["cycle_modes"])
                    energy_sensor: ManagerEnergySensor
                    for energy_sensor in list(entry_data.entities.values()):  # type: ignore
                        try:
                            cycle_modes_new.remove(energy_sensor.cycle_mode)
                            # cycle_mode still present: update
                            energy_sensor.update_name(
                                energy_sensor.formatted_name(name)
                            )
                        except KeyError:
                            # cycle_mode removed from updated config
                            await energy_sensor.async_shutdown(True)
                    # leftovers are those newly added cycle_mode(s)
                    for cycle_mode in cycle_modes_new:
                        ManagerEnergySensor(energy_meter, name, subentry_id, cycle_mode)
                    # no state flush/update for entities since they're all integrating sensors
                    # and need a cycle to be computed/refreshed
            case pmc.ConfigSubentryType.MANAGER_YIELD:
                assert self.losses_meter
                yield_config: ManagerYieldConfig = entry_data.config  # type: ignore
                name = yield_config["name"]
                self.losses_meter._callback_interval_ts = yield_config[
                    "sampling_interval_seconds"
                ]
                entities = entry_data.entities
                # update PowerSensor
                if entity := entities.get("losses_power"):
                    entity.update_name(name)
                # update ManagerEnergySensors
                cycle_modes_new = set(yield_config["cycle_modes"])
                for energy_sensor in [
                    entity
                    for entity in entities.values()
                    if isinstance(entity, ManagerEnergySensor)
                ]:
                    try:
                        cycle_modes_new.remove(energy_sensor.cycle_mode)
                        # cycle_mode still present: update
                        energy_sensor.update_name(energy_sensor.formatted_name(name))
                    except KeyError:
                        # cycle_mode removed from updated config
                        await energy_sensor.async_shutdown(True)
                # leftovers are those newly added cycle_mode(s)
                for cycle_mode in cycle_modes_new:
                    ManagerEnergySensor(
                        self.losses_meter, name, subentry_id, cycle_mode
                    )
                # update yield sensors
                yield_sensors_id_new = {
                    yield_sensor_id
                    for yield_sensor_id in YieldSensorId
                    if yield_config.get(yield_sensor_id)
                }
                for yield_sensor_id in YieldSensorId:
                    try:
                        yield_sensor = entities[yield_sensor_id]
                    except KeyError:
                        # yield_sensor not (yet?) configured
                        continue
                    try:
                        yield_sensors_id_new.remove(yield_sensor_id)
                        # yield_sensor still present: update
                        yield_sensor.update_name(yield_config.get(yield_sensor_id))  # type: ignore
                    except KeyError:
                        # yield_sensor removed from updated config
                        await yield_sensor.async_shutdown(True)
                # leftovers are newly added yield sensors
                for yield_sensor_id in yield_sensors_id_new:
                    YieldSensor(
                        self,
                        yield_sensor_id,
                        config_subentry_id=subentry_id,
                        name=yield_config[yield_sensor_id],  # type: ignore
                    )

    async def _async_subentry_remove(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                # default cleanup will suffice
                pass
            case pmc.ConfigSubentryType.MANAGER_YIELD:
                assert self.losses_meter
                self.losses_meter.stop()
                self.losses_meter.shutdown()

    # interface: self
    def _battery_voltage_callback(self, state: "State | None"):
        try:
            self.battery_voltage = ElectricPotentialConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                VOLTAGE_UNIT,
            )
            self.battery_meter.process(
                self.battery_voltage * self.battery_current,
                state.last_updated_timestamp,  # type: ignore
            )
        except Exception as e:
            self.battery_voltage = 0
            self.battery_meter.process(0, TIME_TS())
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
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
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(
                    self.WARNING, e, "_battery_current_update(state:%s)", state
                )

        # left rectangle integration
        # We assume 'generator convention' for current
        # i.e. positive current = discharging
        d_time = time_ts - self._battery_current_last_ts
        if 0 < d_time < self.maximum_latency_ts:
            charge_out = self.battery_current * d_time / 3600
            self.battery_charge_estimate -= charge_out
            if self.battery_charge_estimate > self.battery_capacity_estimate:
                self.battery_capacity_estimate = self.battery_charge_estimate
            elif self.battery_charge_estimate < 0:
                self.battery_capacity_estimate -= self.battery_charge_estimate
                self.battery_charge_estimate = 0
            if self.battery_charge_sensor:
                self.battery_charge_sensor.update(self.battery_charge_estimate)

        self.battery_current = battery_current
        self._battery_current_last_ts = time_ts

        self.battery_meter.process(self.battery_current * self.battery_voltage, time_ts)

    def _battery_charge_callback(self, state: "State | None"):
        pass

    async def _async_store_save(self, *args):
        # args depends on the source of this call:
        # no args means we're being called by the loop scheduler i.e. periodic save
        # args[0] == event means we're in the final write stage of HA shutting down
        # args[0] == 0 means we're unloading the controller
        if not args:
            self._store_save_callback_unsub = self.schedule_async_callback(
                self.STORE_SAVE_PERIOD, self._async_store_save
            )
        elif args[0]:
            self._final_write_unsub = None

        now = dt_util.now()
        data = ControllerStoreType(
            {
                "time": now.isoformat(),
                "time_ts": now.timestamp(),
                "battery_charge_estimate": self.battery_charge_estimate,
                "battery_capacity_estimate": self.battery_capacity_estimate,
            }
        )
        for meter in self.energy_meters.values():
            data[meter.metering_source.value] = meter.save()
        await self._store.async_save(data)
