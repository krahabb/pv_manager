import enum
import typing

from homeassistant import const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import storage
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    ElectricCurrentConverter,
    ElectricPotentialConverter,
)

from .. import const as pmc, controller
from ..binary_sensor import ProcessorWarningBinarySensor
from ..helpers import validation as hv
from ..manager import Manager
from ..processors import (
    MAXIMUM_LATENCY_DISABLED,
    SAFE_MAXIMUM_POWER_DISABLED,
    SAFE_MINIMUM_POWER_DISABLED,
)
from ..processors.battery_estimator import BatteryEstimator
from ..sensor import BatteryChargeSensor, EnergySensor, PowerSensor, Sensor
from ._energy_meters import (
    TIME_TS,
    BaseMeter,
    BatteryMeter,
    LoadMeter,
    LossesMeter,
    MeterStoreType,
    PvMeter,
    SourceType,
)

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, TypedDict, Unpack

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, EventStateChangedData, State

    from ..controller import EntryData
    from ..helpers.entity import EntityArgs
    from ..processors import ProcessorWarning


VOLTAGE_UNIT = hac.UnitOfElectricPotential.VOLT
CURRENT_UNIT = hac.UnitOfElectricCurrent.AMPERE


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

    def __init__(self, entry_id: str):
        super().__init__(
            Manager.hass,
            self.VERSION,
            f"{pmc.DOMAIN}.{Controller.TYPE}.{entry_id}",
        )


MANAGER_ENERGY_SENSOR_NAME = "Energy"  # default name for ManagerEnergySensor


class ManagerEnergySensorConfig(pmc.EntityConfig, pmc.SubentryConfig):
    """Configure additional energy (metering) sensors for the various energy sources."""

    metering_source: SourceType
    cycle_modes: list[EnergySensor.CycleMode]


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


class Controller(controller.Controller["Controller.Config"]):  # type: ignore
    """Off-grid plant manager: a collection of integrated helpers for a basic off-grid system
    with PV BATTERY and LOAD."""

    if typing.TYPE_CHECKING:

        class BatteryConfig(TypedDict):
            battery_voltage_entity_id: str
            battery_current_entity_id: str
            safe_maximum_battery_power_w: NotRequired[float]
            battery_charge_entity_id: NotRequired[str]
            battery_capacity: float

        class PVConfig(TypedDict):
            pv_power_entity_id: NotRequired[str]
            safe_maximum_pv_power_w: NotRequired[float]

        class LoadConfig(TypedDict):
            load_power_entity_id: NotRequired[str]
            safe_maximum_load_power_w: NotRequired[float]

        class Config(pmc.EntityConfig, controller.Controller.Config):
            battery: "Controller.BatteryConfig"
            pv: "Controller.PVConfig"
            load: "Controller.LoadConfig"
            maximum_latency_seconds: NotRequired[int]
            """Maximum time between source power/energy samples before considering an error in data sampling."""

    TYPE = pmc.ConfigEntryType.OFF_GRID_MANAGER
    PLATFORMS = {Sensor.PLATFORM}

    STORE_SAVE_PERIOD = 3600

    config: "Config"

    energy_meters: "Final[dict[SourceType, BaseMeter]]"
    battery_meter: BatteryMeter
    battery_in_meter: BaseMeter
    battery_out_meter: BaseMeter
    pv_meter: PvMeter
    load_meter: LoadMeter
    losses_meter: LossesMeter | None

    battery_estimator: BatteryEstimator

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
        "battery_estimator",
        # entities
        "battery_charge_sensor",
        "losses_power_sensor",
        "system_yield_sensor",
        "battery_yield_sensor",
        "conversion_yield_sensor",
        "conversion_yield_actual_sensor",
        # callbacks
        "_final_write_unsub",
    )

    @staticmethod
    def get_config_entry_schema(config: "Config | None") -> pmc.ConfigSchema:
        if not config:
            config = {
                "name": "Off grid Manager",
                "battery": {
                    "battery_voltage_entity_id": "",
                    "battery_current_entity_id": "",
                    "battery_capacity": 100,
                },
                "pv": {},
                "load": {},
            }
        return hv.entity_schema(config) | {
            hv.vol.Required("battery"): hv.section(
                hv.vol.Schema(
                    {
                        hv.req_config(
                            "battery_voltage_entity_id", config
                        ): hv.sensor_selector(device_class=Sensor.DeviceClass.VOLTAGE),
                        hv.req_config(
                            "battery_current_entity_id", config
                        ): hv.sensor_selector(device_class=Sensor.DeviceClass.CURRENT),
                        hv.opt_config(
                            "safe_maximum_battery_power_w", config
                        ): hv.positive_number_selector(unit_of_measurement="W"),
                        hv.req_config(
                            "battery_capacity", config
                        ): hv.positive_number_selector(unit_of_measurement="Ah"),
                    }
                ),
                {"collapsed": True},
            ),
            hv.vol.Required("pv"): hv.section(
                hv.vol.Schema(
                    {
                        hv.opt_config("pv_power_entity_id", config): hv.sensor_selector(
                            device_class=Sensor.DeviceClass.POWER
                        ),
                        hv.opt_config(
                            "safe_maximum_pv_power_w", config
                        ): hv.positive_number_selector(unit_of_measurement="W"),
                    }
                ),
                {"collapsed": True},
            ),
            hv.vol.Required("load"): hv.section(
                hv.vol.Schema(
                    {
                        hv.opt_config(
                            "load_power_entity_id", config
                        ): hv.sensor_selector(device_class=Sensor.DeviceClass.POWER),
                        hv.opt_config(
                            "safe_maximum_load_power_w", config
                        ): hv.positive_number_selector(unit_of_measurement="W"),
                    }
                ),
                {"collapsed": True},
            ),
            hv.opt_config("maximum_latency_seconds", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
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
                            options=list(SourceType)
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

    def __init__(self, config_entry: "ConfigEntry"):
        self.energy_meters = {}
        self._store = ControllerStore(config_entry.entry_id)
        self.battery_voltage: float = 0
        self.battery_current: float = 0
        self._battery_current_last_ts: float = 0
        self.battery_charge: float = 0
        self.battery_charge_estimate: float = 0
        self.battery_capacity_estimate: float = 0

        self.losses_meter = None

        self.battery_charge_sensor: BatteryChargeSensor | None = None
        self._final_write_unsub = None

        super().__init__(config_entry)

        # TODO: setup according to some sort of configuration
        BatteryChargeSensor(
            self,
            "battery_charge",
            capacity=self.battery_capacity,
            name="Battery charge",
            parent_attr=Sensor.ParentAttr.DYNAMIC,
        )

        _warning_meters = (self.battery_meter, self.load_meter, self.pv_meter)
        _warning_processors_map: dict[str, set["ProcessorWarning"]] = {}
        for _meter in _warning_meters:
            for _warning_processor in _meter.warnings:
                try:
                    _warning_processors_map[_warning_processor.id].add(
                        _warning_processor
                    )
                except KeyError:
                    _warning_processors_map[_warning_processor.id] = {
                        _warning_processor
                    }
        for _warning_id, _warning_processors in _warning_processors_map.items():
            ProcessorWarningBinarySensor(
                self, f"{_warning_id}_warning", _warning_processors
            )


    def _on_init(self):
        config: "Controller.Config" = self.config # type: ignore
        maximum_latency_seconds = (
            config.get("maximum_latency_seconds") or MAXIMUM_LATENCY_DISABLED
        )
        self.maximum_latency_ts = maximum_latency_seconds

        config_battery = config["battery"]
        self.battery_voltage_entity_id = config_battery["battery_voltage_entity_id"]
        self.battery_current_entity_id = config_battery["battery_current_entity_id"]
        self.battery_charge_entity_id = config_battery.get("battery_charge_entity_id")
        self.battery_capacity = config_battery.get("battery_capacity", 0)
        _safe_maximum_battery_power_w = config_battery.get(
            "safe_maximum_battery_power_w", SAFE_MAXIMUM_POWER_DISABLED
        )
        self.battery_meter = BatteryMeter(
            self,
            {
                "maximum_latency_seconds": maximum_latency_seconds,
                "safe_maximum_power_w": _safe_maximum_battery_power_w,
                "safe_minimum_power_w": (
                    SAFE_MINIMUM_POWER_DISABLED
                    if _safe_maximum_battery_power_w is SAFE_MAXIMUM_POWER_DISABLED
                    else -_safe_maximum_battery_power_w
                ),
            },
        )

        config_pv = config["pv"]
        self.pv_meter = PvMeter(
            self,
            {
                "source_entity_id": config_pv.get("pv_power_entity_id"),
                "maximum_latency_seconds": maximum_latency_seconds,
                "safe_maximum_power_w": config_pv.get(
                    "safe_maximum_pv_power_w", SAFE_MAXIMUM_POWER_DISABLED
                ),
                "safe_minimum_power_w": 0,
            },
        )

        config_load = config["load"]
        self.load_meter = LoadMeter(
            self,
            {
                "source_entity_id": config_load.get("load_power_entity_id"),
                "maximum_latency_seconds": maximum_latency_seconds,
                "safe_maximum_power_w": config_load.get(
                    "safe_maximum_load_power_w", SAFE_MAXIMUM_POWER_DISABLED
                ),
                "safe_minimum_power_w": 0,
            },
        )

        return super()._on_init()

    async def async_setup(self):

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

        self.track_timer(self.STORE_SAVE_PERIOD, self._async_store_save)
        self._final_write_unsub = Manager.hass.bus.async_listen_once(
            hac.EVENT_HOMEASSISTANT_FINAL_WRITE,
            self._async_store_save,
        )

        self.track_state(
            self.battery_voltage_entity_id,
            self._battery_voltage_callback,
            Controller.HassJobType.Callback,
        )
        self.track_state(
            self.battery_current_entity_id,
            self._battery_current_callback,
            Controller.HassJobType.Callback,
        )
        if self.battery_charge_entity_id:
            self.track_state(
                self.battery_charge_entity_id,
                self._battery_charge_callback,
                Controller.HassJobType.Callback,
            )

        await self.pv_meter.async_start()
        await self.load_meter.async_start()

        await super().async_setup()
        # trigger now after adding entities to hass
        if self.losses_meter:
            await self.losses_meter.async_start()

    async def async_shutdown(self):
        if self._final_write_unsub:
            self._final_write_unsub()
            self._final_write_unsub = None

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
                        EnergySensor(
                            self,
                            f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{energy_meter.metering_source}",
                            cycle_mode,
                            energy_meter,
                            name=name,
                            config_subentry_id=subentry_id,
                        )
            case pmc.ConfigSubentryType.MANAGER_YIELD:
                assert not self.losses_meter
                yield_config: ManagerYieldConfig = entry_data.config  # type: ignore
                self.losses_meter = losses_meter = LossesMeter(
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
                    EnergySensor(
                        self,
                        f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{losses_meter.metering_source}",
                        cycle_mode,
                        losses_meter,
                        name=name,
                        config_subentry_id=subentry_id,
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
                    self.async_create_task(
                        losses_meter.async_start(), "LossesMeter.async_start"
                    )

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
                    energy_sensor: EnergySensor
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
                        EnergySensor(
                            self,
                            f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{energy_meter.metering_source}",
                            cycle_mode,
                            energy_meter,
                            name=name,
                            config_subentry_id=subentry_id,
                        )
                    # no state flush/update for entities since they're all integrating sensors
                    # and need a cycle to be computed/refreshed
            case pmc.ConfigSubentryType.MANAGER_YIELD:
                losses_meter = self.losses_meter
                assert losses_meter
                yield_config: ManagerYieldConfig = entry_data.config  # type: ignore
                name = yield_config["name"]
                # TODO: retrigger the track_timer
                losses_meter.update_period_ts = yield_config[
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
                    if isinstance(entity, EnergySensor)
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
                    EnergySensor(
                        self,
                        f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{losses_meter.metering_source}",
                        cycle_mode,
                        losses_meter,
                        name=name,
                        config_subentry_id=subentry_id,
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
                self.losses_meter.shutdown()

    # interface: self
    def _battery_voltage_callback(
        self, event: "Event[EventStateChangedData] | Controller.Event"
    ):
        try:
            state = event.data["new_state"]
            self.battery_voltage = ElectricPotentialConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                VOLTAGE_UNIT,
            )
            self.battery_meter.process(
                self.battery_voltage * self.battery_current, event.time_fired_timestamp
            )
        except Exception as e:
            self.battery_voltage = 0
            self.battery_meter.process(None, event.time_fired_timestamp)
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(
                    self.WARNING, e, "_battery_voltage_update(state:%s)", state
                )

    def _battery_current_callback(
        self, event: "Event[EventStateChangedData] | Controller.Event"
    ):
        time_ts = event.time_fired_timestamp
        try:
            state = event.data["new_state"]
            battery_current = ElectricCurrentConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                CURRENT_UNIT,
            )
        except Exception as e:
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

    def _battery_charge_callback(
        self, event: "Event[EventStateChangedData] | Controller.Event"
    ):
        pass

    async def _async_store_save(self, *args):
        # args depends on the source of this call:
        # no args means we're being called by the loop scheduler i.e. periodic save
        # args[0] == event means we're in the final write stage of HA shutting down
        # args[0] == 0 means we're unloading the controller
        if args:
            if args[0]:
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
