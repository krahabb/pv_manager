from dataclasses import dataclass
import datetime
import enum
import time
import typing

from homeassistant import const as hac
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

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, Unpack

    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.core import Event, HomeAssistant, State
    import voluptuous as vol

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


class MeterStoreType(typing.TypedDict):
    time_ts: float
    energy: float


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


@dataclass(slots=True)
class BaseMeter:
    controller: "Final[Controller]"
    metering_source: "Final[MeteringSource]"
    energy_sensors: "Final[set[EnergySensor]]"
    time_ts: float
    value: float  # last sample in: might be power or energy
    energy: float  # accumulated total energy

    def __init__(
        self, controller: "Controller", metering_source: MeteringSource, time_ts: float
    ):
        self.controller = controller
        self.metering_source = metering_source
        self.energy_sensors = set()
        self.time_ts = time_ts
        self.value = 0
        self.energy = 0
        controller.energy_meters[metering_source] = self

    def add(self, value: float, time_ts: float):
        pass

    def interpolate(self, time_ts: float):
        pass

    def track_state(self, state: "State | None"):
        pass

    def load(self, data: ControllerStoreType):
        try:
            meter_data: MeterStoreType = data[self.metering_source.value]
            self.energy = meter_data["energy"]
        except:
            pass

    def save(self, data: ControllerStoreType):
        # remember to 'reset_partial' so to save all of the energy accumulated so far
        data[self.metering_source.value] = MeterStoreType(
            {
                "time_ts": self.time_ts,
                "energy": self.energy,
            }
        )


class PowerMeter(BaseMeter):

    def add(self, value: float, time_ts: float):
        # left rectangle integration
        d_time = time_ts - self.time_ts
        if 0 < d_time < self.controller.maximum_latency_ts:
            if self.energy_sensors:
                energy = self.value * d_time / 3600
                self.energy += energy
                for sensor in self.energy_sensors:
                    sensor.accumulate(energy, time_ts)
            else:
                self.energy += self.value * d_time / 3600
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
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
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
            sensor.accumulate(value, time_ts)
        self.value = value
        self.time_ts = time_ts


class BatteryPowerMeter(PowerMeter):

    in_meter: EnergyMeter
    out_meter: EnergyMeter

    __slots__ = (
        "in_meter",
        "out_meter",
    )

    def __init__(self, controller, time_ts: float):
        super().__init__(controller, MeteringSource.BATTERY, time_ts)
        self.in_meter = EnergyMeter(controller, MeteringSource.BATTERY_IN, time_ts)
        self.out_meter = EnergyMeter(controller, MeteringSource.BATTERY_OUT, time_ts)

    def add(self, value: float, time_ts: float):
        # left rectangle integration
        d_time = time_ts - self.time_ts
        if 0 < d_time < self.controller.maximum_latency_ts:
            energy = self.value * d_time / 3600
            self.energy += energy
            for sensor in self.energy_sensors:
                sensor.accumulate(energy, time_ts)
            if energy > 0:
                self.out_meter._add_internal(energy, time_ts)
            else:
                self.in_meter._add_internal(-energy, time_ts)

        self.value = value
        self.time_ts = time_ts


class LossesEnergyMeter(EnergyMeter):

    __slots__ = (
        "battery_energy",
        "battery_in_energy",
        "battery_out_energy",
        "load_energy",
        "pv_energy",
        "_callback_interval_ts",
        "_callback_unsub",
    )

    def __init__(self, controller, time_ts: float, callback_interval_ts: float):
        self._callback_interval_ts = callback_interval_ts
        self._callback_unsub = None
        super().__init__(controller, MeteringSource.LOSSES, time_ts)

    def start(self):
        """Called after restoring data in the controller so to initialize incremental counters."""
        self._losses_compute()
        self._callback_unsub = self.controller.schedule_callback(
            self._callback_interval_ts, self._losses_callback
        )

    def stop(self):
        if self._callback_unsub:
            self._callback_unsub.cancel()
            self._callback_unsub = None

    @callback
    def _losses_callback(self):
        time_ts = TIME_TS()
        controller = self.controller
        self._callback_unsub = controller.schedule_callback(
            self._callback_interval_ts, self._losses_callback
        )
        battery_old = self.battery_energy
        battery_in_old = self.battery_in_energy
        battery_out_old = self.battery_out_energy
        load_old = self.load_energy
        pv_old = self.pv_energy
        losses_old = self.energy
        # get the 'new' total
        controller.battery_meter.interpolate(time_ts)
        controller.load_meter.interpolate(time_ts)
        controller.pv_meter.interpolate(time_ts)
        self._losses_compute()
        # compute delta to get the average power in the sampling period
        # we don't check maximum_latency here since it has already been
        # managed in pc, load and battery meters
        d_losses = self.energy - losses_old
        for sensor in self.energy_sensors:
            sensor.accumulate(d_losses, time_ts)

        if controller.losses_power_sensor:
            d_time = time_ts - self.time_ts
            if 0 < d_time < controller.maximum_latency_ts:
                controller.losses_power_sensor.update(d_losses * 3600 / d_time)
            else:
                controller.losses_power_sensor.update(None)

        self.value = d_losses
        self.time_ts = time_ts

        if controller.conversion_yield_actual_sensor:
            try:
                d_load = self.load_energy - load_old
                controller.conversion_yield_actual_sensor.update(
                    round(d_load * 100 / (d_load + d_losses))
                )
            except:
                controller.conversion_yield_actual_sensor.update(None)

    def _losses_compute(self):
        controller = self.controller
        self.battery_energy = battery = controller.battery_meter.energy
        self.battery_in_energy = battery_in = controller.battery_meter.in_meter.energy
        self.battery_out_energy = battery_out = (
            controller.battery_meter.out_meter.energy
        )
        self.load_energy = load = controller.load_meter.energy
        self.pv_energy = pv = controller.pv_meter.energy
        self.energy = losses = pv + battery - load

        # Estimate energy actually stored in the battery:
        # in the long term -> battery_in > battery_out with the difference being the energy 'eaten up'
        # battery_yield = battery_out / (battery_in - battery_stored_energy)
        # battery_stored_energy is hard to compute since it depends on the discharge current/voltage
        # we'll use a conservative approach with the following formula. It's contribution to
        # battery_yeild will nevertheless decay as far as battery_in, battery_out will increase
        battery_stored = (
            controller.battery_charge_estimate * controller.battery_voltage * 0.9
        )

        if controller.conversion_yield_sensor:
            try:
                conversion_yield = load / (load + losses)
                controller.conversion_yield_sensor.update(round(conversion_yield * 100))
            except:
                controller.conversion_yield_sensor.update(None)

        if controller.battery_yield_sensor:
            try:
                # battery_yield = battery_out_energy / (battery_in_energy - battery_stored_energy)
                _temp = battery_in - battery_stored
                if _temp > battery_out:
                    controller.battery_yield_sensor.update(
                        round(battery_out * 100 / _temp)
                    )
                else:
                    controller.battery_yield_sensor.update(None)
            except:
                controller.battery_yield_sensor.update(None)

        if controller.system_yield_sensor:
            try:
                # system_yield = load_energy / (pv_energy - battery_stored_energy)
                _temp = pv - battery_stored
                if _temp > load:
                    controller.system_yield_sensor.update(round(load * 100 / _temp))
                else:
                    controller.system_yield_sensor.update(None)
            except:
                controller.system_yield_sensor.update(None)


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
    battery_meter: BatteryPowerMeter
    pv_meter: PowerMeter
    load_meter: PowerMeter
    losses_meter: LossesEnergyMeter

    losses_power_sensor: PowerSensor
    system_yield_sensor: Sensor
    battery_yield_sensor: Sensor
    conversion_yield_sensor: Sensor
    conversion_yield_actual_sensor: Sensor

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
        "pv_meter",
        "load_meter",
        "losses_meter",
        "losses_power_sensor",
        "system_yield_sensor",
        "battery_yield_sensor",
        "conversion_yield_sensor",
        "conversion_yield_actual_sensor",
        # entities
        "battery_charge_sensor",
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
            hv.req_config("maximum_latency_minutes", config): hv.time_period_selector(
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
                        vol.Required("metering_source"): hv.select_selector(
                            options=list(MeteringSource)
                        ),
                        vol.Required("cycle_modes"): hv.select_selector(
                            options=list(EnergySensor.CycleMode), multiple=True
                        ),
                    }
                else:
                    return hv.entity_schema(config) | {
                        hv.req_config("cycle_modes", config): hv.select_selector(
                            options=list(EnergySensor.CycleMode), multiple=True
                        ),
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
                return hv.entity_schema(config) | {
                    hv.req_config("cycle_modes", config): hv.select_selector(
                        options=list(EnergySensor.CycleMode), multiple=True
                    ),
                    hv.req_config(
                        "sampling_interval_seconds", config
                    ): hv.time_period_selector(),
                    hv.opt_config("system_yield", config): str,
                    hv.opt_config("battery_yield", config): str,
                    hv.opt_config("conversion_yield", config): str,
                    hv.opt_config("conversion_yield_actual", config): str,
                }

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

    def __init__(self, hass, config_entry):
        super().__init__(hass, config_entry)
        self.battery_voltage_entity_id = self.config["battery_voltage_entity_id"]
        self.battery_current_entity_id = self.config["battery_current_entity_id"]
        self.battery_charge_entity_id = self.config.get("battery_charge_entity_id")
        self.battery_capacity = self.config.get("battery_capacity", 0)
        self.pv_power_entity_id = self.config.get("pv_power_entity_id")
        self.load_power_entity_id = self.config.get("load_power_entity_id")
        self.maximum_latency_ts = self.config.get("maximum_latency_minutes", 0) * 60

        self._store = ControllerStore(hass, config_entry.entry_id)

        self.battery_voltage: float = 0
        self.battery_current: float = 0
        self._battery_current_last_ts: float = 0
        self.battery_charge: float = 0
        self.battery_charge_estimate: float = 0
        self.battery_capacity_estimate: float = self.battery_capacity

        self.energy_meters = {}

        self.battery_charge_sensor: BatteryChargeSensor | None = None

        time_ts = TIME_TS()
        self.battery_meter = BatteryPowerMeter(self, time_ts)
        self.pv_meter = PowerMeter(self, MeteringSource.PV, time_ts)
        self.load_meter = PowerMeter(self, MeteringSource.LOAD, time_ts)
        self.losses_meter = None  # type: ignore

        self._store_save_callback_unsub = None
        self._final_write_unsub = None

        for subentry_id, config_subentry in config_entry.subentries.items():
            match config_subentry.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                    energy_sensor_config: ManagerEnergySensorConfig = config_subentry.data  # type: ignore
                    energy_meter = self.energy_meters.get(
                        energy_sensor_config["metering_source"]
                    )
                    if energy_meter:
                        name = energy_sensor_config["name"]
                        if name == MANAGER_ENERGY_SENSOR_NAME:
                            name = f"{energy_meter.metering_source} {name}"
                        for cycle_mode in energy_sensor_config["cycle_modes"]:
                            ManagerEnergySensor(
                                energy_meter, name, subentry_id, cycle_mode
                            )
                case pmc.ConfigSubentryType.MANAGER_YIELD:
                    yield_config: ManagerYieldConfig = config_subentry.data  # type: ignore
                    self.losses_meter = LossesEnergyMeter(
                        self, time_ts, yield_config["sampling_interval_seconds"]
                    )
                    name = yield_config["name"]
                    self.losses_power_sensor = PowerSensor(
                        self,
                        "losses_power",
                        config_subentry_id=subentry_id,
                        name=f"{name} (power)",
                        parent_attr=PowerSensor.ParentAttr.DYNAMIC,
                    )
                    for cycle_mode in yield_config["cycle_modes"]:
                        ManagerEnergySensor(
                            self.losses_meter, name, subentry_id, cycle_mode
                        )
                    for yield_sensor_id in (
                        "system_yield",
                        "battery_yield",
                        "conversion_yield",
                        "conversion_yield_actual",
                    ):
                        name = yield_config.get(yield_sensor_id)
                        setattr(self, f"{yield_sensor_id}_sensor", None)
                        if name:
                            Sensor(
                                self,
                                yield_sensor_id,
                                config_subentry_id=subentry_id,
                                name=name,
                                native_unit_of_measurement="%",
                                parent_attr=Sensor.ParentAttr.DYNAMIC,
                            )

    async def async_init(self):

        if store_data := await self._store.async_load():
            for meter in self.energy_meters.values():
                meter.load(store_data)
            if "battery_charge_estimate" in store_data:
                self.battery_charge_estimate = store_data["battery_charge_estimate"]
            if "battery_capacity_estimate" in store_data:
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
        if self.losses_meter:
            self.losses_meter.start()

        await super().async_init()

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

        self.energy_meters.clear()
        self.battery_meter = self.pv_meter = self.load_meter = self.losses_meter = None  # type: ignore
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

        self.battery_meter.add(self.battery_current * self.battery_voltage, time_ts)

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
            meter.save(data)
        await self._store.async_save(data)
