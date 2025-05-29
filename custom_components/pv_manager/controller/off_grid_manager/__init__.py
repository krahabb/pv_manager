import enum
import typing

from homeassistant import const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import storage
from homeassistant.util import dt as dt_util

from ... import const as pmc, controller
from ...helpers import validation as hv
from ...manager import Manager
from ...processors.battery import BatteryEstimator
from ...processors.estimator_consumption_heuristic import HeuristicConsumptionEstimator
from ...processors.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator
from ...sensor import EnergySensor, PowerSensor, Sensor
from ..devices.estimator_device import (
    EnergyEstimatorDevice,
    SignalEnergyEstimatorDevice,
)
from .energy_meters import (
    BatteryMeter,
    LoadMeter,
    LossesMeter,
    PvMeter,
    SourceType,
)

if typing.TYPE_CHECKING:
    from typing import Any, Final, Iterable, Mapping, NotRequired, TypedDict, Unpack

    from homeassistant.config_entries import ConfigEntry

    from ...controller import EntryData
    from ...controller.devices import Device
    from ...processors import BaseProcessor, EnergyBroadcast
    from ...processors.estimator_energy import SignalEnergyEstimator

    class ControllerStoreType(TypedDict):
        time: str
        time_ts: float

        battery: NotRequired[Mapping[str, Any]]
        load: NotRequired[Mapping[str, Any]]
        losses: NotRequired[Mapping[str, Any]]
        pv: NotRequired[Mapping[str, Any]]


class ControllerStore(storage.Store["ControllerStoreType"]):
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
        device: "Device",
        id: YieldSensorId,
        config_subentry_id: str,
        name: str,
    ):
        super().__init__(
            device,
            id,
            config_subentry_id=config_subentry_id,
            name=name,
        )


class ManagerLossesConfig(pmc.EntityConfig, pmc.SubentryConfig):
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

    update_period: int

    # Yield sensors names/enabler
    system_yield: "NotRequired[str]"
    battery_yield: "NotRequired[str]"
    conversion_yield: "NotRequired[str]"
    conversion_yield_actual: "NotRequired[str]"


class Controller(controller.Controller["Controller.Config"]):  # type: ignore
    """Off-grid plant manager: a collection of integrated helpers for a basic off-grid system
    with PV BATTERY and LOAD."""

    if typing.TYPE_CHECKING:

        class EstimatorConfig(SignalEnergyEstimator.Config):
            enabled: bool
            # pv estimator specific
            weather_entity_id: NotRequired[str]
            weather_model: NotRequired[str]

        class Config(pmc.EntityConfig, controller.Controller.Config):
            battery: BatteryMeter.Config
            pv: PvMeter.Config
            load: LoadMeter.Config
            estimator: "Controller.EstimatorConfig"
            maximum_latency: NotRequired[int]
            """Maximum time between source power/energy samples before considering an error in data sampling."""

        config: Config

        EnergyMeterTuple = tuple[EnergyBroadcast, Device]
        energy_meters: Final[dict[SourceType, EnergyMeterTuple]]
        battery_meter: BatteryMeter
        pv_meter: PvMeter
        load_meter: LoadMeter
        losses_meter: LossesMeter | None

        losses_power_sensor: PowerSensor | None
        system_yield_sensor: Sensor | None
        battery_yield_sensor: Sensor | None
        conversion_yield_sensor: Sensor | None
        conversion_yield_actual_sensor: Sensor | None

    DEFAULT_NAME = "Off grid Manager"
    DEFAULT_CONFIG: "Config" = {
        "name": DEFAULT_NAME,
        "battery": {
            "battery_voltage_entity_id": "",
            "battery_current_entity_id": "",
            "battery_capacity": 100,
        },
        "pv": {},
        "load": {},
        "estimator": {
            "enabled": False,
            "sampling_interval_minutes": 10,
            "history_duration_days": 7,
            "observation_duration_minutes": 30,
            "weather_model": "cubic",
        },
    }

    TYPE = pmc.ConfigEntryType.OFF_GRID_MANAGER
    PLATFORMS = {Sensor.PLATFORM}

    STORE_SAVE_PERIOD = 3600

    __slots__ = (
        # config
        "maximum_latency_ts",
        # state
        "_store",
        "energy_meters",
        "battery_meter",
        "pv_meter",
        "load_meter",
        "losses_meter",
        # entities
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
            config = Controller.DEFAULT_CONFIG

        def _estimator_schema(
            config: "Controller.EstimatorConfig | None",
        ) -> pmc.ConfigSchema:
            if not config:
                config = Controller.DEFAULT_CONFIG["estimator"]
            return {
                hv.req_config("enabled", config): bool,
                hv.req_config(
                    "sampling_interval_minutes", config
                ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
                hv.req_config(
                    "observation_duration_minutes", config
                ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
                hv.req_config("history_duration_days", config): hv.time_period_selector(
                    unit_of_measurement=hac.UnitOfTime.DAYS, max=30
                ),
                hv.opt_config(
                    "weather_entity_id", config
                ): hv.weather_entity_selector(),
                hv.opt_config(
                    "weather_model", config
                ): HeuristicPVEnergyEstimator.weather_model_selector(),
            }

        return hv.entity_schema(config) | {
            hv.vol.Required("battery"): hv.section(
                BatteryMeter.get_config_schema(config.get("battery") or {}),
            ),
            hv.vol.Required("pv"): hv.section(
                PvMeter.get_config_schema(config.get("pv") or {}),
            ),
            hv.vol.Required("load"): hv.section(
                LoadMeter.get_config_schema(config.get("load") or {}),
            ),
            hv.vol.Required("estimator"): hv.section(
                _estimator_schema(config.get("estimator")),
            ),
            hv.opt_config("maximum_latency", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
        }

    @staticmethod
    def get_config_subentry_schema(
        config_entry: "ConfigEntry",
        subentry_type: str,
        config: pmc.ConfigMapping | None,
    ) -> pmc.ConfigSchema:
        match subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                if not config:
                    config = {
                        "name": MANAGER_ENERGY_SENSOR_NAME,
                    }
                    _options = [t.value for t in SourceType]
                    for subentry in config_entry.subentries.values():
                        if (
                            subentry.subentry_type
                            == pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR
                        ):
                            try:
                                _options.remove(subentry.data["metering_source"])
                            except (KeyError, ValueError):
                                pass
                    return {
                        hv.req_config("metering_source", config): hv.select_selector(
                            options=_options
                        ),
                        hv.req_config("name", config): str,
                        hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
                    }
                else:
                    return {
                        hv.req_config("name", config): str,
                        hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
                    }

            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                if not config:
                    config = {
                        "name": "Losses",
                        "update_period": 10,
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
                            "update_period", config
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
            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                return pmc.ConfigSubentryType.MANAGER_LOSSES
        return None

    def __init__(self, config_entry: "ConfigEntry"):
        self.energy_meters = {}
        self._store = ControllerStore(config_entry.entry_id)
        self.losses_meter = None
        self._final_write_unsub = None

        super().__init__(config_entry)

    def _on_init(self):
        config: "Controller.Config" = self.config  # type: ignore
        self.maximum_latency_ts = (
            config.get("maximum_latency") or PvMeter.MAXIMUM_LATENCY_DISABLED
        )

        # we need to build meters here since they need to be in place when the controller
        # will be loading subentries during __init__
        estimator_config = config.get("estimator")
        if estimator_config and estimator_config.get("enabled"):
            load_meter_config = config["load"] | estimator_config
            LoadMeterClass = type(
                "LoadMeterClass",
                (LoadMeter, SignalEnergyEstimatorDevice, HeuristicConsumptionEstimator),
                {},
            )
            self.load_meter = LoadMeterClass(self, load_meter_config)
            pv_meter_config = config["pv"] | estimator_config
            PvMeterClass = type(
                "PvMeterClass",
                (PvMeter, SignalEnergyEstimatorDevice, HeuristicPVEnergyEstimator),
                {},
            )
            self.pv_meter = PvMeterClass(self, pv_meter_config)
            battery_meter_config = config["battery"] | (
                estimator_config | {"forecast_duration_hours": 24}
            )
            BatteryMeterClass = type(
                "BatteryMeterClass",
                (BatteryMeter, EnergyEstimatorDevice, BatteryEstimator),
                {},
            )
            self.battery_meter = BatteryMeterClass(self, battery_meter_config)  # type: ignore
            self.battery_meter.connect_consumption(self.load_meter)
            self.battery_meter.connect_production(self.pv_meter)
        else:
            self.load_meter = LoadMeter(self, config["load"])
            self.pv_meter = PvMeter(self, config["pv"])
            self.battery_meter = BatteryMeter(self, config["battery"])
        return super()._on_init()

    async def async_setup(self):

        if store_data := await self._store.async_load():

            meters: "list[BaseProcessor]" = [self.battery_meter]
            if self.losses_meter:
                meters.append(self.losses_meter)
            for meter in meters:
                meter.restore(store_data[meter.id.value])

        self.track_timer(
            self.STORE_SAVE_PERIOD,
            self._async_store_save,
            self.HassJobType.Coroutinefunction,
        )
        self._final_write_unsub = Manager.hass.bus.async_listen_once(
            hac.EVENT_HOMEASSISTANT_FINAL_WRITE,
            self._async_store_save,
        )

        await super().async_setup()
        # trigger now after adding entities to hass
        if self.losses_meter:
            self.losses_meter.start()

    async def async_shutdown(self):
        if self._final_write_unsub:
            self._final_write_unsub()
            self._final_write_unsub = None

        await self._async_store_save(None)

        for energy_meter_tuple in tuple(reversed(self.energy_meters.values())):
            energy_meter_tuple[0].shutdown()
        assert not self.energy_meters

        await super().async_shutdown()

    @typing.override
    def _subentry_add(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                sensor_config: ManagerEnergySensorConfig = entry_data.config  # type: ignore[assignment]
                try:
                    energy_meter, device = self.energy_meters[
                        sensor_config["metering_source"]
                    ]
                except KeyError:
                    return
                name = sensor_config["name"]
                self._create_subentry_energy_sensors(
                    device,
                    energy_meter,
                    (
                        name
                        if name != MANAGER_ENERGY_SENSOR_NAME
                        else f"{energy_meter.id} {name}"
                    ),
                    subentry_id,
                    sensor_config["cycle_modes"],
                )
            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                assert not self.losses_meter
                losses_config: ManagerLossesConfig = entry_data.config  # type: ignore
                self.losses_meter = energy_meter = LossesMeter(
                    self, losses_config["update_period"]
                )
                name = losses_config["name"]
                PowerSensor(
                    self,
                    "losses_power",
                    config_subentry_id=subentry_id,
                    name=name,
                    parent_attr=PowerSensor.ParentAttr.DYNAMIC,
                )
                self._create_subentry_energy_sensors(
                    self, energy_meter, name, subentry_id, losses_config["cycle_modes"]
                )
                for yield_sensor_id in YieldSensorId:
                    setattr(self, f"{yield_sensor_id}_sensor", None)
                    if name := losses_config.get(yield_sensor_id):
                        YieldSensor(
                            self,
                            yield_sensor_id,
                            config_subentry_id=subentry_id,
                            name=name,
                        )
                if self.config_entry.state == ConfigEntryState.LOADED:
                    energy_meter.start()

    @typing.override
    async def _async_subentry_update(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                sensor_config: ManagerEnergySensorConfig = entry_data.config  # type: ignore[assignment]
                try:
                    energy_meter, device = self.energy_meters[
                        sensor_config["metering_source"]
                    ]
                except KeyError:
                    return
                name = sensor_config["name"]
                if name == MANAGER_ENERGY_SENSOR_NAME:
                    name = f"{energy_meter.id} {name}"
                cycle_modes_new = set(sensor_config["cycle_modes"])
                energy_sensor: EnergySensor
                for energy_sensor in tuple(entry_data.entities.values()):  # type: ignore
                    try:
                        cycle_modes_new.remove(energy_sensor.cycle_mode)
                        # cycle_mode still present: update
                        energy_sensor.update_name(energy_sensor.formatted_name(name))
                    except KeyError:
                        # cycle_mode removed from updated config
                        await energy_sensor.async_shutdown(True)
                # leftovers are those newly added cycle_mode(s)
                self._create_subentry_energy_sensors(
                    device, energy_meter, name, subentry_id, cycle_modes_new
                )
                # no state flush/update for entities since they're all integrating sensors
                # and need a cycle to be computed/refreshed
            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                energy_meter = self.losses_meter
                assert energy_meter
                losses_config: ManagerLossesConfig = entry_data.config  # type: ignore
                name = losses_config["name"]
                # TODO: retrigger the track_timer
                energy_meter.update_period_ts = losses_config["update_period"]
                entities = entry_data.entities
                # update PowerSensor
                if entity := entities.get("losses_power"):
                    entity.update_name(name)
                # update ManagerEnergySensors
                cycle_modes_new = set(losses_config["cycle_modes"])
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
                self._create_subentry_energy_sensors(
                    self, energy_meter, name, subentry_id, cycle_modes_new
                )
                # update yield sensors
                yield_sensors_id_new = {
                    yield_sensor_id
                    for yield_sensor_id in YieldSensorId
                    if losses_config.get(yield_sensor_id)
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
                        yield_sensor.update_name(losses_config.get(yield_sensor_id))  # type: ignore
                    except KeyError:
                        # yield_sensor removed from updated config
                        await yield_sensor.async_shutdown(True)
                # leftovers are newly added yield sensors
                for yield_sensor_id in yield_sensors_id_new:
                    YieldSensor(
                        self,
                        yield_sensor_id,
                        config_subentry_id=subentry_id,
                        name=losses_config[yield_sensor_id],  # type: ignore
                    )

    @typing.override
    async def _async_subentry_remove(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR:
                # default cleanup will suffice
                pass
            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                assert self.losses_meter
                self.losses_meter.shutdown()

    # interface: self
    async def _async_store_save(self, time_or_event_or_none, /):
        # args depends on the source of this call:
        # None: means we're unloading the controller
        # event: means we're in the final write stage of HA shutting down
        # float: schueduled timer
        if time_or_event_or_none and type(time_or_event_or_none) != float:
            self._final_write_unsub = None

        now = dt_util.now()
        data: "ControllerStoreType" = {
            "time": now.isoformat(),
            "time_ts": now.timestamp(),
        }

        data[SourceType.BATTERY.value] = self.battery_meter.store()
        if self.losses_meter:
            data[SourceType.LOSSES.value] = self.losses_meter.store()

        await self._store.async_save(data)

    def _create_subentry_energy_sensors(
        self,
        device: "Device",
        energy_broadcast: "EnergyBroadcast",
        name: str,
        subentry_id: str,
        cycle_modes: "Iterable[EnergySensor.CycleMode]",
    ):
        sensor_id = (
            f"{pmc.ConfigSubentryType.MANAGER_ENERGY_SENSOR}_{energy_broadcast.id}"
        )
        for cycle_mode in cycle_modes:
            EnergySensor(
                device,
                sensor_id,
                cycle_mode,
                energy_broadcast,
                name=name,
                config_subentry_id=subentry_id,
            )
