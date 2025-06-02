import enum
import typing

from homeassistant import const as hac
from homeassistant.helpers import storage
from homeassistant.util import dt as dt_util

from ... import const as pmc, controller
from ...binary_sensor import BinarySensor
from ...helpers import validation as hv
from ...manager import Manager
from ...sensor import EnergySensor, Sensor
from .energy_meters import (
    BatteryMeter,
    LoadMeter,
    PvMeter,
    SourceType,
)

if typing.TYPE_CHECKING:
    from typing import Any, Final, Iterable, Mapping, NotRequired, TypedDict, Unpack

    from homeassistant.config_entries import ConfigEntry

    from ...controller import EntryData
    from ...controller.devices import Device
    from .energy_meters import MeterDevice

    class ControllerStoreType(TypedDict):
        time: str
        time_ts: float
        """REMOVE
        battery: NotRequired[Mapping[str, Any]]
        load: NotRequired[Mapping[str, Any]]
        losses: NotRequired[Mapping[str, Any]]
        pv: NotRequired[Mapping[str, Any]]
        """


class ControllerStore(storage.Store["ControllerStoreType"]):
    VERSION = 1

    def __init__(self, entry_id: str):
        super().__init__(
            Manager.hass,
            self.VERSION,
            f"{pmc.DOMAIN}.{Controller.TYPE}.{entry_id}",
        )


SOURCE_TYPE_METER_MAP: dict[str, type["MeterDevice"]] = {
    SourceType.BATTERY: BatteryMeter,
    SourceType.LOAD: LoadMeter,
    SourceType.PV: PvMeter,
}


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

        """REMOVE
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

        config: Config

        EnergyMeterTuple = tuple[EnergyBroadcast, Device]
        #REMOVE energy_meters: Final[dict[SourceType, EnergyMeterTuple]]
        battery_meter: BatteryMeter
        pv_meter: PvMeter
        load_meter: LoadMeter
        losses_meter: LossesMeter | None
        losses_power_sensor: PowerSensor | None
        system_yield_sensor: Sensor | None
        battery_yield_sensor: Sensor | None
        conversion_yield_sensor: Sensor | None
        conversion_yield_actual_sensor: Sensor | None
        """
        meter_devices: Final[dict[str, dict[str, MeterDevice]]]
        """
        example:
        {
            "battery": {
                subentry_id: BatteryMeter,
            }
        }
        """

    DEFAULT_NAME = "Off grid Manager"
    """REMOVE
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
    """

    TYPE = pmc.ConfigEntryType.OFF_GRID_MANAGER
    PLATFORMS = {Sensor.PLATFORM, BinarySensor.PLATFORM}

    STORE_SAVE_PERIOD = 3600

    __slots__ = (
        # config
        # state
        "_store",
        "meter_devices",
        # REMOVE "losses_meter",
        # entities
        # REMOVE "losses_power_sensor",
        # REMOVE "system_yield_sensor",
        # REMOVE "battery_yield_sensor",
        # REMOVE "conversion_yield_sensor",
        # REMOVE "conversion_yield_actual_sensor",
        # callbacks
        "_final_write_unsub",
    )

    """
    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None") -> pmc.ConfigSchema:
        if not config:
            config = cls.DEFAULT_CONFIG

        def _estimator_schema(
            config: "Controller.EstimatorConfig | None",
        ) -> pmc.ConfigSchema:
            if not config:
                config = cls.DEFAULT_CONFIG["estimator"]
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
    """

    @staticmethod
    def get_config_subentry_schema(
        config_entry: "ConfigEntry",
        subentry_type: str,
        config: pmc.ConfigMapping | None,
    ) -> pmc.ConfigSchema:
        match subentry_type.split("_"):
            case ("manager", source_type, "meter"):
                return SOURCE_TYPE_METER_MAP[source_type].get_config_schema(
                    config  # type:ignore
                )

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
            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                return pmc.ConfigSubentryType.MANAGER_LOSSES
        return None

    def __init__(self, config_entry: "ConfigEntry"):
        # REMOVE self.energy_meters = {}
        self._store = ControllerStore(config_entry.entry_id)
        self.meter_devices = {}
        # REMOVE self.losses_meter = None
        self._final_write_unsub = None
        super().__init__(config_entry)

    async def async_setup(self):

        if store_data := await self._store.async_load():
            """TODO
            meters: "list[BaseProcessor]" = [self.battery_meter]
            if self.losses_meter:
                meters.append(self.losses_meter)
            for meter in meters:
                meter.restore(store_data[meter.id.value])
            """

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
        """REMOVE
        if self.losses_meter:
            self.losses_meter.start()
        """

    async def async_shutdown(self):
        if self._final_write_unsub:
            self._final_write_unsub()
            self._final_write_unsub = None

        await self._async_store_save(None)

        """REMOVE
        for energy_meter_tuple in tuple(reversed(self.energy_meters.values())):
            energy_meter_tuple[0].shutdown()
        assert not self.energy_meters
        """
        await super().async_shutdown()

    @typing.override
    def _subentry_add(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type.split("_"):  # type: ignore
            case ("manager", source_type, "meter"):
                SOURCE_TYPE_METER_MAP[source_type](self, entry_data)

            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                pass
                """REMOVE
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
                """

    @typing.override
    async def _async_subentry_update(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type.split("_"):  # type: ignore
            case ("manager", source_type, "meter"):
                await self.meter_devices[source_type][subentry_id].update_entry(
                    entry_data
                )

            case pmc.ConfigSubentryType.MANAGER_LOSSES:
                """REMOVE
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
                """

    @typing.override
    async def _async_subentry_remove(self, subentry_id: str, entry_data: "EntryData"):
        match entry_data.subentry_type.split("_"):  # type: ignore
            case ("manager", source_type, "meter"):
                self.meter_devices[source_type][subentry_id].shutdown()

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

        """TODO
        data[SourceType.BATTERY.value] = self.battery_meter.store()
        if self.losses_meter:
            data[SourceType.LOSSES.value] = self.losses_meter.store()
        """

        await self._store.async_save(data)
