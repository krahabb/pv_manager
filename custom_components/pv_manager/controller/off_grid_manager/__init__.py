import typing

from ... import const as pmc, controller
from ...binary_sensor import BinarySensor
from ...helpers import validation as hv
from ...helpers.manager import Manager
from ...processors import EventBroadcast
from ...sensor import Sensor
from .devices import (
    BatteryEstimator,
    BatteryMeter,
    HeuristicPVEnergyEstimator,
    LoadEstimator,
    LoadMeter,
    LossesMeter,
    PvEstimator,
    PvMeter,
)

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        Final,
        Iterable,
        Mapping,
        NotRequired,
        TypedDict,
        Unpack,
    )

    from homeassistant.config_entries import ConfigEntry

    from ...controller import EntryData
    from .devices import MeterDevice, OffGridManagerDevice

    SUBENTRY_TYPE_CONFIG_MAP: dict[str | None, Callable]
    SUBENTRY_TYPE_DEVICE_MAP: dict[str | None, type[OffGridManagerDevice]]
    SUBENTRY_TYPE_ESTIMATOR_MAP: dict[str | None, type[OffGridManagerDevice]]


SUBENTRY_TYPE_CONFIG_MAP = {
    pmc.ConfigSubentryType.MANAGER_BATTERY_METER: BatteryMeter.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_LOAD_METER: LoadMeter.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_PV_METER: PvMeter.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_ESTIMATOR: HeuristicPVEnergyEstimator.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_LOSSES: LossesMeter.get_config_schema,
}

SUBENTRY_TYPE_DEVICE_MAP = {
    pmc.ConfigSubentryType.MANAGER_BATTERY_METER: BatteryMeter,
    pmc.ConfigSubentryType.MANAGER_LOAD_METER: LoadMeter,
    pmc.ConfigSubentryType.MANAGER_PV_METER: PvMeter,
    pmc.ConfigSubentryType.MANAGER_LOSSES: LossesMeter,
}

SUBENTRY_TYPE_ESTIMATOR_MAP = {
    pmc.ConfigSubentryType.MANAGER_BATTERY_METER: BatteryMeter,
    pmc.ConfigSubentryType.MANAGER_LOAD_METER: LoadEstimator,
    pmc.ConfigSubentryType.MANAGER_PV_METER: PvEstimator,
    pmc.ConfigSubentryType.MANAGER_ESTIMATOR: BatteryEstimator,
    pmc.ConfigSubentryType.MANAGER_LOSSES: LossesMeter,
}

"""


class ManagerLossesConfig(pmc.EntityConfig, pmc.SubentryConfig):
    ""Configure measuring of pv plant losses through observations of input power/energy measures
    (pv, battery and load). Losses are computed so:

    battery = battery_out - battery_in
    losses + load = pv + battery

    should mainly consist in inverter losses + cabling (or any consumption
    not measured through load) assuming load is the energy measured at the output of the inverter.

    In the long term i.e. excluding battery storage:

    system efficiency = load / pv (should include losses + battery losses)
    conversion efficiency = load / (load + losses) (mainly inverter losses)
    battery efficiency = battery_out / battery_in (sampled at same battery charge)
    ""

    cycle_modes: list[EnergySensor.CycleMode]

    update_period: int

    # Yield sensors names/enabler
    system_yield: "NotRequired[str]"
    battery_yield: "NotRequired[str]"
    conversion_yield: "NotRequired[str]"
    conversion_yield_actual: "NotRequired[str]"
"""


class Controller(controller.Controller["Controller.Config"]):  # type: ignore
    """
    Off-grid plant manager: a collection of integrated helpers for a complete off-grid system
    with PV, BATTERY and LOAD.
    Multiple source entities of the same type can be configured and their energy measures
    are collected (sum) to build a 'complex' estimator for the whole system.
    This estimator is composed of a total PV energy estimatio, a total LOAD estimation,
    and a total BATTERY (total capacity if multiple batteries) estimator.
    This complex estimator would then be able to forecast production, consumption, losses,
    charge level, and many other estimations like TIME TO FULL CHARGE and TIME TO FULL DISCHARGE
    and so on.
    """

    if typing.TYPE_CHECKING:

        estimator_config: Final[pmc.ConfigMapping | None]

        class MeterDevicesT(TypedDict):
            battery: dict[str, BatteryMeter]
            load: dict[str, LoadMeter]  # could be LoadEstimator
            pv: dict[str, PvMeter]  # could be PvEstimator

        meter_devices: Final[MeterDevicesT]
        meter_device_add_event: Final[EventBroadcast[MeterDevice]]
        meter_device_remove_event: Final[EventBroadcast[MeterDevice]]
        battery_estimator: BatteryEstimator | None

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

    __slots__ = (
        "estimator_config",
        "meter_devices",
        "meter_device_add_event",
        "meter_device_remove_event",
        "battery_estimator",
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
    @typing.override
    def get_config_subentry_schema(
        config_entry: "ConfigEntry",
        subentry_type: str,
        config: pmc.ConfigMapping | None,
        /,
    ) -> pmc.ConfigSchema:
        return SUBENTRY_TYPE_CONFIG_MAP[subentry_type](config)

    @staticmethod
    def _get_estimator_config(config_entry: "ConfigEntry"):
        for subentry in config_entry.subentries.values():
            if subentry.subentry_type == pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                return subentry.data
        return None

    def __init__(self, config_entry: "ConfigEntry"):
        self.estimator_config = Controller._get_estimator_config(config_entry)
        self.meter_devices = {"battery": {}, "load": {}, "pv": {}}
        self.meter_device_add_event = EventBroadcast()
        self.meter_device_remove_event = EventBroadcast()
        super().__init__(config_entry)

    @typing.override
    def _subentry_add(self, subentry_id: str, entry_data: "EntryData"):
        try:
            if self.estimator_config:
                SUBENTRY_TYPE_ESTIMATOR_MAP[entry_data.subentry_type](
                    self, self.estimator_config | entry_data.config, subentry_id
                )
            else:
                SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type](
                    self, entry_data.config, subentry_id
                )
        except KeyError:
            match entry_data.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                    # This means the estimator subentry is being added after load
                    raise Controller.EntryReload()

                case pmc.ConfigSubentryType.MANAGER_LOSSES:
                    LossesMeter(self, entry_data.config, subentry_id)
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
        try:
            await self.meter_devices[
                SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type].SOURCE_TYPE
            ][subentry_id].update_entry(entry_data)
        except KeyError:
            match entry_data.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                    raise Controller.EntryReload()

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
        try:
            self.meter_devices[
                SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type].SOURCE_TYPE
            ][subentry_id].shutdown()
        except KeyError:
            match entry_data.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                    raise Controller.EntryReload()
