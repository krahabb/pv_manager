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
    from .devices import MeterDevice, OffGridManagerDevice, SourceType

    class EntryConfig(controller.Controller.Config):
        pass

    SUBENTRY_TYPE_CONFIG_MAP: dict[str | None, Callable]
    SUBENTRY_TYPE_DEVICE_MAP: dict[
        str | None, tuple[SourceType, type[MeterDevice], type[OffGridManagerDevice]]
    ]


SUBENTRY_TYPE_CONFIG_MAP = {
    pmc.ConfigSubentryType.MANAGER_BATTERY_METER: BatteryMeter.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_LOAD_METER: LoadMeter.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_PV_METER: PvMeter.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_ESTIMATOR: BatteryEstimator.get_config_schema,
    pmc.ConfigSubentryType.MANAGER_LOSSES: LossesMeter.get_config_schema,
}

SUBENTRY_TYPE_DEVICE_MAP = {
    pmc.ConfigSubentryType.MANAGER_BATTERY_METER: (
        BatteryMeter.SOURCE_TYPE,
        BatteryMeter,
        BatteryMeter,
    ),
    pmc.ConfigSubentryType.MANAGER_LOAD_METER: (
        LoadMeter.SOURCE_TYPE,
        LoadMeter,
        LoadEstimator,
    ),
    pmc.ConfigSubentryType.MANAGER_PV_METER: (
        PvMeter.SOURCE_TYPE,
        PvMeter,
        PvEstimator,
    ),
    pmc.ConfigSubentryType.MANAGER_LOSSES: (
        LossesMeter.SOURCE_TYPE,
        LossesMeter,
        LossesMeter,
    ),
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


class Controller(controller.Controller["EntryConfig"]):  # type: ignore
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

        type Config = EntryConfig

        meter_devices: Final[dict[SourceType, dict[str, MeterDevice]]]
        meter_device_add_event: Final[EventBroadcast[MeterDevice]]
        meter_device_remove_event: Final[EventBroadcast[MeterDevice]]
        battery_estimator: BatteryEstimator

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
    def _get_estimator_entry(config_entry: "ConfigEntry"):
        for subentry in config_entry.subentries.values():
            if subentry.subentry_type == pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                return subentry
        return None

    def __init__(self, config_entry: "ConfigEntry"):
        self.meter_devices = {
            BatteryMeter.SOURCE_TYPE: {},
            LoadMeter.SOURCE_TYPE: {},
            PvMeter.SOURCE_TYPE: {},
            LossesMeter.SOURCE_TYPE: {},
        }
        self.meter_device_add_event = EventBroadcast()
        self.meter_device_remove_event = EventBroadcast()
        super().__init__(config_entry)

    @typing.override
    def _on_init(self):
        if estimator_entry := Controller._get_estimator_entry(self.config_entry):
            self.battery_estimator = BatteryEstimator(
                self, estimator_entry.data, estimator_entry.subentry_id  # type: ignore
            )
        else:
            self.battery_estimator = None  # type: ignore

    def shutdown(self):
        super().shutdown()

        for meter_device_entries in self.meter_devices.values():
            assert not meter_device_entries

        self.battery_estimator = None  # type: ignore

    @typing.override
    def _subentry_add(self, subentry_id: str, entry_data: "EntryData"):
        try:
            if self.battery_estimator:
                SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type][2](
                    self, self.battery_estimator.config | entry_data.config, subentry_id
                )
            else:
                SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type][1](
                    self, entry_data.config, subentry_id
                )
        except KeyError:
            match entry_data.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                    if not self.battery_estimator:
                        # This means the estimator subentry is being added after load
                        raise Controller.EntryReload()

    @typing.override
    async def _async_subentry_update(self, subentry_id: str, entry_data: "EntryData"):
        try:
            await self.meter_devices[
                SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type][0]
            ][subentry_id].async_update_entry(entry_data)
        except KeyError:
            match entry_data.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                    raise Controller.EntryReload()

    @typing.override
    async def _async_subentry_remove(self, subentry_id: str, entry_data: "EntryData"):
        try:
            self.meter_devices[SUBENTRY_TYPE_DEVICE_MAP[entry_data.subentry_type][0]][
                subentry_id
            ].shutdown()
        except KeyError:
            match entry_data.subentry_type:
                case pmc.ConfigSubentryType.MANAGER_ESTIMATOR:
                    raise Controller.EntryReload()
