"""Constants for pv_manager tests."""

import enum
import typing

from custom_components.pv_manager import const as pmc
from custom_components.pv_manager.helpers import metering

if typing.TYPE_CHECKING:
    from typing import Iterable, NotRequired

    from homeassistant.config_entries import ConfigSubentryDataWithId


class Optional:
    pass


class OptionalString(Optional, str):
    pass


class OptionalInt(Optional, int):
    pass


class OptionalFloat(Optional, float):
    pass


class EntityIdEnum(enum.StrEnum):
    WEATHER = "weather.home"
    BATTERY_VOLTAGE = "sensor.battery_voltage"
    BATTERY_CURRENT = "sensor.battery_current"
    BATTERY_CHARGE = "sensor.battery_charge"
    PV_POWER = "sensor.pv_power"
    LOAD_POWER = "sensor.load_power"
    CONSUMPTION_POWER = "sensor.consumption_power"


ENTITY_REGISTRY_PRELOAD: dict[EntityIdEnum, dict[str, typing.Any]] = {
    EntityIdEnum.CONSUMPTION_POWER: {
        "original_device_class": "power",
        "original_name": "Consumption",
        "unit_of_measurement": "W",
    }
}


class ConfigEntriesItem(typing.TypedDict):
    type: pmc.ConfigEntryType
    data: pmc.ConfigMapping
    options: "NotRequired[pmc.EntryOptionsConfig]"
    subentries_data: "NotRequired[Iterable[ConfigSubentryDataWithId]]"


CE_PV_PLANT_SIMULATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.PV_PLANT_SIMULATOR,
        "data": {
            "name": "PV plant simulator",
            "native_unit_of_measurement": "W",
            "peak_power": 1000,
            "weather_entity_id": OptionalString(EntityIdEnum.WEATHER),
            "battery_voltage": 48,
            "battery_capacity": 100,
            "consumption_baseload_power_w": 100,
            "consumption_daily_extra_power_w": 1000,
            "consumption_daily_fill_factor": 0.2,
            "inverter_zeroload_power_w": 20,
            "inverter_efficiency": 0.9,
        },
    }
)
CE_OFF_GRID_MANAGER = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.OFF_GRID_MANAGER,
        "data": {
            "name": "Off grid manager",
            "battery": {
                "battery_voltage_entity_id": EntityIdEnum.BATTERY_VOLTAGE,
                "battery_current_entity_id": EntityIdEnum.BATTERY_CURRENT,
                "battery_charge_entity_id": OptionalString(""),
                "battery_capacity": 100,
            },
            "pv": {
                "pv_power_entity_id": OptionalString(EntityIdEnum.PV_POWER),
            },
            "load": {
                "load_power_entity_id": OptionalString(EntityIdEnum.LOAD_POWER),
            },
            "maximum_latency_seconds": 10,
        },
    }
)
CE_ENERGY_CALCULATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.ENERGY_CALCULATOR,
        "data": {
            "name": "Energy calculator",
            "source_entity_id": EntityIdEnum.LOAD_POWER,
            "cycle_modes": list(metering.CycleMode),
            "update_period_seconds": 5,
            "maximum_latency_seconds": 10,
        },
    }
)
CE_CONSUMPTION_ESTIMATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR,
        "data": {
            "name": "Consumption estimator",
            "source_entity_id": EntityIdEnum.CONSUMPTION_POWER,
            "sampling_interval_minutes": 10,
            "observation_duration_minutes": 60,
            "history_duration_days": 1,
            "update_period_seconds": OptionalFloat(5),
            "maximum_latency_seconds": 10,
            "safe_maximum_power_w": OptionalFloat(1000),
        },
    }
)
CE_PVENERGY_HEURISTIC_ESTIMATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR,
        "data": {
            "name": "PV Energy estimator",
            "source_entity_id": EntityIdEnum.PV_POWER,
            "sampling_interval_minutes": 10,
            "observation_duration_minutes": 60,
            "history_duration_days": 1,
            "update_period_seconds": OptionalFloat(5),
            "maximum_latency_seconds": 10,
            "safe_maximum_power_w": OptionalFloat(1000),
        },
    }
)

CONFIG_ENTRIES: list[ConfigEntriesItem] = [
    CE_OFF_GRID_MANAGER,
    CE_ENERGY_CALCULATOR,
    CE_CONSUMPTION_ESTIMATOR,
    CE_PVENERGY_HEURISTIC_ESTIMATOR,
]

if pmc.DEBUG:
    CONFIG_ENTRIES.extend(
        [
            CE_PV_PLANT_SIMULATOR,
        ]
    )
