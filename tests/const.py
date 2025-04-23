"""Constants for pv_manager tests."""

import enum
import typing


from custom_components.pv_manager import const as pmc
from custom_components.pv_manager.helpers import metering


class NotRequiredEnum(enum.Enum):
    STRING = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()


class EntityIdEnum(enum.StrEnum):
    BATTERY_VOLTAGE = "sensor.battery_voltage"
    BATTERY_CURRENT = "sensor.battery_current"
    BATTERY_CHARGE = "sensor.battery_charge"
    PV_POWER = "sensor.pv_power"
    LOAD_POWER = "sensor.load_power"
    CONSUMPTION_POWER = "sensor.consumption"


class ConfigEntriesItem(typing.TypedDict):
    type: pmc.ConfigEntryType
    data: dict[str, typing.Any]


CE_PV_PLANT_SIMULATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.PV_PLANT_SIMULATOR,
        "data": {
            "name": "PV plant simulator",
            "native_unit_of_measurement": "kW",
            "peak_power": 1000,
            "weather_entity_id": NotRequiredEnum.STRING,
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
            "battery_voltage_entity_id": EntityIdEnum.BATTERY_VOLTAGE,
            "battery_current_entity_id": EntityIdEnum.BATTERY_CURRENT,
            "battery_charge_entity_id": NotRequiredEnum.STRING,
            "battery_capacity": 100,
            "pv_power_entity_id": NotRequiredEnum.STRING,
            "load_power_entity_id": NotRequiredEnum.STRING,
            "maximum_latency_minutes": 5,
        },
    }
)
CE_ENERGY_CALCULATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.ENERGY_CALCULATOR,
        "data": {
            "name": "Energy calculator",
            "power_entity_id": EntityIdEnum.LOAD_POWER,
            "cycle_modes": list(metering.CycleMode),
            "integration_period_seconds": 5,
            "maximum_latency_seconds": 60,
        },
    }
)
CE_ENERGY_CALCULATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.ENERGY_CALCULATOR,
        "data": {
            "name": "Energy calculator",
            "power_entity_id": EntityIdEnum.LOAD_POWER,
            "cycle_modes": list(metering.CycleMode),
            "integration_period_seconds": 5,
            "maximum_latency_seconds": 60,
        },
    }
)
CE_CONSUMPTION_ESTIMATOR = ConfigEntriesItem(
    {
        "type": pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR,
        "data": {
            "name": "Consumption estimator",
            "observed_entity_id": EntityIdEnum.CONSUMPTION_POWER,
            "refresh_period_minutes": 5,
            "sampling_interval_minutes": 10,
            "observation_duration_minutes": 60,
            "history_duration_days": 1,
            "maximum_latency_minutes": 5,
        },
    }
)

CONFIG_ENTRIES: list[ConfigEntriesItem] = [
    CE_PV_PLANT_SIMULATOR,
    CE_OFF_GRID_MANAGER,
    CE_ENERGY_CALCULATOR,
    CE_CONSUMPTION_ESTIMATOR,
]
