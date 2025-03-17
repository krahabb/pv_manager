import enum
import logging
import typing

import homeassistant.const as hac

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

DOMAIN: typing.Final = "pv_manager"


class ConfigEntryType(enum.StrEnum):
    """The type of config entry"""

    PV_POWER_SIMULATOR = enum.auto()
    PV_ENERGY_CALCULATOR = enum.auto()
    PV_ENERGY_ESTIMATOR = enum.auto()

    @staticmethod
    def get_from_entry(config_entry: "ConfigEntry"):
        return ConfigEntryType((config_entry.unique_id).split(".")[0])  # type: ignore


class ConfigSubentryType(enum.StrEnum):
    #PV_ENERGY_SENSOR = enum.auto()
    pass


CONFIGENTRY_SUBENTRY_MAP: dict[ConfigEntryType, tuple[ConfigSubentryType, ...]] = {
    ConfigEntryType.PV_POWER_SIMULATOR: (),
    #ConfigEntryType.PV_ENERGY_CALCULATOR: (ConfigSubentryType.PV_ENERGY_SENSOR,),
    ConfigEntryType.PV_ENERGY_CALCULATOR: (),
    ConfigEntryType.PV_ENERGY_ESTIMATOR: (),
}

CONF_TYPE: typing.Final = "type"
# sets the logging level x ConfigEntry
CONF_LOGGING_LEVEL: typing.Final = "logging_level"
CONF_LOGGING_VERBOSE: typing.Final = 5
CONF_LOGGING_DEBUG: typing.Final = logging.DEBUG
CONF_LOGGING_INFO: typing.Final = logging.INFO
CONF_LOGGING_WARNING: typing.Final = logging.WARNING
CONF_LOGGING_CRITICAL: typing.Final = logging.CRITICAL
CONF_LOGGING_LEVEL_OPTIONS: typing.Final = {
    logging.NOTSET: "default",
    CONF_LOGGING_CRITICAL: "critical",
    CONF_LOGGING_WARNING: "warning",
    CONF_LOGGING_INFO: "info",
    CONF_LOGGING_DEBUG: "debug",
    CONF_LOGGING_VERBOSE: "verbose",
}


class BaseConfig(typing.TypedDict):
    logging_level: typing.NotRequired[int]


CONF_NAME: typing.Final = hac.CONF_NAME


class EntityConfig(typing.TypedDict):
    name: str


CONF_NATIVE_UNIT_OF_MEASUREMENT: typing.Final = "native_unit_of_measurement"


class SensorConfig(EntityConfig):
    native_unit_of_measurement: str | enum.StrEnum


CONF_PEAK_POWER: typing.Final = "peak_power"
CONF_SIMULATE_WEATHER: typing.Final = "simulate_weather"


CONF_PV_POWER_ENTITY_ID: typing.Final = "pv_power_entity_id"
CONF_INTEGRATION_PERIOD: typing.Final = "integration_period"
CONF_MAXIMUM_LATENCY: typing.Final = "maximum_latency"


CONF_SOURCE_ENTITY_ID: typing.Final = "source_entity_id"
CONF_DAILY_ENERGY_FORECAST_ENTITY: typing.Final = "daily_energy_forecast_entity"


class PVEnergyEstimatorConfigType(BaseConfig):
    """Common config_entry keys"""

    source_entity_id: str
    """The entity_id of the source entity"""
    daily_energy_forecast_entity: EntityConfig
    """create an entity with the daily energy of pv"""
