import enum
import logging
import typing

import homeassistant.const as hac

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

DOMAIN: typing.Final = "pv_manager"

try:
    import json
    import os

    class DEBUG:
        """Define a DEBUG symbol which will be None in case the debug conf is missing so
        that the code can rely on this to enable special behaviors."""

        # this will raise an OSError on non-dev machines missing the
        # debug configuration file so the DEBUG symbol will be invalidated
        data = json.load(
            open(
                file="./custom_components/pv_manager/debug.secret.json",
                mode="r",
                encoding="utf-8",
            )
        )

        @staticmethod
        def get_debug_output_filename(hass: "HomeAssistant", filename):
            path = hass.config.path("custom_components", DOMAIN, "debug")
            os.makedirs(path, exist_ok=True)
            return os.path.join(path, filename)

except Exception:
    DEBUG = None  # type: ignore


class ConfigEntryType(enum.StrEnum):
    """The type of config entry"""

    ENERGY_CALCULATOR = enum.auto()
    PV_ENERGY_ESTIMATOR = enum.auto()
    CONSUMPTION_ESTIMATOR = enum.auto()
    OFF_GRID_MANAGER = enum.auto()
    PV_PLANT_SIMULATOR = enum.auto()

    @staticmethod
    def get_from_entry(config_entry: "ConfigEntry"):
        # might raise ValueError if config_entry.unique_id does not map to a valid enum
        return ConfigEntryType((config_entry.unique_id).split(".")[0])  # type: ignore


class ConfigSubentryType(enum.StrEnum):
    ENERGY_ESTIMATOR_SENSOR = enum.auto()
    MANAGER_ENERGY_SENSOR = enum.auto()


CONFIGENTRY_SUBENTRY_MAP: dict[ConfigEntryType, tuple[ConfigSubentryType, ...]] = {
    ConfigEntryType.PV_ENERGY_ESTIMATOR: (ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR,),
    ConfigEntryType.CONSUMPTION_ESTIMATOR: (
        ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR,
    ),
    ConfigEntryType.OFF_GRID_MANAGER: (ConfigSubentryType.MANAGER_ENERGY_SENSOR,),
}

CONF_TYPE: typing.Final = "type"


class EntryConfig(typing.TypedDict):
    """Base (common) definition for ConfigEntry.data"""

    pass


class SubentryConfig(typing.TypedDict):
    """Base (common) definition for ConfigSubentry.data"""

    pass


# sets the logging level x ConfigEntry
CONF_LOGGING_LEVEL_OPTIONS: typing.Final = {
    "default": logging.NOTSET,
    "critical": logging.CRITICAL,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "verbose": 5,
}


class EntryOptionsConfig(typing.TypedDict):
    """Base (common) definition for ConfigEntry.options"""

    logging_level: typing.NotRequired[str]
    create_diagnostic_entities: typing.NotRequired[bool]


class EntityConfig(typing.TypedDict):
    name: str


class SensorConfig(EntityConfig):
    native_unit_of_measurement: str | enum.StrEnum
