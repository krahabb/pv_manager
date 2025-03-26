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

    PV_ENERGY_CALCULATOR = enum.auto()
    PV_ENERGY_ESTIMATOR = enum.auto()
    PV_POWER_SIMULATOR = enum.auto()

    @staticmethod
    def get_from_entry(config_entry: "ConfigEntry"):
        return ConfigEntryType((config_entry.unique_id).split(".")[0])  # type: ignore


class ConfigSubentryType(enum.StrEnum):
    # PV_ENERGY_SENSOR = enum.auto()
    pass


CONFIGENTRY_SUBENTRY_MAP: dict[ConfigEntryType, tuple[ConfigSubentryType, ...]] = {
    ConfigEntryType.PV_ENERGY_CALCULATOR: (),
    # ConfigEntryType.PV_ENERGY_CALCULATOR: (ConfigSubentryType.PV_ENERGY_SENSOR,),
    ConfigEntryType.PV_ENERGY_ESTIMATOR: (),
    ConfigEntryType.PV_POWER_SIMULATOR: (),
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


class EntityConfig(typing.TypedDict):
    name: str


class SensorConfig(EntityConfig):
    native_unit_of_measurement: str | enum.StrEnum
