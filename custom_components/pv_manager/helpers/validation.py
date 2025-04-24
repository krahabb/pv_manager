import typing

from homeassistant import const as hac
from homeassistant.data_entry_flow import section
from homeassistant.helpers import selector
import voluptuous as vol

from .. import const as pmc

if typing.TYPE_CHECKING:
    from enum import StrEnum
    from typing import Any, Unpack


def optional(key: str, config, default=None):
    return vol.Optional(key, description={"suggested_value": config.get(key, default)})


def opt_default(key: str, default):
    return vol.Optional(key, description={"suggested_value": default})


def opt_config(key: str, config: pmc.ConfigMapping):
    return vol.Optional(key, description={"suggested_value": config.get(key)})


def required(key: str, config, default=None):
    return vol.Required(key, description={"suggested_value": config.get(key, default)})


def req_default(key: str, default):
    return vol.Required(key, description={"suggested_value": default})


def req_config(key: str, config: pmc.ConfigMapping):
    return vol.Required(key, description={"suggested_value": config.get(key)})


def exclusive(key: str, group: str, config, default=None):
    return vol.Exclusive(
        key, group, description={"suggested_value": config.get(key, default)}
    )


def entity_schema(
    config: pmc.EntityConfig | pmc.ConfigMapping = {},
) -> pmc.ConfigSchema:
    return {req_config("name", config): str}


def sensor_schema(
    config: pmc.SensorConfig | pmc.ConfigMapping,
    units: "type[StrEnum]",
) -> pmc.ConfigSchema:
    schema = entity_schema(config)
    schema[req_config("native_unit_of_measurement", config)] = select_selector(
        options=list(units)
    )
    return schema


def select_selector(**kwargs: "Unpack[selector.SelectSelectorConfig]"):
    return selector.SelectSelector(selector.SelectSelectorConfig(**kwargs))


if typing.TYPE_CHECKING:

    class _sensor_selector_args(typing.TypedDict):
        device_class: typing.NotRequired[str | list[str]]


def sensor_selector(**kwargs: "Unpack[_sensor_selector_args]"):
    return selector.EntitySelector(
        {
            "filter": {
                "domain": "sensor",
                **kwargs,
            }
        }
    )


def weather_selector():
    return selector.EntitySelector({"filter": {"domain": "weather"}})


def positive_number_selector(**kwargs: "Unpack[selector.NumberSelectorConfig]"):
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=kwargs.pop("min", 0),
            mode=kwargs.pop("mode", selector.NumberSelectorMode.BOX),
            **kwargs,  # type:ignore
        )
    )


def time_period_selector(**kwargs: "Unpack[selector.NumberSelectorConfig]"):
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=kwargs.pop("min", 0),
            unit_of_measurement=kwargs.pop(
                "unit_of_measurement", hac.UnitOfTime.SECONDS
            ),
            mode=kwargs.pop("mode", selector.NumberSelectorMode.BOX),
            **kwargs,  # type:ignore
        )
    )
