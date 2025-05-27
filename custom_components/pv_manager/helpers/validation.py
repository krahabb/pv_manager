import typing

from homeassistant import const as hac, data_entry_flow
from homeassistant.helpers import selector
import voluptuous as vol

from .. import const as pmc
from ..manager import MeteringCycle

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
    kwargs["mode"] = kwargs.get("mode", selector.SelectSelectorMode.DROPDOWN)
    return selector.SelectSelector(kwargs)


def cycle_modes_selector():
    return selector.SelectSelector(
        {
            "options": list(MeteringCycle.Mode),
            "multiple": True,
            "mode": selector.SelectSelectorMode.DROPDOWN,
        }
    )


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


def weather_entity_selector():
    return selector.EntitySelector({"filter": {"domain": "weather"}})


def number_selector(**kwargs: "Unpack[selector.NumberSelectorConfig]"):
    kwargs["mode"] = kwargs.get("mode", selector.NumberSelectorMode.BOX)
    return selector.NumberSelector(kwargs)


def positive_number_selector(**kwargs: "Unpack[selector.NumberSelectorConfig]"):
    kwargs["min"] = kwargs.get("min", 0)
    kwargs["mode"] = kwargs.get("mode", selector.NumberSelectorMode.BOX)
    return selector.NumberSelector(kwargs)


def time_period_selector(**kwargs: "Unpack[selector.NumberSelectorConfig]"):
    kwargs["min"] = kwargs.get("min", 0)
    kwargs["mode"] = kwargs.get("mode", selector.NumberSelectorMode.BOX)
    kwargs["unit_of_measurement"] = kwargs.get(
        "unit_of_measurement", hac.UnitOfTime.SECONDS
    )
    return selector.NumberSelector(kwargs)


def consumption_estimator_selector():
    # TODO: need to find out a way to filter on entry type
    return selector.ConfigEntrySelector(
        selector.ConfigEntrySelectorConfig(integration=pmc.DOMAIN)
    )


def pvenergy_estimator_selector():
    # TODO: need to find out a way to filter on entry type
    return selector.ConfigEntrySelector(
        selector.ConfigEntrySelectorConfig(integration=pmc.DOMAIN)
    )


class section(data_entry_flow.section):

    def __init__(self, schema: pmc.ConfigSchema, collapsed: bool = True):
        super().__init__(vol.Schema(schema), {"collapsed": collapsed})
