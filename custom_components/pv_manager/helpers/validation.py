import typing

from homeassistant import const as hac
from homeassistant.data_entry_flow import section
from homeassistant.helpers import selector
import voluptuous as vol

from .. import const as pmc

if typing.TYPE_CHECKING:
    from enum import StrEnum


def required(key: str, user_input, default=None):
    return vol.Required(
        key, description={"suggested_value": user_input.get(key, default)}
    )


def entity_schema(
    user_input: pmc.EntityConfig | dict = {},
    **defaults: typing.Unpack[pmc.EntityConfig],
) -> dict:
    return {required(pmc.CONF_NAME, user_input, defaults.get("name")): str}


def sensor_schema(
    user_input: pmc.SensorConfig | dict = {},
    **defaults: typing.Unpack[pmc.SensorConfig],
) -> dict:
    schema = entity_schema(user_input, **defaults)
    default_unit = defaults.get("native_unit_of_measurement")
    schema[required(pmc.CONF_NATIVE_UNIT_OF_MEASUREMENT, user_input, default_unit)] = (
        select_selector(type(default_unit))  # type: ignore
    )
    return schema


def sensor_section(
    user_input: dict = {},
    collapsed: bool = True,
    **defaults: typing.Unpack[pmc.SensorConfig],
):
    return section(
        vol.Schema(sensor_schema(user_input, **defaults)),
        {"collapsed": collapsed},
    )


def select_selector(**kwargs: "typing.Unpack[selector.SelectSelectorConfig]"):
    return selector.SelectSelector(selector.SelectSelectorConfig(**kwargs))


if typing.TYPE_CHECKING:

    class _sensor_selector_args(typing.TypedDict):
        device_class: typing.NotRequired[str]


def sensor_selector(**kwargs: "typing.Unpack[_sensor_selector_args]"):
    return selector.EntitySelector(
        {
            "filter": {
                "domain": "sensor",
                **kwargs,
            }
        }
    )


def time_period_selector(**kwargs):
    return selector.NumberSelector(
        {
            "min": 0,
            "mode": "box",
            "unit_of_measurement": hac.UnitOfTime.SECONDS,
            **kwargs,
        }
    )
