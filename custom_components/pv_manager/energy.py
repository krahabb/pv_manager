import typing

from . import const as pmc

if typing.TYPE_CHECKING:
    from homeassistant.components.energy.types import SolarForecastType
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

    from .controller import Controller


async def async_get_solar_forecast(
    hass: "HomeAssistant", config_entry_id: str
) -> "SolarForecastType | None":

    config_entry: "ConfigEntry[Controller]" = hass.config_entries.async_get_known_entry(config_entry_id)
    try:
        return config_entry.runtime_data.get_solar_forecast()
    except AttributeError:
        return None  # controller not loaded
