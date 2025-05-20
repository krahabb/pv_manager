import typing

from . import const as pmc

if typing.TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.components.energy.types import SolarForecastType

    from .controller.pv_energy_estimator import Controller as PvEnergyEstimator

async def async_get_solar_forecast(
    hass: "HomeAssistant", config_entry_id: str
) -> "SolarForecastType | None":

    config_entry = hass.config_entries.async_get_known_entry(config_entry_id)
    match pmc.ConfigEntryType.get_from_entry(config_entry):
        case pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR:
            try:
                controller: "PvEnergyEstimator" = config_entry.runtime_data
            except AttributeError:
                return None # not loaded
            return controller.get_solar_forecast()

