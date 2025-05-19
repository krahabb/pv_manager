from datetime import timedelta
from time import time
import typing

from homeassistant.util import dt as dt_util

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
            controller: "PvEnergyEstimator" = config_entry.runtime_data
            time = dt_util.start_of_local_day()
            delta = timedelta(hours=1)
            wh_hours = {}
            for i in range(48):
                ts = time.timestamp()
                wh_hours[time.isoformat()] = controller.get_estimated_energy(ts, ts + 3600)
                time = time + delta
            return { "wh_hours": wh_hours}

