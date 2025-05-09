import typing

from homeassistant import config_entries, const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import slugify
import pytest

from custom_components.pv_manager import const as pmc
from custom_components.pv_manager.controller import (
    pv_energy_estimator,
    pv_plant_simulator,
)

from tests import const as tc, helpers

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry, ConfigFlowResult
    from homeassistant.core import HomeAssistant


async def test_estimator_pv_energy_heuristic(
    hass: "HomeAssistant", time_mock: helpers.TimeMocker
):
    HISTORY_DURATION = 1 * 86400
    SIMULATOR_DATA_RATE_PERIOD = 60

    pv_plant_simulator.Controller.SAMPLING_PERIOD = SIMULATOR_DATA_RATE_PERIOD
    pv_simulator_entry = helpers.ConfigEntryMocker[pv_plant_simulator.Controller](
        hass, tc.CE_PV_PLANT_SIMULATOR
    )
    assert await pv_simulator_entry.async_setup()

    try:

        pv_simulator_controller = pv_simulator_entry.controller
        # pre-fill with 1 day worth of history the HA recorder
        await time_mock.async_warp(HISTORY_DURATION, SIMULATOR_DATA_RATE_PERIOD)

        data: "pv_energy_estimator.Controller.Config" = dict(tc.CE_PVENERGY_HEURISTIC_ESTIMATOR["data"])  # type: ignore
        data["source_entity_id"] = (
            pv_simulator_controller.pv_power_simulator_sensor.entity_id
        )
        data["maximum_latency_seconds"] = SIMULATOR_DATA_RATE_PERIOD * 2
        async with helpers.ConfigEntryMocker(
            hass,
            tc.ConfigEntriesItem(
                type=pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR,
                data=data,
            ),
        ) as pv_estimator_mock:

            pass
    finally:
        await pv_simulator_entry.async_unload()
