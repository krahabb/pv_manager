import typing

import pytest
from homeassistant import config_entries, const as hac
from homeassistant.config_entries import ConfigEntryState
from homeassistant.data_entry_flow import FlowResultType

from custom_components.pv_manager import const as pmc

from tests import const as tc, helpers

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry, ConfigFlowResult
    from homeassistant.core import HomeAssistant

async def _cleanup_config_entry(hass: "HomeAssistant", result: "ConfigFlowResult"):
    config_entry: "ConfigEntry" = result["result"]  # type: ignore
    assert config_entry.state == ConfigEntryState.LOADED
    assert await hass.config_entries.async_unload(config_entry.entry_id)

#@pytest.mark.usefixtures("recorder_mock")
async def test_config_flow(hass: "HomeAssistant"):

    config_flow = hass.config_entries.flow

    for ce in tc.CONFIG_ENTRIES:

        entry_type = ce["type"]

        try:
            result = await config_flow.async_init(
                pmc.DOMAIN, context={"source": config_entries.SOURCE_USER}
            )
            result = await helpers.async_assert_flow_menu_to_step(
                config_flow, result, "user", entry_type.value
            )
            user_input = {}
            for _key, _value in ce["data"].items():
                if not isinstance(_value, tc.Optional):
                    user_input[_key] = _value

            result = await config_flow.async_configure(
                result["flow_id"],
                user_input=user_input,
            )
            assert result.get("type") == FlowResultType.CREATE_ENTRY

            config_entry: "ConfigEntry" = result["result"]  # type: ignore
            assert config_entry.unique_id.split(".")[0] == entry_type # type: ignore

            # now cleanup the entry
            await _cleanup_config_entry(hass, result)

        except Exception as e:
            raise Exception(f"Testing config entry :{str(ce)}") from e


async def test_options_flow(hass: "HomeAssistant"):

    options_flow = hass.config_entries.options

    for ce in tc.CONFIG_ENTRIES:

        try:
            async with helpers.ConfigEntryMocker(hass, ce) as ce_mock:

                # try a debug config with diagnostic entities
                result = await options_flow.async_init(ce_mock.config_entry_id)
                user_input = pmc.EntryOptionsConfig({
                    "logging_level": "debug",
                    "create_diagnostic_entities": False,
                })
                result = await options_flow.async_configure(
                    result["flow_id"],
                    user_input=dict(user_input),
                )
                assert result.get("type") == FlowResultType.CREATE_ENTRY

                # try revert it to default behaviour/config
                result = await options_flow.async_init(ce_mock.config_entry_id)
                user_input = pmc.EntryOptionsConfig({
                    "logging_level": "default",
                    "create_diagnostic_entities": False,
                })
                result = await options_flow.async_configure(
                    result["flow_id"],
                    user_input=dict(user_input),
                )
                assert result.get("type") == FlowResultType.CREATE_ENTRY

        except Exception as e:
            raise Exception(f"Testing config entry :{str(ce)}") from e