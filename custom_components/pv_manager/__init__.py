import typing

from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers import config_validation as cv

from . import const as pmc
from .controller import Controller
from .manager import Manager

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


CONFIG_SCHEMA = cv.config_entry_only_config_schema(pmc.DOMAIN)


async def async_setup(hass: "HomeAssistant", config):

    # force initialize
    Manager.get(hass)

    return True


async def async_setup_entry(
    hass: "HomeAssistant", config_entry: "ConfigEntry[Controller]"
):

    try:
        controller_class = await Controller.get_controller_class(
            hass, pmc.ConfigEntryType.get_from_entry(config_entry)
        )
        config_entry.runtime_data = cntrl = controller_class(config_entry)
        await cntrl.async_setup()
        return True
    except Exception as e:
        if hasattr(config_entry, "runtime_data"):
            # TODO: better cleanup maybe invoking a 'safer' async_shutdown
            object.__delattr__(config_entry, "runtime_data")
        raise ConfigEntryError(f"Error initializing Controller class: {str(e)}") from e


async def async_unload_entry(
    hass: "HomeAssistant", config_entry: "ConfigEntry[Controller]"
):
    cntrl = config_entry.runtime_data
    await cntrl.async_shutdown()
    return True
