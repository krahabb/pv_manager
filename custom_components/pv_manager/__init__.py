import typing

from homeassistant.exceptions import ConfigEntryError

from . import const as pmc
from .controller import Controller

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


async def async_setup_entry(
    hass: "HomeAssistant", config_entry: "ConfigEntry[Controller]"
):

    try:
        controller_class = await Controller.get_controller_class(
            hass, pmc.ConfigEntryType.get_from_entry(config_entry)
        )
    except Exception as e:
        raise ConfigEntryError("Invalid Controller class lookup") from e

    try:
        cntrl = controller_class(hass, config_entry)
        await cntrl.async_init()
        return True
    except Exception as e:
        if hasattr(config_entry, "runtime_data"):
            # TODO: better cleanup maybe invoking a 'safer' async_shutdown
            object.__delattr__(config_entry, "runtime_data")
        raise ConfigEntryError("Error initializing Controller class") from e


async def async_unload_entry(
    hass: "HomeAssistant", config_entry: "ConfigEntry[Controller]"
):
    cntrl = config_entry.runtime_data
    await cntrl.async_shutdown()
    return True
