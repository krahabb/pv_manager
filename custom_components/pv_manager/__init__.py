import typing

from . import const as pmc, controller, helpers
from .controller import Controller

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


async def async_setup_entry(
    hass: "HomeAssistant", config_entry: "ConfigEntry[controller.Controller]"
):

    controller_class = await Controller.get_controller_class(
        hass, pmc.ConfigEntryType.get_from_entry(config_entry)
    )

    cntrl = controller_class(hass, config_entry)
    await cntrl.async_init()
    return True


async def async_unload_entry(
    hass: "HomeAssistant", config_entry: "ConfigEntry[controller.Controller]"
):
    cntrl = config_entry.runtime_data
    await cntrl.async_shutdown()
    return True
