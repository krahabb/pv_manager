import typing

from homeassistant.exceptions import ConfigEntryError
from homeassistant.helpers import config_validation as cv, entity_registry

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
    await config_entry.runtime_data.async_shutdown()
    return True


async def async_migrate_entry(hass: "HomeAssistant", config_entry: "ConfigEntry"):

    import asyncio
    await asyncio.sleep(5)

    if config_entry.version > 1:
        # This means the user has downgraded from a future version
        return False

    if config_entry.version == 1:
        kwargs = {}

        if config_entry.minor_version < 2:

            def _migrate_uniqueid(registry_entry: entity_registry.RegistryEntry):
                """
                old format: f"{config_entry.entry_id}_{entity.id}"
                new format: f"{device.uniqued_id}-{entity.id}"

                where device.uniqued_id = f"{config_entry.entry_id}.{device.id}"
                or (controller/main device) = f"{config_entry.entry_id}"
                """
                s = registry_entry.unique_id.split("_")
                entry_id = s.pop(0)
                assert entry_id == config_entry.entry_id
                new_unique_id = f'{entry_id}-{"_".join(s)}'
                return {"new_unique_id": new_unique_id}

            await entity_registry.async_migrate_entries(
                hass, config_entry.entry_id, _migrate_uniqueid
            )

        hass.config_entries.async_update_entry(
            config_entry, minor_version=2, version=1, **kwargs
        )

    Manager.log(
        Manager.DEBUG,
        "Migration to configuration version %s.%s successful",
        config_entry.version,
        config_entry.minor_version,
    )

    return True
