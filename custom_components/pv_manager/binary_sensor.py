import typing

from homeassistant.components import binary_sensor

from . import const as pmc, helpers
from .helpers.entity import Entity

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller
    from .helpers.entity import EntityArgs

    class BinarySensorArgs(EntityArgs):
        device_class: typing.NotRequired[binary_sensor.BinarySensorDeviceClass]
        is_on: typing.NotRequired[bool | None]


async def async_setup_entry(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry[Controller]",
    add_entities: "AddConfigEntryEntitiesCallback",
):
    await config_entry.runtime_data.async_setup_entry_platform(
        binary_sensor.DOMAIN, add_entities
    )


class BinarySensor(Entity, binary_sensor.BinarySensorEntity):

    PLATFORM = binary_sensor.DOMAIN

    DeviceClass = binary_sensor.BinarySensorDeviceClass

    __slots__ = (
        "device_class",
        "is_on",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        **kwargs: "typing.Unpack[BinarySensorArgs]",
    ):
        self.device_class = kwargs.pop("device_class", None)
        self.is_on = kwargs.pop("is_on", None)
        Entity.__init__(self, controller, id, **kwargs)

    def update(self, is_on: bool | None):
        if self.is_on != is_on:
            self.is_on = is_on
            if self.added_to_hass:
                self._async_write_ha_state()
