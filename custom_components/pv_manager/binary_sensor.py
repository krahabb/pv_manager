import typing

from homeassistant.components import binary_sensor

from . import const as pmc, helpers
from .helpers.entity import Entity

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller
    from .controller.common import ProcessorWarning
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
        BinarySensor.PLATFORM, add_entities
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
        self.device_class = kwargs.pop("device_class", self._attr_device_class)
        self.is_on = kwargs.pop("is_on", None)
        Entity.__init__(self, controller, id, **kwargs)

    def update(self, is_on: bool | None):
        if self.is_on != is_on:
            self.is_on = is_on
            self._async_write_ha_state()

    def update_safe(self, is_on: bool | None):
        if self.is_on != is_on:
            self.is_on = is_on
            if self.added_to_hass:
                self._async_write_ha_state()


class ProcessorWarningBinarySensor(BinarySensor):

    _attr_device_class = BinarySensor.DeviceClass.PROBLEM
    _attr_entity_category = BinarySensor.EntityCategory.DIAGNOSTIC
    _attr_parent_attr = None

    _processor_warning: "ProcessorWarning"
    __slots__ = (
        "_processor_warning",
        "_processor_warning_unsub",
    )

    def __init__(
        self,
        controller,
        id,
        processor_warning: "ProcessorWarning",
        **kwargs: "typing.Unpack[BinarySensorArgs]",
    ):
        self._processor_warning = processor_warning
        self._processor_warning_unsub = None
        super().__init__(
            controller,
            id,
            **kwargs,
        )

    async def async_shutdown(self, remove):
        if self._processor_warning_unsub:
            self._processor_warning_unsub()
            self._processor_warning_unsub = None
        await super().async_shutdown(remove)
        self._processor_warning = None # type: ignore

    async def async_added_to_hass(self):
        self.is_on = self._processor_warning.on
        self._processor_warning_unsub = self._processor_warning.listen(self.update)
        return await super().async_added_to_hass()

    async def async_will_remove_from_hass(self):
        if self._processor_warning_unsub:
            self._processor_warning_unsub()
            self._processor_warning_unsub = None
        return await super().async_will_remove_from_hass()
