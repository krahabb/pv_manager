import typing

from homeassistant.components import binary_sensor

from . import const as pmc, helpers
from .helpers.entity import Entity
from .processors import ProcessorWarning

if typing.TYPE_CHECKING:
    from typing import Callable, Iterable, Unpack

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller, Device
    from .helpers.entity import Entity


async def async_setup_entry(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry[Controller]",
    add_entities: "AddConfigEntryEntitiesCallback",
):
    await config_entry.runtime_data.async_setup_entry_platform(
        BinarySensor.PLATFORM, add_entities
    )


class BinarySensor(Entity, binary_sensor.BinarySensorEntity):

    if typing.TYPE_CHECKING:
        class Args(Entity.Args):
            device_class: typing.NotRequired[binary_sensor.BinarySensorDeviceClass]
            is_on: typing.NotRequired[bool | None]


    PLATFORM = binary_sensor.DOMAIN

    DeviceClass = binary_sensor.BinarySensorDeviceClass

    __slots__ = (
        "device_class",
        "is_on",
    )

    def __init__(
        self,
        device: "Device",
        id: str,
        **kwargs: "Unpack[Args]",
    ):
        self.device_class = kwargs.pop("device_class", self._attr_device_class)
        self.is_on = kwargs.pop("is_on", None)
        super().__init__(device, id, **kwargs)

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
    """Manages the state of multiple ProcessorWarning for the same warning id.
    The state is set as the 'or' of the warnings states meaning any warning in the collection
    being on will set the sensor on."""

    _attr_device_class = BinarySensor.DeviceClass.PROBLEM
    _attr_entity_category = BinarySensor.EntityCategory.DIAGNOSTIC
    _attr_parent_attr = None

    _processor_warnings: "Iterable[ProcessorWarning]"
    _processor_warnings_unsub: "Iterable[Callable[[], None]]"

    # optimized properties for the case where we only have one
    _processor_warning: "ProcessorWarning | None"

    __slots__ = (
        "_processor_warnings",
        "_processor_warnings_unsub",
        "_processor_warning",
    )

    def __init__(
        self,
        device: "Device",
        id,
        processor_warning: "Iterable[ProcessorWarning] | ProcessorWarning",
        **kwargs: "Unpack[BinarySensor.Args]",
    ):
        if isinstance(processor_warning, ProcessorWarning):
            self._processor_warning = processor_warning
        else:
            self._processor_warning = None
            self._processor_warnings = processor_warning
        self._processor_warnings_unsub = ()
        super().__init__(
            device,
            id,
            **kwargs,
        )

    async def async_shutdown(self, remove):
        for unsub in self._processor_warnings_unsub:
            unsub()
        await super().async_shutdown(remove)
        self._processor_warnings = None  # type: ignore
        self._processor_warnings_unsub = None  # type: ignore
        self._processor_warning = None

    async def async_added_to_hass(self):
        if self._processor_warning:
            self.is_on = self._processor_warning.on
            self._processor_warnings_unsub = (
                self._processor_warning.listen(self._update_warnings),
            )
        else:
            self._update_warnings(False)
            self._processor_warnings_unsub = [
                _processor_warning.listen(self._update_warnings)
                for _processor_warning in self._processor_warnings
            ]
        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self):
        for unsub in self._processor_warnings_unsub:
            unsub()
        self._processor_warnings_unsub = ()
        await super().async_will_remove_from_hass()

    def _update_warnings(self, is_on):
        if self._processor_warning:
            # Binded to a single source of warning: no need to populate
            # extra_state_attributes or cycle whatever
            self.is_on = is_on
        else:
            sources = []
            for warning in self._processor_warnings:
                if warning.on:
                    sources.append(warning.processor.id)
            if sources:
                self.is_on = True
                self.extra_state_attributes = {"sources": sources}
            else:
                self.is_on = False
                self.extra_state_attributes = None
        if self.added_to_hass:
            self.async_write_ha_state()
