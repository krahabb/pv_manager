import enum
import typing

from homeassistant.helpers import entity, restore_state

from . import Loggable
from ..manager import Manager

if typing.TYPE_CHECKING:
    from typing import ClassVar, Final, NotRequired

    from .. import const as pmc
    from ..controller import Controller, Device
    from ..controller.common.estimator import Estimator

    class EntityArgs(typing.TypedDict):
        config_subentry_id: NotRequired[str]
        name: NotRequired[str | None]
        entity_category: NotRequired[entity.EntityCategory]
        icon: NotRequired[str]
        # translation_key: typing.NotRequired[str]
        parent_attr: NotRequired["ParentAttr | None"]


class ParentAttr(enum.Enum):
    """Specifies if the entity should automatically manage its reference in the parent
    controller. This is used to avoid circular references in the entity tree and report
    the availability of the entity in the controller instance.
    The Entity will (eventually) create a reference member in the parent controller named as
    f{id}_{PLATFORM}."""

    REMOVE = 1
    """The entity will remove its reference at shutdown."""
    STATIC = 2
    """The entity will set a reference at construction and clear it at shutdown."""
    DYNAMIC = 3
    """The entity will set a reference when added to hass and clear it when removed."""


class Entity(Loggable, entity.Entity if typing.TYPE_CHECKING else object):

    PLATFORM: typing.ClassVar[str]

    EntityCategory = entity.EntityCategory
    ParentAttr = ParentAttr

    is_diagnostic: typing.ClassVar[bool] = False

    device: "Final[Device]"

    _attr_parent_attr: ParentAttr | None = ParentAttr.REMOVE
    """By default our entities will automatically remove their reference from the controller
    at shutdown. Be sure to override this at construction or inheritance when a different
    behavior is required."""

    # HA core entity attributes:
    _attr_device_class = None
    _attr_entity_category = None
    _attr_icon = None

    # HA core cache/slots const presets
    available: typing.Final
    assumed_state: typing.Final
    force_update: typing.Final
    should_poll: typing.Final

    __slots__ = (
        "device",
        "config_subentry_id",
        "assumed_state",
        "available",
        "device_info",
        "entity_category",
        "extra_state_attributes",
        "force_update",
        "icon",
        "name",
        "should_poll",
        "unique_id",
        "added_to_hass",
        "_parent_attr",
    )

    def __init__(
        self,
        device: "Device",
        id: str,
        **kwargs: "typing.Unpack[EntityArgs]",
    ):
        controller = device.controller
        self.device = device
        self.config_subentry_id = kwargs.pop("config_subentry_id", None)
        self.assumed_state = False
        self.available = True
        self.device_info = device.device_info
        self.entity_category = kwargs.pop("entity_category", self._attr_entity_category)
        self.extra_state_attributes = None
        self.force_update = False
        self.icon = kwargs.pop("icon", self._attr_icon)
        self.name = kwargs.pop("name", None) or id
        self.should_poll = False
        self.unique_id = "_".join((controller.config_entry.entry_id, id))
        self.added_to_hass = False
        self._parent_attr = kwargs.pop("parent_attr", self._attr_parent_attr)
        for _attr_name, _attr_value in kwargs.items():
            setattr(self, _attr_name, _attr_value)
        Loggable.__init__(self, id, logger=device)

        entities = controller.entries[self.config_subentry_id].entities
        assert id not in entities
        entities[id] = self
        if self._parent_attr is ParentAttr.STATIC:
            setattr(device, f"{id}_{self.PLATFORM}", self)
        elif self._parent_attr is ParentAttr.DYNAMIC:
            setattr(device, f"{id}_{self.PLATFORM}", None)
        try:
            if add_entities := controller.platforms[self.PLATFORM]:
                add_entities((self,), config_subentry_id=self.config_subentry_id)
        except KeyError:
            controller.platforms[self.PLATFORM] = None

    async def async_shutdown(self, remove: bool):
        if self._parent_attr:
            delattr(self.device, f"{self.id}_{self.PLATFORM}")
        self.device.controller.entries[self.config_subentry_id].entities.pop(self.id)
        if remove:
            if self.added_to_hass:
                await self.async_remove(force_remove=True)
            Manager.entity_registry.async_remove(self.entity_id)
        self.device = None  # type: ignore

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.added_to_hass = True
        if self._parent_attr is ParentAttr.DYNAMIC:
            setattr(self.device, f"{self.id}_{self.PLATFORM}", self)
        self.log(self.DEBUG, "added to hass")

    async def async_will_remove_from_hass(self):
        self.added_to_hass = False
        if self._parent_attr is ParentAttr.DYNAMIC:
            setattr(self.device, f"{self.id}_{self.PLATFORM}", None)
        await super().async_will_remove_from_hass()
        self.log(self.DEBUG, "removed from hass")

    def update(self, value):
        """Optimized update and flush to HA without checking the entity is added."""
        pass

    def update_safe(self, value):
        """Update and flush with safety check: only flush if added to HA"""
        pass

    def update_name(self, name: str):
        """Updates entity name and flush to HA state. Useful when updating ConfigEntries/Subentries
        as a shortcut update."""
        self.name = name
        if self.added_to_hass:
            self._async_write_ha_state()


class ExtraStoredDataDict(dict, restore_state.ExtraStoredData):
    """Object to hold extra stored data as a plain dict"""

    def as_dict(self) -> dict[str, typing.Any]:
        return self

    @classmethod
    def from_dict(cls, restored: dict[str, typing.Any]) -> typing.Self | None:
        """Initialize a stored state from a dict."""
        return cls(restored)


class RestoreEntity(restore_state.RestoreEntity):
    pass


class DiagnosticEntity(Entity if typing.TYPE_CHECKING else object):

    is_diagnostic: typing.Final = True

    _attr_parent_attr = None

    # HA core entity attributes:
    _attr_entity_category = entity.EntityCategory.DIAGNOSTIC

    def __init__(self, device: "Device", id: str, *args, **kwargs):
        super().__init__(device, id, *args, **kwargs)
        device.controller.diagnostic_entities[id] = self

    async def async_shutdown(self, remove: bool):
        self.device.controller.diagnostic_entities.pop(self.id)
        await super().async_shutdown(remove)


class EstimatorEntity(Entity if typing.TYPE_CHECKING else object):

    estimator: "Estimator"

    # by default we likely don't want to manage self instances in controller
    _attr_parent_attr = None

    _SLOTS_ = (
        "estimator",
        "_estimator_update_unsub",
        "_estimator_update_func",
    )

    def __init__(
        self,
        device: "Device",
        id: str,
        estimator: "Estimator",
        *args,
        estimator_update_func: typing.Callable[
            ["Estimator"], typing.Any
        ] = lambda e: None,
        **kwargs,
    ):
        self.estimator = estimator
        self._estimator_update_unsub = None
        self._estimator_update_func = estimator_update_func
        super().__init__(device, id, *args, **kwargs)

    async def async_shutdown(self, remove: bool):
        if self._estimator_update_unsub:
            self._estimator_update_unsub()
            self._estimator_update_unsub = None
        self.estimator = None  # type: ignore
        await super().async_shutdown(remove)

    async def async_added_to_hass(self):
        self.on_estimator_update(self.estimator)
        await super().async_added_to_hass()
        self._estimator_update_unsub = self.estimator.listen_update(
            self.on_estimator_update
        )

    async def async_will_remove_from_hass(self):
        if self._estimator_update_unsub:
            self._estimator_update_unsub()
            self._estimator_update_unsub = None
        await super().async_will_remove_from_hass()

    def on_estimator_update(self, estimator: "Estimator"):
        """Called automatically whenever the binded estimator updates (if the entity is loaded).
        Since it could be used to prepare the state before adding to hass it will nevertheless
        need to check for added_to_hass.
        The default implementation could work in simple cases by passing a
        conversion function to the constructor (estimator_update_func):
        - (estimator) -> entity state
        """
        self.update_safe(self._estimator_update_func(estimator))
