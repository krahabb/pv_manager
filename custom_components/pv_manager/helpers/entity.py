import enum
from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import entity, restore_state

from . import Loggable
from .manager import Manager

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        ClassVar,
        Final,
        NotRequired,
        Self,
        TypedDict,
        Unpack,
    )

    from .. import const as pmc
    from ..controller import Controller, Device
    from ..processors import Estimator


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


class Entity(Loggable, entity.Entity if TYPE_CHECKING else object):

    if TYPE_CHECKING:

        class Args(TypedDict):
            config_subentry_id: NotRequired[str | None]
            name: NotRequired[str | None]
            entity_category: NotRequired[entity.EntityCategory]
            icon: NotRequired[str]
            parent_attr: NotRequired[ParentAttr | None]

        PLATFORM: ClassVar[str]
        is_diagnostic: ClassVar[bool]

        device: Final[Device]
        # HA core cache/slots const presets
        available: Final
        assumed_state: Final
        force_update: Final
        has_entity_name: Final
        should_poll: Final
        unique_id: Final[str]

    EntityCategory = entity.EntityCategory
    ParentAttr = ParentAttr

    is_diagnostic = False

    _attr_parent_attr: ParentAttr | None = ParentAttr.REMOVE
    """By default our entities will automatically remove their reference from the controller
    at shutdown. Be sure to override this at construction or inheritance when a different
    behavior is required."""

    # HA core entity attributes:
    _attr_device_class = None
    _attr_entity_category = None
    _attr_icon = None

    __slots__ = (
        "device",
        "config_subentry_id",
        "assumed_state",
        "available",
        "device_info",
        "entity_category",
        "extra_state_attributes",
        "force_update",
        "has_entity_name",
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
        **kwargs: "Unpack[Args]",
    ):
        controller = device.controller
        self.device = device
        self.config_subentry_id = kwargs.pop(
            "config_subentry_id", device.config_subentry_id
        )
        self.assumed_state = False
        self.available = True
        self.device_info = device.device_info
        self.entity_category = kwargs.pop("entity_category", self._attr_entity_category)
        self.extra_state_attributes = None
        self.force_update = False
        self.has_entity_name = True
        self.icon = kwargs.pop("icon", self._attr_icon)
        self.name = kwargs.pop("name", None) or id
        self.should_poll = False
        self.unique_id = f"{device.unique_id}-{id}"
        self.added_to_hass = False
        self._parent_attr = kwargs.pop("parent_attr", self._attr_parent_attr)
        for _attr_name, _attr_value in kwargs.items():
            setattr(self, _attr_name, _attr_value)
        Loggable.__init__(self, id, logger=device)

        entities = controller.entries[self.config_subentry_id].entities
        if self.unique_id in entities:
            raise Exception(
                f"{__name__}:{self.LN()} Duplicated id ({id}) in device ({device.name}:{device.unique_id}) subentry ({controller.entries[self.config_subentry_id].subentry_type}:{self.config_subentry_id})"
            )
        entities[self.unique_id] = self
        if self._parent_attr is ParentAttr.STATIC:
            setattr(device, f"{id}_{self.PLATFORM}", self)
        elif self._parent_attr is ParentAttr.DYNAMIC:
            setattr(device, f"{id}_{self.PLATFORM}", None)
        try:
            if add_entities := controller.platforms[self.PLATFORM]:
                add_entities((self,), config_subentry_id=self.config_subentry_id)
            else:
                assert controller.config_entry.state != ConfigEntryState.LOADED
        except KeyError:
            controller.platforms[self.PLATFORM] = None

    async def async_shutdown(self, remove: bool, /):
        if self._parent_attr:
            delattr(self.device, f"{self.id}_{self.PLATFORM}")
        del self.device.controller.entries[self.config_subentry_id].entities[
            self.unique_id
        ]
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

    def update(self, value, /):
        """Optimized update and flush to HA without checking the entity is added."""
        pass

    def update_safe(self, value, /):
        """Update and flush with safety check: only flush if added to HA"""
        pass

    def update_config(self, /, **kwargs: "Unpack[Args]"):
        """Updates entity attributes and flush to HA state. Useful when updating ConfigEntries/Subentries
        as a shortcut update. Pretty dangerous though if kwargs are not carefully choosen (even among [Args]).
        """
        for _attr_name, _attr_value in kwargs.items():
            setattr(self, _attr_name, _attr_value)
        if self.added_to_hass:
            self._async_write_ha_state()

    def update_name(self, name: str, /):
        """Updates entity name and flush to HA state. Useful when updating ConfigEntries/Subentries
        as a shortcut update."""
        self.name = name
        if self.added_to_hass:
            self._async_write_ha_state()


class ExtraStoredDataDict(dict, restore_state.ExtraStoredData):
    """Object to hold extra stored data as a plain dict"""

    def as_dict(self) -> "dict[str, Any]":
        return self

    @classmethod
    def from_dict(cls, restored: "dict[str, Any]", /) -> "Self | None":
        """Initialize a stored state from a dict."""
        return cls(restored)


class RestoreEntity(restore_state.RestoreEntity):
    pass


class DiagnosticEntity(Entity if TYPE_CHECKING else object):

    is_diagnostic: "Final" = True

    _attr_parent_attr = None

    # HA core entity attributes:
    _attr_entity_category = entity.EntityCategory.DIAGNOSTIC

    def __init__(self, device: "Device", id: str, /, *args, **kwargs):
        super().__init__(device, id, *args, **kwargs)
        device.controller.diagnostic_entities[id] = self

    async def async_shutdown(self, remove: bool, /):
        del self.device.controller.diagnostic_entities[self.id]
        await super().async_shutdown(remove)


class EstimatorEntity[_estimatorT: "Estimator[Any]"](Entity):

    if TYPE_CHECKING:

        type EstimatorUpdateT = Callable[[_estimatorT], Any]

        class Args(Entity.Args):
            estimator_update_func: NotRequired["EstimatorEntity.EstimatorUpdateT"]

        estimator: Final[_estimatorT]
        estimator_update_func: Final["EstimatorEntity.EstimatorUpdateT"]

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
        estimator: "_estimatorT",
        /,
        **kwargs: "Unpack[Args]",
    ):
        self.estimator = estimator
        self._estimator_update_unsub = None
        self.estimator_update_func = kwargs.pop("estimator_update_func", lambda e: None)
        super().__init__(device, id, **kwargs)

    async def async_shutdown(self, remove: bool, /):
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

    def on_estimator_update(self, estimator: "_estimatorT", /):
        # Called automatically whenever the binded estimator updates (if the entity is loaded).
        # Since it could be used to prepare the state before adding to hass it will nevertheless
        # need to check for added_to_hass.
        # The default implementation could work in simple cases by passing a
        # conversion function to the constructor (estimator_update_func):
        # - (estimator) -> entity state
        # But in general it would be more efficient to subclass and override (in
        # that case the _estimator_update_func is ignored)
        self.update_safe(self.estimator_update_func(estimator))
