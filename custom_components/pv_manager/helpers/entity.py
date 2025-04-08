import enum
import typing

from homeassistant.helpers import entity

from . import Loggable

if typing.TYPE_CHECKING:

    from .. import const as pmc
    from ..controller import Controller

    class EntityArgs(typing.TypedDict):
        config_subentry_id: typing.NotRequired[str]
        name: typing.NotRequired[str | None]
        # translation_key: typing.NotRequired[str]
        parent_attr: typing.NotRequired["ParentAttr | None"]


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


    is_diagnostic: bool = False

    controller: "Controller"

    _attr_parent_attr: ParentAttr | None = ParentAttr.REMOVE
    """By default our entities will automatically remove their reference from the controller
    at shutdown. Be sure to override this at construction or inheritance when a different
    behavior is required."""

    __slots__ = (
        "controller",
        "config_subentry_id",
        "name",
        "should_poll",
        "unique_id",
        "added_to_hass",
        "_parent_attr",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        **kwargs: "typing.Unpack[EntityArgs]",
    ):
        self.controller = controller
        self.config_subentry_id = kwargs.pop("config_subentry_id", None)
        self.name = kwargs.pop("name", None) or id
        self.should_poll = False
        self.unique_id = "_".join((controller.config_entry.entry_id, id))
        self.added_to_hass = False
        self._parent_attr = kwargs.pop("parent_attr", self._attr_parent_attr)
        for _attr_name, _attr_value in kwargs.items():
            setattr(self, _attr_name, _attr_value)
        Loggable.__init__(self, id, logger=controller)
        if self.PLATFORM in controller.entities:
            controller.entities[self.PLATFORM][id] = self
            if self.PLATFORM in controller.platforms:
                controller.platforms[self.PLATFORM](
                    [self], config_subentry_id=self.config_subentry_id
                )
        else:
            controller.entities[self.PLATFORM] = {id: self}

        if self._parent_attr is ParentAttr.STATIC:
            setattr(controller, f"{id}_{self.PLATFORM}", self)
        elif self._parent_attr is ParentAttr.DYNAMIC:
            setattr(controller, f"{id}_{self.PLATFORM}", None)

    async def async_shutdown(self):
        if self._parent_attr:
            delattr(self.controller, f"{self.id}_{self.PLATFORM}")
        self.controller.entities[self.PLATFORM].pop(self.id)
        self.controller = None  # type: ignore


    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self.added_to_hass = True
        if self._parent_attr is ParentAttr.DYNAMIC:
            setattr(self.controller, f"{self.id}_{self.PLATFORM}", self)
        self.log(self.DEBUG, "added to hass")

    async def async_will_remove_from_hass(self):
        self.added_to_hass = False
        if self._parent_attr is ParentAttr.DYNAMIC:
            setattr(self.controller, f"{self.id}_{self.PLATFORM}", None)
        await super().async_will_remove_from_hass()
        self.log(self.DEBUG, "removed from hass")

    def update(self, value):
        # stub
        pass


class DiagnosticEntity(Entity if typing.TYPE_CHECKING else object):

    is_diagnostic: typing.Final = True

    _attr_parent_attr: ParentAttr | None = None

    # HA core entity attributes:
    _attr_entity_category = entity.EntityCategory.DIAGNOSTIC
