import typing

from . import Loggable

if typing.TYPE_CHECKING:
    from homeassistant.helpers import entity

    from .. import const as pmc
    from ..controller import Controller

    class EntityArgs(typing.TypedDict):
        config_subentry_id: typing.NotRequired[str]
        name: typing.NotRequired[str | None]
        # translation_key: typing.NotRequired[str]


class Entity(Loggable, entity.Entity if typing.TYPE_CHECKING else object):

    PLATFORM: typing.ClassVar[str]

    __slots__ = (
        "controller",
        "config_subentry_id",
        "name",
        "should_poll",
        "unique_id",
        "_added_to_hass",
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
        self._added_to_hass = False
        for _attr_name, _attr_value in kwargs.items():
            setattr(self, _attr_name, _attr_value)
        Loggable.__init__(self, id, logger=controller)
        platform_entities = controller.entities.setdefault(self.PLATFORM, {})
        platform_entities[id] = self

    async def async_added_to_hass(self):
        await super().async_added_to_hass()
        self._added_to_hass = True
        self.log(self.DEBUG, "added to hass")

    async def async_will_remove_from_hass(self):
        self._added_to_hass = False
        await super().async_will_remove_from_hass()
        self.log(self.DEBUG, "removed from hass")
