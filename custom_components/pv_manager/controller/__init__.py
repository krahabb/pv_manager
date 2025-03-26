import typing

from homeassistant.core import callback
from homeassistant.helpers import entity_registry

from .. import const as pmc, helpers

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Coroutine

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
    import voluptuous as vol

    from ..helpers.entity import Entity


class Controller[_ConfigT: pmc.BaseConfig](helpers.Loggable):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE: typing.ClassVar[pmc.ConfigEntryType]

    PLATFORMS: typing.ClassVar[set[str]] = set()
    """Default entity platforms used by the controller"""

    config: _ConfigT
    entities: typing.Final[dict[str, dict[str, "Entity"]]]
    hass: "HomeAssistant"

    __slots__ = (
        "config_entry",
        "config",
        "entities",
        "hass",
        "_entry_update_listener_unsub",
    )

    @staticmethod
    async def get_controller_class(
        hass: "HomeAssistant", type: pmc.ConfigEntryType
    ) -> "type[Controller]":
        controller_module = await helpers.async_import_module(
            hass, f".controller.{type}"
        )
        return controller_module.Controller

    @staticmethod
    def get_config_entry_schema(user_input) -> dict:
        # to be overriden
        return {}

    @staticmethod
    def get_config_subentry_schema(subentry_type: str, user_input) -> dict:
        # to be overriden
        return {}

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        self.config_entry = config_entry
        self.config = config_entry.data  # type: ignore
        self.entities = {platform: {} for platform in self.PLATFORMS}
        self.hass = hass
        helpers.Loggable.__init__(self, config_entry.title)
        config_entry.runtime_data = self

    # interface: Loggable
    def configure_logger(self):
        """
        Configure a 'logger' and a 'logtag' based off current config for every ConfigEntry.
        """
        self.logtag = f"{self.TYPE}({self.id})"
        # using helpers.getLogger (instead of logger.getChild) to 'wrap' the Logger class ..
        self.logger = logger = helpers.getLogger(f"{helpers.LOGGER.name}.{self.logtag}")
        try:
            logger.setLevel(self.config.get("logging_level", self.DEBUG))
        except Exception as exception:
            # do not use self Loggable interface since we might be not set yet
            helpers.LOGGER.warning(
                "error (%s) setting log level: likely a corrupted configuration entry",
                str(exception),
            )

    def log(self, level: int, msg: str, *args, **kwargs):
        if (logger := self.logger).isEnabledFor(level):
            logger._log(level, msg, args, **kwargs)

    # interface: self
    async def async_init(self):
        self._entry_update_listener_unsub = self.config_entry.add_update_listener(
            self._entry_update_listener
        )
        # Here we're forwarding to all the platform registerd in self.entities.
        # This is by default preset in the constructor with a list of (default) PLATFORMS
        # for the controller class.
        # The list of 'actual' entities could also be enriched by instantiating entities
        # in the (derived) contructor since async_init will be called at loading time right after
        # class instance initialization.
        await self.hass.config_entries.async_forward_entry_setups(
            self.config_entry, self.entities.keys()
        )

    async def async_setup_entry_platform(
        self,
        platform: str,
        add_entities: "AddConfigEntryEntitiesCallback",
    ):
        """Generic async_setup_entry for any platform where entities are instantiated
        in the controller constructor. This should be overriden with it's more specific
        async_setup_entry_{platform} for more optimized initialization"""
        # manage config_subentry forwarding...
        e: dict[str | None, list] = {}
        for entity in self.entities[platform].values():
            if entity.config_subentry_id in e:
                e[entity.config_subentry_id].append(entity)
            else:
                e[entity.config_subentry_id] = [entity]
        for config_subentry_id, entities in e.items():
            add_entities(entities, config_subentry_id=config_subentry_id)

    async def async_shutdown(self):
        if not await self.hass.config_entries.async_unload_platforms(
            self.config_entry, self.entities.keys()
        ):
            return False
        self._entry_update_listener_unsub()
        # removing circular refs here...maybe invoke entity shutdown?
        self.entities.clear()
        return True

    def get_entity_registry(self):
        return entity_registry.async_get(self.hass)
    
    def schedule_async_callback(
        self, delay: float, target: "Callable[..., Coroutine]", *args
    ):
        @callback
        def _callback(_target, *_args):
            self.async_create_task(_target(*_args), "._callback")

        return self.hass.loop.call_later(delay, _callback, target, *args)

    def schedule_callback(self, delay: float, target: "Callable", *args):
        return self.hass.loop.call_later(delay, target, *args)

    @callback
    def async_create_task[_R](
        self,
        target: "Coroutine[Any, Any, _R]",
        name: str,
        eager_start: bool = True,
    ):
        return self.config_entry.async_create_task(
            self.hass, target, f"{self.logtag}{name}", eager_start
        )

    async def _entry_update_listener(
        self, hass: "HomeAssistant", config_entry: "ConfigEntry"
    ):
        self.config = config_entry.data  # type: ignore
