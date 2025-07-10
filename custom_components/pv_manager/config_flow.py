import enum
from functools import partial
from typing import TYPE_CHECKING
import uuid

from homeassistant import config_entries, const as hac, data_entry_flow
from homeassistant.core import callback
import voluptuous as vol

from . import const as pmc
from .controller import Controller
from .helpers import validation as hv

if TYPE_CHECKING:

    from homeassistant.config_entries import CALLBACK_TYPE, ConfigEntry, ConfigSubentry
    from homeassistant.core import HomeAssistant

    SUBENTRY_FLOW_MAP: dict[str, dict[str, type[config_entries.ConfigSubentryFlow]]]
    ENTRY_UPDATE_LISTENERS: dict[ConfigEntry, CALLBACK_TYPE]


class CommonFlow(data_entry_flow.FlowHandler if TYPE_CHECKING else object):
    """Common Mixin style base class for all of our flows.
    This will need special care in implementations in order to set its attributes depending on context.
    """

    if TYPE_CHECKING:
        controller_class: type[Controller]
        """Cached Controller class instance providing schemas and rules to this flow."""
        config_entry: ConfigEntry | None
        """Cached config_entry when reconfiguring or managing subentry flows. Hoping it doesn't mess with HA."""
        current_schema: pmc.ConfigSchema
        """current_schema is updated whenever we show a form so that the context knows the actual
        validation Schema. This is used for example to correctly merge updates in reconfigure flows."""
        current_config: pmc.ConfigMapping
        """current_config is used to generate defaults for schema by passing it to the Controller schema creator.
        Being 'None' or empty means the configuration is new so the controller should fill in some defaults else
        it should just the available config (reconfigure flows)."""

    controller_class = None  # type: ignore
    config_entry = None
    current_schema = {}
    current_config = {}

    def merge_input(self, user_input: pmc.ConfigMapping):
        """Merge current input into current config allowing to preserve values which will not be changed
        while re-configuring. This will ensure also, that optional fields (in schema) are correctly removed
        from config should they be nulled by the user."""
        # TODO: merge subkeys since we're supporting sections in schema
        data = dict(self.current_config)
        schema_keys = {key.schema for key in self.current_schema.keys()}
        data = {
            key: value
            for key, value in self.current_config.items()
            if key in schema_keys
        }
        data.update(user_input)
        for key in schema_keys:
            if key not in user_input:
                data.pop(key, None)
        return data


class ConfigSubentryFlow(CommonFlow, config_entries.ConfigSubentryFlow):  # type: ignore
    """Generalized subentry flow."""

    if TYPE_CHECKING:
        entry_id: str
        subentry_type: str
        config_entry: ConfigEntry
        subentry_id: str | None
        config_subentry: ConfigSubentry

    subentry_id = None

    async def _async_init_controller(self):
        assert not self.controller_class
        self.entry_id, self.subentry_type = self.handler
        self.config_entry = self.hass.config_entries.async_get_known_entry(
            self.entry_id
        )
        self.controller_class = await Controller.get_controller_class(
            self.hass, pmc.ConfigEntryType.get_from_entry(self.config_entry)
        )

    async def async_step_user(self, user_input: pmc.ConfigDict | None = None):

        if user_input:
            return self.async_create_entry(
                title=user_input.get("name", self.subentry_type),
                data=user_input,
                unique_id=self.controller_class.get_config_subentry_unique_id(
                    self.subentry_type, user_input
                ),
            )
        else:
            await self._async_init_controller()
            for subentry in self.config_entry.subentries.values():
                if subentry.unique_id == self.subentry_type:
                    self.config_subentry = subentry
                    self.current_config = subentry.data
                    return await self._async_show_form("reconfigure")

        return await self._async_show_form()

    async def async_step_reconfigure(self, user_input: pmc.ConfigDict | None = None):

        if user_input:
            return self.async_update_and_abort(
                self.config_entry,
                self.config_subentry,
                title=user_input.get("name", self.subentry_type),
                data=self.merge_input(user_input),
            )
        else:
            await self._async_init_controller()
            self.subentry_id = self._reconfigure_subentry_id
            self.config_subentry = self.config_entry.subentries[self.subentry_id]
            self.current_config = self.config_subentry.data

        return await self._async_show_form()

    async def _async_show_form(self, step_id: str | None = None):
        self.current_schema = self.controller_class.get_config_subentry_schema(
            self.config_entry, self.subentry_type, self.current_config
        )
        return super().async_show_form(
            step_id=step_id,
            data_schema=vol.Schema(self.current_schema),
        )


SUBENTRY_FLOW_MAP = {
    pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR: {
        pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR: ConfigSubentryFlow,
    },
    pmc.ConfigEntryType.CONSUMPTION_ESTIMATOR: {
        pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR: ConfigSubentryFlow,
    },
    pmc.ConfigEntryType.OFF_GRID_MANAGER: {
        pmc.ConfigSubentryType.MANAGER_BATTERY_METER: ConfigSubentryFlow,
        pmc.ConfigSubentryType.MANAGER_LOAD_METER: ConfigSubentryFlow,
        pmc.ConfigSubentryType.MANAGER_PV_METER: ConfigSubentryFlow,
        pmc.ConfigSubentryType.MANAGER_ESTIMATOR: ConfigSubentryFlow,
        pmc.ConfigSubentryType.MANAGER_LOSSES: ConfigSubentryFlow,
    },
}
UNIQUE_SUBENTRY_TYPES = [
    subentry_type for subentry_type in pmc.ConfigSubentryType if subentry_type.unique
]
ENTRY_UPDATE_LISTENERS = {}


async def _entry_update_listener(hass: "HomeAssistant", config_entry: "ConfigEntry"):
    """Called on every entry change for entries that support unique subentry types.
    This callback will make sure that the 'supported_subentry_types' property gets updated
    depending on unique ones being already (or not) configured."""
    # Raw approach: just invalidate the underlying property and let the code invoke
    # async_get_supported_subentry_types since the "update" logic is the same and we would
    # just duplicate the code
    object.__setattr__(config_entry, "_supported_subentry_types", None)


class ConfigFlow(CommonFlow, config_entries.ConfigFlow, domain=pmc.DOMAIN):  # type: ignore
    # The schema version of the entries that it creates
    # Home Assistant will call your migrate method if the version changes
    VERSION = 1
    MINOR_VERSION = 2

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Create the options flow."""
        return OptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: "ConfigEntry"
    ) -> dict[str, type[config_entries.ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        try:
            supported_subentry_types = SUBENTRY_FLOW_MAP[
                pmc.ConfigEntryType.get_from_entry(config_entry)
            ]
            unique_subentry_types = [
                subentry_type
                for subentry_type in UNIQUE_SUBENTRY_TYPES
                if subentry_type in supported_subentry_types
            ]
            if not unique_subentry_types:
                return supported_subentry_types

            # This config_entry has some unique subentry types so we have to check
            # if these are already configured. We also setup a listener for changes so that
            # we can dynamically update the ConfigEntry.supported_subentry_types
            if config_entry not in ENTRY_UPDATE_LISTENERS:
                ENTRY_UPDATE_LISTENERS[config_entry] = config_entry.add_update_listener(
                    _entry_update_listener
                )

            configured_unique_subentry_types = [
                subentry.subentry_type
                for subentry in config_entry.subentries.values()
                if subentry.subentry_type in unique_subentry_types
            ]
            if not configured_unique_subentry_types:
                return supported_subentry_types

            return {
                subentry_type: subentry_flow_class
                for subentry_type, subentry_flow_class in supported_subentry_types.items()
                if subentry_type not in configured_unique_subentry_types
            }

        except:
            # no log since the entry is likely unloadable and more detailed logging should
            # be available in async_setup_entry
            return {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # our 'steps' are all generically implemented through Controller generalization
        for type in pmc.ConfigEntryType:
            setattr(
                self, f"async_step_{type}", partial(self._async_step_controller, type)
            )

    async def async_step_user(self, user_input=None):
        if pmc.DEBUG:
            menu_options = list(pmc.ConfigEntryType)
        else:
            DEBUG_ENTRY_TYPES = {pmc.ConfigEntryType.PV_PLANT_SIMULATOR}
            menu_options = []
            for entry_type in pmc.ConfigEntryType:
                if entry_type not in DEBUG_ENTRY_TYPES:
                    menu_options.append(entry_type.value)
        return self.async_show_menu(menu_options=menu_options)

    async def async_step_reconfigure(self, user_input):
        self.config_entry = self._get_reconfigure_entry()
        return await getattr(
            self,
            f"async_step_{pmc.ConfigEntryType.get_from_entry(self.config_entry)}",
        )(None)

    async def _async_step_controller(
        self, controller_type: pmc.ConfigEntryType, user_input: pmc.ConfigDict | None
    ):
        if user_input is not None:
            if self.config_entry:
                return self.async_update_reload_and_abort(
                    self.config_entry,
                    title=user_input.get("name", controller_type),
                    data=self.merge_input(user_input),
                )
            else:
                return self.async_create_entry(
                    title=user_input.get("name", controller_type),
                    data=user_input,
                )

        if self.config_entry:
            self.current_config = self.config_entry.data
        else:
            await self.async_set_unique_id(
                ".".join((controller_type, uuid.uuid4().hex))
            )

        self.controller_class = await Controller.get_controller_class(
            self.hass, controller_type
        )
        self.current_schema = self.controller_class.get_config_schema(
            self.current_config
        )
        return self.async_show_form(
            step_id=controller_type,
            data_schema=vol.Schema(self.current_schema),
        )


class OptionsFlow(config_entries.OptionsFlow):
    async def async_step_init(self, user_input: pmc.ConfigMapping | None):
        """Manage the options."""
        if user_input:
            return self.async_create_entry(data=user_input)

        user_input = self.config_entry.options
        if not user_input:
            user_input = {
                "logging_level": "default",
                "create_diagnostic_entities": False,
            }

        return self.async_show_form(
            data_schema=vol.Schema(
                {
                    hv.req_config("logging_level", user_input): hv.select_selector(
                        options=list(pmc.CONF_LOGGING_LEVEL_OPTIONS.keys())
                    ),
                    hv.req_config("create_diagnostic_entities", user_input): bool,
                }
            ),
        )
