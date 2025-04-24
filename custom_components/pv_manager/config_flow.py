import enum
from functools import partial
from types import MappingProxyType
import typing
import uuid

from homeassistant import config_entries, const as hac, data_entry_flow
from homeassistant.core import callback
import voluptuous as vol

from . import const as pmc
from .controller import Controller
from .helpers import validation as hv

if typing.TYPE_CHECKING:
    from typing import Any

    from homeassistant.config_entries import ConfigEntry, ConfigSubentry


class CommonFlow(data_entry_flow.FlowHandler if typing.TYPE_CHECKING else object):
    """Common Mixin style base class for all of our flows.
    This will need special care in implementations in order to set its attributes depending on context.
    """

    controller_class: type[Controller] = None  # type: ignore
    """Cached Controller class instance providing schemas and rules to this flow."""
    current_schema: pmc.ConfigSchema = {}
    """current_schema is updated whenever we show a form so that the context knows the actual
    validation Schema. This is used for example to correctly merge updates in reconfigure flows."""
    current_config: pmc.ConfigMapping = {}
    """current_config is used to generate defaults for schema by passing it to the Controller schema creator.
    Being 'None' or empty means the configuration is new so the controller should fill in some defaults else
    it should just the available config (reconfigure flows)."""

    def merge_input(self, user_input: pmc.ConfigMapping):
        """Merge current input into current config allowing to preserve values which will not be changed
        while re-configuring. This will ensure also, that optional fields (in schema) are correctly removed
        from config should they be nulled by the user."""
        data = dict(self.current_config)
        data.update(user_input)
        for key in self.current_schema.keys():
            s_key = key.schema
            if s_key not in user_input:
                data.pop(s_key, None)
        return data


class ConfigSubentryFlow(CommonFlow, config_entries.ConfigSubentryFlow):  # type: ignore
    """Generalized subentry flow."""

    ENTRY_TYPE: typing.ClassVar[pmc.ConfigEntryType]
    SUBENTRY_TYPE: typing.ClassVar[pmc.ConfigSubentryType]

    """
    TODO: self.handler should be a tuple like: (entry_id, subentry_type)
    so we could simplify our generalization
    """

    async def async_step_user(self, user_input: pmc.ConfigDict | None = None):

        if user_input:
            return self.async_create_entry(
                title=user_input.get("name", self.SUBENTRY_TYPE),
                data=user_input,
                unique_id=self.controller_class.get_config_subentry_unique_id(
                    self.SUBENTRY_TYPE, user_input
                ),
            )

        return await self._async_show_form()

    async def async_step_reconfigure(self, user_input: pmc.ConfigDict | None = None):

        if user_input:
            return self.async_update_and_abort(
                self._get_reconfigure_entry(),
                self._get_reconfigure_subentry(),
                title=user_input.get("name", self.SUBENTRY_TYPE),
                data=self.merge_input(user_input),
            )

        self.current_config = self._get_reconfigure_subentry().data
        return await self._async_show_form()

    async def _async_show_form(
        self,
    ):
        if not self.controller_class:
            self.controller_class = await Controller.get_controller_class(
                self.hass, self.ENTRY_TYPE
            )

        self.current_schema = self.controller_class.get_config_subentry_schema(
            self.SUBENTRY_TYPE, self.current_config
        )
        return super().async_show_form(
            data_schema=vol.Schema(self.current_schema),
        )


SUBENTRY_FLOW_MAP = {entry_type: {} for entry_type in pmc.ConfigEntryType}
for entry_type, subentry_tuple in pmc.CONFIGENTRY_SUBENTRY_MAP.items():
    subentry_flows = SUBENTRY_FLOW_MAP[entry_type]
    for subentry_type in subentry_tuple:

        class _ConfigSubentryFlow(ConfigSubentryFlow):
            ENTRY_TYPE = entry_type
            SUBENTRY_TYPE = subentry_type

        subentry_flows[subentry_type.value] = _ConfigSubentryFlow


class ConfigFlow(CommonFlow, config_entries.ConfigFlow, domain=pmc.DOMAIN):  # type: ignore
    # The schema version of the entries that it creates
    # Home Assistant will call your migrate method if the version changes
    VERSION = 1
    MINOR_VERSION = 1

    reconfigure_entry: "ConfigEntry | None" = None

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
            return SUBENTRY_FLOW_MAP[pmc.ConfigEntryType.get_from_entry(config_entry)]
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
                    menu_options.append(entry_type)
        return self.async_show_menu(menu_options=menu_options)

    async def async_step_reconfigure(self, user_input):
        self.reconfigure_entry = self._get_reconfigure_entry()
        return await getattr(
            self,
            f"async_step_{pmc.ConfigEntryType.get_from_entry(self.reconfigure_entry)}",
        )(None)

    async def _async_step_controller(
        self, controller_type: pmc.ConfigEntryType, user_input: pmc.ConfigDict | None
    ):
        if user_input:
            if self.reconfigure_entry:
                return self.async_update_reload_and_abort(
                    self.reconfigure_entry,
                    title=user_input.get("name", controller_type),
                    data=self.merge_input(user_input),
                )
            else:
                return self.async_create_entry(
                    title=user_input.get("name", controller_type),
                    data=user_input,
                )

        if self.reconfigure_entry:
            self.current_config = self.reconfigure_entry.data
        else:
            await self.async_set_unique_id(
                ".".join((controller_type, uuid.uuid4().hex))
            )

        self.controller_class = await Controller.get_controller_class(
            self.hass, controller_type
        )
        self.current_schema = self.controller_class.get_config_entry_schema(
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
