import enum
from functools import partial
import typing
import uuid

from homeassistant import config_entries, const as hac
from homeassistant.core import callback
from homeassistant.data_entry_flow import section
import voluptuous as vol

from . import const as pmc
from .controller import Controller
from .helpers import validation as hv

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry, ConfigSubentry


class ConfigSubentryFlow(config_entries.ConfigSubentryFlow):
    """Generalized subentry flow."""

    ENTRY_TYPE: typing.ClassVar[pmc.ConfigEntryType]
    SUBENTRY_TYPE: typing.ClassVar[pmc.ConfigSubentryType]

    reconfigure_subentry: "ConfigSubentry | None" = None

    """
    TODO: self.handler should be a tuple like: (entry_id, subentry_type)
    so we could simplify our generalization
    """

    async def async_step_user(self, user_input):

        if user_input:
            if self.reconfigure_subentry:
                return self.async_update_and_abort(
                    self._get_reconfigure_entry(),
                    self.reconfigure_subentry,
                    title=user_input.get("name", self.SUBENTRY_TYPE),
                    data=user_input,
                )
            else:
                return self.async_create_entry(
                    title=user_input.get("name", self.SUBENTRY_TYPE),
                    data=user_input,
                )

        if self.reconfigure_subentry:
            user_input = self.reconfigure_subentry.data
        else:
            user_input = {}

        controller_class = await Controller.get_controller_class(
            self.hass, self.ENTRY_TYPE
        )
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                controller_class.get_config_subentry_schema(
                    self.SUBENTRY_TYPE, user_input
                )
            ),
        )

    async def async_step_reconfigure(self, user_input: dict | None):
        self.reconfigure_subentry = self._get_reconfigure_subentry()
        return await self.async_step_user(None)


SUBENTRY_FLOW_MAP = {entry_type: {} for entry_type in pmc.ConfigEntryType}
for entry_type, subentry_tuple in pmc.CONFIGENTRY_SUBENTRY_MAP.items():
    subentry_flows = SUBENTRY_FLOW_MAP[entry_type]
    for subentry_type in subentry_tuple:

        class _ConfigSubentryFlow(ConfigSubentryFlow):
            ENTRY_TYPE = entry_type
            SUBENTRY_TYPE = subentry_type

        subentry_flows[subentry_type.value] = _ConfigSubentryFlow


class ConfigFlow(config_entries.ConfigFlow, domain=pmc.DOMAIN):
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
        return self.async_show_menu(step_id="user", menu_options=menu_options)

    async def async_step_reconfigure(self, user_input):
        self.reconfigure_entry = self._get_reconfigure_entry()
        return await getattr(
            self,
            f"async_step_{pmc.ConfigEntryType.get_from_entry(self.reconfigure_entry)}",
        )(None)

    async def _async_step_controller(
        self, controller_type: pmc.ConfigEntryType, user_input
    ):
        if user_input:
            if self.reconfigure_entry:
                return self.async_update_reload_and_abort(
                    self.reconfigure_entry,
                    title=user_input.get("name", controller_type),
                    data=user_input,
                )
            else:
                return self.async_create_entry(
                    title=user_input.get("name", controller_type),
                    data=user_input,
                )

        if self.reconfigure_entry:
            user_input = self.reconfigure_entry.data
        else:
            user_input = {}
            await self.async_set_unique_id(
                ".".join((controller_type, uuid.uuid4().hex))
            )

        controller_class = await Controller.get_controller_class(
            self.hass, controller_type
        )
        return self.async_show_form(
            step_id=controller_type,
            data_schema=vol.Schema(
                controller_class.get_config_entry_schema(user_input)
            ),
        )


class OptionsFlow(config_entries.OptionsFlow):
    async def async_step_init(self, user_input):
        """Manage the options."""
        if user_input:
            return self.async_create_entry(data=user_input)

        user_input = self.config_entry.options

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        hv.required(
                            "logging_level", user_input, "default"
                        ): hv.select_selector(
                            options=list(pmc.CONF_LOGGING_LEVEL_OPTIONS.keys())
                        ),
                        hv.required(
                            "create_diagnostic_entities", user_input, False
                        ): bool,
                    }
                ),
                self.config_entry.options,
            ),
        )
