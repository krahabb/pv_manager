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

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry


class ConfigSubentryFlow(config_entries.ConfigSubentryFlow):
    """Generalized subentry flow."""

    ENTRY_TYPE: typing.ClassVar[pmc.ConfigEntryType]
    SUBENTRY_TYPE: typing.ClassVar[pmc.ConfigSubentryType]

    """
    TODO: self.handler should be a tuple like: (entry_id, subentry_type)
    so we could simplify our generalization
    """

    async def async_step_user(self, user_input: pmc.SensorConfig | dict | None):
        """User flow to add a new location."""
        if user_input:
            return self.async_create_entry(
                title=user_input.get("name", self.SUBENTRY_TYPE),
                data=user_input,
            )

        user_input = {}
        controller_class = await Controller.get_controller_class(
            self.hass, self.ENTRY_TYPE
        )
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(controller_class.get_config_subentry_schema(self.SUBENTRY_TYPE, user_input)),
        )

SUBENTRY_FLOW_MAP = {}
for entry_type, subentry_tuple in pmc.CONFIGENTRY_SUBENTRY_MAP.items():
    subentry_flows = {}
    for subentry_type in subentry_tuple:
        class _ConfigSubentryFlow(ConfigSubentryFlow):
            ENTRY_TYPE = entry_type
            SUBENTRY_TYPE = subentry_type
        subentry_flows[subentry_type] = _ConfigSubentryFlow
    SUBENTRY_FLOW_MAP[entry_type] = subentry_flows


class ConfigFlow(config_entries.ConfigFlow, domain=pmc.DOMAIN):
    # The schema version of the entries that it creates
    # Home Assistant will call your migrate method if the version changes
    VERSION = 1
    MINOR_VERSION = 1

    reconfigure_entry: "ConfigEntry | None" = None

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: "ConfigEntry"
    ) -> dict[str, type[config_entries.ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return SUBENTRY_FLOW_MAP[pmc.ConfigEntryType.get_from_entry(config_entry)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # our 'steps' are all generically implemented through Controller generalization
        for type in pmc.ConfigEntryType:
            setattr(
                self, f"async_step_{type}", partial(self._async_step_controller, type)
            )

    async def async_step_user(self, user_input=None):
        return self.async_show_menu(step_id="user", menu_options=list(pmc.ConfigEntryType))

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

