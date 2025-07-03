import typing

from homeassistant import const as hac
from homeassistant.util import slugify

from .. import const as pmc, helpers
from ..helpers import validation as hv
from ..helpers.manager import Manager
from ..sensor import Sensor
from .devices import Device
from .devices.estimator_device import (
    EnergyEstimatorDevice,
    EnergyEstimatorSensor,
)

if typing.TYPE_CHECKING:
    from typing import Any, Callable, ClassVar, Coroutine, Final, TypedDict, Unpack

    from homeassistant.components.energy.types import SolarForecastType
    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from ..helpers.entity import DiagnosticEntity, Entity


class EntryData[_ConfigT: pmc.EntryConfig | pmc.SubentryConfig]:
    """Cached entry/subentry data."""

    if typing.TYPE_CHECKING:
        subentry_id: Final[str | None]  # type: ignore
        subentry_type: Final[str | None]  # type: ignore
        config: _ConfigT
        entities: Final[dict[str, Entity]]  # type: ignore

    __slots__ = (
        "subentry_id",
        "subentry_type",
        "config",
        "entities",
    )

    @staticmethod
    def Entry(config_entry: "ConfigEntry"):
        entry = EntryData()
        entry.subentry_id = None  # type: ignore
        entry.subentry_type = None  # type: ignore
        entry.config = config_entry.data
        entry.entities = {}  # type: ignore
        return entry

    @staticmethod
    def SubEntry(subentry: "ConfigSubentry"):
        entry = EntryData()
        entry.subentry_id = subentry.subentry_id  # type: ignore
        entry.subentry_type = subentry.subentry_type  # type: ignore
        entry.config = subentry.data
        entry.entities = {}  # type: ignore
        return entry


class Controller[_ConfigT: pmc.EntryConfig](Device):
    """Base controller class for managing ConfigEntry behavior."""

    class EntryReload(Exception):
        pass

    if typing.TYPE_CHECKING:

        class Config(pmc.EntryConfig):
            pass

        class Args(TypedDict):
            pass

        logger: helpers.logging.Logger

        TYPE: ClassVar[pmc.ConfigEntryType]
        PLATFORMS: ClassVar[set[str]]

        config: _ConfigT
        options: pmc.EntryOptionsConfig
        platforms: Final[dict[str, AddConfigEntryEntitiesCallback | None]]
        """Dict of add_entities callbacks."""
        devices: Final[list[Device]]
        entries: Final[dict[str | None, EntryData]]
        """Cached copy of subentries used to manage subentry add/remove/update."""
        diagnostic_entities: Final[dict[str, DiagnosticEntity]]

    PLATFORMS = set()
    """Default entity platforms used by the controller. This is used to prepopulate the entities dict so
    that we'll (at least) forward entry setup to these platforms (even if we've still not registered any entity for those).
    This in turn allows us to later create and register entities since we'll register the add_entities callback in our platforms."""

    __slots__ = (
        "config_entry",
        "config",
        "options",
        "platforms",
        "devices",
        "entries",
        "diagnostic_entities",
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
    def get_config_subentry_schema(
        config_entry: "ConfigEntry",
        subentry_type: str,
        config: pmc.ConfigMapping | None,
    ) -> pmc.ConfigSchema:
        # to be overriden
        return {}

    @staticmethod
    def get_config_subentry_unique_id(
        subentry_type: str, user_input: pmc.ConfigMapping
    ) -> str | None:
        # to be overriden for more complex behaviors
        _config_subentry_type = pmc.ConfigSubentryType(subentry_type)
        return _config_subentry_type.value if _config_subentry_type.unique else None

    def __init__(self, config_entry: "ConfigEntry"):
        self.config_entry = config_entry
        self.config = config_entry.data  # type: ignore
        self.options = config_entry.options  # type: ignore
        self.platforms = {platform: None for platform in self.PLATFORMS}
        self.devices = []
        self.entries = {None: EntryData.Entry(config_entry)}
        self.diagnostic_entities = {}
        logger = helpers.getLogger(
            f"{helpers.LOGGER.name}.{slugify(config_entry.title)}"
        )
        logger.setLevel(
            pmc.CONF_LOGGING_LEVEL_OPTIONS.get(
                self.options.get("logging_level", "default"), self.DEFAULT
            )
        )
        super().__init__(
            config_entry.entry_id, controller=self, logger=logger, config=self.config
        )

        entries = self.entries
        for subentry_id, subentry in config_entry.subentries.items():
            entries[subentry_id] = entry_data = EntryData.SubEntry(subentry)
            self._subentry_add(subentry_id, entry_data)

        if self.options.get("create_diagnostic_entities"):
            self._create_diagnostic_entities()

    # interface: Loggable
    def log(self, level: int, msg: str, *args, **kwargs):
        if (logger := self.logger).isEnabledFor(level):
            logger._log(level, msg, args, **kwargs)

    # interface: self
    async def async_setup(self):
        self.track_callback(
            "_entry_update_listener",
            self.config_entry.add_update_listener(self._entry_update_listener),
        )

        self.devices.sort(key=lambda device: device.priority)
        for device in self.devices:
            await device.async_start()
        await Manager.hass.config_entries.async_forward_entry_setups(
            self.config_entry, self.platforms
        )

    async def async_shutdown(self):
        if not await Manager.hass.config_entries.async_unload_platforms(
            self.config_entry, self.platforms.keys()
        ):
            raise Exception("Failed to unload platforms")

        # removing circular refs here
        for entry_data in self.entries.values():
            for entity in tuple(entry_data.entities.values()):
                await entity.async_shutdown(False)
            assert not entry_data.entities
        assert not self.diagnostic_entities
        self.platforms.clear()

        for device in tuple(reversed(self.devices)):
            device.shutdown()
        assert not self.devices

    async def async_setup_entry_platform(
        self,
        platform: str,
        add_entities: "AddConfigEntryEntitiesCallback",
    ):
        """Generic async_setup_entry for any platform where entities are instantiated
        in the controller constructor."""
        # cache the add_entities callback so we can dinamically add later
        self.platforms[platform] = add_entities
        # manage config_subentry forwarding...
        for subentry_id, entry_data in self.entries.items():
            entities = [
                entity
                for entity in entry_data.entities.values()
                if entity.PLATFORM is platform
            ]
            if entities:
                add_entities(entities, config_subentry_id=subentry_id)

    def schedule_reload(self):
        Manager.hass.config_entries.async_schedule_reload(self.config_entry.entry_id)

    def get_solar_forecast(self) -> "SolarForecastType | None":
        """Returns the forecasts array for HA energy integration.
        This is here to setup the entry-point to be called by the energy platform.
        Specialized controllers should point this to an actual implementation."""
        return None

    async def _entry_update_listener(
        self, hass: "HomeAssistant", config_entry: "ConfigEntry"
    ):
        try:
            entries = self.entries
            if self.config != config_entry.data:
                self.config = config_entry.data  # type: ignore
                entry_data = entries[None]
                entry_data.config = config_entry.data
                # eventually we can 'kick' a subentry_update
                # self._subentry_update(None, entry_data)

            # dispatch detailed subentry updates so that inheriteds don't have to bother
            # every time scanning and matching variations
            removed_entries = set()
            for subentry_id in entries:
                if subentry_id and (subentry_id not in config_entry.subentries):
                    removed_entries.add(subentry_id)
            for subentry_id in removed_entries:
                entry_data = entries[subentry_id]
                await self._async_subentry_remove(subentry_id, entry_data)
                # removed leftover entities (eventually)
                for entity in tuple(entry_data.entities.values()):
                    try:
                        await entity.async_shutdown(True)
                    except Exception as e:
                        self.log_exception(self.WARNING, e, "entity.async_shutdown")

                assert not entry_data.entities
                # remove only after shutting down entities
                del entries[subentry_id]

            for subentry_id, subentry in config_entry.subentries.items():
                try:
                    entry_data = entries[subentry_id]
                    if entry_data.config is not subentry.data:
                        entry_data.config = subentry.data
                        await self._async_subentry_update(subentry_id, entry_data)
                except KeyError:
                    # new subentry
                    entries[subentry_id] = entry_data = EntryData.SubEntry(subentry)
                    self._subentry_add(subentry_id, entry_data)

            if self.options != config_entry.options:
                self.options = config_entry.options  # type: ignore
                self.logger.setLevel(
                    pmc.CONF_LOGGING_LEVEL_OPTIONS.get(
                        self.options.get("logging_level", "default"), self.DEFAULT
                    )
                )
                if self.options.get("create_diagnostic_entities"):
                    self._create_diagnostic_entities()
                else:
                    await self._async_destroy_diagnostic_entities()

        except Controller.EntryReload:
            self.schedule_reload()

    def _create_diagnostic_entities(self):
        """Dynamically create some diagnostic entities depending on configuration"""
        pass

    async def _async_destroy_diagnostic_entities(self):
        """Cleanup diagnostic entities when configuration option changes."""
        for entity in list(self.diagnostic_entities.values()):
            await entity.async_shutdown(True)
        assert not self.diagnostic_entities

    def _subentry_add(self, subentry_id: str, entry_data: EntryData):
        """Placeholder method invoked whan a new subentry is added and at start.
        Raises 'EntryReload' if not able to handle adding an entry 'on the fly' so that the
        config_entry will be reloaded."""

    async def _async_subentry_update(self, subentry_id: str, entry_data: EntryData):
        """Placeholder method invoked whan a subentry is updated.
        Raises 'EntryReload' if not able to handle updating an entry 'on the fly' so that the
        config_entry will be reloaded."""

    async def _async_subentry_remove(self, subentry_id: str, entry_data: EntryData):
        """Placeholder method invoked whan a subentry is removed.
        Raises 'EntryReload' if not able to handle removing an entry 'on the fly' so that the
        config_entry will be reloaded."""


class EnergyEstimatorController[_ConfigT: "EnergyEstimatorController.Config"](  # type: ignore
    Controller[_ConfigT], EnergyEstimatorDevice
):

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimatorDevice.Config, Controller.Config):
            pass

    PLATFORMS = {Sensor.PLATFORM}

    @staticmethod
    def get_config_subentry_schema(
        config_entry: "ConfigEntry",
        subentry_type: str,
        config: pmc.ConfigMapping | None,
    ) -> pmc.ConfigSchema:
        match subentry_type:
            case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                if not config:
                    config = {
                        "name": "Energy forecast",
                        "forecast_duration_hours": 1,
                    }
                return hv.entity_schema(
                    config,
                ) | {
                    hv.req_config(
                        "forecast_duration_hours", config
                    ): hv.time_period_selector(
                        min=1, unit_of_measurement=hac.UnitOfTime.HOURS
                    ),
                }

        return {}

    # interface: Controller
    @typing.override
    def _subentry_add(
        self, subentry_id: str, entry_data: "EntryData[EnergyEstimatorSensor.Config]"
    ):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                EnergyEstimatorSensor(
                    self,
                    f"{pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR}_{subentry_id}",
                    name=entry_data.config.get(
                        "name", pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR
                    ),
                    config_subentry_id=subentry_id,
                    forecast_duration_ts=int(
                        entry_data.config.get("forecast_duration_hours", 1) * 3600
                    ),
                )

    @typing.override
    async def _async_subentry_update(self, subentry_id: str, entry_data: EntryData):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                entity: EnergyEstimatorSensor
                for entity in entry_data.entities.values():  # type: ignore
                    entity.name = entry_data.config.get(
                        "name", pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR
                    )
                    entity.forecast_duration_ts = int(
                        entry_data.config.get("forecast_duration_hours", 1) * 3600
                    )
                    entity.on_estimator_update(self)
