import typing

from homeassistant import const as hac
from homeassistant.util import slugify

from .. import const as pmc, helpers
from ..binary_sensor import ProcessorWarningBinarySensor
from ..helpers import validation as hv
from ..helpers.entity import EstimatorEntity
from ..manager import Manager
from ..processors.estimator import EnergyEstimator
from ..sensor import Sensor
from .devices import Device

if typing.TYPE_CHECKING:
    from typing import Any, Callable, ClassVar, Coroutine, Final, TypedDict, Unpack

    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.core import HomeAssistant, State
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from ..helpers.entity import DiagnosticEntity, Entity, EntityArgs


class EntryData[_ConfigT: pmc.EntryConfig | pmc.SubentryConfig]:
    """Cached entry/subentry data."""

    subentry_id: "Final[str | None]"
    subentry_type: "Final[str | None]"
    config: _ConfigT | pmc.ConfigMapping
    entities: "Final[dict[str, Entity]]"

    __slots__ = (
        "subentry_id",
        "subentry_type",
        "config",
        "entities",
    )

    @staticmethod
    def Entry(config_entry: "ConfigEntry"):
        entry = EntryData()
        entry.subentry_id = None
        entry.subentry_type = None
        entry.config = config_entry.data
        entry.entities = {}
        return entry

    @staticmethod
    def SubEntry(subentry: "ConfigSubentry"):
        entry = EntryData()
        entry.subentry_id = subentry.subentry_id
        entry.subentry_type = subentry.subentry_type
        entry.config = subentry.data
        entry.entities = {}
        return entry


class Controller[_ConfigT: pmc.EntryConfig](Device):
    """Base controller class for managing ConfigEntry behavior."""

    if typing.TYPE_CHECKING:

        class Config(pmc.EntryConfig):
            pass

        class Args(TypedDict):
            pass

    TYPE: "ClassVar[pmc.ConfigEntryType]"

    PLATFORMS: "ClassVar[set[str]]" = set()
    """Default entity platforms used by the controller. This is used to prepopulate the entities dict so
    that we'll (at least) forward entry setup to these platforms (even if we've still not registered any entity for those).
    This in turn allows us to later create and register entities since we'll register the add_entities callback in our platforms."""

    logger: helpers.logging.Logger

    config: _ConfigT
    options: pmc.EntryOptionsConfig
    platforms: "Final[dict[str, AddConfigEntryEntitiesCallback | None]]"
    """Dict of add_entities callbacks."""
    devices: "Final[dict[str, Device]]"
    entries: "Final[dict[str | None, EntryData]]"
    """Cached copy of subentries used to manage subentry add/remove/update."""
    diagnostic_entities: "Final[dict[str, DiagnosticEntity]]"

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
    def get_config_entry_schema(config: pmc.ConfigMapping | None) -> pmc.ConfigSchema:
        # to be overriden
        return {}

    @staticmethod
    def get_config_subentry_schema(
        subentry_type: str, config: pmc.ConfigMapping | None
    ) -> pmc.ConfigSchema:
        # to be overriden
        return {}

    @staticmethod
    def get_config_subentry_unique_id(
        subentry_type: str, user_input: pmc.ConfigMapping
    ) -> str | None:
        # to be overriden
        return None

    def __init__(self, config_entry: "ConfigEntry"):
        self.config_entry = config_entry
        self.config = config_entry.data  # type: ignore
        self.options = config_entry.options  # type: ignore
        self.platforms = {platform: None for platform in self.PLATFORMS}
        self.devices = {}
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
        for subentry_id, subentry in self.config_entry.subentries.items():
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
        for device in self.devices.values():
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
        for subentry_id, entry_data in self.entries.items():
            for entity in tuple(entry_data.entities.values()):
                await entity.async_shutdown(False)
            assert not entry_data.entities
        assert not self.diagnostic_entities
        self.platforms.clear()

        for device in tuple(reversed(self.devices.values())):
            device.shutdown()
        self.devices.clear()

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

    async def _entry_update_listener(
        self, hass: "HomeAssistant", config_entry: "ConfigEntry"
    ):
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
                await entity.async_shutdown(True)
            assert not entry_data.entities
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

    def _create_diagnostic_entities(self):
        """Dynamically create some diagnostic entities depending on configuration"""
        pass

    async def _async_destroy_diagnostic_entities(self):
        """Cleanup diagnostic entities when configuration option changes."""
        for entity in list(self.diagnostic_entities.values()):
            await entity.async_shutdown(True)
        assert not self.diagnostic_entities

    def _subentry_add(self, subentry_id: str, entry_data: EntryData):
        pass

    async def _async_subentry_update(self, subentry_id: str, entry_data: EntryData):
        pass

    async def _async_subentry_remove(self, subentry_id: str, entry_data: EntryData):
        pass


class EnergyEstimatorSensorConfig(pmc.EntityConfig, pmc.SubentryConfig):
    """Configure additional estimation sensors reporting forecasted energy over an amount of time."""

    forecast_duration_hours: int


class EnergyEstimatorSensor(EstimatorEntity, Sensor):

    controller: "EnergyEstimatorController"

    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_unit_of_measurement = hac.UnitOfEnergy.WATT_HOUR

    __slots__ = ("forecast_duration_ts",)

    def __init__(
        self,
        controller: "EnergyEstimatorController",
        id,
        *,
        forecast_duration_ts: float = 0,
        **kwargs: "Unpack[EntityArgs]",
    ):
        self.forecast_duration_ts = forecast_duration_ts
        super().__init__(
            controller,
            id,
            controller.estimator,
            state_class=None,
            **kwargs,
        )

    @typing.override
    def on_estimator_update(self, estimator: EnergyEstimator):
        self.native_value = round(
            estimator.get_estimated_energy(
                estimator.observed_time_ts,
                estimator.observed_time_ts + self.forecast_duration_ts,
            )
        )
        if self.added_to_hass:
            self._async_write_ha_state()


class TodayEnergyEstimatorSensor(EnergyEstimatorSensor):

    @typing.override
    def on_estimator_update(self, estimator: EnergyEstimator):
        self.extra_state_attributes = estimator.get_state_dict()
        self.native_value = round(
            estimator.today_energy
            + estimator.get_estimated_energy(
                estimator.observed_time_ts, estimator.tomorrow_ts
            )
        )
        if self.added_to_hass:
            self._async_write_ha_state()


class TomorrowEnergyEstimatorSensor(EnergyEstimatorSensor):

    @typing.override
    def on_estimator_update(self, estimator: EnergyEstimator):
        self.native_value = round(
            estimator.get_estimated_energy(
                estimator.tomorrow_ts, estimator.tomorrow_ts + 86400
            )
        )
        if self.added_to_hass:
            self._async_write_ha_state()


class EnergyEstimatorController[_ConfigT: "EnergyEstimatorController.Config"](  # type: ignore
    Controller[_ConfigT]
):

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimator.Config, Controller.Config):
            pass

    PLATFORMS = {Sensor.PLATFORM}

    estimator: EnergyEstimator

    __slots__ = (
        # configuration
        # state
        "estimator",
    )

    @staticmethod
    def get_config_entry_schema(
        config: "Config | None",
    ) -> pmc.ConfigSchema:
        if not config:
            config = {
                "source_entity_id": "",
                "sampling_interval_minutes": 10,
                "observation_duration_minutes": 20,
                "history_duration_days": 7,
                "maximum_latency_seconds": 60,
            }
        return {
            hv.req_config("source_entity_id", config): hv.sensor_selector(
                device_class=[Sensor.DeviceClass.POWER, Sensor.DeviceClass.ENERGY]
            ),
            hv.req_config(
                "sampling_interval_minutes",
                config,
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.req_config(
                "observation_duration_minutes", config
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.req_config("history_duration_days", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.DAYS, max=30
            ),
            hv.opt_config("update_period_seconds", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
            hv.opt_config("maximum_latency_seconds", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
            hv.opt_config("safe_maximum_power_w", config): hv.positive_number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
        }

    @staticmethod
    def get_config_subentry_schema(
        subentry_type: str, config: pmc.ConfigMapping | None
    ) -> pmc.ConfigSchema:
        match subentry_type:
            case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                if not config:
                    config = {
                        "name": "Energy estimation",
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

    def __init__(
        self, config_entry: "ConfigEntry", estimator_class: type[EnergyEstimator]
    ):

        self.estimator = estimator_class(
            self.TYPE,
            config=config_entry.data,  # type: ignore
        )

        super().__init__(config_entry)

        for warning in self.estimator.warnings:
            ProcessorWarningBinarySensor(self, f"{warning.id}_warning", warning)

        TodayEnergyEstimatorSensor(
            self,
            "today_energy_estimate",
            name=f"{self.config.get("name", "Estimated energy")} (today)",
        )
        TomorrowEnergyEstimatorSensor(
            self,
            "tomorrow_energy_estimate",
            name=f"{self.config.get("name", "Estimated energy")} (tomorrow)",
        )

    # interface: Controller
    async def async_setup(self):
        await self.estimator.async_start()
        await super().async_setup()

    async def async_shutdown(self):
        self.estimator.shutdown()
        self.estimator = None  # type: ignore
        await super().async_shutdown()

    def _subentry_add(
        self, subentry_id: str, entry_data: EntryData[EnergyEstimatorSensorConfig]
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
                    forecast_duration_ts=entry_data.config.get(
                        "forecast_duration_hours", 1
                    )
                    * 3600,
                )

    async def _async_subentry_update(self, subentry_id: str, entry_data: EntryData):
        match entry_data.subentry_type:
            case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                entity: EnergyEstimatorSensor
                for entity in entry_data.entities.values():  # type: ignore
                    entity.name = entry_data.config.get(
                        "name", pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR
                    )
                    entity.forecast_duration_ts = (
                        entry_data.config.get("forecast_duration_hours", 1) * 3600
                    )
                    entity.on_estimator_update(self.estimator)

    # interface: self
