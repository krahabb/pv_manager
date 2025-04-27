import datetime as dt
import time
import typing

from homeassistant import const as hac
from homeassistant.components.recorder import get_instance as recorder_instance, history
from homeassistant.core import callback
from homeassistant.helpers import (
    event,
    json,
)
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    EnergyConverter,
    PowerConverter,
)

from .. import const as pmc, helpers
from ..helpers import validation as hv
from ..manager import Manager
from ..sensor import Sensor
from .common import estimator

if typing.TYPE_CHECKING:
    from types import MappingProxyType
    from typing import Any, Callable, ClassVar, Coroutine, Final, Mapping, Unpack

    from homeassistant.config_entries import ConfigEntry, ConfigSubentry
    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.device_registry import DeviceInfo
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
    import voluptuous as vol

    from ..helpers.entity import DiagnosticEntity, Entity, EntityArgs


class EntryData[_ConfigT: pmc.EntryConfig | pmc.SubentryConfig]:

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


class Controller[_ConfigT: pmc.EntryConfig](helpers.Loggable):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE: "ClassVar[pmc.ConfigEntryType]"

    PLATFORMS: "ClassVar[set[str]]" = set()
    """Default entity platforms used by the controller. This is used to prepopulate the entities dict so
    that we'll (at least) forward entry setup to these platforms (even if we've still not registered any entity for those).
    This in turn allows us to later create and register entities since we'll register the add_entities callback in our platforms."""

    hass: "Final[HomeAssistant]"
    device_info: "Final[DeviceInfo]"

    config: _ConfigT
    options: pmc.EntryOptionsConfig
    platforms: "Final[dict[str, AddConfigEntryEntitiesCallback | None]]"
    """Dict of add_entities callbacks."""
    entries: "Final[dict[str | None, EntryData]]"
    """Cached copy of subentries used to manage subentry add/remove/update."""
    diagnostic_entities: "Final[dict[str, DiagnosticEntity]]"
    _callbacks_unsub: "Final[set[Callable[[], None]]]"
    """Dict of callbacks to be unsubscribed when the entry is unloaded."""

    __slots__ = (
        "config_entry",
        "hass",
        "device_info",
        "config",
        "options",
        "platforms",
        "entries",
        "diagnostic_entities",
        "_callbacks_unsub",
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

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        self.config_entry = config_entry
        self.hass = hass
        self.device_info = {"identifiers": {(pmc.DOMAIN, config_entry.entry_id)}}
        self.config = config_entry.data  # type: ignore
        self.options = config_entry.options  # type: ignore
        self.platforms = {platform: None for platform in self.PLATFORMS}
        entries = self.entries = {None: EntryData.Entry(config_entry)}
        self.diagnostic_entities = {}
        self._callbacks_unsub = set()
        helpers.Loggable.__init__(self, config_entry.title)
        Manager.device_registry.async_get_or_create(
            config_entry_id=config_entry.entry_id,
            name=config_entry.title,
            model=self.TYPE,
            **self.device_info,  # type: ignore
        )

        for subentry_id, subentry in config_entry.subentries.items():
            entries[subentry_id] = entry_data = EntryData.SubEntry(subentry)
            self._subentry_add(subentry_id, entry_data)

        if self.options.get("create_diagnostic_entities"):
            self._create_diagnostic_entities()

        config_entry.runtime_data = self

    # interface: Loggable
    def configure_logger(self):
        """
        Configure a 'logger' and a 'logtag' based off current config for every ConfigEntry.
        """
        self.logtag = f"{self.TYPE}({self.id})"
        # using helpers.getLogger (instead of logger.getChild) to 'wrap' the Logger class ..
        self.logger = helpers.getLogger(f"{helpers.LOGGER.name}.{self.logtag}")
        self.logger.setLevel(
            pmc.CONF_LOGGING_LEVEL_OPTIONS.get(
                self.options.get("logging_level", "default"), self.DEFAULT
            )
        )

    def log(self, level: int, msg: str, *args, **kwargs):
        if (logger := self.logger).isEnabledFor(level):
            logger._log(level, msg, args, **kwargs)

    # interface: self
    async def async_init(self):
        self._callbacks_unsub.add(
            self.config_entry.add_update_listener(self._entry_update_listener)
        )
        # Here we're forwarding to all the platform registerd in self.entities.
        # This is by default preset in the constructor with a list of (default) PLATFORMS
        # for the controller class.
        # The list of 'actual' entities could also be enriched by instantiating entities
        # in the (derived) contructor since async_init will be called at loading time right after
        # class instance initialization.
        await self.hass.config_entries.async_forward_entry_setups(
            self.config_entry, self.platforms
        )

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

    async def async_shutdown(self):
        if not await self.hass.config_entries.async_unload_platforms(
            self.config_entry, self.platforms.keys()
        ):
            raise Exception("Failed to unload platforms")

        for unsub in self._callbacks_unsub:
            unsub()

        # removing circular refs here
        for subentry_id, entry_data in self.entries.items():
            for entity in list(entry_data.entities.values()):
                await entity.async_shutdown(False)
            assert not entry_data.entities
        assert not self.diagnostic_entities
        self.platforms.clear()

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

    def track_state(
        self,
        entity_id: str,
        action: "Callable[[Event[event.EventStateChangedData]], Any]",
    ):
        """Track a state change for the given entity_id."""
        self._callbacks_unsub.add(
            event.async_track_state_change_event(self.hass, entity_id, action)
        )

    def track_state_update(
        self, entity_id: str, target: "Callable[[State | None], None]"
    ):
        """Start tracking state updates for the given entity_id and immediately call target with current state."""

        @callback
        def _track_state_callback(
            event: "Event[event.EventStateChangedData]",
        ):
            target(event.data.get("new_state"))

        self._callbacks_unsub.add(
            event.async_track_state_change_event(
                self.hass, entity_id, _track_state_callback
            )
        )
        target(self.hass.states.get(entity_id))

    async def async_track_state_update(
        self,
        entity_id: str,
        target: "Callable[[State | None], Coroutine[Any, Any, Any]]",
    ):
        """Start tracking state updates for the given entity_id and immediately call target with current state."""

        @callback
        def _track_state_callback(
            event: "Event[event.EventStateChangedData]",
        ):
            self.async_create_task(
                target(event.data.get("new_state")),
                f"_track_state_callback({entity_id})",
            )

        self._callbacks_unsub.add(
            event.async_track_state_change_event(
                self.hass, entity_id, _track_state_callback
            )
        )
        await target(self.hass.states.get(entity_id))

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
            for entity in list(entry_data.entities.values()):
                await entity.async_shutdown(True)
            assert not entry_data.entities
            entries.pop(subentry_id)

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


class EnergyEstimatorSensor(Sensor):

    controller: "EnergyEstimatorController"

    _attr_parent_attr = None

    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_unit_of_measurement = hac.UnitOfEnergy.WATT_HOUR

    __slots__ = ("forecast_duration_ts",)

    def __init__(
        self,
        controller,
        id,
        *,
        forecast_duration_ts: float = 0,
        **kwargs: "Unpack[EntityArgs]",
    ):
        self.forecast_duration_ts = forecast_duration_ts
        super().__init__(
            controller,
            id,
            state_class=None,
            **kwargs,
        )

    async def async_added_to_hass(self):
        self.update_estimate(self.controller.estimator, False)
        await super().async_added_to_hass()
        self.controller.estimator_sensors.add(self)

    async def async_will_remove_from_hass(self):
        self.controller.estimator_sensors.remove(self)
        await super().async_will_remove_from_hass()

    def update_estimate(self, estimator: estimator.Estimator, flush: bool):
        self.native_value = round(
            estimator.get_estimated_energy(
                estimator.observed_time_ts,
                estimator.observed_time_ts + self.forecast_duration_ts,
            )
        )
        if flush:
            self._async_write_ha_state()


class TodayEnergyEstimatorSensor(EnergyEstimatorSensor):

    def update_estimate(self, estimator: estimator.Estimator, flush: bool):
        self.extra_state_attributes = estimator.get_state_dict()
        self.native_value = round(
            estimator.today_energy
            + estimator.get_estimated_energy(
                estimator.observed_time_ts, estimator.tomorrow_ts
            )
        )
        if flush:
            self._async_write_ha_state()


class TomorrowEnergyEstimatorSensor(EnergyEstimatorSensor):

    def update_estimate(self, estimator: estimator.Estimator, flush: bool):
        self.native_value = round(
            estimator.get_estimated_energy(
                estimator.tomorrow_ts, estimator.tomorrow_ts + 86400
            )
        )
        if flush:
            self._async_write_ha_state()


class EnergyEstimatorControllerConfig(pmc.EntryConfig, estimator.EstimatorConfig):

    observed_entity_id: str
    """Entity ID of the energy/power observed entity"""
    refresh_period_minutes: int
    """Time between model updates (polling of input pv sensor) beside listening to state changes"""


class EnergyEstimatorController[_ConfigT: EnergyEstimatorControllerConfig](
    Controller[_ConfigT]
):

    PLATFORMS = {Sensor.PLATFORM}

    estimator_sensors: set[EnergyEstimatorSensor]

    __slots__ = (
        # configuration
        "observed_entity_id",
        "refresh_period_ts",
        # state
        "estimator",
        "estimator_sensors",
        "_state_convert_func",
        "_state_convert_unit",
        "_refresh_callback_unsub",
        "_restore_history_task",
        "_restore_history_exit",
    )

    @staticmethod
    def get_config_entry_schema(
        config: EnergyEstimatorControllerConfig | None,
    ) -> pmc.ConfigSchema:
        if not config:
            config = {
                "observed_entity_id": "",
                "sampling_interval_minutes": 10,
                "maximum_latency_minutes": 1,
                "observation_duration_minutes": 20,
                "history_duration_days": 7,
                "refresh_period_minutes": 5,
            }
        return {
            hv.req_config("observed_entity_id", config): hv.sensor_selector(
                device_class=[Sensor.DeviceClass.POWER, Sensor.DeviceClass.ENERGY]
            ),
            hv.req_config(
                "sampling_interval_minutes",
                config,
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.req_config("maximum_latency_minutes", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.MINUTES
            ),
            hv.req_config(
                "observation_duration_minutes", config
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.req_config("history_duration_days", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.DAYS, max=30
            ),
            hv.req_config("refresh_period_minutes", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.MINUTES
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
        self,
        hass: "HomeAssistant",
        config_entry: "ConfigEntry",
        estimator_class: "type[estimator.Estimator]",
        **estimator_kwargs,
    ):
        config = config_entry.data
        self.observed_entity_id = config["observed_entity_id"]
        reg_entry = Manager.entity_registry.async_get(self.observed_entity_id)
        if not reg_entry:
            raise ValueError(
                f"Observed entity {self.observed_entity_id} not found in entity registry"
            )
        if reg_entry.unit_of_measurement in hac.UnitOfPower:
            self._state_convert_func = PowerConverter.convert
            self._state_convert_unit = hac.UnitOfPower.WATT
            estimator_class = type(
                "PowerObserverEstimator",
                (estimator.PowerObserver, estimator_class),
                {},
            )
        elif reg_entry.unit_of_measurement in hac.UnitOfEnergy:
            self._state_convert_func = EnergyConverter.convert
            self._state_convert_unit = hac.UnitOfEnergy.WATT_HOUR
            estimator_class = type(
                "EnergyObserverEstimator",
                (estimator.EnergyObserver, estimator_class),
                {},
            )
        else:
            raise ValueError(
                f"Unsupported unit of measurement {reg_entry.unit_of_measurement} for observed entity: {self.observed_entity_id}"
            )

        self.refresh_period_ts = config.get("refresh_period_minutes", 5) * 60
        self.estimator = estimator_class(
            tzinfo=dt_util.get_default_time_zone(),
            **(estimator_kwargs | config),  # type: ignore
        )

        self.estimator_sensors = set()
        self._refresh_callback_unsub = None
        self._restore_history_task = None
        self._restore_history_exit = False

        super().__init__(hass, config_entry)

        TodayEnergyEstimatorSensor(
            self,
            "today_energy_estimate",
            name=f"{config.get("name", "Estimated energy")} (today)",
        )
        TomorrowEnergyEstimatorSensor(
            self,
            "tomorrow_energy_estimate",
            name=f"{config.get("name", "Estimated energy")} (tomorrow)",
        )

    # interface: Controller
    async def async_init(self):
        self._restore_history_task = recorder_instance(
            self.hass
        ).async_add_executor_job(
            self._restore_history,
            helpers.datetime_from_epoch(
                time.time() - self.estimator.history_duration_ts
            ),
        )
        await self._restore_history_task
        self._restore_history_task = None

        self._refresh_callback_unsub = self.schedule_callback(
            self.refresh_period_ts, self._refresh_callback
        )
        self.estimator.on_update_estimate = self._update_estimate
        self.estimator.update_estimate()
        self.track_state_update(self.observed_entity_id, self._process_observation)
        await super().async_init()

    async def async_shutdown(self):
        if self._restore_history_task:
            if not self._restore_history_task.done():
                self._restore_history_exit = True
                await self._restore_history_task
            self._restore_history_task = None
        if self._refresh_callback_unsub:
            self._refresh_callback_unsub.cancel()
            self._refresh_callback_unsub = None
        self.estimator.on_update_estimate = None
        self.estimator: "estimator.Estimator" = None  # type: ignore
        await super().async_shutdown()
        assert not self.estimator_sensors

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
                    if entity.added_to_hass:
                        entity.update_estimate(self.estimator, True)

    # interface: self
    def _refresh_callback(self):
        self._refresh_callback_unsub = self.schedule_callback(
            self.refresh_period_ts, self._refresh_callback
        )
        self._process_observation(self.hass.states.get(self.observed_entity_id))

    def _process_observation(self, state: "State | None"):
        try:
            self.estimator.add_observation(
                estimator.Observation(
                    time.time(),
                    self._state_convert_func(
                        float(state.state),  # type: ignore
                        state.attributes["unit_of_measurement"],  # type: ignore
                        self._state_convert_unit,
                    ),
                )
            )
        except Exception as e:
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(self.WARNING, e, "updating estimate")

    def _update_estimate(self, estimator: "estimator.Estimator"):

        for sensor in self.estimator_sensors:
            sensor.update_estimate(estimator, True)

    def _restore_history(self, history_start_time: dt.datetime):
        if self._restore_history_exit:
            return

        observed_entity_states = history.state_changes_during_period(
            self.hass,
            history_start_time,
            None,
            self.observed_entity_id,
            no_attributes=False,
        )

        for state in observed_entity_states[self.observed_entity_id]:
            if self._restore_history_exit:
                return
            try:
                self.estimator.add_observation(
                    estimator.Observation(
                        state.last_updated_timestamp,
                        self._state_convert_func(
                            float(state.state),
                            state.attributes["unit_of_measurement"],
                            self._state_convert_unit,
                        ),
                    )
                )
            except:
                # in case the state doesn't represent a proper value
                # just discard it
                pass

        if pmc.DEBUG:
            filepath = pmc.DEBUG.get_debug_output_filename(
                self.hass,
                f"model_{self.observed_entity_id}_{self.config.get('name', self.TYPE).lower().replace(" ", "_")}.json",
            )
            json.save_json(filepath, self.estimator.as_dict())
