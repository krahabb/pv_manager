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
    from typing import Any, Callable, ClassVar, Coroutine, Final, Unpack

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.device_registry import DeviceInfo
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from ..helpers.entity import Entity, EntityArgs


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
    platforms: "Final[dict[str, AddConfigEntryEntitiesCallback]]"
    """Dict of add_entities callbacks."""
    entities: "Final[dict[str, dict[str, Entity]]]"
    """Dict of registered entities for this controller/entry. This will be scanned in order to forward entry setup during
    initialization."""
    _callbacks_unsub: "Final[set[Callable[[], None]]]"
    """Dict of callbacks to be unsubscribed when the entry is unloaded."""

    __slots__ = (
        "config_entry",
        "hass",
        "device_info",
        "config",
        "options",
        "platforms",
        "entities",
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
    def get_config_entry_schema(user_input) -> dict:
        # to be overriden
        return {}

    @staticmethod
    def get_config_subentry_schema(subentry_type: str, user_input) -> dict:
        # to be overriden
        return {}

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        self.config_entry = config_entry
        self.hass = hass
        self.device_info = {"identifiers": {(pmc.DOMAIN, config_entry.entry_id)}}
        self.config = config_entry.data  # type: ignore
        self.options = config_entry.options  # type: ignore
        self.platforms = {}
        self.entities = {platform: {} for platform in self.PLATFORMS}
        self._callbacks_unsub = set()
        helpers.Loggable.__init__(self, config_entry.title)
        Manager.device_registry.async_get_or_create(
            config_entry_id=config_entry.entry_id,
            name=config_entry.title,
            model=self.TYPE,
            **self.device_info,  # type: ignore
        )
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
            self.config_entry, self.entities.keys()
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
        entities_per_config_subentry: dict[str | None, list] = {}
        for entity in self.entities[platform].values():
            if entity.config_subentry_id in entities_per_config_subentry:
                entities_per_config_subentry[entity.config_subentry_id].append(entity)
            else:
                entities_per_config_subentry[entity.config_subentry_id] = [entity]
        for config_subentry_id, entities in entities_per_config_subentry.items():
            add_entities(entities, config_subentry_id=config_subentry_id)

    async def async_shutdown(self):
        if not await self.hass.config_entries.async_unload_platforms(
            self.config_entry, self.platforms.keys()
        ):
            raise Exception("Failed to unload platforms")

        for unsub in self._callbacks_unsub:
            unsub()
        # removing circular refs here
        for entities_per_platform in self.entities.values():
            for entity in list(entities_per_platform.values()):
                await entity.async_shutdown()
            assert not entities_per_platform
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
        self.config = config_entry.data  # type: ignore
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
        """Cleanup diagnostic entities, when the entry is unloaded. If 'remove' is True
        it will be removed from the entity registry as well."""
        ent_reg = Manager.entity_registry
        for entities_per_platform in self.entities.values():
            for entity in list(entities_per_platform.values()):
                if entity.is_diagnostic:
                    if entity.added_to_hass:
                        await entity.async_remove(force_remove=True)
                    await entity.async_shutdown()
                    ent_reg.async_remove(entity.entity_id)


class EnergyEstimatorSensorConfig(pmc.EntityConfig, pmc.SubentryConfig):
    """Configure additional estimation sensors reporting forecasted energy over an amount of time."""

    forecast_duration_hours: int


class EnergyEstimatorSensor(Sensor):

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
            device_class=Sensor.DeviceClass.ENERGY,
            state_class=None,
            native_unit_of_measurement=hac.UnitOfEnergy.WATT_HOUR,
            **kwargs,
        )


class EnergyEstimatorControllerConfig(pmc.EntryConfig, estimator.EstimatorConfig):

    observed_entity_id: str
    """Entity ID of the energy/power observed entity"""
    refresh_period_minutes: int
    """Time between model updates (polling of input pv sensor) beside listening to state changes"""


class EnergyEstimatorController[_ConfigT: EnergyEstimatorControllerConfig](
    Controller[_ConfigT]
):

    PLATFORMS = {Sensor.PLATFORM}

    __slots__ = (
        # configuration
        "observed_entity_id",
        "refresh_period_ts",
        # state
        "estimator",
        "today_energy_estimate_sensor",
        "tomorrow_energy_estimate_sensor",
        "estimator_sensors",
        "_state_convert_func",
        "_state_convert_unit",
        "_refresh_callback_unsub",
        "_restore_history_task",
        "_restore_history_exit",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict):

        return {
            hv.required("observed_entity_id", user_input): hv.sensor_selector(
                device_class=[Sensor.DeviceClass.POWER, Sensor.DeviceClass.ENERGY]
            ),
            hv.required(
                "sampling_interval_minutes", user_input, 10
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.required(
                "maximum_latency_minutes", user_input, 1
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.required(
                "observation_duration_minutes", user_input, 20
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
            hv.required(
                "history_duration_days", user_input, 14
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.DAYS, max=30),
            hv.required(
                "refresh_period_minutes", user_input, 5
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
        }

    @staticmethod
    def get_config_subentry_schema(subentry_type: str, user_input):
        match subentry_type:
            case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                return hv.entity_schema(
                    user_input,
                    name="Energy estimation",
                ) | {
                    hv.required(
                        "forecast_duration_hours", user_input, 1
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
        super().__init__(hass, config_entry)

        self.observed_entity_id = self.config["observed_entity_id"]
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

        self.refresh_period_ts = self.config.get("refresh_period_minutes", 5) * 60
        self.estimator = estimator_class(
            tzinfo=dt_util.get_default_time_zone(),
            **(estimator_kwargs | self.config),  # type: ignore
        )
        self.today_energy_estimate_sensor = EnergyEstimatorSensor(
            self,
            "today_energy_estimate",
            name=f"{self.config.get("name", "Estimated energy")} (today)",
        )
        self.tomorrow_energy_estimate_sensor = EnergyEstimatorSensor(
            self,
            "tomorrow_energy_estimate",
            name=f"{self.config.get("name", "Estimated energy")} (tomorrow)",
        )

        self.estimator_sensors: dict[str, EnergyEstimatorSensor] = {}
        for subentry_id, config_subentry in config_entry.subentries.items():
            match config_subentry.subentry_type:
                case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                    self.estimator_sensors[subentry_id] = EnergyEstimatorSensor(
                        self,
                        f"{pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR}_{subentry_id}",
                        name=config_subentry.data.get("name"),
                        config_subentry_id=subentry_id,
                        forecast_duration_ts=config_subentry.data.get(
                            "forecast_duration_hours", 0
                        )
                        * 3600,
                        parent_attr=None,
                    )

        self._refresh_callback_unsub = None
        self._restore_history_task = None
        self._restore_history_exit = False

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
        self.estimator_sensors.clear()
        await super().async_shutdown()

    async def _entry_update_listener(
        self, hass: "HomeAssistant", config_entry: "ConfigEntry"
    ):
        # check if config subentries changed
        estimator_sensors = dict(self.estimator_sensors)
        for subentry_id, config_subentry in config_entry.subentries.items():
            match config_subentry.subentry_type:
                case pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR:
                    subentry_config = config_subentry.data
                    if subentry_id in estimator_sensors:
                        # entry already present: just update it
                        estimator_sensor = estimator_sensors.pop(subentry_id)
                        estimator_sensor.name = config_subentry.data.get(
                            "name", estimator_sensor.id
                        )
                        estimator_sensor.forecast_duration_ts = (
                            subentry_config.get("forecast_duration_hours", 1) * 3600
                        )
                    else:
                        self.estimator_sensors[subentry_id] = EnergyEstimatorSensor(
                            self,
                            f"{pmc.ConfigSubentryType.ENERGY_ESTIMATOR_SENSOR}_{subentry_id}",
                            name=subentry_config.get("name"),
                            config_subentry_id=subentry_id,
                            forecast_duration_ts=subentry_config.get(
                                "forecast_duration_hours", 0
                            )
                            * 3600,
                            parent_attr=None,
                        )

        for subentry_id in estimator_sensors.keys():
            # these were removed subentries
            estimator_sensor = self.estimator_sensors.pop(subentry_id)
            await estimator_sensor.async_shutdown()

        await super()._entry_update_listener(hass, config_entry)

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

        self.today_energy_estimate_sensor.extra_state_attributes = (
            estimator.get_state_dict()
        )
        self.today_energy_estimate_sensor.update_safe(
            estimator.today_energy
            + estimator.get_estimated_energy(
                estimator.observed_time_ts, estimator.tomorrow_ts
            )
        )
        self.tomorrow_energy_estimate_sensor.update_safe(
            estimator.get_estimated_energy(
                estimator.tomorrow_ts, estimator.tomorrow_ts + 86400
            )
        )

        for sensor in self.estimator_sensors.values():
            sensor.update_safe(
                self.estimator.get_estimated_energy(
                    estimator.observed_time_ts,
                    estimator.observed_time_ts + sensor.forecast_duration_ts,
                )
            )

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
