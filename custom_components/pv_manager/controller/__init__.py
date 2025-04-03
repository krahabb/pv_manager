import datetime as dt
import time
import typing

from homeassistant import const as hac
from homeassistant.components.recorder import get_instance as recorder_instance, history
from homeassistant.core import callback
from homeassistant.helpers import (
    entity_registry,
    event,
    json,
)
from homeassistant.util.unit_conversion import (
    EnergyConverter,
    PowerConverter,
)

from .. import const as pmc, helpers
from ..helpers import validation as hv
from ..sensor import Sensor
from .common import estimator

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Coroutine

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
    import voluptuous as vol

    from ..helpers.entity import Entity


class Controller[_ConfigT: pmc.EntryConfig](helpers.Loggable):
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


class EnergyEstimatorControllerConfig(pmc.EntryConfig, estimator.EstimatorConfig):

    observed_entity_id: str
    """Entity ID of the energy/power observed entity"""
    refresh_period_minutes: int
    """Time between model updates (polling of input pv sensor) beside listening to state changes"""


class EnergyEstimatorController[_ConfigT: EnergyEstimatorControllerConfig](
    Controller[_ConfigT]
):

    __slots__ = (
        # configuration
        "observed_entity_id",
        "refresh_period_ts",
        # state
        "estimator",
        "_state_convert_func",
        "_state_convert_unit",
        "_observed_entity_tracking_unsub",
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

    def __init__(
        self,
        hass: "HomeAssistant",
        config_entry: "ConfigEntry",
        estimator_class: "type[estimator.Estimator]",
        **estimator_kwargs,
    ):
        super().__init__(hass, config_entry)

        self.observed_entity_id = self.config["observed_entity_id"]
        ent_reg = self.get_entity_registry()
        reg_entry = ent_reg.async_get(self.observed_entity_id)
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
        self.estimator = estimator_class(**(estimator_kwargs | self.config))  # type: ignore
        self._refresh_callback_unsub = None
        self._observed_entity_tracking_unsub = None

    async def async_init(self):
        self._restore_history_exit = False
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

        self._observed_entity_tracking_unsub = event.async_track_state_change_event(
            self.hass,
            self.observed_entity_id,
            self._observed_entity_tracking_callback,
        )
        self._refresh_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )
        return await super().async_init()

    async def async_shutdown(self):
        if await super().async_shutdown():
            if self._restore_history_task:
                if not self._restore_history_task.done():
                    self._restore_history_exit = True
                    await self._restore_history_task
                self._restore_history_task = None
            if self._refresh_callback_unsub:
                self._refresh_callback_unsub.cancel()
                self._refresh_callback_unsub = None
            if self._observed_entity_tracking_unsub:
                self._observed_entity_tracking_unsub()
                self._observed_entity_tracking_unsub = None
            self.estimator: "estimator.Estimator" = None  # type: ignore
            return True
        return False

    @callback
    def _observed_entity_tracking_callback(
        self, event: "Event[event.EventStateChangedData]"
    ):
        self._process_observation(event.data.get("new_state"))

    async def _async_refresh_callback(self):
        self._refresh_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )
        self._process_observation(self.hass.states.get(self.observed_entity_id))

    def _process_observation(self, tracked_state: "State | None"):
        if tracked_state:
            try:
                if self.estimator.add_observation(
                    estimator.Observation(
                        time.time(),
                        self._state_convert_func(
                            float(tracked_state.state),
                            tracked_state.attributes["unit_of_measurement"],
                            self._state_convert_unit,
                        ),
                    )
                ):
                    self._update_estimate()

                return

            except Exception as e:
                self.log_exception(self.DEBUG, e, "updating estimate")

    def _update_estimate(self):
        pass

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
