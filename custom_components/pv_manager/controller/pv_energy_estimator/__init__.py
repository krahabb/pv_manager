"""
Controller for pv energy production estimation
"""

import datetime as dt
import time
import typing
import voluptuous as vol

import astral
import astral.sun
from homeassistant import const as hac
from homeassistant.components.recorder import get_instance as recorder_instance, history
from homeassistant.core import callback
from homeassistant.helpers import (
    event,
    json,
    sun as sun_helpers,
)
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    DistanceConverter,
    EnergyConverter,
    PowerConverter,
    TemperatureConverter,
)

from ... import const as pmc, controller, helpers
from ...helpers import validation as hv
from ...sensor import Sensor
from .estimator import EnergyObserver, Observation, PowerObserver, WeatherHistory
from .estimator_heuristic import HeuristicEstimator

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State


class ControllerConfig(typing.TypedDict):
    pv_power_entity_id: typing.NotRequired[str]
    """The source entity_id of the pv power of the system"""
    pv_energy_entity_id: typing.NotRequired[str]
    """The source entity_id of the pv energy of the system"""
    weather_entity_id: typing.NotRequired[str]
    """The entity used for weather forecast in the system"""

    # model parameters
    sampling_interval_minutes: int
    """Time resolution of model data"""
    observation_duration_minutes: int  # minutes
    """The time window for calculating current energy production from incoming energy observation."""
    history_duration_days: int
    """Number of (backward) days of data to keep in the model (used to build the estimates for the time forward)."""
    refresh_period_minutes: int
    """Time between model updates (polling of input pv sensor) beside listening to state changes"""
    maximum_latency_minutes: int
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.BaseConfig):
    """TypedDict for ConfigEntry data"""


class Controller(controller.Controller[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    __slots__ = (
        # configuration
        "pv_source_entity_id",
        "weather_entity_id",
        "refresh_period_ts",
        # state
        "estimator",
        "weather_state",
        "pv_energy_estimator_sensor",
        "_sun_offset",
        "_state_convert_func",
        "_state_convert_unit",
        "_pv_entity_tracking_unsub",
        "_weather_tracking_unsub",
        "_estimate_callback_unsub",
        "_restore_history_task",
        "_restore_history_exit",
    )

    @staticmethod
    def get_config_entry_schema(user_input: dict):

        return hv.entity_schema(
            user_input,
            name="PV energy estimation",
        ) | {
            # we should allow configuring either pv_power_entity_id or pv_energy_entity_id
            # but data entry flow looks like not allowing this XOR-like schema
            hv.optional("pv_power_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.optional("pv_energy_entity_id", user_input): hv.sensor_selector(
                device_class=Sensor.DeviceClass.ENERGY
            ),
            hv.optional("weather_entity_id", user_input): hv.weather_selector(),
            hv.required(
                "sampling_interval_minutes", user_input, 10
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
            hv.required(
                "maximum_latency_minutes", user_input, 1
            ): hv.time_period_selector(unit_of_measurement=hac.UnitOfTime.MINUTES),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)

        self.refresh_period_ts = self.config.get("refresh_period_minutes", 5) * 60
        # TODO: get sun_offset from local tz
        self._sun_offset = dt.timedelta(hours=1)

        if "pv_power_entity_id" in self.config:
            estimator_class = type(
                "PowerObserverHourlyEstimator", (PowerObserver, HeuristicEstimator), {}
            )
            self._state_convert_func = PowerConverter.convert
            self._state_convert_unit = hac.UnitOfPower.WATT
            self.pv_source_entity_id = self.config["pv_power_entity_id"]
        elif "pv_energy_entity_id" in self.config:
            estimator_class = type(
                "EnergyObserverHourlyEstimator",
                (EnergyObserver, HeuristicEstimator),
                {},
            )
            self._state_convert_func = EnergyConverter.convert
            self._state_convert_unit = hac.UnitOfEnergy.WATT_HOUR
            self.pv_source_entity_id = self.config["pv_energy_entity_id"]
        else:
            raise Exception(
                "missing either 'pv_power_entity_id' or 'pv_energy_entity_id' in config"
            )

        self.weather_entity_id = self.config.get("weather_entity_id")
        self.weather_state = None

        location, elevation = sun_helpers.get_astral_location(hass)
        self.estimator = estimator_class(
            sampling_interval_ts=self.config.get("sampling_interval_minutes", 10) * 60,
            observation_duration_ts=self.config.get("observation_duration_minutes", 20)
            * 60,
            history_duration_ts=self.config.get("history_duration_days", 7) * 86400,
            maximum_latency_ts=self.config.get("maximum_latency_minutes", 1) * 60,
            astral_observer=astral.sun.Observer(
                location.latitude, location.longitude, elevation
            ),
        )

        self.pv_energy_estimator_sensor = Sensor(
            self,
            pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR,
            device_class=Sensor.DeviceClass.ENERGY,
            state_class=None,
            name=self.config["name"],
            native_unit_of_measurement=hac.UnitOfEnergy.WATT_HOUR,
        )

        for _t in range(len(self.estimator.estimations)):
            Sensor(
                self,
                f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}",
                device_class=Sensor.DeviceClass.ENERGY,
                state_class=None,
                name=f"{self.config["name"]} ({_t+1:02})",
                native_unit_of_measurement=hac.UnitOfEnergy.WATT_HOUR,
            )

    async def async_init(self):
        self._restore_history_exit = False
        self._restore_history_task = recorder_instance(
            self.hass
        ).async_add_executor_job(self._restore_history)
        await self._restore_history_task
        self._restore_history_task = None

        self.estimator.update_estimate()
        accumulated_energy = 0
        for _t in range(len(self.estimator.estimations)):
            accumulated_energy += self.estimator.estimations[_t].energy
            sensor: Sensor = self.entities[Sensor.PLATFORM][
                f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}"
            ]  # type:ignore
            sensor.update(accumulated_energy)

        self.pv_energy_estimator_sensor.update(self.estimator.forecast_today_energy)

        self._pv_entity_tracking_unsub = event.async_track_state_change_event(
            self.hass,
            self.pv_source_entity_id,
            self._pv_entity_tracking_callback,
        )

        if self.weather_entity_id:
            self._weather_tracking_unsub = event.async_track_state_change_event(
                self.hass,
                self.weather_entity_id,
                self._weather_tracking_callback,
            )
            await self._async_update_weather(
                self.hass.states.get(self.weather_entity_id)
            )
        else:
            self._weather_tracking_unsub = None

        self._estimate_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )

        self._update_estimate(self.hass.states.get(self.pv_source_entity_id))

        await super().async_init()

    async def async_shutdown(self):
        if await super().async_shutdown():
            if self._pv_entity_tracking_unsub:
                self._pv_entity_tracking_unsub()
                self._pv_entity_tracking_unsub = None
            if self._weather_tracking_unsub:
                self._weather_tracking_unsub()
                self._weather_tracking_unsub = None
            if self._estimate_callback_unsub:
                self._estimate_callback_unsub.cancel()
                self._estimate_callback_unsub = None
            if self._restore_history_task:
                if not self._restore_history_task.done():
                    self._restore_history_exit = True
                    await self._restore_history_task
                self._restore_history_task = None
            self.pv_energy_estimator_sensor: Sensor = None  # type: ignore
            return True
        return False

    @callback
    def _pv_entity_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        self._update_estimate(event.data.get("new_state"))

    @callback
    def _weather_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        self.async_create_task(
            self._async_update_weather(event.data.get("new_state")),
            "_async_update_weather",
        )

    async def _async_refresh_callback(self):
        self._estimate_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )
        self._update_estimate(self.hass.states.get(self.pv_source_entity_id))

    def _update_estimate(self, tracked_state: "State | None"):
        if tracked_state:
            try:
                if self.estimator.add_observation(
                    Observation(
                        time.time(),
                        self._state_convert_func(
                            float(tracked_state.state),
                            tracked_state.attributes["unit_of_measurement"],
                            self._state_convert_unit,
                        ),
                    )
                ):
                    self.estimator.update_estimate()
                    accumulated_energy = 0
                    for _t in range(len(self.estimator.estimations)):
                        accumulated_energy += self.estimator.estimations[_t].energy
                        sensor: Sensor = self.entities[Sensor.PLATFORM][
                            f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}"
                        ]  # type:ignore
                        sensor.update(accumulated_energy)

                    self.pv_energy_estimator_sensor.update(
                        self.estimator.forecast_today_energy
                    )

                return

            except Exception as e:
                self.log_exception(self.DEBUG, e, "updating estimate")

        self.pv_energy_estimator_sensor.update(None)
        for _t in range(len(self.estimator.estimations)):
            sensor: Sensor = self.entities[Sensor.PLATFORM][
                f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}"
            ]  # type:ignore
            sensor.update(None)

    async def _async_update_weather(self, weather_state: "State | None"):
        self.weather_state = weather_state
        if weather_state:
            self.estimator.add_weather(Controller._weather_from_state(weather_state))

            service_response = await self.hass.services.async_call(
                "weather",
                "get_forecasts",
                service_data={
                    "type": "hourly",
                    "entity_id": self.weather_entity_id,
                },
                blocking=True,
                return_response=True,
            )
            # self.log(self.DEBUG, "updated weather forecasts %s", str(service_response))

    def _restore_history(self):
        if self._restore_history_exit:
            return
        now_ts = time.time()

        history_start_time = helpers.datetime_from_epoch(
            now_ts - self.estimator.history_duration_ts
        )

        if self.weather_entity_id:
            weather_states = history.state_changes_during_period(
                self.hass,
                history_start_time,
                None,
                self.weather_entity_id,
                no_attributes=False,
            )
            for weather_state in weather_states[self.weather_entity_id]:
                if self._restore_history_exit:
                    return
                try:
                    self.estimator.add_weather(
                        Controller._weather_from_state(weather_state)
                    )
                except:
                    pass

        pv_entity_states = history.state_changes_during_period(
            self.hass,
            history_start_time,
            None,
            self.pv_source_entity_id,
            no_attributes=False,
        )

        for state in pv_entity_states[self.pv_source_entity_id]:
            if self._restore_history_exit:
                return
            try:
                self.estimator.add_observation(
                    Observation(
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
                self.hass, f"model_{self.pv_source_entity_id}.json"
            )
            dump_model = {
                "samples": list(self.estimator.history_samples),
                "weather": list(self.estimator.weather_samples),
                "model": self.estimator.model,
            }
            json.save_json(filepath, dump_model)

    @staticmethod
    def _weather_from_state(weather_state: "State"):
        attributes = weather_state.attributes
        if "visibility" in attributes:
            visibility = DistanceConverter.convert(
                attributes["visibility"],
                attributes["visibility_unit"],
                hac.UnitOfLength.KILOMETERS,
            )
        else:
            visibility = None
        return WeatherHistory(
            time=weather_state.last_updated,
            time_ts=weather_state.last_updated_timestamp,
            temperature=TemperatureConverter.convert(
                float(attributes["temperature"]),
                attributes["temperature_unit"],
                hac.UnitOfTemperature.CELSIUS,
            ),
            cloud_coverage=attributes.get("cloud_coverage"),
            visibility=visibility,
        )
