"""
Controller for pv energy production estimation
"""

import datetime as dt
import time
import typing

import astral
import astral.sun
from homeassistant import const as hac
from homeassistant.components import weather
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
import voluptuous as vol

from ... import const as pmc, controller, helpers
from ...helpers import validation as hv
from ...sensor import Sensor
from .estimator import EnergyObserver, Observation, PowerObserver, WeatherHistory
from .estimator_heuristic import HeuristicEstimator

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State

    from ...helpers.entity import EntityArgs


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


class EntryConfig(ControllerConfig, pmc.EntityConfig, pmc.EntryConfig):
    """TypedDict for ConfigEntry data"""


class PVEnergyEstimatorSensorConfig(pmc.EntityConfig, pmc.SubentryConfig):
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
        **kwargs: "typing.Unpack[EntityArgs]",
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
        "today_energy_estimate_sensor",
        "tomorrow_energy_estimate_sensor",
        "estimator_sensors",
        "_sun_offset",
        "_state_convert_func",
        "_state_convert_unit",
        "_pv_entity_tracking_unsub",
        "_weather_tracking_unsub",
        "_estimate_callback_unsub",
        "_restore_history_task",
        "_restore_history_exit",
    )

    # interface: Controller
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

    @staticmethod
    def get_config_subentry_schema(subentry_type: str, user_input):
        match subentry_type:
            case pmc.ConfigSubentryType.PV_ENERGY_ESTIMATOR_SENSOR:
                return hv.entity_schema(
                    user_input,
                    name="PV energy estimation",
                ) | {
                    hv.required(
                        "forecast_duration_hours", user_input, 1
                    ): hv.time_period_selector(
                        min=1, unit_of_measurement=hac.UnitOfTime.HOURS
                    ),
                }

        return {}

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

        self.today_energy_estimate_sensor = EnergyEstimatorSensor(
            self,
            "today_energy_estimate",
            name="Today energy estimate"
        )
        self.tomorrow_energy_estimate_sensor = EnergyEstimatorSensor(
            self,
            "tomorrow_energy_estimate",
            name="Tomorrow energy estimate"
        )

        # remove legacy debug entities
        ent_reg = self.get_entity_registry()
        for _t in range(24):
            unique_id = "_".join(
                (
                    self.config_entry.entry_id,
                    f"{pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR}_{_t}",
                )
            )
            entity_id = ent_reg.async_get_entity_id(
                Sensor.PLATFORM, pmc.DOMAIN, unique_id
            )
            if entity_id:
                ent_reg.async_remove(entity_id)

        self.estimator_sensors: dict[str, EnergyEstimatorSensor] = {}
        for subentry_id, config_subentry in config_entry.subentries.items():
            match config_subentry.subentry_type:
                case pmc.ConfigSubentryType.PV_ENERGY_ESTIMATOR_SENSOR:
                    self.estimator_sensors[subentry_id] = EnergyEstimatorSensor(
                        self,
                        f"{pmc.ConfigSubentryType.PV_ENERGY_ESTIMATOR_SENSOR}_{subentry_id}",
                        name=config_subentry.data.get("name"),
                        config_subentry_id=subentry_id,
                        forecast_duration_ts=config_subentry.data.get(
                            "forecast_duration_hours", 0
                        )
                        * 3600,
                    )

    async def async_init(self):
        self._restore_history_exit = False
        self._restore_history_task = recorder_instance(
            self.hass
        ).async_add_executor_job(self._restore_history)
        await self._restore_history_task
        self._restore_history_task = None

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

        self._update_estimate()

        self._estimate_callback_unsub = self.schedule_async_callback(
            self.refresh_period_ts, self._async_refresh_callback
        )

        self._process_observation(self.hass.states.get(self.pv_source_entity_id))

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
            self.today_energy_estimate_sensor: EnergyEstimatorSensor = None  # type: ignore
            self.tomorrow_energy_estimate_sensor: EnergyEstimatorSensor = None  # type: ignore
            self.estimator_sensors.clear()
            self.estimator: HeuristicEstimator = None  # type: ignore
            return True
        return False

    async def _entry_update_listener(
        self, hass: "HomeAssistant", config_entry: "ConfigEntry"
    ):
        # check if config subentries changed
        estimator_sensors = dict(self.estimator_sensors)
        for subentry_id, config_subentry in config_entry.subentries.items():
            match config_subentry.subentry_type:
                case pmc.ConfigSubentryType.PV_ENERGY_ESTIMATOR_SENSOR:
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
                        # TODO: this isnt working as of now since we need to forward platform entry setup
                        self.estimator_sensors[subentry_id] = EnergyEstimatorSensor(
                            self,
                            f"{pmc.ConfigSubentryType.PV_ENERGY_ESTIMATOR_SENSOR}_{subentry_id}",
                            name=subentry_config.get("name"),
                            config_subentry_id=subentry_id,
                            forecast_duration_ts=subentry_config.get(
                                "forecast_duration_hours", 0
                            )
                            * 3600,
                        )

        for subentry_id in estimator_sensors.keys():
            # these were removed subentries
            estimator_sensor = self.estimator_sensors.pop(subentry_id)
            self.entities[Sensor.PLATFORM].pop(estimator_sensor.id)

        await super()._entry_update_listener(hass, config_entry)

    # interface: self
    @callback
    def _pv_entity_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        self._process_observation(event.data.get("new_state"))

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
        self._process_observation(self.hass.states.get(self.pv_source_entity_id))

    def _process_observation(self, tracked_state: "State | None"):
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
                    self._update_estimate()

                return

            except Exception as e:
                self.log_exception(self.DEBUG, e, "updating estimate")

    def _update_estimate(self):
        self.estimator.update_estimate()
        self.today_energy_estimate_sensor.update(self.estimator.today_forecast_energy)
        self.tomorrow_energy_estimate_sensor.update(self.estimator.tomorrow_forecast_energy)

        now = time.time()
        for sensor in self.estimator_sensors.values():
            sensor.update(
                self.estimator.get_estimated_energy(
                    now, now + sensor.forecast_duration_ts
                )
            )

    async def _async_update_weather(self, weather_state: "State | None"):
        self.weather_state = weather_state
        if weather_state:
            self.estimator.add_weather(Controller._weather_from_state(weather_state))

            forecasts: list[WeatherHistory] = []
            try:
                response = await self.hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    service_data={
                        "type": "hourly",
                        "entity_id": self.weather_entity_id,
                    },
                    blocking=True,
                    return_response=True,
                )
                forecasts = [
                    self._weather_from_forecast(f)
                    for f in response[self.weather_entity_id]["forecast"]  # type:ignore
                ]

            except Exception as e:
                self.log_exception(self.DEBUG, e, "requesting hourly weather forecasts")

            try:
                response = await self.hass.services.async_call(
                    "weather",
                    "get_forecasts",
                    service_data={
                        "type": "daily",
                        "entity_id": self.weather_entity_id,
                    },
                    blocking=True,
                    return_response=True,
                )
                daily_weather_forecasts = [
                    self._weather_from_forecast(f)
                    for f in response[self.weather_entity_id]["forecast"]  # type:ignore
                ]
                if daily_weather_forecasts:
                    if forecasts:
                        # We're adding daily forecasts at the end of our (eventual) hourly forecasts
                        # When doing so, we take special care as to not overlap the end of the hourly
                        # list with the beginning of the daily list
                        last_hourly_forecast = forecasts[-1]
                        last_hourly_forecast_end_ts = (
                            last_hourly_forecast.time_ts + 3600
                        )
                        index = 0
                        for daily_forecast in daily_weather_forecasts:

                            if daily_forecast.time_ts < last_hourly_forecast_end_ts:
                                index += 1
                                continue

                            if (
                                daily_forecast.time_ts > last_hourly_forecast_end_ts
                            ) and index:
                                # this is not the first daily so we add an 'interpolation' between
                                # the end of the hourly list with the beginning of the daily one
                                daily_forecast_prev = daily_weather_forecasts[index - 1]
                                daily_forecast_prev.time_ts = (
                                    last_hourly_forecast_end_ts
                                )
                                daily_forecast_prev.time = helpers.datetime_from_epoch(
                                    last_hourly_forecast_end_ts
                                )
                                forecasts.append(daily_forecast_prev)

                            forecasts += daily_weather_forecasts[index:]
                            break

                    else:
                        forecasts = daily_weather_forecasts

            except Exception as e:
                self.log_exception(self.DEBUG, e, "requesting daily weather forecasts")

            self.estimator.set_weather_forecasts(forecasts)

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
                self.hass,
                f"model_{self.pv_source_entity_id}_{self.config['name'].lower().replace(" ", "_")}.json",
            )
            dump_model = {
                "samples": list(self.estimator.history_samples),
                "weather": list(self.estimator.weather_history),
                "model": self.estimator.model,
            }
            json.save_json(filepath, dump_model)

    _WEATHER_CONDITION_TO_CLOUD: typing.ClassVar[dict[str | None, float | None]] = {
        None: None,
        weather.ATTR_CONDITION_CLEAR_NIGHT: 0,
        weather.ATTR_CONDITION_CLOUDY: 100,
        weather.ATTR_CONDITION_EXCEPTIONAL: 80,
        weather.ATTR_CONDITION_FOG: 80,
        weather.ATTR_CONDITION_HAIL: 80,
        weather.ATTR_CONDITION_LIGHTNING: 70,
        weather.ATTR_CONDITION_LIGHTNING_RAINY: 70,
        weather.ATTR_CONDITION_PARTLYCLOUDY: 50,
        weather.ATTR_CONDITION_POURING: 80,
        weather.ATTR_CONDITION_RAINY: 60,
        weather.ATTR_CONDITION_SNOWY: 100,
        weather.ATTR_CONDITION_SNOWY_RAINY: 100,
        weather.ATTR_CONDITION_SUNNY: 0,
        weather.ATTR_CONDITION_WINDY: 0,
        weather.ATTR_CONDITION_WINDY_VARIANT: 0,
    }

    @staticmethod
    def _weather_from_state(weather_state: "State"):
        attributes = weather_state.attributes

        if "cloud_coverage" in attributes:
            cloud_coverage = attributes["cloud_coverage"]
        else:
            cloud_coverage = Controller._WEATHER_CONDITION_TO_CLOUD.get(
                weather_state.state
            )

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
            cloud_coverage=cloud_coverage,
            visibility=visibility,
        )

    def _weather_from_forecast(self, forecast: dict):
        assert self.weather_state

        time = dt_util.as_utc(dt_util.parse_datetime(forecast["datetime"]))  # type: ignore
        time_ts = time.timestamp()

        weather_attributes = self.weather_state.attributes

        if "temperature" in forecast:
            temperature = TemperatureConverter.convert(
                float(forecast["temperature"]),
                weather_attributes["temperature_unit"],
                hac.UnitOfTemperature.CELSIUS,
            )
        else:
            temperature = None

        if "cloud_coverage" in forecast:
            cloud_coverage = forecast["cloud_coverage"]
        else:
            cloud_coverage = self._WEATHER_CONDITION_TO_CLOUD.get(
                forecast.get("condition")
            )

        return WeatherHistory(
            time=time,
            time_ts=time_ts,
            temperature=temperature,
            cloud_coverage=cloud_coverage,
            visibility=None,
        )
