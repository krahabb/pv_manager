"""
Controller for pv energy production estimation
"""

import datetime as dt
import enum
import time
import typing

import astral
import astral.sun
from homeassistant import const as hac
from homeassistant.components import weather
from homeassistant.components.recorder import history
from homeassistant.core import callback
from homeassistant.helpers import (
    event,
    sun as sun_helpers,
)
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    DistanceConverter,
    TemperatureConverter,
)

from .. import const as pmc, controller, helpers
from ..helpers import validation as hv
from ..sensor import DiagnosticSensor, Sensor
from .common.estimator_pvenergy_heuristic import (
    Estimator_PVEnergy_Heuristic,
    TimeSpanEnergyModel,
    WeatherSample,
)

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import Event, HomeAssistant, State

    from ..helpers.entity import EntityArgs


class ControllerConfig(controller.EnergyEstimatorControllerConfig):
    weather_entity_id: typing.NotRequired[str]
    """The entity used for weather forecast in the system"""


class EntryConfig(ControllerConfig, pmc.EntityConfig):
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


class DiagnosticSensorsEnum(enum.StrEnum):
    observed_ratio = enum.auto()
    weather_cloud_constant = enum.auto()


class Controller(controller.EnergyEstimatorController[EntryConfig]):
    """Base controller class for managing ConfigEntry behavior."""

    TYPE = pmc.ConfigEntryType.PV_ENERGY_ESTIMATOR

    estimator: Estimator_PVEnergy_Heuristic

    __slots__ = (
        # configuration
        "weather_entity_id",
        # state
        "weather_state",
        "_weather_tracking_unsub",
        "today_energy_estimate_sensor",
        "tomorrow_energy_estimate_sensor",
        "estimator_sensors",
    )

    # interface: Controller
    @staticmethod
    def get_config_entry_schema(user_input: dict):

        return (
            hv.entity_schema(
                user_input,
                name="PV energy estimation",
            )
            | {
                hv.optional("weather_entity_id", user_input): hv.weather_selector(),
            }
            | controller.EnergyEstimatorController.get_config_entry_schema(user_input)
        )

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

        location, elevation = sun_helpers.get_astral_location(hass)

        super().__init__(
            hass,
            config_entry,
            Estimator_PVEnergy_Heuristic,
            astral_observer=astral.sun.Observer(
                location.latitude, location.longitude, elevation
            ),
        )

        self.weather_entity_id = self.config.get("weather_entity_id")
        self._weather_tracking_unsub = None
        self.weather_state = None

        self.today_energy_estimate_sensor = EnergyEstimatorSensor(
            self, "today_energy_estimate", name="Today energy estimate"
        )
        self.tomorrow_energy_estimate_sensor = EnergyEstimatorSensor(
            self, "tomorrow_energy_estimate", name="Tomorrow energy estimate"
        )

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
        await super().async_init()
        if self.weather_entity_id:
            self._weather_tracking_unsub = event.async_track_state_change_event(
                self.hass,
                self.weather_entity_id,
                self._weather_tracking_callback,
            )
            await self._async_update_weather(
                self.hass.states.get(self.weather_entity_id)
            )

        self._update_estimate()
        self._process_observation(self.hass.states.get(self.observed_entity_id))

    async def async_shutdown(self):
        if await super().async_shutdown():
            if self._weather_tracking_unsub:
                self._weather_tracking_unsub()
                self._weather_tracking_unsub = None
            self.today_energy_estimate_sensor: EnergyEstimatorSensor = None  # type: ignore
            self.tomorrow_energy_estimate_sensor: EnergyEstimatorSensor = None  # type: ignore
            self.estimator_sensors.clear()
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
            await estimator_sensor.async_shutdown()

        await super()._entry_update_listener(hass, config_entry)

    async def _async_create_diagnostic_entities(self):
        sensors = self.entities[Sensor.PLATFORM]
        for diagnostic_sensor_enum in DiagnosticSensorsEnum:
            if diagnostic_sensor_enum not in sensors:
                DiagnosticSensor(self, diagnostic_sensor_enum)

    async def _async_destroy_diagnostic_entities(self):
        await super()._async_destroy_diagnostic_entities()

    # interface: EnergyEstimatorController
    def _update_estimate(self):
        estimator = self.estimator

        estimator.update_estimate()

        sensors = self.entities[Sensor.PLATFORM]

        self.today_energy_estimate_sensor.extra_state_attributes = {
            # "today_ts": estimator._today_local_ts,
            # "today": helpers.datetime_from_epoch(estimator._today_local_ts).isoformat(),
            # "tomorrow_ts": estimator._tomorrow_local_ts,
            # "tomorrow": helpers.datetime_from_epoch(estimator._tomorrow_local_ts).isoformat(),
            # "observed_time_ts": estimator.observed_time_ts,
            "observed_time": helpers.datetime_from_epoch(
                estimator.observed_time_ts
            ).isoformat(),
            # "observed_ratio": estimator.observed_ratio,
            "model_energy_max": estimator._model_energy_max,
            # "model_Wc": TimeSpanEnergyModel.Wc,
            "weather": estimator.get_weather_at(estimator.observed_time_ts),
        }
        self.today_energy_estimate_sensor.update(
            estimator.today_energy
            + estimator.get_estimated_energy(
                estimator.observed_time_ts, estimator._tomorrow_local_ts
            )
        )
        self.tomorrow_energy_estimate_sensor.update(
            estimator.get_estimated_energy(
                estimator._tomorrow_local_ts, estimator._tomorrow_local_ts + 86400
            )
        )

        now = time.time()
        for sensor in self.estimator_sensors.values():
            sensor.update(
                estimator.get_estimated_energy(now, now + sensor.forecast_duration_ts)
            )

        if DiagnosticSensorsEnum.observed_ratio in sensors:
            sensors[DiagnosticSensorsEnum.observed_ratio].update(
                estimator.observed_ratio
            )
        if DiagnosticSensorsEnum.weather_cloud_constant in sensors:
            sensors[DiagnosticSensorsEnum.weather_cloud_constant].update(
                TimeSpanEnergyModel.Wc
            )

    def _restore_history(self, history_start_time: dt.datetime):
        if self._restore_history_exit:
            return

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

        super()._restore_history(history_start_time)

    # interface: self
    @callback
    def _weather_tracking_callback(self, event: "Event[event.EventStateChangedData]"):
        self.async_create_task(
            self._async_update_weather(event.data.get("new_state")),
            "_async_update_weather",
        )

    async def _async_update_weather(self, weather_state: "State | None"):
        self.weather_state = weather_state
        if weather_state:
            self.estimator.add_weather(Controller._weather_from_state(weather_state))

            forecasts: list[WeatherSample] = []
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

    _WEATHER_CONDITION_TO_CLOUD: typing.Final[dict[str | None, float | None]] = {
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

        condition = weather_state.state
        if "cloud_coverage" in attributes:
            cloud_coverage = attributes["cloud_coverage"]
        else:
            cloud_coverage = Controller._WEATHER_CONDITION_TO_CLOUD.get(condition)

        if "visibility" in attributes:
            visibility = DistanceConverter.convert(
                attributes["visibility"],
                attributes["visibility_unit"],
                hac.UnitOfLength.KILOMETERS,
            )
        else:
            visibility = None

        return WeatherSample(
            time=weather_state.last_updated,
            time_ts=weather_state.last_updated_timestamp,
            condition=condition,
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

        condition = forecast.get("condition")
        if "cloud_coverage" in forecast:
            cloud_coverage = forecast["cloud_coverage"]
        else:
            cloud_coverage = self._WEATHER_CONDITION_TO_CLOUD.get(condition)

        return WeatherSample(
            time=time,
            time_ts=time_ts,
            condition=condition,
            temperature=temperature,
            cloud_coverage=cloud_coverage,
            visibility=None,
        )
