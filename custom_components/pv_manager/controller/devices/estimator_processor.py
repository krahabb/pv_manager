import typing

from homeassistant import const as hac

from . import Device
from ... import const as pmc
from ...helpers import validation as hv
from ...helpers.entity import EstimatorEntity
from ...processors.estimator import EnergyEstimator, Estimator
from ...sensor import Sensor
from .energy_processor import EnergyProcessorDevice

if typing.TYPE_CHECKING:
    from typing import Any, Callable, ClassVar, Coroutine, Final, Unpack

    from ...helpers.entity import Entity


class EstimatorDevice(Estimator, Device):
    if typing.TYPE_CHECKING:

        class Config(Estimator.Config, Device.Config):
            pass

        class Args(Estimator.Args, Device.Args):
            config: "EstimatorDevice.Config"


class EnergyEstimatorSensor(EstimatorEntity, Sensor):
    """Entity reporting the forecasted energy over a (future) time span."""

    if typing.TYPE_CHECKING:

        class Config(pmc.EntityConfig, pmc.SubentryConfig):
            forecast_duration_hours: int

        class Args(Entity.Args):
            pass

    device: "EnergyEstimatorDevice"

    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_unit_of_measurement = hac.UnitOfEnergy.WATT_HOUR

    __slots__ = ("forecast_duration_ts",)

    def __init__(
        self,
        device: "EnergyEstimatorDevice",
        id,
        *,
        forecast_duration_ts: float = 0,
        **kwargs: "Unpack[Args]",
    ):
        self.forecast_duration_ts = forecast_duration_ts
        super().__init__(
            device,
            id,
            device,
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


class EnergyEstimatorDevice(EnergyEstimator, EstimatorDevice, EnergyProcessorDevice):

    if typing.TYPE_CHECKING:

        class Config(
            EnergyEstimator.Config,
            EstimatorDevice.Config,
            EnergyProcessorDevice.Config,
            pmc.EntityConfig,
        ):
            pass

        class Args(
            EnergyEstimator.Args, EstimatorDevice.Args, EnergyProcessorDevice.Args
        ):
            config: "EnergyEstimatorDevice.Config"

    @classmethod
    def get_config_schema(cls, config: "Config | None") -> "pmc.ConfigSchema":
        if not config:
            config = {
                "name": cls.DEFAULT_NAME,
                "source_entity_id": "",
                "sampling_interval_minutes": 10,
                "observation_duration_minutes": 20,
                "history_duration_days": 7,
                "maximum_latency_seconds": 60,
            }
        return {
            hv.req_config("name", config): str,
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

    def __init__(self, id, **kwargs: "Unpack[Args]"):

        super().__init__(id, **kwargs)

        TodayEnergyEstimatorSensor(
            self,
            "today_energy_estimate",
            name=f"{self.config.get("name", self.__class__.DEFAULT_NAME)} (today)",
        )
        TomorrowEnergyEstimatorSensor(
            self,
            "tomorrow_energy_estimate",
            name=f"{self.config.get("name", self.__class__.DEFAULT_NAME)} (tomorrow)",
        )
