import typing

from homeassistant import const as hac

from . import ProcessorDevice, SignalEnergyProcessorDevice
from ... import const as pmc
from ...helpers import validation as hv
from ...helpers.entity import EstimatorEntity
from ...processors.estimator_energy import (
    EnergyEstimator,
    Estimator,
    SignalEnergyEstimator,
)
from ...sensor import Sensor

if typing.TYPE_CHECKING:
    from typing import Any, Callable, ClassVar, Coroutine, Final, NotRequired, Unpack

    from ...helpers.entity import Entity


class EstimatorDevice(ProcessorDevice, Estimator):
    if typing.TYPE_CHECKING:

        class Config(Estimator.Config, ProcessorDevice.Config):
            pass

        class Args(Estimator.Args, ProcessorDevice.Args):
            config: "EstimatorDevice.Config"


class EnergyEstimatorSensor(EstimatorEntity, Sensor):
    """Entity reporting the forecasted energy over a (future) time span."""

    if typing.TYPE_CHECKING:

        class Config(pmc.EntityConfig, pmc.SubentryConfig):
            forecast_duration_hours: int

        class Args(Entity.Args):
            forecast_duration_ts: NotRequired[int]
            estimator: NotRequired[EnergyEstimator]

        device: "EnergyEstimatorDevice"
        forecast_duration_ts: int

    _unrecorded_attributes = frozenset(
        (
            "forecast",
            "state",
        )
    )
    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_unit_of_measurement = hac.UnitOfEnergy.WATT_HOUR

    __slots__ = ("forecast_duration_ts",)

    def __init__(
        self,
        device: "EnergyEstimatorDevice",
        id,
        /,
        **kwargs: "Unpack[Args]",
    ):
        self.forecast_duration_ts = kwargs.pop("forecast_duration_ts", 0)
        super().__init__(
            device,
            id,
            kwargs.pop("estimator", device),
            state_class=None,
            **kwargs,  # type: ignore
        )

    @typing.override
    def on_estimator_update(self, estimator: EnergyEstimator):
        self.native_value = round(
            estimator.get_estimated_energy(
                estimator.estimation_time_ts,
                estimator.estimation_time_ts + self.forecast_duration_ts,
            )
        )
        if self.added_to_hass:
            self._async_write_ha_state()


class TodayEnergyEstimatorSensor(EnergyEstimatorSensor):

    @typing.override
    def on_estimator_update(self, estimator: EnergyEstimator):
        self.extra_state_attributes = estimator.as_state_dict()
        self.native_value = round(
            estimator.today_energy
            + estimator.get_estimated_energy(
                estimator.estimation_time_ts, estimator.tomorrow_ts
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


class EnergyEstimatorDevice(EstimatorDevice, EnergyEstimator):

    if typing.TYPE_CHECKING:

        class Config(
            EnergyEstimator.Config,
            EstimatorDevice.Config,
            pmc.EntityConfig,
        ):
            pass

        class Args(EnergyEstimator.Args, EstimatorDevice.Args):
            config: "EnergyEstimatorDevice.Config"

    @classmethod
    def get_config_schema(cls, config: "Config | None") -> pmc.ConfigSchema:
        _config = config or {"name": cls.DEFAULT_NAME}
        return hv.entity_schema(_config) | super().get_config_schema(config)

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
