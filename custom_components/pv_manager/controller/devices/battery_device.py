from typing import TYPE_CHECKING

from . import EnergyMeterDevice
from ...helpers import validation as hv
from ...processors.battery import BatteryEstimator, BatteryProcessor
from ...sensor import BatteryChargeSensor, EnergySensor, Sensor
from .estimator_device import EnergyEstimatorDevice

if TYPE_CHECKING:
    from typing import Final, NotRequired, Unpack

    from .. import EntryData
    from ... import const as pmc


class BatteryProcessorDevice(EnergyMeterDevice, BatteryProcessor):

    if TYPE_CHECKING:

        class Config(BatteryProcessor.Config, EnergyMeterDevice.Config):
            cycle_modes_in: NotRequired[list[EnergySensor.CycleMode]]
            cycle_modes_out: NotRequired[list[EnergySensor.CycleMode]]

        class Args(BatteryProcessor.Args, EnergyMeterDevice.Args):
            config: "BatteryProcessorDevice.Config"

        config: Config

        battery_charge_sensor: BatteryChargeSensor | None

    _SLOTS_ = ("battery_charge_sensor",)

    @classmethod
    def get_config_schema(cls, config: "Config | None", /) -> "pmc.ConfigSchema":
        _config = config or {}
        return super().get_config_schema(config) | {
            hv.opt_config("cycle_modes_in", _config): hv.cycle_modes_selector(),
            hv.opt_config("cycle_modes_out", _config): hv.cycle_modes_selector(),
        }

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        config = self.config
        self._create_energy_sensors(
            "energy_in_sensor",
            f"{config.get("name", self.__class__.DEFAULT_NAME)} (in)",
            self.energy_broadcast_in,
            config.get("cycle_modes_in", ()),
        )
        self._create_energy_sensors(
            "energy_out_sensor",
            f"{config.get("name", self.__class__.DEFAULT_NAME)} (out)",
            self.energy_broadcast_out,
            config.get("cycle_modes_out", ()),
        )

        BatteryChargeSensor(
            self,
            "battery_charge",
            capacity=self.battery_capacity,
            name="Battery charge",
            parent_attr=BatteryChargeSensor.ParentAttr.DYNAMIC,
        )
        self.charge_processor.listen_energy(
            lambda charge, time_ts: (
                self.battery_charge_sensor.accumulate(-charge)
                if self.battery_charge_sensor
                else None
            )
        )

    async def update_entry(self, entry_data: "EntryData", /):
        await super().update_entry(entry_data)
        config = self.config
        await self._update_energy_sensors(
            "energy_in_sensor",
            "cycle_modes_in",
            f"{config.get("name", self.__class__.DEFAULT_NAME)} (in)",
            self.energy_broadcast_in,
            entry_data,
        )
        await self._update_energy_sensors(
            "energy_out_sensor",
            "cycle_modes_out",
            f"{config.get("name", self.__class__.DEFAULT_NAME)} (out)",
            self.energy_broadcast_out,
            entry_data,
        )


class BatteryEstimatorDevice(EnergyEstimatorDevice, BatteryEstimator):

    if TYPE_CHECKING:

        class Config(BatteryEstimator.Config, EnergyEstimatorDevice.Config):
            cycle_modes_in: NotRequired[list[EnergySensor.CycleMode]]
            cycle_modes_out: NotRequired[list[EnergySensor.CycleMode]]

        class Args(BatteryEstimator.Args, EnergyEstimatorDevice.Args):
            config: "BatteryEstimatorDevice.Config"

        config: Config

        today_charge_estimate_sensor: Sensor
        tomorrow_charge_estimate_sensor: Sensor

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self.listen_update(self.on_update_estimate)

        self.today_charge_estimate_sensor = Sensor(
            self,
            "today_charge_estimate",
            name="Charge estimate (today)",
            native_unit_of_measurement="Ah",
            suggested_display_precision=0,
            parent_attr=Sensor.ParentAttr.REMOVE,
        )
        self.tomorrow_charge_estimate_sensor = Sensor(
            self,
            "tomorrow_charge_estimate",
            name="Charge estimate (tomorrow)",
            native_unit_of_measurement="Ah",
            suggested_display_precision=0,
            parent_attr=Sensor.ParentAttr.REMOVE,
        )

    def on_update_estimate(self, estimator: "BatteryEstimator", /):
        f = estimator.get_forecast(estimator.estimation_time_ts, estimator.tomorrow_ts)
        today_charge_estimate = estimator.charge_in - estimator.charge_out + f.charge
        self.today_charge_estimate_sensor.update_safe(today_charge_estimate)
        f = estimator.get_forecast(estimator.tomorrow_ts, estimator.tomorrow_ts + 86400)
        self.tomorrow_charge_estimate_sensor.update_safe(
            today_charge_estimate + f.charge
        )
