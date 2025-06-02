import typing

from . import SignalEnergyProcessorDevice
from ...helpers import validation as hv
from ...processors.battery import BatteryEstimator, BatteryProcessor
from ...sensor import BatteryChargeSensor, EnergySensor
from .estimator_device import EnergyEstimatorDevice

if typing.TYPE_CHECKING:
    from typing import Final, NotRequired

    from .. import EntryData
    from ... import const as pmc


class BatteryProcessorDevice(SignalEnergyProcessorDevice, BatteryProcessor):

    if typing.TYPE_CHECKING:

        class Config(SignalEnergyProcessorDevice.Config, BatteryProcessor.Config):
            cycle_modes_in: NotRequired[list[EnergySensor.CycleMode]]
            cycle_modes_out: NotRequired[list[EnergySensor.CycleMode]]

        class Args(SignalEnergyProcessorDevice.Args, BatteryProcessor.Args):
            config: "BatteryProcessorDevice.Config"

        config: Config

        battery_charge_sensor: BatteryChargeSensor | None

    _SLOTS_ = ("battery_charge_sensor",)

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None", /) -> "pmc.ConfigSchema":
        _config = config or {}
        return super().get_config_schema(config) | {
            hv.opt_config("cycle_modes_in", _config): hv.cycle_modes_selector(),
            hv.opt_config("cycle_modes_out", _config): hv.cycle_modes_selector(),
        }

    def __init__(self, id, /, **kwargs):
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

    @typing.override
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
    pass
