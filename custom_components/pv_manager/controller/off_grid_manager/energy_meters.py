import enum
from time import time as TIME_TS
import typing

from homeassistant import const as hac

from ... import const as pmc
from ...helpers import validation as hv
from ...manager import Manager
from ...processors import (
    EnergyBroadcast,
    SignalEnergyProcessor,
)
from ...processors.battery import BatteryProcessor
from ...sensor import BatteryChargeSensor, Sensor
from ..devices import SignalEnergyProcessorDevice

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, TypedDict, Unpack

    from . import Controller as OffGridManager
    from ...processors.estimator_energy import SignalEnergyEstimator


SourceType = SignalEnergyProcessorDevice.SourceType


class BaseMeter(EnergyBroadcast):

    if typing.TYPE_CHECKING:

        class StoreType(TypedDict):
            pass

        class Args(EnergyBroadcast.Args):
            pass

        id: Final[SourceType]  # type: ignore

    def restore(self, data: "StoreType"):
        pass

    def store(self) -> "StoreType":
        return {}


class MeterDevice(SignalEnergyProcessorDevice, BaseMeter):
    if typing.TYPE_CHECKING:

        class Config(SignalEnergyProcessorDevice.Config):
            pass

        class Args(SignalEnergyProcessorDevice.Args):
            config: "MeterDevice.Config"

        controller: OffGridManager

    def __init__(
        self,
        metering_source: SourceType,
        controller: "OffGridManager",
        config: "Config",
    ):
        super().__init__(
            metering_source,
            controller=controller,
            model=f"{metering_source}_meter",
            config=config,
            name=f"{controller.config.get("name", controller.TYPE)} {metering_source}",
        )
        self.maximum_latency_ts = controller.maximum_latency_ts
        controller.energy_meters[metering_source] = (self, self)

    def shutdown(self):
        del self.controller.energy_meters[self.id]
        setattr(self.controller, f"{self.id}_meter", None)
        super().shutdown()


class BatteryMeter(MeterDevice, BatteryProcessor):

    if typing.TYPE_CHECKING:

        class Config(MeterDevice.Config, BatteryProcessor.Config):
            pass

        class StoreType(BatteryProcessor.StoreType):
            pass

        battery_charge_sensor: BatteryChargeSensor | None

    __slots__ = (
        # sensors
        "battery_charge_sensor",
    ) + BatteryProcessor._SLOTS_

    @staticmethod
    def get_config_schema(config: "Config") -> "pmc.ConfigSchema":
        return {
            hv.req_config("battery_voltage_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.VOLTAGE
            ),
            hv.req_config("battery_current_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.CURRENT
            ),
            hv.req_config("battery_capacity", config): hv.positive_number_selector(
                unit_of_measurement="Ah"
            ),
            hv.opt_config("input_max", config): hv.positive_number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
        }

    def __init__(self, controller: "OffGridManager", config: "Config"):
        self.battery_charge_sensor = None

        super().__init__(SourceType.BATTERY, controller, config)

        self.configure(SignalEnergyProcessor.InputMode.POWER)
        controller.energy_meters[SourceType.BATTERY_IN] = (
            self.energy_broadcast_in,
            self,
        )
        controller.energy_meters[SourceType.BATTERY_OUT] = (
            self.energy_broadcast_out,
            self,
        )
        # TODO: setup according to some sort of configuration
        BatteryChargeSensor(
            self,
            "battery_charge",
            capacity=self.battery_capacity,
            name="Battery charge",
            parent_attr=Sensor.ParentAttr.DYNAMIC,
        )
        self.battery_charge_broadcast.listen(
            lambda battery_charge: (
                self.battery_charge_sensor.update(battery_charge)
                if self.battery_charge_sensor
                else None
            )
        )

    @typing.override
    def shutdown(self):
        meters = self.controller.energy_meters
        del meters[SourceType.BATTERY_OUT]
        del meters[SourceType.BATTERY_IN]
        super().shutdown()


class EnergyEstimatorMeterDevice(MeterDevice):
    """Partial common base class for meters (pv and load) that could be built as estimators.
    This is done at runtime by building a new class composition including the
    needed components if the OffGridManager has a specific config subentry for that.
    TODO"""

    if typing.TYPE_CHECKING:

        class Config(MeterDevice.Config):
            pass

    @staticmethod
    def get_config_schema(config: "Config") -> "pmc.ConfigSchema":
        return {
            hv.opt_config("source_entity_id", config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.POWER
            ),
            hv.opt_config("input_max", config): hv.positive_number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
        }

    def __init__(self, metering_source, controller, config):
        super().__init__(metering_source, controller, config)
        self.input_min = 0
        self.energy_min = 0


class PvMeter(EnergyEstimatorMeterDevice):

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimatorMeterDevice.Config):
            pass

    def __init__(self, controller: "OffGridManager", config: "Config"):
        super().__init__(SourceType.PV, controller, config)


class LoadMeter(EnergyEstimatorMeterDevice):

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimatorMeterDevice.Config):
            pass

    def __init__(self, controller: "OffGridManager", config: "Config"):
        super().__init__(SourceType.LOAD, controller, config)


class LossesMeter(BaseMeter):

    if typing.TYPE_CHECKING:

        class StoreType(BaseMeter.StoreType):
            battery_energy: float
            battery_in_energy: float
            battery_out_energy: float
            load_energy: float
            pv_energy: float
            losses_energy: float  # redundant tho

        controller: Final[OffGridManager]
        battery_meter: Final[BatteryMeter]
        load_meter: Final[LoadMeter]
        pv_meter: Final[PvMeter]
        battery_meter_old: float
        battery_meter_in_old: float
        battery_meter_out_old: float
        load_meter_old: float
        pv_meter_old: float

    __slots__ = (
        # config
        "controller",
        "battery_meter",
        "load_meter",
        "pv_meter",
        "update_period_ts",
        # counters
        "battery_energy",
        "battery_in_energy",
        "battery_out_energy",
        "load_energy",
        "pv_energy",
        "losses_energy",
        # cached previous samples from meters
        "battery_meter_old",
        "battery_meter_in_old",
        "battery_meter_out_old",
        "load_meter_old",
        "pv_meter_old",
        # state
        "_timer_unsub",
    )

    def __init__(self, controller: "OffGridManager", update_period_seconds: float):
        self.controller = controller
        self.battery_meter = controller.battery_meter
        self.load_meter = controller.load_meter
        self.pv_meter = controller.pv_meter
        self.update_period_ts = update_period_seconds

        # set in case load is not called when new
        self.battery_energy = 0
        self.battery_in_energy = 0
        self.battery_out_energy = 0
        self.load_energy = 0
        self.pv_energy = 0
        self.losses_energy = 0

        self._timer_unsub = None
        super().__init__(SourceType.LOSSES, logger=controller)
        controller.energy_meters[SourceType.LOSSES] = (self, controller)

    def start(self):
        self._losses_compute()
        self.time_ts = TIME_TS()
        self.battery_meter_old = self.battery_meter.energy
        self.battery_meter_in_old = self.battery_meter.energy_in
        self.battery_meter_out_old = self.battery_meter.energy_out
        self.load_meter_old = self.load_meter.energy
        self.pv_meter_old = self.pv_meter.energy
        self._timer_unsub = Manager.schedule(
            self.update_period_ts, self._timer_callback
        )

    def shutdown(self):
        if self._timer_unsub:
            self._timer_unsub.cancel()
            self._timer_unsub = None
        del self.controller.energy_meters[self.id]
        setattr(self.controller, f"{self.id}_meter", None)
        super().shutdown()
        self.controller = None  # type: ignore
        self.battery_meter = None  # type: ignore
        self.load_meter = None  # type: ignore
        self.pv_meter = None  # type: ignore

    @typing.override
    def restore(self, data: "StoreType"):
        with self.exception_warning("loading meter data"):
            self.battery_energy = data["battery_energy"]
            self.battery_in_energy = data["battery_in_energy"]
            self.battery_out_energy = data["battery_out_energy"]
            self.load_energy = data["load_energy"]
            self.pv_energy = data["pv_energy"]
            self.losses_energy = data["losses_energy"]

    @typing.override
    def store(self) -> "StoreType":
        return {
            "battery_energy": self.battery_energy,
            "battery_in_energy": self.battery_in_energy,
            "battery_out_energy": self.battery_out_energy,
            "load_energy": self.load_energy,
            "pv_energy": self.pv_energy,
            "losses_energy": self.losses_energy,
        }

    def _timer_callback(self):
        self._timer_unsub = Manager.schedule(
            self.update_period_ts, self._timer_callback
        )
        time_ts = TIME_TS()
        # get the 'new' total
        battery_meter = self.battery_meter
        battery_meter.update(time_ts)
        self.battery_energy += battery_meter.energy - self.battery_meter_old
        self.battery_meter_old = battery_meter.energy
        load_meter = self.load_meter
        load_meter.update(time_ts)
        d_load = load_meter.energy - self.load_meter_old
        self.load_energy += d_load
        self.load_meter_old = load_meter.energy
        pv_meter = self.pv_meter
        pv_meter.update(time_ts)
        self.pv_energy += pv_meter.energy - self.pv_meter_old
        self.pv_meter_old = pv_meter.energy

        losses_old = self.losses_energy
        self._losses_compute()
        # compute delta to get the average power in the sampling period
        # we don't check maximum_latency here since it has already been
        # managed in pv, load and battery meters
        d_losses = self.losses_energy - losses_old
        for energy_listener in self.energy_listeners:
            energy_listener(d_losses, time_ts)

        controller = self.controller
        if controller.losses_power_sensor:
            d_time = time_ts - self.time_ts
            if 0 < d_time < controller.maximum_latency_ts:
                controller.losses_power_sensor.update(round(d_losses * 3600 / d_time))
            else:
                controller.losses_power_sensor.update(None)

        self.time_ts = time_ts

        if controller.conversion_yield_actual_sensor:
            try:
                controller.conversion_yield_actual_sensor.update(
                    round(d_load * 100 / (d_load + d_losses))
                )
            except:
                controller.conversion_yield_actual_sensor.update(None)

    def _losses_compute(self):

        battery = self.battery_energy
        battery_in = self.battery_in_energy
        battery_out = self.battery_out_energy
        load = self.load_energy
        pv = self.pv_energy
        self.losses_energy = losses = pv + battery - load

        # Estimate energy actually stored in the battery:
        # in the long term -> battery_in > battery_out with the difference being the energy 'eaten up'
        # battery_yield = battery_out / (battery_in - battery_stored_energy)
        # battery_stored_energy is hard to compute since it depends on the discharge current/voltage
        # we'll use a conservative approach with the following formula. It's contribution to
        # battery_yield will nevertheless decay as far as battery_in, battery_out will increase
        battery_meter = self.battery_meter
        battery_stored = (
            battery_meter.battery_charge_estimate * (battery_meter.battery_voltage or 0) * 0.9
        )

        controller = self.controller
        if controller.conversion_yield_sensor:
            try:
                conversion_yield = load / (load + losses)
                controller.conversion_yield_sensor.update(round(conversion_yield * 100))
            except:
                controller.conversion_yield_sensor.update(None)

        if controller.battery_yield_sensor:
            try:
                # battery_yield = battery_out_energy / (battery_in_energy - battery_stored_energy)
                _temp = battery_in - battery_stored
                if _temp > battery_out:
                    controller.battery_yield_sensor.update(
                        round(battery_out * 100 / _temp)
                    )
                else:
                    controller.battery_yield_sensor.update(None)
            except:
                controller.battery_yield_sensor.update(None)

        if controller.system_yield_sensor:
            try:
                # system_yield = load_energy / (pv_energy - battery_stored_energy)
                _temp = pv - battery_stored
                if _temp > load:
                    controller.system_yield_sensor.update(round(load * 100 / _temp))
                else:
                    controller.system_yield_sensor.update(None)
            except:
                controller.system_yield_sensor.update(None)
