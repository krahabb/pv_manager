import enum
from time import time
import typing

from homeassistant import const as hac

from custom_components.pv_manager.controller.devices.estimator_device import EnergyEstimatorDevice

from ... import const as pmc
from ...helpers import validation as hv
from ...helpers.dataattr import DataAttr, DataAttrParam
from ...manager import Manager
from ...processors import (
    BaseProcessor,
    EnergyBroadcast,
)
from ...processors.battery import BatteryEstimator, BatteryProcessor
from ...sensor import BatteryChargeSensor, Sensor
from ..devices import SignalEnergyProcessorDevice
from ..devices.estimator_device import EnergyEstimatorDevice

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, Self, TypedDict, Unpack

    from . import Controller as OffGridManager


SourceType = SignalEnergyProcessorDevice.SourceType


class MeterDevice(SignalEnergyProcessorDevice):
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
        self.charge_processor.listen_energy(
            lambda charge, time_ts: (
                self.battery_charge_sensor.accumulate(-charge)
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


class BatteryEstimatorMeter(BatteryMeter, BatteryEstimator, EnergyEstimatorDevice):

    def __init__(self, controller, config):
        super().__init__(controller, config)
        self.listen_update(self.on_update_estimate)

        self.today_charge_estimate_sensor = Sensor(
            self,
            "today_charge_estimate",
            name="Charge estimate (today)",
            native_unit_of_measurement="Ah",
            suggested_display_precision=0,
            parent_attr=Sensor.ParentAttr.REMOVE
        )
        self.tomorrow_charge_estimate_sensor = Sensor(
            self,
            "tomorrow_charge_estimate",
            name="Charge estimate (tomorrow)",
            native_unit_of_measurement="Ah",
            suggested_display_precision=0,
            parent_attr=Sensor.ParentAttr.REMOVE
        )

    def on_update_estimate(self, estimator: "BatteryEstimator"):
        f = estimator.get_forecast(estimator.estimation_time_ts, estimator.tomorrow_ts)
        today_charge_estimate = estimator.charge + f.charge
        self.today_charge_estimate_sensor.update_safe(today_charge_estimate)
        f = estimator.get_forecast(estimator.tomorrow_ts, estimator.tomorrow_ts + 86400)
        self.tomorrow_charge_estimate_sensor.update_safe(today_charge_estimate + f.charge)

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


class LossesMeter(BaseProcessor, EnergyBroadcast):

    if typing.TYPE_CHECKING:
        controller: Final[OffGridManager]
        battery_meter: Final[BatteryMeter]
        load_meter: Final[LoadMeter]
        pv_meter: Final[PvMeter]

    battery_energy: DataAttr[float, DataAttrParam.stored] = 0
    battery_in_energy: DataAttr[float, DataAttrParam.stored] = 0
    battery_out_energy: DataAttr[float, DataAttrParam.stored] = 0
    load_energy: DataAttr[float, DataAttrParam.stored] = 0
    pv_energy: DataAttr[float, DataAttrParam.stored] = 0
    losses_energy: DataAttr[float, DataAttrParam.stored] = 0

    __slots__ = (
        # config
        "controller",
        "battery_meter",
        "load_meter",
        "pv_meter",
        "update_period_ts",
        # internal state
        "_load_energy_old",
        "_timer_unsub",
        "_battery_callback_unsub",
        "_load_callback_unsub",
        "_pv_callback_unsub",
    )

    def __init__(self, controller: "OffGridManager", update_period: float):
        self.controller = controller
        self.battery_meter = controller.battery_meter
        self.load_meter = controller.load_meter
        self.pv_meter = controller.pv_meter
        self.update_period_ts = update_period

        self._timer_unsub = None
        super().__init__(SourceType.LOSSES, logger=controller, config={})
        controller.energy_meters[SourceType.LOSSES] = (self, controller)

    def start(self):
        self._losses_compute()
        self.time_ts = time()
        self._battery_callback_unsub = self.battery_meter.listen_energy(
            self._battery_callback
        )
        self._load_callback_unsub = self.load_meter.listen_energy(self._load_callback)
        self._pv_callback_unsub = self.pv_meter.listen_energy(self._pv_callback)
        self._timer_unsub = Manager.schedule(
            self.update_period_ts, self._timer_callback
        )

    def shutdown(self):
        if self._timer_unsub:
            self._timer_unsub.cancel()
            self._timer_unsub = None
        self._battery_callback_unsub()
        self._load_callback_unsub()
        self._pv_callback_unsub()

        del self.controller.energy_meters[self.id]
        setattr(self.controller, f"{self.id}_meter", None)
        super().shutdown()
        self.controller = None  # type: ignore
        self.battery_meter = None  # type: ignore
        self.load_meter = None  # type: ignore
        self.pv_meter = None  # type: ignore

    def _battery_callback(self, energy: float, time_ts: float):
        self.battery_energy += energy
        if energy > 0:
            self.battery_out_energy += energy
        else:
            self.battery_in_energy -= energy

    def _load_callback(self, energy: float, time_ts: float):
        self.load_energy += energy

    def _pv_callback(self, energy: float, time_ts: float):
        self.pv_energy += energy

    def _timer_callback(self):
        self._timer_unsub = Manager.schedule(
            self.update_period_ts, self._timer_callback
        )
        time_ts = time()
        # get the 'new' total
        self.battery_meter.update(time_ts)
        self.load_meter.update(time_ts)
        self.pv_meter.update(time_ts)

        d_load = self.load_energy - self._load_energy_old
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
        self._load_energy_old = load = self.load_energy
        pv = self.pv_energy
        self.losses_energy = losses = pv + battery - load

        # Estimate energy actually stored in the battery:
        # in the long term -> battery_in > battery_out with the difference being the energy 'eaten up'
        # battery_yield = battery_out / (battery_in - battery_stored_energy)
        # battery_stored_energy is hard to compute since it depends on the discharge current/voltage
        # we'll use a conservative approach with the following formula. It's contribution to
        # battery_yield will nevertheless decay as far as battery_in, battery_out will increase

        controller = self.controller
        if controller.conversion_yield_sensor:
            try:
                conversion_yield = load / (load + losses)
                controller.conversion_yield_sensor.update(round(conversion_yield * 100))
            except:
                controller.conversion_yield_sensor.update(None)

        """
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
        """
