from time import time as TIME_TS
import typing

from homeassistant import const as hac
from homeassistant.util.unit_conversion import (
    ElectricCurrentConverter,
    ElectricPotentialConverter,
)

from ... import const as pmc
from ...sensor import BatteryChargeSensor, EnergySensor, PowerSensor, Sensor
from ...processors import (
    BaseEnergyProcessor,
    EnergyInputMode,
    SourceType,
    SAFE_MAXIMUM_POWER_DISABLED,
    SAFE_MINIMUM_POWER_DISABLED,
)
from ..devices.energy_processor import EnergyProcessorDevice

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, TypedDict, Unpack

    from homeassistant.core import Event, EventStateChangedData

    from ...helpers.entity import EntityArgs
    from . import Controller as OffGridManager

VOLTAGE_UNIT = hac.UnitOfElectricPotential.VOLT
CURRENT_UNIT = hac.UnitOfElectricCurrent.AMPERE
POWER_UNIT = hac.UnitOfPower.WATT
ENERGY_UNIT = hac.UnitOfEnergy.WATT_HOUR


class MeterStoreType(typing.TypedDict):
    time_ts: float | None
    energy: float


class BaseMeter(BaseEnergyProcessor):

    if typing.TYPE_CHECKING:

        class Config(BaseEnergyProcessor.Config):
            pass

        class Args(BaseEnergyProcessor.Args):
            device: "MeterDevice"

    device: "Final[MeterDevice]"
    metering_source: "Final[SourceType]"

    _SLOTS_ = (
        "device",
        "metering_source",
    )

    def __init__(self, metering_source: SourceType, **kwargs: "Unpack[Args]"):
        self.device = kwargs["device"]
        self.metering_source = metering_source
        super().__init__(metering_source, **kwargs)
        controller = self.device.controller
        controller.energy_meters[metering_source] = self
        setattr(controller, f"{self.metering_source}_meter", self)

    def shutdown(self):
        controller = self.device.controller
        del controller.energy_meters[self.metering_source]
        setattr(controller, f"{self.metering_source}_meter", None)
        super().shutdown()
        self.device = None  # type: ignore

    def load(self, data: MeterStoreType):
        self.output = data["energy"]

    def save(self):
        return MeterStoreType(
            {
                "time_ts": self.time_ts,
                "energy": self.output,
            }
        )


class MeterDevice(EnergyProcessorDevice, BaseMeter):
    if typing.TYPE_CHECKING:

        class Config(EnergyProcessorDevice.Config, BaseMeter.Config):
            pass

    controller: "Final[OffGridManager]"

    def __init__(
        self,
        metering_source: SourceType,
        controller: "OffGridManager",
        config: "Config",
    ):
        super().__init__(
            metering_source,
            controller=controller,
            device=self,  # type: ignore
            model=f"{metering_source}_meter",
            config=config,
            name=metering_source.value.upper(),
        )


class BatteryMeter(MeterDevice):

    if typing.TYPE_CHECKING:

        class Config(MeterDevice.Config):
            battery_voltage_entity_id: str
            battery_current_entity_id: str
            battery_charge_entity_id: NotRequired[str]
            battery_capacity: float

    battery_charge_sensor: BatteryChargeSensor | None

    __slots__ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        # state
        "battery_voltage",
        "battery_current",
        "_battery_current_last_ts",
        "battery_charge",
        "battery_charge_estimate",
        "battery_capacity_estimate",
        # meters
        "in_meter",
        "out_meter",
        # sensors
        "battery_charge_sensor",
    )

    def __init__(self, controller: "OffGridManager", config: "Config"):
        self.battery_voltage_entity_id = config["battery_voltage_entity_id"]
        self.battery_current_entity_id = config["battery_current_entity_id"]
        self.battery_charge_entity_id = config.get("battery_charge_entity_id")
        self.battery_capacity = config.get("battery_capacity", 0)

        self.battery_voltage: float = 0
        self.battery_current: float = 0
        self._battery_current_last_ts: float = 0
        self.battery_charge: float = 0
        self.battery_charge_estimate: float = 0
        self.battery_capacity_estimate: float = 0
        self.battery_charge_sensor = None

        super().__init__(SourceType.BATTERY, controller, config)
        # our config doesnt carry safe_minimum_power so we'll update it to reflect safe_maximum_power
        if self.safe_maximum_power != SAFE_MAXIMUM_POWER_DISABLED:
            self.safe_minimum_power = -self.safe_maximum_power
            self.safe_minimum_power_cal = self.safe_minimum_power / 3600

        self.configure(EnergyInputMode.POWER)
        self.in_meter = BaseMeter(SourceType.BATTERY_IN, device=self, config={})
        self.out_meter = BaseMeter(SourceType.BATTERY_OUT, device=self, config={})
        self._energy_listeners.add(self._energy_callback)

        # TODO: setup according to some sort of configuration
        BatteryChargeSensor(
            self,
            "battery_charge",
            capacity=self.battery_capacity,
            name="Battery charge",
            parent_attr=Sensor.ParentAttr.DYNAMIC,
        )

    async def async_start(self):
        self.track_state(
            self.battery_voltage_entity_id,
            self._battery_voltage_callback,
            self.HassJobType.Callback,
        )
        self.track_state(
            self.battery_current_entity_id,
            self._battery_current_callback,
            self.HassJobType.Callback,
        )
        if self.battery_charge_entity_id:
            self.track_state(
                self.battery_charge_entity_id,
                self._battery_charge_callback,
                self.HassJobType.Callback,
            )

    def _energy_callback(self, energy: float, time_ts: float):
        if energy > 0:
            in_out_meter = self.out_meter
        else:
            in_out_meter = self.in_meter
            energy = -energy
        in_out_meter.time_ts = time_ts
        in_out_meter.input = energy
        in_out_meter.output += energy
        for listener in in_out_meter._energy_listeners:
            listener(energy, time_ts)

    def _battery_voltage_callback(
        self, event: "Event[EventStateChangedData] | BatteryMeter.Event"
    ):
        try:
            state = event.data["new_state"]
            self.battery_voltage = ElectricPotentialConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                VOLTAGE_UNIT,
            )
            self.process(
                self.battery_voltage * self.battery_current, event.time_fired_timestamp
            )
        except Exception as e:
            self.battery_voltage = 0
            self.process(None, event.time_fired_timestamp)
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(
                    self.WARNING, e, "_battery_voltage_update(state:%s)", state
                )

    def _battery_current_callback(
        self, event: "Event[EventStateChangedData] | BatteryMeter.Event"
    ):
        time_ts = event.time_fired_timestamp
        try:
            state = event.data["new_state"]
            battery_current = ElectricCurrentConverter.convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                CURRENT_UNIT,
            )
        except Exception as e:
            battery_current = 0
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(
                    self.WARNING, e, "_battery_current_update(state:%s)", state
                )

        # left rectangle integration
        # We assume 'generator convention' for current
        # i.e. positive current = discharging
        d_time = time_ts - self._battery_current_last_ts
        if 0 < d_time < self.maximum_latency_ts:
            charge_out = self.battery_current * d_time / 3600
            self.battery_charge_estimate -= charge_out
            if self.battery_charge_estimate > self.battery_capacity_estimate:
                self.battery_capacity_estimate = self.battery_charge_estimate
            elif self.battery_charge_estimate < 0:
                self.battery_capacity_estimate -= self.battery_charge_estimate
                self.battery_charge_estimate = 0
            if self.battery_charge_sensor:
                self.battery_charge_sensor.update(self.battery_charge_estimate)

        self.battery_current = battery_current
        self._battery_current_last_ts = time_ts

        self.process(self.battery_current * self.battery_voltage, time_ts)

    def _battery_charge_callback(
        self, event: "Event[EventStateChangedData] | BatteryMeter.Event"
    ):
        pass


class PvMeter(MeterDevice):

    if typing.TYPE_CHECKING:

        class Config(MeterDevice.Config):
            pass

    def __init__(self, controller: "OffGridManager", config: "Config"):
        super().__init__(SourceType.PV, controller, config)


class LoadMeter(MeterDevice):

    if typing.TYPE_CHECKING:

        class Config(MeterDevice.Config):
            pass

    def __init__(self, controller: "OffGridManager", config: "Config"):
        super().__init__(SourceType.LOAD, controller, config)


class LossesMeter(BaseMeter):

    __slots__ = (
        "battery_energy",
        "battery_in_energy",
        "battery_out_energy",
        "load_energy",
        "pv_energy",
    )

    def __init__(self, controller: "OffGridManager", update_period_seconds: float):
        super().__init__(
            SourceType.LOSSES,
            device=controller,  # type: ignore
            config={"update_period_seconds": update_period_seconds},
        )

    async def async_start(self):
        self._losses_compute()
        self.time_ts = TIME_TS()
        await super().async_start()

    def _update_callback(self):
        time_ts = TIME_TS()
        controller = self.device.controller
        battery_old = self.battery_energy
        battery_in_old = self.battery_in_energy
        battery_out_old = self.battery_out_energy
        load_old = self.load_energy
        pv_old = self.pv_energy
        losses_old = self.output
        # get the 'new' total
        controller.battery_meter.update(time_ts)
        controller.load_meter.update(time_ts)
        controller.pv_meter.update(time_ts)
        self._losses_compute()
        # compute delta to get the average power in the sampling period
        # we don't check maximum_latency here since it has already been
        # managed in pc, load and battery meters
        d_losses = self.output - losses_old
        for energy_listener in self._energy_listeners:
            energy_listener(d_losses, time_ts)

        if controller.losses_power_sensor:
            d_time = time_ts - self.time_ts
            if 0 < d_time < controller.maximum_latency_ts:
                controller.losses_power_sensor.update(round(d_losses * 3600 / d_time))
            else:
                controller.losses_power_sensor.update(None)

        self.input = d_losses
        self.time_ts = time_ts

        if controller.conversion_yield_actual_sensor:
            try:
                d_load = self.load_energy - load_old
                controller.conversion_yield_actual_sensor.update(
                    round(d_load * 100 / (d_load + d_losses))
                )
            except:
                controller.conversion_yield_actual_sensor.update(None)

    def _losses_compute(self):
        controller = self.device.controller
        battery_meter = controller.battery_meter
        self.battery_energy = battery = battery_meter.output
        self.battery_in_energy = battery_in = controller.battery_in_meter.output
        self.battery_out_energy = battery_out = controller.battery_out_meter.output
        self.load_energy = load = controller.load_meter.output
        self.pv_energy = pv = controller.pv_meter.output
        self.output = losses = pv + battery - load

        # Estimate energy actually stored in the battery:
        # in the long term -> battery_in > battery_out with the difference being the energy 'eaten up'
        # battery_yield = battery_out / (battery_in - battery_stored_energy)
        # battery_stored_energy is hard to compute since it depends on the discharge current/voltage
        # we'll use a conservative approach with the following formula. It's contribution to
        # battery_yeild will nevertheless decay as far as battery_in, battery_out will increase
        battery_stored = (
            battery_meter.battery_charge_estimate * battery_meter.battery_voltage * 0.9
        )

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
