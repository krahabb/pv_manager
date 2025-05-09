from time import time as TIME_TS
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.util.unit_conversion import (
    EnergyConverter,
    PowerConverter,
)

from .. import const as pmc
from ..processors import BaseEnergyProcessor, EnergyInputMode, SourceType

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, Unpack

    from homeassistant.core import State

    from . import Controller
    from ..helpers.entity import EntityArgs
    from .off_grid_manager import Controller as OffGridManager

POWER_UNIT = hac.UnitOfPower.WATT
ENERGY_UNIT = hac.UnitOfEnergy.WATT_HOUR


class MeterStoreType(typing.TypedDict):
    time_ts: float | None
    energy: float


class BaseMeter(BaseEnergyProcessor):

    controller: "Final[OffGridManager]"
    metering_source: "Final[SourceType]"

    _SLOTS_ = (
        "controller",
        "metering_source",
    )

    def __init__(
        self,
        controller: "OffGridManager",
        metering_source: SourceType,
        config: "BaseEnergyProcessor.Config",
    ):
        self.controller = controller
        self.metering_source = metering_source
        super().__init__(metering_source, logger=controller, config=config)
        controller.energy_meters[metering_source] = self
        setattr(controller, f"{self.metering_source}_meter", self)

    def shutdown(self):
        super().shutdown()
        self.controller.energy_meters.pop(self.metering_source)  # type: ignore
        setattr(self.controller, f"{self.metering_source}_meter", None)
        self.controller = None  # type: ignore

    def load(self, data: MeterStoreType):
        self.output = data["energy"]

    def save(self):
        return MeterStoreType(
            {
                "time_ts": self.time_ts,
                "energy": self.output,
            }
        )


class BatteryMeter(BaseMeter):

    __slots__ = (
        "in_meter",
        "out_meter",
    )

    def __init__(self, controller: "OffGridManager", config: "BaseMeter.Config"):
        super().__init__(controller, SourceType.BATTERY, config)
        self.configure(EnergyInputMode.POWER)
        self.in_meter = BaseMeter(controller, SourceType.BATTERY_IN, {})
        self.out_meter = BaseMeter(controller, SourceType.BATTERY_OUT, {})
        self._energy_listeners.add(self._energy_callback)

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


class PvMeter(BaseMeter):
    def __init__(self, controller: "OffGridManager", config: "BaseMeter.Config"):
        super().__init__(controller, SourceType.PV, config)


class LoadMeter(BaseMeter):
    def __init__(self, controller: "OffGridManager", config: "BaseMeter.Config"):
        super().__init__(controller, SourceType.LOAD, config)


class LossesMeter(BaseMeter):

    controller: "OffGridManager"

    __slots__ = (
        "battery_energy",
        "battery_in_energy",
        "battery_out_energy",
        "load_energy",
        "pv_energy",
    )

    def __init__(self, controller: "OffGridManager", update_period_seconds: float):
        super().__init__(
            controller,
            SourceType.LOSSES,
            {"update_period_seconds": update_period_seconds},
        )

    async def async_start(self):
        self._losses_compute()
        self.time_ts = TIME_TS()
        await super().async_start()

    def _update_callback(self):
        time_ts = TIME_TS()
        controller = self.controller
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
        controller = self.controller
        self.battery_energy = battery = controller.battery_meter.output
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
            controller.battery_charge_estimate * controller.battery_voltage * 0.9
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
