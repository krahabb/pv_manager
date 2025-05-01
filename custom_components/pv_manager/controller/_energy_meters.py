import abc
from datetime import tzinfo
import enum
import time
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import (
    EnergyConverter,
    PowerConverter,
)

from .. import const as pmc
from .common import BaseEnergyProcessor, EnergyInputMode
from .common.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator

if typing.TYPE_CHECKING:
    from typing import Any, Final, NotRequired, Unpack

    from astral import sun
    from homeassistant.core import Event, HomeAssistant, State

    from ..helpers.entity import EntityArgs
    from .common.estimator_pvenergy_heuristic import PVEnergyEstimatorConfig
    from .off_grid_manager import Controller

POWER_UNIT = hac.UnitOfPower.WATT
ENERGY_UNIT = hac.UnitOfEnergy.WATT_HOUR

# define a common name so to eventually switch to time.monotonic for time integration
TIME_TS = time.time


class MeteringSource(enum.StrEnum):
    BATTERY = enum.auto()
    BATTERY_IN = enum.auto()
    BATTERY_OUT = enum.auto()
    LOAD = enum.auto()
    LOSSES = enum.auto()
    PV = enum.auto()


class MeterStoreType(typing.TypedDict):
    time_ts: float | None
    energy: float


class BaseMeter(BaseEnergyProcessor if typing.TYPE_CHECKING else object):
    """
    This class acts as an interface built on top of BaseEnergyProcessor
    but it has no concrete inheritance since we're using it as a
    'virtual base class-like' in our hierarchy involving Estimator
    which also is a (concrete) descendant of BaseEnergyProcessor.

    For actual energy meters not implemented through Estimator
    we'll then have to declare a concrete class (see ConcreteBaseMeter)
    """

    controller: "Final[Controller]"
    metering_source: "Final[MeteringSource]"

    _SLOTS_ = (
        "controller",
        "metering_source",
        "_state_convert_func",
    )

    def __init__(self, controller: "Controller", metering_source: MeteringSource):
        self.controller = controller
        self.metering_source = metering_source
        self._state_convert_func = self._state_convert_detect
        controller.energy_meters[metering_source] = self

    def shutdown(self):
        self.controller.energy_meters.pop(self.metering_source)
        setattr(self.controller, f"{self.metering_source}_meter", None)
        self.controller = None  # type: ignore
        # TODO: unregister track_state?

    def track_state(self, state: "State | None"):
        try:
            self.process(
                self._state_convert_func(
                    float(state.state),  # type: ignore
                    state.attributes["unit_of_measurement"],  # type: ignore
                    self.input_unit,
                ),
                state.last_updated_timestamp,  # type: ignore
            )  # type: ignore
        except Exception as e:
            # this is expected and silently managed when state == None or 'unknown'
            self.process(0, TIME_TS())
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.controller.log_exception(
                    self.controller.WARNING,
                    e,
                    "BaseMeter(%s).track_state (state:%s)",
                    self.metering_source.name,
                    state,
                )

    def _state_convert_detect(
        self, value: float, from_unit: str | None, to_unit: str | None
    ) -> float:
        """Installed as _state_convert_func at init time this will detect the type of observed entity
        by inspecting the unit and install the proper converter."""
        if from_unit in hac.UnitOfPower:
            self._state_convert_func = PowerConverter.convert
            self.configure(EnergyInputMode.POWER)
        elif from_unit in hac.UnitOfEnergy:
            self._state_convert_func = EnergyConverter.convert
            self.configure(EnergyInputMode.ENERGY)
        else:
            # TODO: raise issue?
            raise ValueError(f"Unsupported unit of measurement '{from_unit}'")
        return self._state_convert_func(value, from_unit, self.input_unit)

    def load(self, data: MeterStoreType):
        self.output = data["energy"]

    def save(self):
        return MeterStoreType(
            {
                "time_ts": self.time_ts,
                "energy": self.output,
            }
        )


class _BaseMeter(BaseMeter, BaseEnergyProcessor):
    """Concrete base class for BaseMeter(s) not implemented through estimators."""

    __slots__ = BaseMeter._SLOTS_

    def __init__(self, controller: "Controller", metering_source: MeteringSource):
        BaseEnergyProcessor.__init__(self)
        BaseMeter.__init__(self, controller, metering_source)

    def shutdown(self):
        BaseMeter.shutdown(self)
        BaseEnergyProcessor.shutdown(self)


class BatteryMeter(_BaseMeter):

    def __init__(self, controller: "Controller"):
        _BaseMeter.__init__(self, controller, MeteringSource.BATTERY)
        self.configure(EnergyInputMode.POWER)
        controller.battery_in_meter = _BaseMeter(controller, MeteringSource.BATTERY_IN)
        controller.battery_out_meter = _BaseMeter(
            controller, MeteringSource.BATTERY_OUT
        )
        self.energy_listeners.add(self._energy_callback)

    def _energy_callback(self, energy: float, time_ts: float):
        if energy > 0:
            in_out_meter = self.controller.battery_out_meter
        else:
            in_out_meter = self.controller.battery_in_meter
            energy = -energy
        in_out_meter.time_ts = time_ts
        in_out_meter.input = energy
        in_out_meter.output += energy
        for listener in in_out_meter.energy_listeners:
            listener(energy, time_ts)


class PvMeter(_BaseMeter):
    def __init__(self, controller: "Controller"):
        _BaseMeter.__init__(self, controller, MeteringSource.PV)


class LoadMeter(_BaseMeter):
    def __init__(self, controller: "Controller"):
        _BaseMeter.__init__(self, controller, MeteringSource.LOAD)


class LossesMeter(_BaseMeter):

    __slots__ = (
        "battery_energy",
        "battery_in_energy",
        "battery_out_energy",
        "load_energy",
        "pv_energy",
        "_callback_interval_ts",
        "_callback_unsub",
    )

    def __init__(self, controller: "Controller", callback_interval_ts: float):
        self._callback_interval_ts = callback_interval_ts
        self._callback_unsub = None
        _BaseMeter.__init__(self, controller, MeteringSource.LOSSES)

    def start(self):
        """Called after restoring data in the controller so to initialize incremental counters."""
        self._losses_compute()
        self.time_ts = TIME_TS()
        self._callback_unsub = self.controller.schedule_callback(
            self._callback_interval_ts, self._losses_callback
        )

    def stop(self):
        if self._callback_unsub:
            self._callback_unsub.cancel()
            self._callback_unsub = None

    @callback
    def _losses_callback(self):
        time_ts = TIME_TS()
        controller = self.controller
        self._callback_unsub = controller.schedule_callback(
            self._callback_interval_ts, self._losses_callback
        )
        battery_old = self.battery_energy
        battery_in_old = self.battery_in_energy
        battery_out_old = self.battery_out_energy
        load_old = self.load_energy
        pv_old = self.pv_energy
        losses_old = self.output
        # get the 'new' total
        controller.battery_meter.interpolate(time_ts)
        controller.load_meter.interpolate(time_ts)
        controller.pv_meter.interpolate(time_ts)
        self._losses_compute()
        # compute delta to get the average power in the sampling period
        # we don't check maximum_latency here since it has already been
        # managed in pc, load and battery meters
        d_losses = self.output - losses_old
        for energy_listener in self.energy_listeners:
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


class EstimatorMeter(BaseMeter):
    pass


class HeuristicPVEnergyEstimatorMeter(EstimatorMeter, HeuristicPVEnergyEstimator):

    __slots__ = EstimatorMeter._SLOTS_

    def __init__(
        self,
        controller,
        astral_observer: "sun.Observer",
        tzinfo: tzinfo,
        **kwargs: "Unpack[PVEnergyEstimatorConfig]",
    ):
        BaseMeter.__init__(self, controller, MeteringSource.PV)
        HeuristicPVEnergyEstimator.__init__(
            self, astral_observer=astral_observer, tzinfo=tzinfo, **kwargs
        )
