from dataclasses import dataclass
from datetime import datetime
import typing

from homeassistant import const as hac
from homeassistant.core import callback

from . import EnergyBroadcast, GenericBroadcast, SignalEnergyProcessor
from ..helpers import datetime_from_epoch
from .estimator_energy import SAMPLING_INTERVAL_MODULO, EnergyBalanceEstimator

if typing.TYPE_CHECKING:
    from typing import Callable, Final, NotRequired, Unpack

    from homeassistant.core import Event, EventStateChangedData


class BatteryProcessor(SignalEnergyProcessor):

    if typing.TYPE_CHECKING:

        class Config(SignalEnergyProcessor.Config):
            battery_voltage_entity_id: NotRequired[str]
            battery_current_entity_id: NotRequired[str]
            battery_charge_entity_id: NotRequired[str]
            battery_capacity: NotRequired[float]

        class Args(SignalEnergyProcessor.Args):
            config: "BatteryProcessor.Config"

        class StoreType(SignalEnergyProcessor.StoreType):
            charge_estimate: float
            capacity_estimate: float

        config: Config
        battery_voltage_entity_id: Final[str | None]
        battery_current_entity_id: Final[str | None]
        battery_charge_entity_id: Final[str | None]
        battery_capacity: Final[float]
        energy_broadcast_in: Final[EnergyBroadcast]
        energy_broadcast_out: Final[EnergyBroadcast]
        battery_charge_broadcast: Final[GenericBroadcast]
        battery_voltage: float | None
        battery_current: float | None
        _battery_current_last_ts: float
        battery_charge: float
        battery_charge_estimate: float
        battery_capacity_estimate: float
        enery_in: float
        energy_out: float
        _current_convert: SignalEnergyProcessor.ConvertFuncType
        _voltage_convert: SignalEnergyProcessor.ConvertFuncType

    _SLOTS_ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        # references
        "energy_broadcast_in",
        "energy_broadcast_out",
        # state
        "battery_voltage",
        "battery_current",
        "_battery_current_last_ts",
        "battery_charge",
        "battery_charge_estimate",
        "battery_capacity_estimate",
        "energy_in",
        "energy_out",
        # misc
        "_current_convert",
        "_voltage_convert",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        # typically we'd want negative limit to be equal to the positive one
        if self.input_min is SignalEnergyProcessor.SAFE_MINIMUM_POWER_DISABLED:
            self.input_min = -self.input_max
            self.energy_min = self.energy_max
        self.energy_broadcast_in = EnergyBroadcast(
            BatteryProcessor.SourceType.BATTERY_IN, logger=self
        )
        self.energy_broadcast_out = EnergyBroadcast(
            BatteryProcessor.SourceType.BATTERY_OUT, logger=self
        )
        self.battery_charge_broadcast = GenericBroadcast("battery_charge", logger=self)
        config = self.config
        self.battery_voltage_entity_id = config.get("battery_voltage_entity_id")
        self.battery_current_entity_id = config.get("battery_current_entity_id")
        self.battery_charge_entity_id = config.get("battery_charge_entity_id")
        self.battery_capacity = config.get("battery_capacity", 0)

        self.battery_voltage = None
        self.battery_current = None
        self._battery_current_last_ts = 0
        self.battery_charge = 0
        self.battery_charge_estimate = 0
        self.battery_capacity_estimate = 0
        self.energy_in = 0
        self.energy_out = 0
        self.energy_listeners.add(self._energy_callback)

    @typing.override
    async def async_start(self):
        if self.battery_current_entity_id:
            self._current_convert = (
                SignalEnergyProcessor.Converter.ElectricCurrentConverter.convert
            )
            self.track_state(self.battery_current_entity_id, self._current_callback)
        if self.battery_voltage_entity_id:
            self._voltage_convert = (
                SignalEnergyProcessor.Converter.ElectricPotentialConverter.convert
            )
            self.track_state(self.battery_voltage_entity_id, self._voltage_callback)
        if self.battery_charge_entity_id:
            self.track_state(self.battery_charge_entity_id, self._charge_callback)

    def shutdown(self):
        self.energy_broadcast_in.shutdown()
        self.energy_broadcast_out.shutdown()
        super().shutdown()

    @typing.override
    def restore(self, data: "StoreType"):
        with self.exception_warning("loading meter data"):
            self.battery_charge_estimate = data["charge_estimate"]
            self.battery_capacity_estimate = data["capacity_estimate"]

    @typing.override
    def store(self) -> "StoreType":
        return {
            "charge_estimate": self.battery_charge_estimate,
            "capacity_estimate": self.battery_capacity_estimate,
        }

    def _energy_callback(self, energy: float, time_ts: float):
        if energy > 0:
            self.energy_out += energy
            for listener in self.energy_broadcast_out.energy_listeners:
                listener(energy, time_ts)
        else:
            energy = -energy
            self.energy_in += energy
            for listener in self.energy_broadcast_in.energy_listeners:
                listener(energy, time_ts)

    @callback
    def _voltage_callback(
        self, event: "Event[EventStateChangedData] | BatteryProcessor.Event"
    ):
        battery_voltage = None
        try:
            state = event.data["new_state"]
            battery_voltage = self._voltage_convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                SignalEnergyProcessor.Unit.VOLTAGE_UNIT,
            )
            self.process(
                battery_voltage * self.battery_current, event.time_fired_timestamp  # type: ignore
            )
        except Exception as e:
            self.process(None, event.time_fired_timestamp)
            if (
                state
                and (self.battery_current is not None)
                and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE)
            ):
                self.log_exception(
                    self.WARNING, e, "_voltage_callback(state:%s)", state
                )
        self.battery_voltage = battery_voltage

    @callback
    def _current_callback(
        self, event: "Event[EventStateChangedData] | BatteryProcessor.Event"
    ):
        time_ts = event.time_fired_timestamp
        try:
            # left rectangle integration
            # We assume 'generator convention' for current
            # i.e. positive current = discharging
            d_time = time_ts - self._battery_current_last_ts
            if 0 < d_time < self.maximum_latency_ts:
                charge_out = self.battery_current * d_time / 3600  # type: ignore
                self.battery_charge_estimate -= charge_out
                if self.battery_charge_estimate > self.battery_capacity:
                    self.battery_charge_estimate = self.battery_capacity
                elif self.battery_charge_estimate < 0:
                    self.battery_charge_estimate = 0
                self.battery_charge_broadcast.broadcast(self.battery_charge_estimate)
            else:
                # signal if the current updates are lacking since
                # we could have voltage updates in between that would
                # prevent our warning to trigger. The other branch of the condition
                # doesn't need to reset the warning since it would be done in
                # the process method
                if not self.warning_maximum_latency.on:
                    self.warning_maximum_latency.toggle()

        except Exception as e:
            # self.battery_current == None is expected
            if self.battery_current is not None:
                self.log_exception(self.WARNING, e, "_current_callback")

        battery_current = None
        try:
            state = event.data["new_state"]
            battery_current = self._current_convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                SignalEnergyProcessor.Unit.CURRENT_UNIT,
            )
            self.process(battery_current * self.battery_voltage, time_ts)  # type: ignore
        except Exception as e:
            self.process(None, time_ts)
            if (
                state
                and (self.battery_voltage is not None)
                and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE)
            ):
                self.log_exception(
                    self.WARNING, e, "_current_callback(state:%s)", state
                )

        self.battery_current = battery_current
        self._battery_current_last_ts = time_ts

    @callback
    def _charge_callback(
        self, event: "Event[EventStateChangedData] | BatteryProcessor.Event"
    ):
        pass


class BatteryEstimator(EnergyBalanceEstimator, BatteryProcessor):

    @dataclass(slots=True)
    class Sample:
        time: "Final[datetime]"
        """The sample time start"""
        time_begin_ts: "Final[int]"
        time_end_ts: "Final[int]"
        battery_energy: float
        battery_energy_in: float
        battery_energy_out: float
        source_energy: float
        load_energy: float
        samples: int

        def __init__(
            self, energy: float, time_ts: float, estimator: "BatteryEstimator"
        ):
            time_ts = int(time_ts)
            time_ts -= time_ts % estimator.sampling_interval_ts
            self.time = datetime_from_epoch(time_ts)
            self.time_begin_ts = time_ts
            self.time_end_ts = time_ts + estimator.sampling_interval_ts
            # TODO: grab initial values of production and consumption
            self.battery_energy = energy
            self.battery_energy_in = 0
            self.battery_energy_out = 0
            self.source_energy = (
                estimator.production_estimator.energy
                if estimator.production_estimator
                else 0
            )
            self.load_energy = (
                estimator.consumption_estimator.energy
                if estimator.consumption_estimator
                else 0
            )
            self.samples = 1

    class Forecast(EnergyBalanceEstimator.Forecast):

        __slots__ = ()

    if typing.TYPE_CHECKING:

        class Config(EnergyBalanceEstimator.Config, BatteryProcessor.Config):
            pass

        class Args(EnergyBalanceEstimator.Args, BatteryProcessor.Args):
            config: "BatteryEstimator.Config"

        class StoreType(BatteryProcessor.StoreType):
            battery_energy: float
            battery_energy_in: float
            battery_energy_out: float
            source_energy: float
            load_energy: float
            losses_energy: float
            conversion_yield: float
            conversion_yield_avg: float
            conversion_yield_total: float

        config: Config
        forecasts: list[Forecast]
        _sample_curr: Sample

        # losses are computed as: source + battery - load
        # where:
        # - source: generated energy
        # - battery: energy from the battery to the load (i.e. positive discharging)
        # - load: energy output from inverter (i.e. measure of all the loads)
        source: float
        load: float
        losses: float

        # conversion_yield is the ratio: losses / load and gives a raw figure of
        # inverter and cabling losses
        conversion_yield: (
            float  # computed over a single sample (i.e. over sampling_interval)
        )
        conversion_yield_avg: float  # avg of conversion_yield
        conversion_yield_total: float  # computed over the whole accumulation

    DEFAULT_NAME = "Battery estimator"

    _SLOTS_ = ()

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self.source = 0
        self.load = 0
        self.losses = 0
        self.conversion_yield = 1
        self.conversion_yield_avg = 1
        self.conversion_yield_total = 1

    # interface: Estimator
    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "battery_energy": self.energy,
            "battery_energy_in": self.energy_in,
            "battery_energy_out": self.energy_out,
            "source_energy": self.source,
            "load_energy": self.load,
            "losses_energy": self.losses,
            "conversion_yield": self.conversion_yield,
            "conversion_yield_avg": self.conversion_yield_avg,
            "conversion_yield_total": self.conversion_yield_total,
        }

    @typing.override
    def restore(self, data: "StoreType"):
        super().restore(data)
        with self.exception_warning("loading meter data"):
            self.energy = data["battery_energy"]
            self.energy_in = data["battery_energy_in"]
            self.energy_out = data["battery_energy_out"]
            self.source = data["source_energy"]
            self.load = data["load_energy"]
            self.losses = data["losses_energy"]
            self.conversion_yield = data["conversion_yield"]
            self.conversion_yield_avg = data["conversion_yield_avg"]
            self.conversion_yield_total = data["conversion_yield_total"]

    @typing.override
    def store(self) -> "StoreType":
        try:
            # self.energy, energy_in, energy_out are updated at every input while
            # source, load, losses are only updated at every sample (sample_curr) termination
            # so we 'pull-back' the battery energy currently accumulating in the sample to align
            # the storage
            sample_curr = self._sample_curr
            battery_energy = self.energy - sample_curr.battery_energy
            battery_energy_in = self.energy_in - sample_curr.battery_energy_in
            battery_energy_out = self.energy_out - sample_curr.battery_energy_out
        except AttributeError:
            battery_energy = self.energy
            battery_energy_in = self.energy_in
            battery_energy_out = self.energy_out

        return super().store() | {  # type: ignore
            "battery_energy": battery_energy,
            "battery_energy_in": battery_energy_in,
            "battery_energy_out": battery_energy_out,
            "source_energy": self.source,
            "load_energy": self.load,
            "losses_energy": self.losses,
            "conversion_yield": self.conversion_yield,
            "conversion_yield_avg": self.conversion_yield_avg,
            "conversion_yield_total": self.conversion_yield_total,
        }

    @typing.override
    def _energy_callback(self, energy: float, time_ts: float):

        try:
            sample_curr = self._sample_curr
            if sample_curr.time_begin_ts < time_ts < sample_curr.time_end_ts:
                sample_curr.battery_energy += energy
                sample_curr.samples += 1
            else:
                self.today_energy += sample_curr.battery_energy
                self.observations_per_sample_avg = (
                    self.observations_per_sample_avg * EnergyBalanceEstimator.OPS_DECAY
                    + sample_curr.samples * (1 - EnergyBalanceEstimator.OPS_DECAY)
                )
                # TODO: update any parameter (likely yields...) from new sample (sample_curr)
                # before discarding the sample...(or store it)
                if self.production_estimator:
                    sample_curr.source_energy = source = (
                        self.production_estimator.energy - sample_curr.source_energy
                    )
                    self.source += source
                else:
                    source = 0
                if self.consumption_estimator:
                    sample_curr.load_energy = load = (
                        self.consumption_estimator.energy - sample_curr.load_energy
                    )
                    self.load += load
                    losses = source + sample_curr.battery_energy - load
                    self.losses += losses
                    try:
                        if load:
                            self.conversion_yield = load / (load + losses)
                            self.conversion_yield_avg = (
                                self.conversion_yield_avg
                                * EnergyBalanceEstimator.OPS_DECAY
                                + self.conversion_yield
                                * (1 - EnergyBalanceEstimator.OPS_DECAY)
                            )
                        self.conversion_yield_total = self.load / (
                            self.load + self.losses
                        )
                    except:
                        # just protect in case something goes to 0
                        pass

                self._sample_curr = sample_curr = BatteryEstimator.Sample(
                    energy, time_ts, self
                )
                self.estimation_time_ts = sample_curr.time_begin_ts
                if self.estimation_time_ts >= self.tomorrow_ts:
                    # new day
                    self._observed_energy_daystart(self.estimation_time_ts)

                self.update_estimate()

        except AttributeError as error:
            assert error.name == "_sample_curr"
            self._sample_curr = sample_curr = BatteryEstimator.Sample(
                energy, time_ts, self
            )

        if energy > 0:
            sample_curr.battery_energy_out += energy
            self.energy_out += energy
            for listener in self.energy_broadcast_out.energy_listeners:
                listener(energy, time_ts)
        else:
            energy = -energy
            sample_curr.battery_energy_in += energy
            self.energy_in += energy
            for listener in self.energy_broadcast_in.energy_listeners:
                listener(energy, time_ts)

    @typing.override
    def update_estimate(self):

        production_estimator = self.production_estimator
        consumption_estimator = self.consumption_estimator

        forecast_ts = self.estimation_time_ts
        sampling_interval_ts = self.sampling_interval_ts
        for forecast in self.forecasts:
            forecast_next_ts = forecast_ts + sampling_interval_ts
            forecast.time_ts = forecast_ts
            if production_estimator:
                forecast.production = production_estimator.get_estimated_energy(
                    forecast_ts, forecast_next_ts
                )
            else:
                forecast.production = 0
            if consumption_estimator:
                forecast.consumption = consumption_estimator.get_estimated_energy(
                    forecast_ts, forecast_next_ts
                )
                if self.conversion_yield_avg:
                    forecast.consumption = (
                        forecast.consumption / self.conversion_yield_avg
                    )
            else:
                forecast.consumption = 0
            forecast_ts = forecast_next_ts

        for listener in self._update_listeners:
            listener(self)
