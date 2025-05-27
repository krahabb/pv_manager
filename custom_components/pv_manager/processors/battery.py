from dataclasses import dataclass
from datetime import datetime
from time import time
import typing

from homeassistant import const as hac
from homeassistant.core import callback

from . import EnergyBroadcast, GenericBroadcast, SignalEnergyProcessor
from ..helpers import datetime_from_epoch
from .estimator_energy import EnergyBalanceEstimator, SignalEnergyEstimator

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
        charge_processor: Final[SignalEnergyProcessor]
        energy_broadcast_in: Final[EnergyBroadcast]
        energy_broadcast_out: Final[EnergyBroadcast]
        battery_voltage: float | None
        battery_current: float | None
        battery_charge: float
        battery_capacity_estimate: float
        enery_in: float
        energy_out: float
        _current_convert: SignalEnergyProcessor.ConvertFuncType
        _current_unit: str
        _voltage_convert: SignalEnergyProcessor.ConvertFuncType
        _voltage_unit: str

    _SLOTS_ = (
        # config
        "battery_voltage_entity_id",
        "battery_current_entity_id",
        "battery_charge_entity_id",
        "battery_capacity",
        # references
        "charge_processor",
        "energy_broadcast_in",
        "energy_broadcast_out",
        "battery_charge_broadcast",
        # state
        "battery_voltage",
        "battery_current",
        "battery_charge",
        "battery_capacity_estimate",
        "energy_in",
        "energy_out",
        # misc
        "_current_convert",
        "_current_unit",
        "_voltage_convert",
        "_voltage_unit",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        # typically we'd want negative limit to be equal to the positive one
        if self.input_min is SignalEnergyProcessor.SAFE_MINIMUM_POWER_DISABLED:
            self.input_min = -self.input_max
            self.energy_min = self.energy_max
        config = self.config
        self.battery_voltage_entity_id = config.get("battery_voltage_entity_id")
        self.battery_current_entity_id = config.get("battery_current_entity_id")
        self.battery_charge_entity_id = config.get("battery_charge_entity_id")
        self.battery_capacity = config.get("battery_capacity", 0)
        # Actual class design considers we're monitoring voltage and current to extract
        # energy and power but we might want a flexible design where we monitor
        # power and current or power and charge. This will need to be investigated (TODO).
        self.configure(SignalEnergyProcessor.Unit.POWER)
        self.charge_processor = SignalEnergyProcessor(
            "charge_processor",
            logger=self,
            config={
                "maximum_latency_seconds": self.maximum_latency_ts,
            },
        )
        self.energy_broadcast_in = EnergyBroadcast(
            BatteryProcessor.SourceType.BATTERY_IN, logger=self
        )
        self.energy_broadcast_out = EnergyBroadcast(
            BatteryProcessor.SourceType.BATTERY_OUT, logger=self
        )

        self.battery_voltage = None
        self.battery_current = None
        self.battery_charge = 0
        self.battery_capacity_estimate = 0
        self.energy_in = 0
        self.energy_out = 0

    @typing.override
    async def async_start(self):
        if self.battery_current_entity_id:
            unit = SignalEnergyProcessor.Unit.CURRENT
            self._current_convert = unit.convert
            self._current_unit = unit.default
            self.charge_processor.configure(SignalEnergyProcessor.Unit.CURRENT)
            self.track_state(self.battery_current_entity_id, self._current_callback)
        if self.battery_voltage_entity_id:
            unit = SignalEnergyProcessor.Unit.VOLTAGE
            self._voltage_convert = unit.convert
            self._voltage_unit = unit.default
            self.track_state(self.battery_voltage_entity_id, self._voltage_callback)

    def shutdown(self):
        self.charge_processor.shutdown()
        self.energy_broadcast_in.shutdown()
        self.energy_broadcast_out.shutdown()
        super().shutdown()

    @typing.override
    def restore(self, data: "StoreType"):
        with self.exception_warning("loading meter data"):
            self.charge_processor.output = -data["charge_estimate"]
            self.battery_capacity_estimate = data["capacity_estimate"]

    @typing.override
    def store(self) -> "StoreType":
        return {
            "charge_estimate": -self.charge_processor.output,
            "capacity_estimate": self.battery_capacity_estimate,
        }

    def update(self, time_ts):
        self.charge_processor.update(time_ts)
        return super().update(time_ts)

    @typing.override
    def process(self, input: float | None, time_ts: float) -> float | None:
        energy = SignalEnergyProcessor.process(self, input, time_ts)
        if energy:
            if energy > 0:
                self.energy_out += energy
                for listener in self.energy_broadcast_out.energy_listeners:
                    listener(energy, time_ts)
            else:
                energy = -energy
                self.energy_in += energy
                for listener in self.energy_broadcast_in.energy_listeners:
                    listener(energy, time_ts)
        return energy

    @callback
    def _current_callback(
        self, event: "Event[EventStateChangedData] | BatteryProcessor.Event"
    ):
        battery_current = None
        time_ts = event.time_fired_timestamp
        state = event.data["new_state"]
        try:
            battery_current = self._current_convert(
                float(state.state),  # type: ignore
                state.attributes["unit_of_measurement"],  # type: ignore
                self._current_unit,
            )
            self.process(
                self.battery_voltage * battery_current, time_ts  # type: ignore
            )
        except Exception as e:
            # do some checks: the idea is to foward a 'signal disconnect'
            # wherever is needed
            self.process(None, time_ts)
            if (
                battery_current is None
            ):  # error thrown in conversion like state unavailable or so
                if state and state.state not in (
                    hac.STATE_UNKNOWN,
                    hac.STATE_UNAVAILABLE,
                ):
                    self.log_exception(
                        self.WARNING, e, "_current_callback(state:%s)", state
                    )
            elif self.battery_voltage is not None:
                # unexpected though
                self.log_exception(self.WARNING, e, "_current_callback")

        self.battery_current = battery_current
        self.charge_processor.process(battery_current, time_ts)

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
                self._voltage_unit,
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


class BatteryEstimator(EnergyBalanceEstimator, SignalEnergyEstimator, BatteryProcessor):  # type: ignore

    @dataclass(slots=True)
    class Sample(SignalEnergyEstimator.Sample):
        battery_energy_in: float
        battery_energy_out: float
        battery_charge_begin: float
        battery_charge_in: float
        battery_charge_out: float
        source_energy: float
        load_energy: float

        def __init__(self, time_ts: float, estimator: "BatteryEstimator", /):
            SignalEnergyEstimator.Sample.__init__(self, time_ts, estimator)
            self.battery_energy_in = self.battery_energy_out = 0
            self.battery_charge_begin = -estimator.charge_processor.output
            self.battery_charge_in = self.battery_charge_out = 0
            self.source_energy = estimator.production_estimator.output
            self.load_energy = estimator.consumption_estimator.output

    class Forecast(EnergyBalanceEstimator.Forecast):

        losses: float
        losses_min: float
        losses_max: float

        __slots__ = (
            "losses",
            "losses_min",
            "losses_max",
        )

        def __init__(self, time_begin_ts: int, time_end_ts: int, /):
            EnergyBalanceEstimator.Forecast.__init__(self, time_begin_ts, time_end_ts)
            self.losses = 0
            self.losses_min = 0
            self.losses_max = 0

        def add(self, forecast: "BatteryEstimator.Forecast", /):
            EnergyBalanceEstimator.Forecast.add(self, forecast)
            self.losses += forecast.losses
            self.losses_min += forecast.losses_min
            self.losses_max += forecast.losses_max

        def addmul(self, forecast: "BatteryEstimator.Forecast", ratio: float, /):
            EnergyBalanceEstimator.Forecast.addmul(self, forecast, ratio)
            self.losses += forecast.losses * ratio
            self.losses_min += forecast.losses_min * ratio
            self.losses_max += forecast.losses_max * ratio

    class _FakeEstimator(EnergyBalanceEstimator._FakeEstimator, SignalEnergyEstimator):
        def _check_sample_curr(self, time_ts: float, /):
            self._sample_curr.__init__(time_ts, self)
            return self._sample_curr

        def _process_sample_curr(
            self, sample_curr: SignalEnergyEstimator.Sample, time_ts: float, /
        ):
            sample_curr.__init__(time_ts, self)
            return sample_curr

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

        # (override base typehint)
        config: Config
        forecasts: Final[list[Forecast]]  # type: ignore
        _forecasts_recycle: Final[list[Forecast]]  # type: ignore
        production_estimator: SignalEnergyEstimator
        consumption_estimator: SignalEnergyEstimator
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

        def _check_sample_curr(self, time_ts: float, /) -> Sample:
            return super()._check_sample_curr(time_ts)  # type: ignore

    DEFAULT_NAME = "Battery estimator"

    _FAKE_ESTIMATOR = _FakeEstimator(
        "",
        config={
            "history_duration_days": 0,
            "observation_duration_minutes": 0,
            "sampling_interval_minutes": SignalEnergyEstimator.SAMPLING_INTERVAL_MODULO,
        },
    )

    _SLOTS_ = (
        "source",
        "load",
        "losses",
        "conversion_yield",
        "conversion_yield_avg",
        "conversion_yield_total",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self.source = 0
        self.load = 0
        self.losses = 0
        self.conversion_yield = 1
        self.conversion_yield_avg = 1
        self.conversion_yield_total = 1
        self.charge_processor.listen_energy(self._charge_callback)

    # interface: Estimator
    async def async_start(self):
        await super().async_start()
        # since we're not (yet?) restoring history for battery estimation we have to
        # fix today_energy since it gets lost when unloading the entry.
        # We could eventually store the estimator state but that's another story
        # since most of its state is initialized in async_init and that's actually called
        # after restoring (we have to think about this. TODO)
        # BEWARE: ensure the battery estimator is started after the production/consumption
        self.today_energy = self.consumption_estimator.today_energy
        if self.conversion_yield_avg:
            self.today_energy /= self.conversion_yield_avg
        self.today_energy -= self.production_estimator.today_energy

    def get_state_dict(self):
        """Returns a synthetic state string for the estimator.
        Used for debugging purposes."""
        return super().get_state_dict() | {
            "battery_energy": self.output,
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
            self.output = data["battery_energy"]
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
        # self.energy, energy_in, energy_out are updated at every input while
        # source, load, losses are only updated at every sample (sample_curr) termination
        # so we 'pull-back' the battery energy currently accumulating in the sample to align
        # the storage
        sample_curr = self._sample_curr
        return super().store() | {  # type: ignore
            "battery_energy": self.output
            - (sample_curr.battery_energy_out - sample_curr.battery_energy_in),
            "battery_energy_in": self.energy_in - sample_curr.battery_energy_in,
            "battery_energy_out": self.energy_out - sample_curr.battery_energy_out,
            "source_energy": self.source,
            "load_energy": self.load,
            "losses_energy": self.losses,
            "conversion_yield": self.conversion_yield,
            "conversion_yield_avg": self.conversion_yield_avg,
            "conversion_yield_total": self.conversion_yield_total,
        }

    @typing.override
    def process(self, input: float | None, time_ts: float) -> float | None:
        sample_curr = self._check_sample_curr(time_ts)
        energy = SignalEnergyProcessor.process(self, input, time_ts)
        if energy is not None:
            # don't accumulate energy in sample_curr..it'll be computed then
            sample_curr.samples += 1
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
        return energy

    def _charge_callback(self, charge: float, time_ts: float, /):
        # No need to call self._check_sample_curr
        # since process has been surely called in the same context
        if charge > 0:
            self._sample_curr.battery_charge_out += charge
        else:
            self._sample_curr.battery_charge_in += charge

    @typing.override
    def _process_sample_curr(self, sample_curr: Sample, time_ts: float, /):
        """Use this callback to flush/store samples or so and update estimates."""
        if sample_curr.samples:
            time_end_curr_ts = sample_curr.time_end_ts
            battery = sample_curr.battery_energy_out - sample_curr.battery_energy_in
            self.today_energy += battery
            self.observations_per_sample_avg = (
                self.observations_per_sample_avg * EnergyBalanceEstimator.OPS_DECAY
                + sample_curr.samples * (1 - EnergyBalanceEstimator.OPS_DECAY)
            )
            # TODO: update any parameter (likely yields...) from new sample (sample_curr)
            # before discarding the sample...(or store it)
            sample_curr.source_energy = source = (
                self.production_estimator.output - sample_curr.source_energy
            )
            self.source += source
            sample_curr.load_energy = load = (
                self.consumption_estimator.output - sample_curr.load_energy
            )
            self.load += load
            losses = source + battery - load
            self.losses += losses
            try:
                if load:
                    self.conversion_yield = load / (load + losses)
                    self.conversion_yield_avg = (
                        self.conversion_yield_avg * EnergyBalanceEstimator.OPS_DECAY
                        + self.conversion_yield * (1 - EnergyBalanceEstimator.OPS_DECAY)
                    )
                self.conversion_yield_total = self.load / (self.load + self.losses)
            except:
                # just protect in case something goes to 0
                pass
        else:
            # trigger a reset since previous sample did not collect any data
            time_end_curr_ts = 0

        # since we're discarding the old sample we try this hack
        sample_curr.__init__(time_ts, self)
        if time_end_curr_ts != sample_curr.time_begin_ts:
            # when samples are not consecutive
            self.reset()

        # make sure those are time aligned else the estimate calculations will mess up
        self.production_estimator._check_sample_curr(time_ts)
        self.consumption_estimator._check_sample_curr(time_ts)
        self.update_estimate()

        return sample_curr

    @typing.override
    def update_estimate(self):
        # bypass EnergyBalanceEstimator raw estimate
        SignalEnergyEstimator.update_estimate(self)

    @typing.override
    def _ensure_forecasts(self, count: int, /):
        estimation_time_ts = self.estimation_time_ts
        sampling_interval_ts = self.sampling_interval_ts
        forecasts = self.forecasts
        _forecasts_recycle = self._forecasts_recycle
        production_estimator = self.production_estimator
        consumption_estimator = self.consumption_estimator

        time_ts = estimation_time_ts + len(forecasts) * sampling_interval_ts
        time_end_ts = estimation_time_ts + count * sampling_interval_ts

        while time_ts < time_end_ts:
            time_next_ts = time_ts + sampling_interval_ts
            try:
                _f = _forecasts_recycle.pop()
                _f.__init__(time_ts, time_next_ts)
            except IndexError:
                _f = self.__class__.Forecast(time_ts, time_next_ts)

            _f_p = production_estimator.get_forecast(time_ts, time_next_ts)
            _f.production = _f_p.energy
            _f.production_min = _f_p.energy_min
            _f.production_max = _f_p.energy_max

            _f_c = consumption_estimator.get_forecast(time_ts, time_next_ts)
            _f.consumption = _f_c.energy
            _f.consumption_min = _f_c.energy_min
            _f.consumption_max = _f_c.energy_max

            losses_ratio = (
                (1 / self.conversion_yield_avg) - 1 if self.conversion_yield_avg else 0
            )
            _f.losses = _f.consumption * losses_ratio
            _f.losses_min = _f.consumption_min * losses_ratio
            _f.losses_max = _f.consumption_max * losses_ratio

            _f.energy = _f.consumption + _f.losses - _f.production
            _f.energy_min = _f.consumption_min + _f.losses_min - _f.production_max
            _f.energy_max = _f.consumption_max + _f.losses_max - _f.production_min

            forecasts.append(_f)
            time_ts = time_next_ts

    # interface: EnergyBalanceEstimator
    @typing.override
    def connect_production(self, estimator: "SignalEnergyEstimator"):
        self.production_estimator = estimator

    @typing.override
    def connect_consumption(self, estimator: "SignalEnergyEstimator"):
        self.consumption_estimator = estimator
