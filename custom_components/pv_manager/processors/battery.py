from dataclasses import dataclass
from datetime import datetime
from time import time
import typing

from homeassistant import const as hac
from homeassistant.core import callback

from . import EnergyBroadcast, SignalEnergyProcessor
from ..helpers import datetime_from_epoch, validation as hv
from ..helpers.dataattr import DataAttr, DataAttrParam
from ..sensor import Sensor
from .estimator_energy import EnergyBalanceEstimator, SignalEnergyEstimator

if typing.TYPE_CHECKING:
    from typing import Callable, Final, NotRequired, Self, Unpack

    from homeassistant.core import Event, EventStateChangedData

    from .. import const as pmc


class BatteryProcessor(SignalEnergyProcessor):

    if typing.TYPE_CHECKING:

        class Config(SignalEnergyProcessor.Config):
            battery_voltage_entity_id: NotRequired[str]
            battery_current_entity_id: NotRequired[str]
            battery_charge_entity_id: NotRequired[str]
            battery_capacity: NotRequired[float]

        class Args(SignalEnergyProcessor.Args):
            config: "BatteryProcessor.Config"

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
        _current_convert: SignalEnergyProcessor.ConvertFuncType
        _current_unit: str
        _voltage_convert: SignalEnergyProcessor.ConvertFuncType
        _voltage_unit: str

    energy_in: DataAttr[float, DataAttrParam.stored] = 0
    energy_out: DataAttr[float, DataAttrParam.stored] = 0
    charge: DataAttr[float, DataAttrParam.stored] = 0
    charge_in: DataAttr[float, DataAttrParam.stored] = 0
    charge_out: DataAttr[float, DataAttrParam.stored] = 0
    capacity_estimate: DataAttr[float, DataAttrParam.stored] = 0

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
        # misc
        "_current_convert",
        "_current_unit",
        "_voltage_convert",
        "_voltage_unit",
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None", /) -> "pmc.ConfigSchema":
        _config = config or {}
        schema = {
            hv.req_config("battery_voltage_entity_id", _config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.VOLTAGE
            ),
            hv.req_config("battery_current_entity_id", _config): hv.sensor_selector(
                device_class=Sensor.DeviceClass.CURRENT
            ),
            hv.req_config("battery_capacity", _config): hv.positive_number_selector(
                unit_of_measurement="Ah"
            ),
        } | super().get_config_schema(config)
        del schema["source_entity_id"] # type: ignore
        return schema

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
                "maximum_latency": self.maximum_latency_ts,
            },
        )
        self.charge_processor.listen_energy(self._charge_callback)
        self.energy_broadcast_in = EnergyBroadcast(
            BatteryProcessor.SourceType.BATTERY_IN, logger=self
        )
        self.energy_broadcast_out = EnergyBroadcast(
            BatteryProcessor.SourceType.BATTERY_OUT, logger=self
        )
        self.battery_voltage = None
        self.battery_current = None

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
            self.disconnect(time_ts)
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
            self.disconnect(event.time_fired_timestamp)
            if (
                state
                and (self.battery_current is not None)
                and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE)
            ):
                self.log_exception(
                    self.WARNING, e, "_voltage_callback(state:%s)", state
                )
        self.battery_voltage = battery_voltage

    def _charge_callback(self, charge: float, time_ts: float, /):
        self.charge -= charge
        if charge > 0:
            self.charge_out += charge
        else:
            self.charge_in -= charge


class BatteryEstimator(EnergyBalanceEstimator, SignalEnergyEstimator, BatteryProcessor):  # type: ignore

    @dataclass(slots=True, eq=False)
    class Sample(SignalEnergyEstimator.Sample):
        energy_in: float
        energy_out: float
        charge_in: float
        charge_out: float
        production: float
        consumption: float

        def __init__(self, time_ts: float, estimator: "BatteryEstimator", /):
            SignalEnergyEstimator.Sample.__init__(self, time_ts, estimator)
            self.energy_in = self.energy_out = 0
            self.charge_in = self.charge_out = 0
            self.production = self.consumption = 0

    class Forecast(EnergyBalanceEstimator.Forecast):

        charge: DataAttr[float] = 0
        charge_min: DataAttr[float] = 0
        charge_max: DataAttr[float] = 0
        losses: DataAttr[float] = 0
        losses_min: DataAttr[float] = 0
        losses_max: DataAttr[float] = 0

        def add(self, forecast: "Self", /):
            EnergyBalanceEstimator.Forecast.add(self, forecast)
            self.charge += forecast.charge
            self.charge_min += forecast.charge_min
            self.charge_max += forecast.charge_max
            self.losses += forecast.losses
            self.losses_min += forecast.losses_min
            self.losses_max += forecast.losses_max

        def addmul(self, forecast: "Self", ratio: float, /):
            EnergyBalanceEstimator.Forecast.addmul(self, forecast, ratio)
            self.charge += forecast.charge * ratio
            self.charge_min += forecast.charge_min * ratio
            self.charge_max += forecast.charge_max * ratio
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

        # (override base typehint)
        config: Config
        forecasts: Final[list[Forecast]]  # type: ignore
        _forecasts_recycle: Final[list[Forecast]]  # type: ignore
        production_estimator: SignalEnergyEstimator
        consumption_estimator: SignalEnergyEstimator
        _sample_curr: Sample

        @typing.override
        def get_forecast(self, time_begin_ts: int, time_end_ts: int, /) -> Forecast: # type: ignore
            pass

        @typing.override
        def _check_sample_curr(self, time_ts: float, /) -> Sample: # type: ignore
            pass

    DEFAULT_NAME = "Battery estimator"

    _FAKE_ESTIMATOR = _FakeEstimator(
        "",
        config={
            "history_duration_days": 0,
            "observation_duration_minutes": 0,
            "sampling_interval_minutes": SignalEnergyEstimator.SAMPLING_INTERVAL_MODULO,
        },
    )

    # losses are computed as: production + battery - consumption
    # where:
    # - production: generated (pv) energy
    # - battery: energy from the battery to the load (i.e. positive discharging)
    # - consumption: energy output from inverter (i.e. measure of all the loads)
    production: DataAttr[float, DataAttrParam.stored] = 0
    consumption: DataAttr[float, DataAttrParam.stored] = 0
    losses: DataAttr[float, DataAttrParam.stored] = 0

    # conversion_yield is the ratio: losses / load and gives a raw figure of
    # inverter and cabling losses
    conversion_yield: DataAttr[float | None, DataAttrParam.stored] = None
    conversion_yield_avg: DataAttr[float, DataAttrParam.stored] = 1
    conversion_yield_total: DataAttr[float, DataAttrParam.stored] = 1

    # TODO: move the stored data to a class hierarchy among processors so
    # that's easier to serialize ( store, restore, state_dict)
    charging_factor: DataAttr[float | None, DataAttrParam.stored] = None
    charging_factor_avg: DataAttr[float, DataAttrParam.stored] = 48
    charging_factor_total: DataAttr[float, DataAttrParam.stored] = 48
    discharging_factor: DataAttr[float | None, DataAttrParam.stored] = None
    discharging_factor_avg: DataAttr[float, DataAttrParam.stored] = 48
    discharging_factor_total: DataAttr[float, DataAttrParam.stored] = 48
    charging_efficiency: DataAttr[float | None, DataAttrParam.stored] = None
    charging_efficiency_avg: DataAttr[float, DataAttrParam.stored] = 1
    charging_efficiency_total: DataAttr[float, DataAttrParam.stored] = 1

    _SLOTS_ = (
        "_production_callback_unsub",
        "_consumption_callback_unsub",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        # TODO : slots (not now since they're used for state serialization)
        self._production_callback_unsub = None
        self._consumption_callback_unsub = None

    # interface: EnergyBalanceEstimator
    @typing.override
    def connect_production(self, estimator: "SignalEnergyEstimator"):
        self.production_estimator = estimator

    @typing.override
    def connect_consumption(self, estimator: "SignalEnergyEstimator"):
        self.consumption_estimator = estimator

    async def async_start(self):
        await super().async_start()
        # since we're not (yet?) restoring history for battery estimation we have to
        # fix today_energy since it gets lost when unloading the entry.
        # We could eventually store the estimator state but that's another story
        # since most of its state is initialized in async_init and that's actually called
        # after restoring (we have to think about this. TODO)
        # BEWARE: ensure the battery estimator is started after the production/consumption
        production_estimator = self.production_estimator
        consumption_estimator = self.consumption_estimator
        """REMOVE
        self.today_energy = consumption_estimator.today_energy
        if self.conversion_yield_avg:
            self.today_energy /= self.conversion_yield_avg
        self.today_energy -= production_estimator.today_energy
        """
        self._production_callback_unsub = production_estimator.listen_energy(
            self._production_callback
        )
        self._consumption_callback_unsub = consumption_estimator.listen_energy(
            self._consumption_callback
        )

    def shutdown(self):
        if self._production_callback_unsub:
            self._production_callback_unsub()
            self._production_callback_unsub = None
        if self._consumption_callback_unsub:
            self._consumption_callback_unsub()
            self._consumption_callback_unsub = None
        super().shutdown()

    @typing.override
    def process(self, input: float | None, time_ts: float) -> float | None:
        sample_curr = self._check_sample_curr(time_ts)
        energy = SignalEnergyProcessor.process(self, input, time_ts)
        if energy is not None:
            # don't accumulate energy in sample_curr..it'll be computed then
            sample_curr.samples += 1
            if energy > 0:
                sample_curr.energy_out += energy
                for listener in self.energy_broadcast_out.energy_listeners:
                    listener(energy, time_ts)
            else:
                energy = -energy
                sample_curr.energy_in += energy
                for listener in self.energy_broadcast_in.energy_listeners:
                    listener(energy, time_ts)
        return energy

    @typing.override
    def _process_sample_curr(self, sample_curr: Sample, time_ts: float, /):
        """Use this callback to flush/store samples or so and update estimates."""
        self.conversion_yield = None
        self.charging_factor = None
        self.discharging_factor = None
        self.charging_efficiency = None
        if sample_curr.samples:
            time_end_curr_ts = sample_curr.time_end_ts
            self.observations_per_sample_avg = (
                self.observations_per_sample_avg * EnergyBalanceEstimator.OPS_DECAY
                + sample_curr.samples * (1 - EnergyBalanceEstimator.OPS_DECAY)
            )
            energy_in = sample_curr.energy_in
            energy_out = sample_curr.energy_out
            charge_in = sample_curr.charge_in
            charge_out = sample_curr.charge_out
            battery = energy_out - energy_in
            self.today_energy += battery
            production = sample_curr.production
            consumption = sample_curr.consumption
            losses = production + battery - consumption
            self.losses += losses
            self.energy_in += energy_in
            self.energy_out += energy_out
            self.charge += charge_in - charge_out

            if production:
                self.production += production

            if consumption:
                self.consumption += consumption
                try:
                    self.conversion_yield_total = self.consumption / (
                        self.consumption + self.losses
                    )
                    self.conversion_yield = consumption / (consumption + losses)
                    self.conversion_yield_avg = (
                        self.conversion_yield_avg * EnergyBalanceEstimator.OPS_DECAY
                        + self.conversion_yield * (1 - EnergyBalanceEstimator.OPS_DECAY)
                    )
                except:  # just protect in case something goes to 0
                    pass

            if charge_in:
                self.charge_in += charge_in
                self.charging_factor_total = self.energy_in / self.charge_in
                self.charging_factor = energy_in / charge_in
                self.charging_factor_avg = (
                    self.charging_factor_avg * EnergyBalanceEstimator.OPS_DECAY
                    + self.charging_factor * (1 - EnergyBalanceEstimator.OPS_DECAY)
                )

            if charge_out:
                self.charge_out += charge_out
                self.discharging_factor_total = self.energy_out / self.charge_out
                self.discharging_factor = energy_out / charge_out
                self.discharging_factor_avg = (
                    self.discharging_factor_avg * EnergyBalanceEstimator.OPS_DECAY
                    + self.discharging_factor * (1 - EnergyBalanceEstimator.OPS_DECAY)
                )
                if charge_in:
                    self.charging_efficiency = (
                        self.discharging_factor / self.charging_factor  # type: ignore
                    )
                    self.charging_efficiency_avg = (
                        self.charging_efficiency_avg * EnergyBalanceEstimator.OPS_DECAY
                        + self.charging_efficiency
                        * (1 - EnergyBalanceEstimator.OPS_DECAY)
                    )

            try:
                self.charging_efficiency_total = (
                    self.discharging_factor_total / self.charging_factor_total
                )
            except:
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

            if _f.energy > 0:
                _f.charge = -_f.energy / self.discharging_factor_avg
            else:
                _f.charge = -_f.energy / self.charging_factor_avg
            if _f.energy_min > 0:
                _f.charge_max = -_f.energy_min / self.discharging_factor_avg
            else:
                _f.charge_max = -_f.energy_min / self.charging_factor_avg
            if _f.energy_max > 0:
                _f.charge_min = -_f.energy_max / self.discharging_factor_avg
            else:
                _f.charge_min = -_f.energy_max / self.charging_factor_avg

            forecasts.append(_f)
            time_ts = time_next_ts

    # interface: BatteryProcessor
    @typing.override
    def _charge_callback(self, charge: float, time_ts: float, /):
        # No need to call self._check_sample_curr
        # since process has been surely called in the same context
        if charge > 0:
            self._sample_curr.charge_out += charge
        else:
            self._sample_curr.charge_in -= charge

    # interface: self
    def _production_callback(self, energy: float, time_ts: float):
        sample_curr = self._check_sample_curr(time_ts)
        sample_curr.production += energy

    def _consumption_callback(self, energy: float, time_ts: float):
        sample_curr = self._check_sample_curr(time_ts)
        sample_curr.consumption += energy
