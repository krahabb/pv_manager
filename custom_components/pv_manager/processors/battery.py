import typing

from homeassistant import const as hac
from homeassistant.core import callback

from . import EnergyBroadcast, SignalEnergyProcessor
from ..helpers import validation as hv
from ..helpers.dataattr import DataAttr, DataAttrParam
from ..sensor import Sensor
from .estimator_energy import EnergyBalanceEstimator, EnergyEstimator

if typing.TYPE_CHECKING:
    from typing import Callable, Final, NotRequired, Self, Unpack

    from homeassistant.core import Event, EventStateChangedData

    from .. import const as pmc
    from .estimator_energy import SignalEnergyEstimator


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
        del schema["source_entity_id"]  # type: ignore
        return schema

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        # typically we'd want negative limit to be equal to the positive one
        if self.input_min is SignalEnergyProcessor.SAFE_MINIMUM_POWER_DISABLED:
            self.input_min = -self.input_max
            self.energy_min = -self.energy_max
        config = self.config
        self.battery_voltage_entity_id = config.get("battery_voltage_entity_id")
        self.battery_current_entity_id = config.get("battery_current_entity_id")
        self.battery_charge_entity_id = config.get("battery_charge_entity_id")
        self.battery_capacity = config.get("battery_capacity", 0)
        # Actual class design considers we're monitoring voltage and current to extract
        # energy and power but we might want a flexible design where we monitor
        # power and current or power and charge. This will need to be investigated (TODO).
        self.configure(SignalEnergyProcessor.Unit.POWER)
        self.listen_energy(self._process_energy)
        self.charge_processor = SignalEnergyProcessor(
            "charge_processor",
            logger=self,
            config={
                "maximum_latency": self.maximum_latency_ts,
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

    def update(self, time_ts, /):
        self.charge_processor.update(time_ts)
        return super().update(time_ts)

    def _process_energy(self, energy: float, time_ts: float, /):
        """Energy listener called by SignalEnergyProcessor(self).process when a sample of energy
        has been calculated."""
        if energy > 0:
            for listener in self.energy_broadcast_out.energy_listeners:
                listener(energy, time_ts)
        else:
            energy = -energy
            for listener in self.energy_broadcast_in.energy_listeners:
                listener(energy, time_ts)

    @callback
    def _current_callback(
        self, event: "Event[EventStateChangedData] | BatteryProcessor.Event", /
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
        self, event: "Event[EventStateChangedData] | BatteryProcessor.Event", /
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


class BatteryEstimator(EnergyEstimator):
    """A 'Battery estimator':
    This works by collecting production(pv) and consumption data (and estimates)
    together with battery energy and charge balance to build a forecast of
    energy balance and charge balance for the battery itself.
    This is also able to estimate system losses (i.e. energy not measured by the 3 type of sources)
    and battery charging efficiency.
    Battery data need to be forwarded to 'process_energy' and 'process_charge'
    as produced by a BatteryProcessor.
    This class is able to collect data from several estimators for load and pv and from
    several batteries (BatteryProcessor) in order to collect them and provide estimation/forecasts
    for a battery system which may be composed by multiple devices of the same type.
    For example we could have 2 or 3 PV sources that feed 2 batteries (in parallel as if it were
    a single larger battery). Load (consumption) also can be configured with multiple
    meters/estimators if the system supply diefferent loads (say an inverter
    and DC loads connected to the battery bank)."""

    class Sample(EnergyEstimator.Sample):
        energy_in: DataAttr[float] = 0
        energy_out: DataAttr[float] = 0
        charge_in: DataAttr[float] = 0
        charge_out: DataAttr[float] = 0
        production: DataAttr[float] = 0
        consumption: DataAttr[float] = 0

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

    if typing.TYPE_CHECKING:

        class Config(EnergyEstimator.Config):
            pass

        class Args(EnergyEstimator.Args):
            config: "BatteryEstimator.Config"

        # (override base typehint)
        config: Config
        forecasts: Final[list[Forecast]]  # type: ignore
        _forecasts_recycle: Final[list[Forecast]]  # type: ignore
        _battery_processors: Final[
            dict[BatteryProcessor, BatteryProcessor.EnergyListenerUnsub]
        ]
        _consumption_estimators: Final[
            dict[SignalEnergyEstimator, SignalEnergyEstimator.EnergyListenerUnsub]
        ]
        _production_estimators: Final[
            dict[SignalEnergyEstimator, SignalEnergyEstimator.EnergyListenerUnsub]
        ]
        _sample_curr: Sample

        @typing.override
        def get_forecast(self, time_begin_ts: int, time_end_ts: int, /) -> Forecast:  # type: ignore
            pass

        @typing.override
        def _check_sample_curr(self, time_ts: float, /) -> Sample:  # type: ignore
            pass

    DEFAULT_NAME = "Battery estimator"

    energy_in: DataAttr[float, DataAttrParam.stored] = 0
    energy_out: DataAttr[float, DataAttrParam.stored] = 0
    charge_in: DataAttr[float, DataAttrParam.stored] = 0
    charge_out: DataAttr[float, DataAttrParam.stored] = 0
    production: DataAttr[float, DataAttrParam.stored] = 0
    consumption: DataAttr[float, DataAttrParam.stored] = 0
    # losses are computed as: production + battery - consumption
    # where:
    # - production: generated (pv) energy
    # - battery: energy from the battery to the load (i.e. positive discharging)
    # - consumption: energy output from inverter (i.e. measure of all the loads)
    losses: DataAttr[float, DataAttrParam.stored] = 0

    ## PERFORMANCE COUNTERS
    # conversion_losses is the ratio: losses / load and gives a raw figure of
    # inverter efficiency and cabling losses
    conversion_losses: DataAttr[float, DataAttrParam.stored] = 0
    conversion_losses_avg: DataAttr[float, DataAttrParam.stored] = 0
    conversion_losses_total: DataAttr[float, DataAttrParam.stored] = 0
    # conversion_yield is: load / (load + losses) and gives the conversion efficiency
    conversion_yield: DataAttr[float | None, DataAttrParam.stored] = None
    conversion_yield_avg: DataAttr[float, DataAttrParam.stored] = 1
    conversion_yield_total: DataAttr[float, DataAttrParam.stored] = 1
    # battery charging/discharging efficiency
    charging_factor: DataAttr[float | None, DataAttrParam.stored] = None
    charging_factor_avg: DataAttr[float, DataAttrParam.stored] = 48
    charging_factor_total: DataAttr[float, DataAttrParam.stored] = 48
    discharging_factor: DataAttr[float | None, DataAttrParam.stored] = None
    discharging_factor_avg: DataAttr[float, DataAttrParam.stored] = 48
    discharging_factor_total: DataAttr[float, DataAttrParam.stored] = 48
    charging_efficiency: DataAttr[float | None, DataAttrParam.stored] = None
    charging_efficiency_avg: DataAttr[float, DataAttrParam.stored] = 1
    charging_efficiency_total: DataAttr[float, DataAttrParam.stored] = 1
    # overall system efficiency: ratio of output (load) vs input (pv)
    # since the battery could store and return some energy the ratio will be affected
    # by 'available battery energy' but this term will asymptotically become irrelevant.
    system_yield: DataAttr[float | None, DataAttrParam.stored] = None
    system_yield_avg: DataAttr[float, DataAttrParam.stored] = 1
    system_yield_total: DataAttr[float, DataAttrParam.stored] = 1

    _SLOTS_ = (
        "_battery_processors",
        "_consumption_estimators",
        "_production_estimators",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self._battery_processors = {}
        self._consumption_estimators = {}
        self._production_estimators = {}

    def shutdown(self):
        for _unsub in self._consumption_estimators.values():
            _unsub()
        self._consumption_estimators.clear()
        for _unsub in self._production_estimators.values():
            _unsub()
        self._production_estimators.clear()
        for _unsub in self._battery_processors.values():
            _unsub()
        self._battery_processors.clear()
        super().shutdown()

    @typing.override
    def _process_sample_curr(self, sample_curr: Sample, time_ts: float, /):
        """Use this callback to flush/store samples or so and update estimates."""
        self.conversion_yield = None
        self.charging_factor = None
        self.discharging_factor = None
        self.charging_efficiency = None
        self.system_yield = None
        if sample_curr.samples:
            avg_c0: "Final" = self.__class__.OPS_DECAY
            avg_c1: "Final" = 1 - avg_c0
            time_end_curr_ts = sample_curr.time_end_ts
            self.observations_per_sample_avg = (
                self.observations_per_sample_avg * avg_c0 + sample_curr.samples * avg_c1
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

            if production:
                self.production += production

            if consumption:
                self.consumption += consumption
                try:
                    self.conversion_losses_total = self.losses / self.consumption
                    self.conversion_losses = losses / consumption
                    self.conversion_losses_avg = (
                        self.conversion_losses_avg * avg_c0
                        + self.conversion_losses * avg_c1
                    )
                    self.conversion_yield_total = self.consumption / (
                        self.consumption + self.losses
                    )
                    self.conversion_yield = consumption / (consumption + losses)
                    self.conversion_yield_avg = (
                        self.conversion_yield_avg * avg_c0
                        + self.conversion_yield * avg_c1
                    )
                except:  # just protect in case something goes to 0
                    pass

            if charge_in:
                self.charge_in += charge_in
                self.charging_factor_total = self.energy_in / self.charge_in
                self.charging_factor = energy_in / charge_in
                self.charging_factor_avg = (
                    self.charging_factor_avg * avg_c0 + self.charging_factor * avg_c1
                )

            if charge_out:
                self.charge_out += charge_out
                self.discharging_factor_total = self.energy_out / self.charge_out
                self.discharging_factor = energy_out / charge_out
                self.discharging_factor_avg = (
                    self.discharging_factor_avg * avg_c0
                    + self.discharging_factor * avg_c1
                )
                if charge_in:
                    self.charging_efficiency = (
                        self.discharging_factor / self.charging_factor  # type: ignore
                    )
                    self.charging_efficiency_avg = (
                        self.charging_efficiency_avg * avg_c0
                        + self.charging_efficiency * avg_c1
                    )
                if production:
                    self.system_yield = (
                        consumption + (charge_in - charge_out) * self.discharging_factor
                    ) / production
                    self.system_yield_avg = (
                        self.system_yield_avg * avg_c0 + self.system_yield * avg_c1
                    )

            try:
                self.charging_efficiency_total = (
                    self.discharging_factor_total / self.charging_factor_total
                )
                self.system_yield_total = (
                    self.consumption
                    + (self.charge_in - self.charge_out) * self.discharging_factor_total
                ) / self.production
            except:
                pass
        else:
            # trigger a reset since previous sample did not collect any data
            time_end_curr_ts = 0

        sample_curr.__init__(time_ts, self)  # recycle sample_curr
        if time_end_curr_ts != sample_curr.time_begin_ts:
            # when samples are not consecutive
            for _processor in self._battery_processors:
                _processor.reset()
            for _processor in self._consumption_estimators:
                _processor.reset()
            for _processor in self._production_estimators:
                _processor.reset()

        # make sure those are time aligned else the estimate calculations will mess up
        # TODO: reorganize synchronization of _check_sample_curr among all the estimators
        # since they should be all aligned to the same sampling period/boundary.
        # This is to have a predictable sequence so that all estimators are updated in
        # a 'controlled' way
        for _consumption_estimator in self._consumption_estimators:
            _consumption_estimator._check_sample_curr(time_ts)
        for _production_estimator in self._production_estimators:
            _production_estimator._check_sample_curr(time_ts)

        self.update_estimate()

        return sample_curr

    @typing.override
    def _ensure_forecasts(self, time_end_ts: int, /):
        estimation_time_ts = self.estimation_time_ts
        sampling_interval_ts = self.sampling_interval_ts
        forecasts = self.forecasts
        _forecasts_recycle = self._forecasts_recycle

        _production_estimators = self._production_estimators
        for _production_estimator in _production_estimators:
            _production_estimator._ensure_forecasts(time_end_ts)
        _consumption_estimators = self._consumption_estimators
        for _consumption_estimator in _consumption_estimators:
            _consumption_estimator._ensure_forecasts(time_end_ts)

        time_ts = estimation_time_ts + len(forecasts) * sampling_interval_ts

        while time_ts < time_end_ts:
            time_next_ts = time_ts + sampling_interval_ts
            try:
                _f = _forecasts_recycle.pop()
                _f.__init__(time_ts, time_next_ts)
            except IndexError:
                _f = self.__class__.Forecast(time_ts, time_next_ts)

            for _production_estimator in _production_estimators:
                _f_p = _production_estimator.get_forecast(time_ts, time_next_ts)
                _f.production += _f_p.energy
                _f.production_min += _f_p.energy_min
                _f.production_max += _f_p.energy_max

            for _consumption_estimator in _consumption_estimators:
                _f_c = _consumption_estimator.get_forecast(time_ts, time_next_ts)
                _f.consumption += _f_c.energy
                _f.consumption_min += _f_c.energy_min
                _f.consumption_max += _f_c.energy_max

            losses_ratio = self.conversion_losses_avg
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

    # interface: self
    def connect_battery(self, processor: BatteryProcessor):
        _unsub_battery_energy = processor.listen_energy(self.process_energy)
        _unsub_battery_charge = processor.charge_processor.listen_energy(
            self.process_charge
        )

        def _unsub():
            _unsub_battery_energy()
            _unsub_battery_charge()

        self._battery_processors[processor] = _unsub

    def disconnect_battery(self, processor: BatteryProcessor):
        self._battery_processors.pop(processor)()

    @typing.override
    def process_energy(self, energy: float, time_ts: float):
        sample_curr = self._check_sample_curr(time_ts)
        sample_curr.samples += 1
        if energy > 0:
            sample_curr.energy_out += energy
        else:
            sample_curr.energy_in -= energy

    def process_charge(self, charge: float, time_ts: float, /):
        sample_curr = self._check_sample_curr(time_ts)
        if charge > 0:
            sample_curr.charge_out += charge
        else:
            sample_curr.charge_in -= charge

    def connect_consumption(self, estimator: "SignalEnergyEstimator"):
        self._consumption_estimators[estimator] = estimator.listen_energy(
            self.process_consumption
        )

    def disconnect_consumption(self, estimator: "SignalEnergyEstimator"):
        self._consumption_estimators.pop(estimator)()

    def process_consumption(self, energy: float, time_ts: float):
        sample_curr = self._check_sample_curr(time_ts)
        sample_curr.consumption += energy

    def connect_production(self, estimator: "SignalEnergyEstimator"):
        self._production_estimators[estimator] = estimator.listen_energy(
            self.process_production
        )

    def disconnect_production(self, estimator: "SignalEnergyEstimator"):
        self._production_estimators.pop(estimator)()

    def process_production(self, energy: float, time_ts: float):
        sample_curr = self._check_sample_curr(time_ts)
        sample_curr.production += energy
