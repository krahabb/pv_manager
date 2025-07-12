from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntryState

from .. import devices
from ... import const as pmc
from ...helpers import validation as hv
from ...helpers.dataattr import DataAttr, DataAttrParam
from ...helpers.entity import EstimatorEntity
from ...processors.estimator_consumption_heuristic import HeuristicConsumptionEstimator
from ...processors.estimator_energy import SignalEnergyEstimator
from ...processors.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator
from ...sensor import PowerSensor, Sensor
from ..devices import battery_device, estimator_device

if TYPE_CHECKING:
    from typing import Callable, ClassVar, Final, NotRequired

    from . import Controller as OffGridManager, EntryData

SourceType = devices.SignalEnergyProcessorDevice.SourceType


class OffGridManagerDevice(devices.ProcessorDevice):
    if TYPE_CHECKING:

        class Config(devices.ProcessorDevice.Config):
            pass

        # override base types
        config: Config
        controller: Final[OffGridManager]  # type: ignore
        config_subentry_id: Final[str]  # type: ignore

        SOURCE_TYPE: ClassVar[SourceType]

    def __init__(
        self, controller: "OffGridManager", config: "Config", subentry_id: str, /
    ):
        _class = self.__class__
        super().__init__(
            f"{_class.SOURCE_TYPE}_{subentry_id}",
            controller=controller,
            config_subentry_id=subentry_id,
            model=_class.__name__,
            config=config,
        )
        if controller.config_entry.state == ConfigEntryState.LOADED:
            self.async_create_task(self.async_start(), "async_start", False)


class MeterDevice(OffGridManagerDevice, devices.SignalEnergyProcessorDevice):
    """Represents a single source of energy measures for a BATTERY, a LOAD or a PV
    generator. This is likely paired to a SignalEnergyProcessor in the inheritance model to
    implement a BatteryMeter, LoadMeter or PvMeter. If estimation is configured,
    LoadMeter(s) and PvMeter(s) are replaced by their 'extended' estimator versions,
    but still behaves as SignalEnergyProcessors from configured sources.
    BatteryEstimator instead (a single instance for the OffGridManager),
    being a 'collector' of measured and estimated data plays its own game
    and doesn't replace BatteryMeter(s).
    """

    if TYPE_CHECKING:

        class Config(
            devices.SignalEnergyProcessorDevice.Config, OffGridManagerDevice.Config
        ):
            pass

        config: Config

    def __init__(
        self, controller: "OffGridManager", config: "Config", subentry_id: str, /
    ):
        super().__init__(controller, config, subentry_id)
        controller.meter_devices[self.__class__.SOURCE_TYPE][
            self.config_subentry_id
        ] = self
        controller.meter_device_add_event.broadcast(self)

    def shutdown(self):
        del self.controller.meter_devices[self.__class__.SOURCE_TYPE][
            self.config_subentry_id
        ]
        self.controller.meter_device_remove_event.broadcast(self)
        super().shutdown()


class BatteryMeter(MeterDevice, battery_device.BatteryProcessorDevice):

    SOURCE_TYPE = SourceType.BATTERY

    def __init__(self, controller: "OffGridManager", config, subentry_id: str, /):
        super().__init__(controller, config, subentry_id)
        if battery_estimator := controller.battery_estimator:
            battery_estimator.connect_battery(self)

    def shutdown(self):
        if battery_estimator := self.controller.battery_estimator:
            battery_estimator.disconnect_battery(self)
        super().shutdown()


class LoadMeter(MeterDevice):

    SOURCE_TYPE = SourceType.LOAD


class PvMeter(MeterDevice):

    SOURCE_TYPE = SourceType.PV


class EstimatorDevice(OffGridManagerDevice, estimator_device.EnergyEstimatorDevice):
    if TYPE_CHECKING:

        class Config(
            estimator_device.EnergyEstimatorDevice.Config, OffGridManagerDevice.Config
        ):
            pass

        config: Config


class BatteryEstimator(EstimatorDevice, battery_device.BatteryEstimatorDevice):
    if TYPE_CHECKING:

        class Config(
            battery_device.BatteryEstimatorDevice.Config,
            EstimatorDevice.Config,
            HeuristicPVEnergyEstimator.Config,
        ):
            pass

        config: Config

    SOURCE_TYPE = SourceType.BATTERY

    __slots__ = ()

    @classmethod
    def get_config_schema(cls, config: "Config"):
        return HeuristicPVEnergyEstimator.get_config_schema(config)

    def __init__(self, controller: "OffGridManager", config, subentry_id: str, /):
        super().__init__(controller, config, subentry_id)
        controller.stored_objects[self.id] = self


class SignalEnergyEstimatorDevice(EstimatorDevice, SignalEnergyEstimator):
    pass


class LoadEstimator(
    LoadMeter, SignalEnergyEstimatorDevice, HeuristicConsumptionEstimator
):
    def __init__(self, controller: "OffGridManager", config, subentry_id: str, /):
        super().__init__(controller, config, subentry_id)
        controller.battery_estimator.connect_consumption(self)

    def shutdown(self):
        self.controller.battery_estimator.disconnect_consumption(self)
        super().shutdown()


class PvEstimator(PvMeter, SignalEnergyEstimatorDevice, HeuristicPVEnergyEstimator):

    def __init__(self, controller: "OffGridManager", config, subentry_id: str, /):
        super().__init__(controller, config, subentry_id)
        controller.battery_estimator.connect_production(self)

    def shutdown(self):
        self.controller.battery_estimator.disconnect_production(self)
        super().shutdown()


class LossesMeter(MeterDevice):

    class YieldSensor(EstimatorEntity[BatteryEstimator], Sensor):

        _attr_parent_attr = Sensor.ParentAttr.DYNAMIC
        # _attr_native_unit_of_measurement = "%"
        _attr_suggested_display_precision = 2

        __slots__ = EstimatorEntity._SLOTS_

        def update_config(
            self,
            name: str,
            estimator_update_func: "LossesMeter.YieldSensor.EstimatorUpdateT",
            /,
        ):
            if self.estimator_update_func != estimator_update_func:
                self.estimator_update_func = estimator_update_func  # type: ignore[Final]
                self.native_value = estimator_update_func(self.estimator)
            super().update_name(name)

    if TYPE_CHECKING:

        class Config(MeterDevice.Config):
            battery_yield: NotRequired[str]
            battery_yield_mode: NotRequired[str]
            conversion_yield: NotRequired[str]
            conversion_yield_mode: NotRequired[str]
            system_yield: NotRequired[str]
            system_yield_mode: NotRequired[str]

        YIELD_SENSOR_IDS: Final[dict[str, dict[str, YieldSensor.EstimatorUpdateT]]]
        YIELD_SENSOR_MODES: Final[list[str]]
        losses_power_sensor: PowerSensor | None
        battery_yield_sensor: YieldSensor | None
        conversion_yield_sensor: YieldSensor | None
        system_yield_sensor: YieldSensor | None
        _load_meters: dict[MeterDevice, Callable]

    WARNINGS = ()  # override SignalEnergyProcessor: no use here

    SOURCE_TYPE = SourceType.LOSSES

    YIELD_SENSOR_IDS = {
        "battery_yield": {
            "instant": lambda e: e.charging_efficiency,
            "average": lambda e: e.charging_efficiency_avg,
            "total": lambda e: e.charging_efficiency_total,
        },
        "conversion_yield": {
            "instant": lambda e: e.conversion_yield,
            "average": lambda e: e.conversion_yield_avg,
            "total": lambda e: e.conversion_yield_total,
        },
        "system_yield": {
            "instant": lambda e: (
                None
                if e.conversion_yield is None or e.charging_efficiency is None
                else e.conversion_yield * e.charging_efficiency
            ),  # TODO: compute system_yield in battery estimator
            "average": lambda e: e.conversion_yield_avg * e.charging_efficiency_avg,
            "total": lambda e: e.conversion_yield_total * e.charging_efficiency_total,
        },
    }
    YIELD_SENSOR_MODES = ["instant", "average", "total"]

    __slots__ = (
        "losses_power_sensor",
        "system_yield_sensor",
        "battery_yield_sensor",
        "conversion_yield_sensor",
        "_losses",
        "_load_meters",
        "_meter_device_add_unsub",
        "_meter_device_remove_unsub",
        "_battery_estimator_update_unsub",
    )

    @classmethod
    def get_config_schema(cls, config: "Config | None", /) -> pmc.ConfigSchema:
        _config = config or {
            "battery_yield": "Battery yield",
            "battery_yield_mode": "average",
            "conversion_yield": "Conversion yield",
            "conversion_yield_mode": "average",
            "system_yield": "System yield",
            "system_yield_mode": "average",
        }
        # ensure we don't pull in SignalEnergyProcessor.Config
        schema = devices.EnergyMeterDevice.get_config_schema(config)
        _s = hv.select_selector(options=LossesMeter.YIELD_SENSOR_MODES)
        for _sensor_id in LossesMeter.YIELD_SENSOR_IDS:
            schema |= {
                hv.opt_config(_sensor_id, _config): str,
                hv.opt_config(f"{_sensor_id}_mode", _config): _s,
            }
        return schema

    def __init__(
        self, controller: "OffGridManager", config: "Config", subentry_id: str, /
    ):
        super().__init__(controller, config, subentry_id)
        self.priority = self.Priority.LOW
        self._losses = 0
        self._load_meters = None  # type: ignore
        self._meter_device_add_unsub = None
        self._battery_estimator_update_unsub = None
        if name := config.get("name"):
            PowerSensor(
                self,
                "losses_power",
                config_subentry_id=subentry_id,
                name=name,
                parent_attr=PowerSensor.ParentAttr.DYNAMIC,
            )

        if battery_estimator := controller.battery_estimator:
            for (
                yield_sensor_id,
                mode_dict,
            ) in LossesMeter.YIELD_SENSOR_IDS.items():
                if name := config.get(yield_sensor_id):
                    LossesMeter.YieldSensor(
                        self,
                        yield_sensor_id,
                        battery_estimator,
                        config_subentry_id=subentry_id,
                        name=name,
                        estimator_update_func=mode_dict.get(
                            config.get(f"{yield_sensor_id}_mode", "average"),
                            mode_dict["average"],
                        ),
                    )

    async def async_start(self):
        controller = self.controller
        self._load_meters = {
            load_meter: load_meter.listen_energy(self._process_load_energy)
            for load_meter in controller.meter_devices[SourceType.LOAD].values()
        }
        self._meter_device_add_unsub = controller.meter_device_add_event.listen(
            self._meter_device_add
        )
        self._meter_device_remove_unsub = controller.meter_device_remove_event.listen(
            self._meter_device_remove
        )
        await super().async_start()
        self._process_load_energy()
        if battery_estimator := controller.battery_estimator:
            self._losses = battery_estimator.losses
            self._battery_estimator_update_unsub = battery_estimator.listen_update(
                self._battery_estimator_update
            )

    def shutdown(self):
        if self._battery_estimator_update_unsub:
            self._battery_estimator_update_unsub()
            self._battery_estimator_update_unsub = None
        if self._meter_device_add_unsub:
            self._meter_device_add_unsub()
            self._meter_device_add_unsub = None
        if self._load_meters:
            for _unsub in self._load_meters.values():
                _unsub()
        super().shutdown()

    async def async_update_entry(self, entry_data: "EntryData[Config]", /):
        subentry_id = entry_data.subentry_id
        config = entry_data.config
        entities = entry_data.entities

        losses_power_sensor = entities.get(f"{self.unique_id}-losses_power")
        if name := config.get("name"):
            if losses_power_sensor:
                losses_power_sensor.update_name(name)
            else:
                PowerSensor(
                    self,
                    "losses_power",
                    config_subentry_id=subentry_id,
                    name=name,
                    parent_attr=PowerSensor.ParentAttr.DYNAMIC,
                )
        elif losses_power_sensor:
            await losses_power_sensor.async_shutdown(True)

        if battery_estimator := self.controller.battery_estimator:
            # update yield sensors
            yield_sensors_id_new = {
                yield_sensor_id: mode_dict
                for yield_sensor_id, mode_dict in LossesMeter.YIELD_SENSOR_IDS.items()
                if config.get(yield_sensor_id)
            }
            for yield_sensor_id, mode_dict in LossesMeter.YIELD_SENSOR_IDS.items():
                try:
                    yield_sensor: LossesMeter.YieldSensor = entities[f"{self.unique_id}-{yield_sensor_id}"]  # type: ignore
                except KeyError:
                    # yield_sensor not previously configured
                    continue
                try:
                    del yield_sensors_id_new[yield_sensor_id]
                    # yield_sensor still present: update
                    yield_sensor.update_config(
                        config[yield_sensor_id],
                        mode_dict.get(
                            config.get(f"{yield_sensor_id}_mode", "average"),
                            mode_dict["average"],
                        ),
                    )
                except KeyError:
                    # yield_sensor removed from updated config
                    await yield_sensor.async_shutdown(True)
            # leftovers are newly added yield sensors
            for yield_sensor_id, mode_dict in yield_sensors_id_new.items():
                LossesMeter.YieldSensor(
                    self,
                    yield_sensor_id,
                    battery_estimator,
                    config_subentry_id=subentry_id,
                    name=config[yield_sensor_id],
                    estimator_update_func=mode_dict.get(
                        config.get(f"{yield_sensor_id}_mode", "average"),
                        mode_dict["average"],
                    ),
                )

        await super().async_update_entry(entry_data)

    def _process_load_energy(self, *args):

        controller = self.controller
        load_power = 0
        for load_meter in self._load_meters:
            load_power += load_meter.input or 0

        if battery_estimator := controller.battery_estimator:
            losses_power = load_power * battery_estimator.conversion_losses
            if self.losses_power_sensor:
                self.losses_power_sensor.update(losses_power)

    def _battery_estimator_update(self, estimator: BatteryEstimator, /):
        losses = estimator.losses
        d_losses = losses - self._losses
        for energy_listener in self.energy_listeners:
            energy_listener(d_losses, estimator.estimation_time_ts)
        self._losses = losses

    def _meter_device_add(self, meter_device: MeterDevice, /):
        if meter_device.SOURCE_TYPE is SourceType.LOAD:
            self._load_meters[meter_device] = meter_device.listen_energy(
                self._process_load_energy
            )
            self._process_load_energy()

    def _meter_device_remove(self, meter_device: MeterDevice, /):
        if meter_device.SOURCE_TYPE is SourceType.LOAD:
            self._load_meters.pop(meter_device)()
            self._process_load_energy()


"""REMOVE
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
