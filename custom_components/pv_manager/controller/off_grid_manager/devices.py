from typing import TYPE_CHECKING

from homeassistant.config_entries import ConfigEntryState

from .. import devices
from ...helpers import validation as hv
from ...helpers.dataattr import DataAttr, DataAttrParam
from ...processors import battery
from ...processors.estimator_consumption_heuristic import HeuristicConsumptionEstimator
from ...processors.estimator_energy import SignalEnergyEstimator
from ...processors.estimator_pvenergy_heuristic import HeuristicPVEnergyEstimator
from ..devices import battery_device, estimator_device

if TYPE_CHECKING:
    from typing import ClassVar, Final

    from . import Controller as OffGridManager

SourceType = devices.SignalEnergyProcessorDevice.SourceType


class OffGridManagerDevice(devices.ProcessorDevice):
    if TYPE_CHECKING:

        class Config(devices.ProcessorDevice.Config):
            pass

        # override base types
        config: Config
        controller: Final[OffGridManager]  # type: ignore
        config_subentry_id: Final[str]  # type: ignore

        SOURCE_TYPE: ClassVar[str]

    def __init__(
        self, controller: "OffGridManager", config: "Config", subentry_id: str, /
    ):
        _class = self.__class__
        super().__init__(
            f"{_class.SOURCE_TYPE}_{subentry_id}",  # this should prove to be unique in the controller context
            controller=controller,
            config_subentry_id=subentry_id,
            model=_class.__name__,
            name=config.get("name") or _class.DEFAULT_NAME or _class.SOURCE_TYPE,
            config=config,
        )


class MeterDevice(OffGridManagerDevice, devices.SignalEnergyProcessorDevice):
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
        if controller.config_entry.state == ConfigEntryState.LOADED:
            self.async_create_task(self.async_start(), "async_start")

    def shutdown(self):
        del self.controller.meter_devices[self.__class__.SOURCE_TYPE][
            self.config_subentry_id
        ]
        super().shutdown()


class BatteryMeter(MeterDevice, battery_device.BatteryProcessorDevice):

    if TYPE_CHECKING:

        class Config(battery_device.BatteryProcessorDevice.Config, MeterDevice.Config):
            pass

        config: Config

    SOURCE_TYPE = SourceType.BATTERY


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
            battery_device.BatteryEstimatorDevice.Config, EstimatorDevice.Config
        ):
            pass

        config: Config

    SOURCE_TYPE = SourceType.BATTERY

    def __init__(
        self, controller: "OffGridManager", config: "Config", subentry_id: str, /
    ):
        super().__init__(controller, config, subentry_id)
        controller.stored_objects[self.id] = self

    async def async_start(self):

        meter_devices = self.controller.meter_devices
        for battery in meter_devices["battery"].values():
            self.connect_battery(battery)
        for load in meter_devices["load"].values():
            self.connect_consumption(load)  # type: ignore
        for pv in meter_devices["pv"].values():
            self.connect_production(pv)  # type: ignore

        return await super().async_start()


class SignalEnergyEstimatorDevice(EstimatorDevice, SignalEnergyEstimator):
    pass


class LoadEstimator(
    LoadMeter, SignalEnergyEstimatorDevice, HeuristicConsumptionEstimator
):
    pass


class PvEstimator(PvMeter, SignalEnergyEstimatorDevice, HeuristicPVEnergyEstimator):
    pass


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
