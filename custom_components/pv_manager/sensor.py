from datetime import datetime, timedelta
import enum
import time
import typing

from homeassistant.components import sensor
from homeassistant.core import HassJob, callback
from homeassistant.helpers import event
from homeassistant.util import dt as dt_util


from . import const as pmc
from .helpers import entity as he

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller
    from .helpers.entity import EntityArgs

    SensorStateType = sensor.StateType | sensor.date | sensor.datetime | sensor.Decimal

    class SensorArgs(EntityArgs):
        device_class: typing.NotRequired[sensor.SensorDeviceClass | None]
        state_class: typing.NotRequired[sensor.SensorStateClass | None]
        native_value: typing.NotRequired[SensorStateType]
        native_unit_of_measurement: typing.NotRequired[str]


async def async_setup_entry(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry[Controller]",
    add_entities: "AddConfigEntryEntitiesCallback",
):
    await config_entry.runtime_data.async_setup_entry_platform(
        sensor.DOMAIN, add_entities
    )


class Sensor(he.Entity, sensor.SensorEntity):

    PLATFORM = sensor.DOMAIN

    DeviceClass = sensor.SensorDeviceClass
    StateClass = sensor.SensorStateClass

    DEVICE_CLASS_TO_STATE_CLASS: dict[
        sensor.SensorDeviceClass | None, sensor.SensorStateClass | None
    ] = {
        None: StateClass.MEASUREMENT,  # generic numeric sensors
        DeviceClass.CURRENT: StateClass.MEASUREMENT,
        DeviceClass.ENUM: None,
        DeviceClass.ENERGY: StateClass.TOTAL_INCREASING,
        DeviceClass.POWER: StateClass.MEASUREMENT,
        DeviceClass.VOLTAGE: StateClass.MEASUREMENT,
    }

    _attr_device_class: typing.ClassVar[sensor.SensorDeviceClass | None] = None
    _attr_native_value = None
    _attr_native_unit_of_measurement: typing.ClassVar[str | None] = None

    __slots__ = (
        "device_class",
        "state_class",
        "native_value",
        "native_unit_of_measurement",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        **kwargs: "typing.Unpack[SensorArgs]",
    ):
        self.device_class = kwargs.pop("device_class", self._attr_device_class)
        self.native_value = kwargs.pop("native_value", self._attr_native_value)
        self.native_unit_of_measurement = kwargs.pop(
            "native_unit_of_measurement", self._attr_native_unit_of_measurement
        )
        if "state_class" in kwargs:
            self.state_class = kwargs.pop("state_class")
        else:
            self.state_class = self.DEVICE_CLASS_TO_STATE_CLASS.get(self.device_class)
        he.Entity.__init__(self, controller, id, **kwargs)

    def update(self, native_value: "SensorStateType"):
        if self.native_value != native_value:
            self.native_value = native_value
            if self.added_to_hass:
                self._async_write_ha_state()


class DiagnosticSensor(he.DiagnosticEntity, Sensor):
    pass


class CycleMode(enum.StrEnum):
    NONE = enum.auto()
    YEARLY = enum.auto()
    MONTHLY = enum.auto()
    WEEKLY = enum.auto()
    DAILY = enum.auto()
    HOURLY = enum.auto()


class EnergySensor(Sensor, he.RestoreEntity):

    CycleMode = CycleMode

    controller: "Controller"

    native_value: float
    _integral_value: float

    _attr_device_class = Sensor.DeviceClass.ENERGY
    _attr_native_value = 0
    _attr_native_unit_of_measurement = Sensor.hac.UnitOfEnergy.WATT_HOUR

    __slots__ = (
        "cycle_mode",
        "last_reset",  # HA property
        "last_reset_dt",
        "last_reset_ts",  # UTC timestamp of last reset
        "next_reset_dt",
        "next_reset_ts",  # UTC timestamp of next reset
        "_integral_value",
        "_async_track_cycle_job",
        "_async_track_cycle_unsub",
    )

    def __init__(
        self,
        controller: "Controller",
        id: str,
        cycle_mode: CycleMode,
        **kwargs: "typing.Unpack[EntityArgs]",
    ):
        self.cycle_mode: typing.Final = cycle_mode
        self.last_reset = None
        self._integral_value = 0
        self._async_track_cycle_unsub = None

        if cycle_mode != CycleMode.NONE:
            name = kwargs.pop("name", id)
            kwargs["name"] = f"{name} ({cycle_mode})"

        Sensor.__init__(
            self,
            controller,
            f"{id}_{cycle_mode}",
            state_class=self.StateClass.TOTAL,
            **kwargs,
        )

    async def async_added_to_hass(self):
        restored_data = self._async_get_restored_data()
        if self.cycle_mode == CycleMode.NONE:
            try:
                extra_data = restored_data.extra_data.as_dict()  # type: ignore
                self._integral_value = extra_data["native_value"]
            except:
                pass
        else:
            self._compute_cycle(False)
            try:
                if (
                    self.last_reset_ts
                    < restored_data.state.last_changed_timestamp  # type: ignore
                    < self.next_reset_ts
                ):
                    extra_data = restored_data.extra_data.as_dict()  # type: ignore
                    self._integral_value = extra_data["native_value"]
            except:
                pass
            self._async_track_cycle_job = HassJob(
                self._async_track_cycle, f"_async_track_cycle({self.cycle_mode})"
            )
            self._async_track_cycle_unsub = event.async_track_point_in_utc_time(self.hass, self._async_track_cycle_job, self.next_reset_dt)  # type: ignore

        self.native_value = int(self._integral_value)
        await super().async_added_to_hass()

    async def async_will_remove_from_hass(self):
        if self._async_track_cycle_unsub:
            self._async_track_cycle_unsub()
            self._async_track_cycle_unsub = None
        return await super().async_will_remove_from_hass()

    @property
    def extra_restore_state_data(self):
        return he.ExtraStoredDataDict({"native_value": self._integral_value})

    def accumulate(self, energy_wh: float):
        if self.cycle_mode == CycleMode.NONE:
            self._integral_value += energy_wh
        else:
            now_ts = time.time()
            if now_ts >= self.next_reset_ts:
                if self._async_track_cycle_unsub:
                    self._async_track_cycle_unsub()
                    self._async_track_cycle_unsub = None
                self._compute_cycle(self.added_to_hass)
                self._integral_value = energy_wh
            else:
                self._integral_value += energy_wh

        _rounded = int(self._integral_value)
        if self.native_value != _rounded:
            self.native_value = _rounded
            if self.added_to_hass:
                self._async_write_ha_state()

    async def _async_track_cycle(self, _dt: datetime):
        self._async_track_cycle_unsub = None
        self._compute_cycle(True)
        self._integral_value -= self.native_value
        self.native_value = 0
        self._async_write_ha_state()

    def _compute_cycle(self, reschedule: bool):
        """'now' is local time."""
        now = dt_util.now()
        match self.cycle_mode:
            case CycleMode.YEARLY:
                self.last_reset_dt = datetime(
                    year=now.year, month=1, day=1, tzinfo=now.tzinfo
                )
                self.next_reset_dt = datetime(
                    year=now.year + 1, month=1, day=1, tzinfo=now.tzinfo
                )
            case CycleMode.MONTHLY:
                self.last_reset_dt = datetime(
                    year=now.year, month=now.month, day=1, tzinfo=now.tzinfo
                )
                self.next_reset_dt = (self.last_reset_dt + timedelta(days=32)).replace(
                    day=1
                )
            case CycleMode.WEEKLY:
                today = datetime(
                    year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                )
                self.last_reset_dt = today - timedelta(days=today.weekday())
                self.next_reset_dt = self.last_reset_dt + timedelta(weeks=1)
            case CycleMode.DAILY:
                self.last_reset_dt = datetime(
                    year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo
                )
                self.next_reset_dt = self.last_reset_dt + timedelta(days=1)
            case CycleMode.HOURLY:
                self.last_reset_dt = datetime(
                    year=now.year,
                    month=now.month,
                    day=now.day,
                    hour=now.hour,
                    tzinfo=now.tzinfo,
                )
                self.next_reset_dt = self.last_reset_dt + timedelta(hours=1)

        self.last_reset_dt = self.last_reset_dt.astimezone(dt_util.UTC)
        self.last_reset_ts = self.last_reset_dt.timestamp()
        self.next_reset_dt = self.next_reset_dt.astimezone(dt_util.UTC)
        self.next_reset_ts = self.next_reset_dt.timestamp()

        self.last_reset = self.last_reset_dt
        if reschedule:
            self._async_track_cycle_unsub = event.async_track_point_in_utc_time(self.hass, self._async_track_cycle_job, self.next_reset_dt)  # type: ignore
            self.log(
                self.DEBUG,
                "Scheduled next cycle at: %s",
                self.next_reset_dt.isoformat(),
            )
