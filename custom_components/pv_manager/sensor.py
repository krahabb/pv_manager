import typing

from homeassistant.components import sensor

from . import const as pmc, helpers
from .helpers.entity import Entity

if typing.TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

    from .controller import Controller
    from .helpers.entity import EntityArgs

    SensorStateType = sensor.StateType | sensor.date | sensor.datetime | sensor.Decimal

    class SensorArgs(EntityArgs):
        device_class: typing.NotRequired[sensor.SensorDeviceClass]
        state_class: typing.NotRequired[sensor.SensorStateClass|None]
        native_value: typing.NotRequired[SensorStateType]
        native_unit_of_measurement: typing.NotRequired[str]


async def async_setup_entry(
    hass: "HomeAssistant",
    config_entry: "ConfigEntry[Controller]",
    add_entities: "AddConfigEntryEntitiesCallback",
):
    await config_entry.runtime_data.async_setup_entry_platform(sensor.DOMAIN, add_entities)


class Sensor(Entity, sensor.SensorEntity):

    PLATFORM = sensor.DOMAIN

    DeviceClass = sensor.SensorDeviceClass
    StateClass = sensor.SensorStateClass

    DEVICE_CLASS_TO_STATE_CLASS: dict[sensor.SensorDeviceClass | None, sensor.SensorStateClass] = {
        sensor.SensorDeviceClass.POWER: sensor.SensorStateClass.MEASUREMENT,
        sensor.SensorDeviceClass.ENERGY: sensor.SensorStateClass.TOTAL_INCREASING,
        sensor.SensorDeviceClass.VOLTAGE: sensor.SensorStateClass.MEASUREMENT,
        sensor.SensorDeviceClass.CURRENT: sensor.SensorStateClass.MEASUREMENT,
    }

    _attr_device_class: typing.ClassVar[sensor.SensorDeviceClass | None] = None
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
        self.native_value = kwargs.pop("native_value", None)
        self.native_unit_of_measurement = kwargs.pop("native_unit_of_measurement", self._attr_native_unit_of_measurement)
        if "state_class" in kwargs:
            self.state_class = kwargs.pop("state_class")
        else:
            self.state_class = self.DEVICE_CLASS_TO_STATE_CLASS.get(self.device_class)
        Entity.__init__(self, controller, id, **kwargs)

    def update(self, native_value: "SensorStateType"):
        if self.native_value != native_value:
            self.native_value = native_value
            if self._added_to_hass:
                self._async_write_ha_state()


class RestoreSensor(Sensor, sensor.RestoreSensor):
    pass

