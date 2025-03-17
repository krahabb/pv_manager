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

    class SensorArgs(EntityArgs):
        device_class: typing.NotRequired[sensor.SensorDeviceClass]


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
    }

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
        self.device_class = kwargs.pop("device_class", None)
        self.state_class = self.DEVICE_CLASS_TO_STATE_CLASS.get(self.device_class)
        Entity.__init__(self, controller, id, **kwargs)


class RestoreSensor(Sensor, sensor.RestoreSensor):
    pass

