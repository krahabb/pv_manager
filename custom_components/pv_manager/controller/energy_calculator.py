import typing

from homeassistant import const as hac

from .. import const as pmc, controller
from ..helpers import validation as hv
from ..sensor import EnergySensor
from .devices import SignalEnergyProcessorDevice

if typing.TYPE_CHECKING:
    from typing import Unpack

    from homeassistant.config_entries import ConfigEntry

    class EntryConfig(
        SignalEnergyProcessorDevice.Config,
        pmc.EntityConfig,
        controller.Controller.Config,
    ):
        cycle_modes: list[EnergySensor.CycleMode]
        """list of 'metering' sensors to configure"""


class Controller(controller.Controller["EntryConfig"], SignalEnergyProcessorDevice):  # type: ignore
    """Energy calculator controller."""

    if typing.TYPE_CHECKING:
        Config = EntryConfig

    TYPE = pmc.ConfigEntryType.ENERGY_CALCULATOR
    DEFAULT_NAME = "Energy"

    PLATFORMS = {EnergySensor.PLATFORM}

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None") -> pmc.ConfigSchema:
        if not config:
            config = {
                "name": Controller.DEFAULT_NAME,
                "source_entity_id": "",
                "cycle_modes": [EnergySensor.CycleMode.TOTAL],
            }
        return hv.entity_schema(config) | {
            hv.req_config("source_entity_id", config): hv.sensor_selector(
                device_class=EnergySensor.DeviceClass.POWER
            ),
            hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
            hv.opt_config("update_period", config): hv.time_period_selector(),
            hv.opt_config("maximum_latency", config): hv.time_period_selector(),
            hv.opt_config("input_max", config): hv.number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
            hv.opt_config("input_min", config): hv.number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
        }

    def __init__(self, config_entry: "ConfigEntry"):
        super().__init__(config_entry)

        config = self.config
        name = config.get("name", Controller.DEFAULT_NAME)

        # TODO: rename id pv_energy_sensor
        for cycle_mode in config["cycle_modes"]:
            EnergySensor(
                self,
                "energy_sensor",
                cycle_mode,
                self,
                name=name,
            )
