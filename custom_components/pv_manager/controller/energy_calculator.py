import typing

from homeassistant import const as hac
from homeassistant.core import callback

from .. import const as pmc, controller
from ..binary_sensor import ProcessorWarningBinarySensor
from ..helpers import validation as hv
from ..sensor import EnergySensor
from ._energy_meters import TIME_TS, EnergyMeter, SourceType
from .common import EnergyProcessorConfig

if typing.TYPE_CHECKING:

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


class ControllerConfig(typing.TypedDict):
    power_entity_id: str
    """The source entity_id of the pv power"""
    cycle_modes: list[EnergySensor.CycleMode]
    """list of 'metering' sensors to configure"""
    integration_period_seconds: typing.NotRequired[int]
    """If set, calculates accumulation of energy independently of input updates"""


class EntryConfig(
    ControllerConfig, EnergyProcessorConfig, pmc.EntityConfig, pmc.EntryConfig
):
    """TypedDict for ConfigEntry data"""


class Controller(controller.Controller[EntryConfig]):
    """Energy calculator controller."""

    TYPE = pmc.ConfigEntryType.ENERGY_CALCULATOR

    PLATFORMS = {EnergySensor.PLATFORM}

    energy_meter: EnergyMeter

    __slots__ = (
        "energy_meter",
        "integration_period_ts",
        "_integration_callback_unsub",
    )

    @staticmethod
    def get_config_entry_schema(config: EntryConfig | None) -> pmc.ConfigSchema:
        if not config:
            config = {
                "name": "Energy",
                "power_entity_id": "",
                "cycle_modes": [EnergySensor.CycleMode.TOTAL],
                "integration_period_seconds": 5,
            }
        return hv.entity_schema(config) | {
            hv.req_config("power_entity_id", config): hv.sensor_selector(
                device_class=EnergySensor.DeviceClass.POWER
            ),
            hv.req_config("cycle_modes", config): hv.cycle_modes_selector(),
            hv.opt_config(
                "integration_period_seconds", config
            ): hv.time_period_selector(),
            hv.opt_config("maximum_latency_seconds", config): hv.time_period_selector(),
            hv.opt_config("safe_maximum_power_w", config): hv.positive_number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
            hv.opt_config("safe_minimum_power_w", config): hv.positive_number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
        }

    def __init__(self, hass: "HomeAssistant", config_entry: "ConfigEntry"):
        super().__init__(hass, config_entry)

        config = self.config
        name = config.get("name")

        self.energy_meter = EnergyMeter(self, SourceType.UNKNOWN, **config)
        for warning in self.energy_meter.warnings:
            ProcessorWarningBinarySensor(self, f"{warning.id}_warning", warning)

        for cycle_mode in config["cycle_modes"]:
            EnergySensor(
                self,
                "pv_energy_sensor",
                cycle_mode,
                self.energy_meter,
                name=name,
            )

        self.track_state_update(
            config["power_entity_id"], self.energy_meter.track_state
        )

        self.integration_period_ts = config.get("integration_period_seconds", 0)
        self._integration_callback_unsub = (
            self.schedule_callback(
                self.integration_period_ts, self._integration_callback
            )
            if self.integration_period_ts
            else None
        )

    async def async_shutdown(self):
        if self._integration_callback_unsub:
            self._integration_callback_unsub.cancel()
            self._integration_callback_unsub = None
        self.energy_meter.shutdown()
        await super().async_shutdown()

    @callback
    def _integration_callback(self):
        """Called on a timer (if 'integration_period' is set) to accumulate energy in-between
        pv_power changes. In general this shouldn't be needed provided pv_power refreshes frequently
        since the accumulation is also done in _pv_power_tracking_callback"""
        self._integration_callback_unsub = self.schedule_callback(
            self.integration_period_ts, self._integration_callback
        )
        self.energy_meter.interpolate(TIME_TS())
