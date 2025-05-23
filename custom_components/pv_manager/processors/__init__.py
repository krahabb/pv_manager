import abc
import enum
from time import time as TIME_TS
import typing

from homeassistant import const as hac
from homeassistant.util.unit_conversion import (
    EnergyConverter,
    PowerConverter,
)

from ..helpers import Loggable
from ..helpers.callback import CallbackTracker
from ..manager import Manager

if typing.TYPE_CHECKING:
    from typing import NotRequired, TypedDict, Unpack

    from homeassistant.core import Event, EventStateChangedData, State


class SourceType(enum.StrEnum):
    UNKNOWN = enum.auto()
    BATTERY = enum.auto()
    BATTERY_IN = enum.auto()
    BATTERY_OUT = enum.auto()
    LOAD = enum.auto()
    LOSSES = enum.auto()
    PV = enum.auto()


class ProcessorWarning:

    CALLBACK_TYPE = typing.Callable[[bool], None]

    processor: "BaseProcessor"
    id: str  # could be any object but it should have a 'proper' str behaviour
    on: bool
    _listeners: set[CALLBACK_TYPE]

    __slots__ = (
        "processor",
        "id",
        "on",
        "_listeners",
    )

    def __init__(self, processor: "BaseProcessor", id):
        self.processor = processor
        self.id = id
        self.on = False
        self._listeners = set()
        setattr(processor, f"warning_{id}", self)

    def shutdown(self):
        self._listeners.clear()
        setattr(self.processor, f"warning_{self.id}", None)
        self.processor = None  # type: ignore

    def listen(self, callback_func: CALLBACK_TYPE):
        self._listeners.add(callback_func)

        def _unsub():
            try:
                self._listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    def toggle(self):
        self.on = not self.on
        for listener in self._listeners:
            listener(self.on)


class BaseProcessor[_input_t](CallbackTracker, Loggable):

    if typing.TYPE_CHECKING:

        class Config(TypedDict):
            source_entity_id: NotRequired[str | None]
            update_period_seconds: NotRequired[float | None]

        class Args(Loggable.Args):
            config: "BaseProcessor.Config"

    config: "Config"
    time_ts: float
    input: _input_t | None

    WARNINGS: typing.ClassVar[typing.Iterable[str]] = ()
    warnings: typing.Final[set[ProcessorWarning]]

    _SLOTS_ = (
        "config",
        "source_entity_id",
        "update_period_ts",
        "time_ts",
        "input",
        "warnings",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        self.config = kwargs["config"]
        self.source_entity_id = self.config.get("source_entity_id")
        self.update_period_ts = self.config.get("update_period_seconds", 0)
        self.time_ts = None  # type: ignore
        self.input = None
        self.warnings = {
            ProcessorWarning(self, warning_id) for warning_id in self.WARNINGS
        }
        super().__init__(id, **kwargs)

    async def async_start(self):
        if self.source_entity_id:
            self.track_state(
                self.source_entity_id,
                self._source_entity_update,
                BaseProcessor.HassJobType.Callback,
            )

        if self.update_period_ts:
            self.track_timer(self.update_period_ts, self._update_callback)

    def shutdown(self):
        """Used to remove references when wanting to shutdown resources usage."""
        super().shutdown()
        for warning in self.warnings:
            warning.shutdown()

    def configure(self, *args, **kwargs):
        pass

    def process(self, input: _input_t, time_ts: float) -> typing.Any:
        self.time_ts = time_ts
        self.input = input

    def update(self, time_ts: float):
        self.time_ts = time_ts

    def as_dict(self):
        """Used for serialization to debug files or so.
        Returns the configuration."""
        return {
            "source_entity_id": self.source_entity_id,
            "update_period_ts": self.update_period_ts,
        }

    def get_state_dict(self):
        """Returns a synthetic state dict.
        Used for debugging purposes."""
        return {
            "warnings": {warning.id: warning.on for warning in self.warnings},
        }

    def _state_convert(self, value: float, from_unit: str | None, to_unit: str | None):
        pass

    def _source_entity_update(
        self, event: "Event[EventStateChangedData] | CallbackTracker.Event"
    ):
        pass

    def _update_callback(self):
        self.update(TIME_TS())


class EnergyInputMode(enum.Enum):
    POWER = (False, "W")
    ENERGY = (True, "Wh")


class EnergyProcessorWarningId(enum.StrEnum):
    maximum_latency = enum.auto()
    out_of_range = enum.auto()


# these defaults are applied when the corresponding config option is not set or is 0
# they means by default we're measuring positive energies with (almost) no latency checks
# In code they're checked anyway but the set limits should be high enough to not pose any real issue
# preferring this behavior over spending time against checking for 'disabled'
MAXIMUM_LATENCY_DISABLED = 1e6
SAFE_MAXIMUM_POWER_DISABLED = 1e6
SAFE_MINIMUM_POWER_DISABLED = -1e6


class BaseEnergyProcessor(BaseProcessor[float]):

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config):
            maximum_latency_seconds: NotRequired[float]
            """Maximum time between source pv power/energy samples before considering an error in data sampling."""
            safe_maximum_power_w: NotRequired[float]
            """Maximum power expected at the input used to filter out outliers from processing. If not set disables the check."""
            safe_minimum_power_w: NotRequired[float]
            """Minimum power expected at the input used to filter out outliers from processing. If not set disables the check."""

        class Args(BaseProcessor.Args):
            config: "BaseEnergyProcessor.Config"

    InputMode = EnergyInputMode

    input_mode: typing.Final[bool]
    input_unit: typing.Final[str | None]
    """input_mode is not implicitly initialized since it must be configured by calling
    configure() before process()."""

    output: float

    WARNINGS = EnergyProcessorWarningId

    # built and set automatically on super().__init__
    warning_maximum_latency: ProcessorWarning
    warning_out_of_range: ProcessorWarning

    ENERGY_LISTENER_TYPE = typing.Callable[[float, float], None]
    _energy_listeners: typing.Final[set[ENERGY_LISTENER_TYPE]]

    _SLOTS_ = (
        "maximum_latency_ts",
        "safe_maximum_power",
        "safe_maximum_power_cal",
        "safe_minimum_power",
        "safe_minimum_power_cal",
        "input_mode",
        "input_unit",
        "output",
        "warning_maximum_latency",
        "warning_out_of_range",
        "_energy_listeners",
    )

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):

        config = kwargs["config"]
        self.maximum_latency_ts = (
            config.get("maximum_latency_seconds", 0) or MAXIMUM_LATENCY_DISABLED
        )
        self.safe_maximum_power = config.get(
            "safe_maximum_power_w", SAFE_MAXIMUM_POWER_DISABLED
        )
        self.safe_maximum_power_cal = self.safe_maximum_power / 3600
        self.safe_minimum_power = config.get(
            "safe_minimum_power_w", SAFE_MINIMUM_POWER_DISABLED
        )
        self.safe_minimum_power_cal = self.safe_minimum_power / 3600
        self.input_unit = None
        self.output = 0
        self._energy_listeners = set()
        super().__init__(id, **kwargs)

    def listen_energy(self, callback_func: ENERGY_LISTENER_TYPE):
        self._energy_listeners.add(callback_func)

        def _unsub():
            try:
                self._energy_listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    @typing.override
    def shutdown(self):
        self._energy_listeners.clear()
        super().shutdown()

    @typing.override
    def configure(
        self,
        input_mode: EnergyInputMode,
    ):
        self.input_mode, self.input_unit = input_mode.value  # type: ignore

    @typing.override
    def process(self, input: float | None, time_ts: float) -> float | None:

        try:
            d_ts = time_ts - self.time_ts
            if 0 <= d_ts < self.maximum_latency_ts:

                if self.warning_maximum_latency.on:
                    self.warning_maximum_latency.toggle()

                if self.input_mode:
                    energy = input - self.input  # type: ignore
                    if (
                        self.safe_minimum_power_cal
                        <= energy / d_ts
                        <= self.safe_maximum_power_cal
                    ):
                        if self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                        self.output += energy
                        for energy_listener in self._energy_listeners:
                            energy_listener(energy, time_ts)
                    else:
                        # assume an energy reset or out of range
                        energy = None
                        if not self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                else:
                    # power left rect integration
                    if self.safe_minimum_power <= self.input <= self.safe_maximum_power:  # type: ignore
                        if self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                        energy = self.input * d_ts / 3600  # type: ignore
                        self.output += energy
                        for energy_listener in self._energy_listeners:
                            energy_listener(energy, time_ts)
                    else:
                        # discard the out of range observation
                        energy = None
                        if not self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()

            else:
                energy = None
                if not self.warning_maximum_latency.on:
                    self.warning_maximum_latency.toggle()

            self.input = input
            self.time_ts = time_ts
            return energy

        except TypeError as error:
            # expected when input or self.input are 'None'
            # TODO: add a warning processor for signaling 'input unavailable'
            self.input = input
            self.time_ts = time_ts
            return None
        except AttributeError as error:
            if error.name == "input_mode":
                raise Exception("configure() need to be called before process()")
            raise error

    @typing.override
    def update(self, time_ts: float):
        if self.input_mode:
            # TODO: interpolate energy
            pass
        else:
            self.process(self.input, time_ts)

    @typing.override
    def as_dict(self):
        return {
            "maximum_latency_seconds": (
                None
                if self.maximum_latency_ts is MAXIMUM_LATENCY_DISABLED
                else self.maximum_latency_ts
            ),
            "safe_maximum_power_w": (
                None
                if self.safe_maximum_power is SAFE_MAXIMUM_POWER_DISABLED
                else self.safe_maximum_power
            ),
            "safe_minimum_power_w": (
                None
                if self.safe_minimum_power is SAFE_MINIMUM_POWER_DISABLED
                else self.safe_minimum_power
            ),
        }

    @typing.override
    def _source_entity_update(
        self, event: "Event[EventStateChangedData] | CallbackTracker.Event"
    ):
        try:
            state = event.data["new_state"]
            self.process(
                self._state_convert(
                    float(state.state),  # type: ignore
                    state.attributes["unit_of_measurement"],  # type: ignore
                    self.input_unit,
                ),
                event.time_fired_timestamp,
            )  # type: ignore
        except Exception as e:
            # this is expected and silently managed when state == None or 'unknown'
            # TODO: put the BaseProcessor in warning like if it was a maximum_latency
            # or set a new warning id like no_signal
            self.process(None, event.time_fired_timestamp)
            if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
                self.log_exception(
                    self.WARNING,
                    e,
                    "track_state (state:%s)",
                    state,
                )

    @typing.override
    def _state_convert(
        self, value: float, from_unit: str | None, to_unit: str | None
    ) -> float:
        """Installed as _state_convert at init time this will detect the type of observed entity
        by inspecting the unit and install the proper converter."""
        if from_unit in hac.UnitOfPower:
            self._state_convert = PowerConverter.convert
            self.configure(self.InputMode.POWER)
        elif from_unit in hac.UnitOfEnergy:
            self._state_convert = EnergyConverter.convert
            self.configure(self.InputMode.ENERGY)
        else:
            # TODO: raise issue?
            raise ValueError(f"Unsupported unit of measurement '{from_unit}'")
        return self._state_convert(value, from_unit, self.input_unit)
