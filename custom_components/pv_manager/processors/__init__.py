"""
processors:
base classes for signal processing
These are mainly abstract and are designed to be included in complex
diamond pattern hierarchies. This is especially true for
estimators which should always include a signal processor to be feeded
"""

import abc
import enum
from time import time as TIME_TS
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.util import unit_conversion

from ..helpers import Loggable, datetime_from_epoch, validation as hv
from ..helpers.callback import CallbackTracker
from ..manager import Manager
from ..sensor import Sensor

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        ClassVar,
        Final,
        Iterable,
        NotRequired,
        TypedDict,
        Unpack,
    )

    from homeassistant.core import Event, EventStateChangedData, State

    from .. import const as pmc


class GenericBroadcast(Loggable):

    if typing.TYPE_CHECKING:
        LISTENER_TYPE = Callable[..., Any]
        _listeners: set[LISTENER_TYPE]

    _SLOTS_ = ("_listeners",)

    def shutdown(self):
        try:
            self._listeners.clear()
        except AttributeError:
            pass
        super().shutdown()

    def listen(self, callback_func: "LISTENER_TYPE"):
        try:
            _listeners = self._listeners
        except AttributeError:
            self._listeners = _listeners = set()

        _listeners.add(callback_func)

        def _unsub():
            try:
                _listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    def broadcast(self, *args):
        try:
            for listener in self._listeners:
                listener(*args)
        except AttributeError:
            pass


class EnergyBroadcast(Loggable):

    if typing.TYPE_CHECKING:
        ENERGY_LISTENER_TYPE = Callable[[float, float], None]
        energy_listeners: set[ENERGY_LISTENER_TYPE]

    _SLOTS_ = ("energy_listeners",)

    def __init__(self, id, **kwargs):
        self.energy_listeners = set()
        super().__init__(id, **kwargs)

    def listen_energy(self, callback_func: "ENERGY_LISTENER_TYPE"):
        listeners = self.energy_listeners
        listeners.add(callback_func)

        def _unsub():
            try:
                listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    def shutdown(self):
        self.energy_listeners.clear()
        super().shutdown()


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


class BaseProcessor(CallbackTracker, Loggable):

    @typing.final
    class Unit(enum.StrEnum):
        """Default units being used in our processing."""

        VOLTAGE_UNIT = hac.UnitOfElectricPotential.VOLT
        CURRENT_UNIT = hac.UnitOfElectricCurrent.AMPERE
        POWER_UNIT = hac.UnitOfPower.WATT
        ENERGY_UNIT = hac.UnitOfEnergy.WATT_HOUR

    Converter: typing.Final = unit_conversion
    """Access to the homeassistant.util.unit_conversion module to reach a unit converter."""

    class SourceType(enum.StrEnum):
        BATTERY = enum.auto()
        BATTERY_IN = enum.auto()
        BATTERY_OUT = enum.auto()
        LOAD = enum.auto()
        LOSSES = enum.auto()
        PV = enum.auto()

    if typing.TYPE_CHECKING:

        class Config(TypedDict):
            pass

        class Args(Loggable.Args):
            config: "BaseProcessor.Config"

        class StoreType(TypedDict):
            pass

        type ConvertFuncType = Callable[[float, str | None, str | None], float]

        DEFAULT_NAME: ClassVar[str]

        config: Config

        WARNINGS: ClassVar[Iterable[str]]
        warnings: Final[set[ProcessorWarning]]

    DEFAULT_NAME = ""
    WARNINGS = ()

    _SLOTS_ = (
        "config",
        "warnings",
    )

    @classmethod
    def get_config_schema(cls, config: "Config | None") -> "pmc.ConfigSchema":
        return {}

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self.config = kwargs["config"]
        self.warnings = {
            ProcessorWarning(self, warning_id) for warning_id in self.WARNINGS
        }

    async def async_start(self):
        pass

    def shutdown(self):
        """Used to remove references when wanting to shutdown resources usage."""
        super().shutdown()
        for warning in self.warnings:
            warning.shutdown()

    def as_dict(self):
        """Used for serialization to debug files or so.
        Returns the configuration."""
        return {}

    def get_state_dict(self):
        """Returns a synthetic state dict.
        Used for debugging purposes."""
        return {
            "warnings": {warning.id: warning.on for warning in self.warnings},
        }

    def restore(self, data: "StoreType"):
        pass

    def store(self) -> "StoreType":
        return {}


class SignalProcessor[_input_t](BaseProcessor):

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config):
            source_entity_id: NotRequired[str | None]
            update_period_seconds: NotRequired[float | None]

        class Args(BaseProcessor.Args):
            config: "SignalProcessor.Config"

    config: "Config"
    time_ts: float
    input: _input_t | None

    _SLOTS_ = (
        "source_entity_id",
        "update_period_ts",
        "time_ts",
        "input",
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None") -> "pmc.ConfigSchema":
        if config is None:
            config = {}
        return {
            hv.req_config("source_entity_id", config): hv.sensor_selector(),
            hv.opt_config("update_period_seconds", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
        }

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        self.time_ts = None  # type: ignore
        self.input = None
        super().__init__(id, **kwargs)
        self.source_entity_id = self.config.get("source_entity_id")
        self.update_period_ts = self.config.get("update_period_seconds", 0)

    @typing.override
    async def async_start(self):
        if self.source_entity_id:
            self.track_state(self.source_entity_id, self._source_entity_update)
        if self.update_period_ts:
            self.track_timer(self.update_period_ts, self._update_callback)

    @typing.override
    def as_dict(self):
        """Used for serialization to debug files or so.
        Returns the configuration."""
        return {
            "source_entity_id": self.source_entity_id,
            "update_period_ts": self.update_period_ts,
        }

    def process(self, input: _input_t, time_ts: float) -> typing.Any:
        self.time_ts = time_ts
        self.input = input

    def update(self, time_ts: float):
        self.time_ts = time_ts

    def _source_convert(self, value: float, from_unit: str | None, to_unit: str | None):
        pass

    @callback
    def _source_entity_update(
        self, event: "Event[EventStateChangedData] | CallbackTracker.Event"
    ):
        pass

    @callback
    def _update_callback(self):
        self.update(TIME_TS())


class SignalEnergyProcessor(SignalProcessor[float], EnergyBroadcast):
    """Computes the energy of a signal i.e. integration over time.
    Accepts input as a plain signal for a classical left rectangle integration or an energy input signal
    so that it 'just' computes differences.
    Input signals can be 'clipped' (maximum/minimum power). This feature allows the integration to only work for
    a limited range of inputs so to cut-out noise (sometimes we have wrong readings in source sensors) or
    in the same way, only integrate positive/negative signals.
    maximum_latency instead acts as a warning (when input signal doesn't update in a while) and also as a safety
    to prevent integrating long lasting signals that could have just been 'stalling'."""

    if typing.TYPE_CHECKING:

        class Config(SignalProcessor.Config):
            maximum_latency_seconds: NotRequired[float]
            """Maximum time between source pv power/energy samples before considering an error in data sampling."""
            input_max: NotRequired[float]
            """Maximum power expected at the input used to filter out outliers from processing. If not set disables the check."""
            input_min: NotRequired[float]
            """Minimum power expected at the input used to filter out outliers from processing. If not set disables the check."""

        class Args(SignalProcessor.Args):
            config: "SignalEnergyProcessor.Config"

        input_mode: Final[bool]
        input_unit: Final[str | None]
        """input_mode is not implicitly initialized since it must be configured by calling
        configure() before process()."""
        energy: float

    # these defaults are applied when the corresponding config option is not set or is 0
    # they means by default we're measuring positive energies with (almost) no latency checks
    # In code they're checked anyway but the set limits should be high enough to not pose any real issue
    # preferring this behavior over spending time against checking for 'disabled'
    MAXIMUM_LATENCY_DISABLED = 1e6
    SAFE_MAXIMUM_POWER_DISABLED = 1e6
    SAFE_MINIMUM_POWER_DISABLED = -1e6

    class InputMode(enum.Enum):
        POWER = (False, BaseProcessor.Unit.POWER_UNIT)
        ENERGY = (True, BaseProcessor.Unit.ENERGY_UNIT)

    class WARNINGS(enum.StrEnum):
        maximum_latency = enum.auto()
        out_of_range = enum.auto()

    # built and set automatically on super().__init__
    warning_maximum_latency: ProcessorWarning
    warning_out_of_range: ProcessorWarning

    _SLOTS_ = (
        "maximum_latency_ts",
        "input_max",
        "energy_max",
        "input_min",
        "energy_min",
        "input_mode",
        "input_unit",
        "energy",
        "warning_maximum_latency",
        "warning_out_of_range",
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None") -> "pmc.ConfigSchema":
        if config is None:
            config = {}
        return {
            hv.opt_config("source_entity_id", config): hv.sensor_selector(
                device_class=[Sensor.DeviceClass.POWER, Sensor.DeviceClass.ENERGY]
            ),
            hv.opt_config("update_period_seconds", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
            hv.opt_config("maximum_latency_seconds", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
            hv.opt_config("input_max", config): hv.number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
            hv.opt_config("input_min", config): hv.number_selector(
                unit_of_measurement=hac.UnitOfPower.WATT
            ),
        }

    def __init__(
        self,
        id,
        **kwargs: "Unpack[Args]",
    ):

        config = kwargs["config"]
        self.maximum_latency_ts = (
            config.get("maximum_latency_seconds", 0)
            or SignalEnergyProcessor.MAXIMUM_LATENCY_DISABLED
        )
        self.input_max = config.get(
            "input_max", SignalEnergyProcessor.SAFE_MAXIMUM_POWER_DISABLED
        )
        self.energy_max = self.input_max / 3600
        self.input_min = config.get(
            "input_min", SignalEnergyProcessor.SAFE_MINIMUM_POWER_DISABLED
        )
        self.energy_min = self.input_min / 3600
        self.input_unit = None
        self.energy = 0
        super().__init__(id, **kwargs)

    @typing.override
    def as_dict(self):
        return {
            "maximum_latency_seconds": (
                None
                if self.maximum_latency_ts
                is SignalEnergyProcessor.MAXIMUM_LATENCY_DISABLED
                else self.maximum_latency_ts
            ),
            "input_max": (
                None
                if self.input_max is SignalEnergyProcessor.SAFE_MAXIMUM_POWER_DISABLED
                else self.input_max
            ),
            "input_min": (
                None
                if self.input_min is SignalEnergyProcessor.SAFE_MINIMUM_POWER_DISABLED
                else self.input_min
            ),
        }

    def configure(self, input_mode: InputMode):
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
                    if self.energy_min <= energy / d_ts <= self.energy_max:
                        if self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                        self.energy += energy
                        for energy_listener in self.energy_listeners:
                            energy_listener(energy, time_ts)
                    else:
                        # assume an energy reset or out of range
                        energy = None
                        if not self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                else:
                    # power left rect integration
                    if self.input_min <= self.input <= self.input_max:  # type: ignore
                        if self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                        energy = self.input * d_ts / 3600  # type: ignore
                        self.energy += energy
                        for energy_listener in self.energy_listeners:
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
        try:
            if self.input_mode:
                # TODO: interpolate energy
                pass
            else:
                self.process(self.input, time_ts)
        except Exception as e:
            # This might happen if we use interpolation on 'invalid' states
            # i.e. when entities don't update or we've still not fully initialized
            # This might be a subtle error though but we just log when debugging
            self.log_exception(self.DEBUG, e, "calling update()", timeout=1800)

    @callback
    @typing.override
    def _source_entity_update(
        self, event: "Event[EventStateChangedData] | CallbackTracker.Event"
    ):
        try:
            state = event.data["new_state"]
            self.process(
                self._source_convert(
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
    def _source_convert(
        self, value: float, from_unit: str | None, to_unit: str | None
    ) -> float:
        """Installed as _source_convert at init time this will detect the type of source entity
        by inspecting the unit and install the proper converter."""
        if from_unit in hac.UnitOfPower:
            self._source_convert = BaseProcessor.Converter.PowerConverter.convert
            self.configure(self.InputMode.POWER)
        elif from_unit in hac.UnitOfEnergy:
            self._source_convert = BaseProcessor.Converter.EnergyConverter.convert
            self.configure(self.InputMode.ENERGY)
        else:
            # TODO: raise issue?
            raise ValueError(f"Unsupported unit of measurement '{from_unit}'")
        return self._source_convert(value, from_unit, self.input_unit)


class Estimator(BaseProcessor):

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config):
            pass

        class Args(BaseProcessor.Args):
            pass

        estimation_time_ts: int

        UPDATE_LISTENER_TYPE = Callable[["Estimator"], None]
        _update_listeners: Final[set[UPDATE_LISTENER_TYPE]]

    _SLOTS_ = (
        "estimation_time_ts",
        "_update_listeners",
    )

    def __init__(self, id, **kwargs: "Unpack[Args]"):
        super().__init__(id, **kwargs)
        self.estimation_time_ts = 0
        self._update_listeners = set()

    @typing.override
    def shutdown(self):
        self._update_listeners.clear()
        super().shutdown()

    @typing.override
    def get_state_dict(self):
        return super().get_state_dict() | {
            "estimation_time": datetime_from_epoch(self.estimation_time_ts).isoformat(),
        }

    # interface: self
    def listen_update(self, callback_func: "UPDATE_LISTENER_TYPE"):
        self._update_listeners.add(callback_func)

        def _unsub():
            try:
                self._update_listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    @abc.abstractmethod
    def update_estimate(self):
        for listener in self._update_listeners:
            listener(self)
