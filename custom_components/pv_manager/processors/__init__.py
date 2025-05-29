"""
processors:
base classes for signal processing
These are mainly abstract and are designed to be included in complex
diamond pattern hierarchies. This is especially true for
estimators which should always include a signal processor to be feeded
"""

import abc
from dataclasses import asdict, dataclass
import enum
import inspect
import typing

from homeassistant import const as hac
from homeassistant.core import callback
from homeassistant.util import unit_conversion

from ..helpers import Loggable, datetime_from_epoch, validation as hv
from ..helpers.callback import CallbackTracker
from ..helpers.dataattr import DataAttr, DataAttrClass
from ..manager import Manager
from ..sensor import Sensor

if typing.TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        ClassVar,
        Final,
        Iterable,
        Mapping,
        NotRequired,
        TypedDict,
        Unpack,
    )

    from homeassistant.core import Event, EventStateChangedData, State

    from .. import const as pmc


class GenericBroadcast[*_argsT]:

    if typing.TYPE_CHECKING:
        type LISTENER_TYPE = Callable[[*_argsT], Any]
        _listeners: set[LISTENER_TYPE]

    __slots__ = ("_listeners",)

    def shutdown(self):
        try:
            self._listeners.clear()
        except AttributeError:
            pass

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

    def broadcast(self, *args: *_argsT):
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

    def __init__(self, id, **kwargs: "Unpack[Loggable.Args]"):
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
    """This class represent a warning signal controlled by any owning class instance.
    This is typically a Processor but it could be anything in theory. It manipulates a bit
    the owning instance attributes to speed up access and creates some 'friedly'.
    In fact, some of it's state is held there."""

    if typing.TYPE_CHECKING:

        type ACTIVATE_TYPE = Callable[[], None]
        type DEACTIVATE_TYPE = Callable[[], None]

        processor: "BaseProcessor"
        id: str  # could be any object but it should have a 'proper' str behaviour

        type CALLBACK_TYPE = Callable[[bool], None]
        _listeners: set[CALLBACK_TYPE]

    __slots__ = (
        "processor",
        "id",
        "_listeners",
    )

    def __init__(self, processor: "BaseProcessor", id):
        self.processor = processor
        self.id = id
        self._listeners = set()
        setattr(processor, f"warning_{id}", self)
        setattr(processor, f"warning_{id}_on", False)
        setattr(processor, f"warning_{id}_activate", self.activate)
        setattr(processor, f"warning_{id}_deactivate", self.deactivate)

    def shutdown(self):
        self._listeners.clear()
        processor = self.processor
        id = self.id
        delattr(processor, f"warning_{id}")
        delattr(processor, f"warning_{id}_on")
        delattr(processor, f"warning_{id}_activate")
        delattr(processor, f"warning_{id}_deactivate")
        self.processor = None  # type: ignore

    @property
    def on(self):
        return getattr(self.processor, f"warning_{self.id}_on")

    def listen(self, callback_func: "CALLBACK_TYPE", /):
        self._listeners.add(callback_func)

        def _unsub():
            try:
                self._listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    def activate(self):
        setattr(self.processor, f"warning_{self.id}_on", True)
        for listener in self._listeners:
            listener(True)

    def deactivate(self):
        setattr(self.processor, f"warning_{self.id}_on", False)
        for listener in self._listeners:
            listener(False)


class BaseProcessor(CallbackTracker, Loggable, DataAttrClass):

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
            ProcessorWarning(self, warning_id) for warning_id in self.__class__.WARNINGS
        }

    async def async_start(self):
        pass

    def shutdown(self):
        """Used to remove references when wanting to shutdown resources usage."""
        super().shutdown()
        for warning in self.warnings:
            warning.shutdown()

    def as_diagnostic_dict(self):
        """Used for serialization to debug files or so.
        Returns the configuration."""
        return {}

    def as_state_dict(self):
        """Returns a synthetic state dict.
        Used for debugging purposes."""
        return {"state": self.as_formatted_dict()}


class UnitOfElectricCharge(enum.StrEnum):
    AMPERE_HOUR = "Ah"


class ElectricChargeConverter(unit_conversion.BaseUnitConverter):
    """Utility to convert electric potential values."""

    UNIT_CLASS = "charge"
    _UNIT_CONVERSION: dict[str | None, float] = {
        UnitOfElectricCharge.AMPERE_HOUR: 1,
    }
    VALID_UNITS = {
        UnitOfElectricCharge.AMPERE_HOUR,
    }


class _UnitDef:
    if typing.TYPE_CHECKING:
        units: Final[type[enum.StrEnum]]
        default: Final[enum.StrEnum]
        converter: Final[type[unit_conversion.BaseUnitConverter]]
        convert: Final["SignalProcessor.ConvertFuncType"]
        convert_to_default: Final[Mapping[str | None, float]]

    __slots__ = (
        "units",
        "default",
        "converter",
        "convert",
        "convert_to_default",
        "_name_",
        "_value_",
        "__dict__",
    )

    def __init__(
        self, default: enum.StrEnum, converter: type[unit_conversion.BaseUnitConverter]
    ):
        self.units = type(default)
        self.default = default
        self.converter = converter
        self.convert = converter.convert
        self.convert_to_default = {
            unit: converter.get_unit_ratio(default, unit)
            for unit in converter.VALID_UNITS
        }
        self._value_ = self


class SignalProcessor[_input_t](BaseProcessor):

    @typing.final
    class Unit(_UnitDef, enum.Enum):
        """Default units being used in our signal processing."""

        def __new__(cls, *values):
            return _UnitDef(*values)

        VOLTAGE = (
            hac.UnitOfElectricPotential.VOLT,
            unit_conversion.ElectricPotentialConverter,
        )
        CURRENT = (
            hac.UnitOfElectricCurrent.AMPERE,
            unit_conversion.ElectricCurrentConverter,
        )
        POWER = (hac.UnitOfPower.WATT, unit_conversion.PowerConverter)
        ENERGY = (hac.UnitOfEnergy.WATT_HOUR, unit_conversion.EnergyConverter)
        CHARGE = (UnitOfElectricCharge.AMPERE_HOUR, ElectricChargeConverter)

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config):
            source_entity_id: NotRequired[str | None]
            update_period: NotRequired[float | None]

        class Args(BaseProcessor.Args):
            config: "SignalProcessor.Config"

        type ConvertFuncType = Callable[[float, str | None, str | None], float]

        config: "Config"
        time_ts: float
        input: _input_t | None
        unit: Final[Unit]
        input_unit: Final[str | None]
        input_convert: Final[Mapping[str | None, float]]

    _SLOTS_ = (
        "source_entity_id",
        "update_period_ts",
        "unit",
        "time_ts",
        "input",
        "unit",
        "input_unit",
        "input_convert",
    )

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None", /) -> "pmc.ConfigSchema":
        if config is None:
            config = {}
        return {
            hv.req_config("source_entity_id", config): hv.sensor_selector(),
            hv.opt_config("update_period", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
        }

    def __init__(self, id, /, **kwargs: "Unpack[Args]"):
        self.time_ts = None  # type: ignore
        self.input = None
        self.input_unit = None
        super().__init__(id, **kwargs)
        self.source_entity_id = self.config.get("source_entity_id")
        self.update_period_ts = self.config.get("update_period", 0)

    @typing.override
    async def async_start(self):
        if self.source_entity_id:
            self.track_state(self.source_entity_id, self._source_entity_update)
        if self.update_period_ts:
            self.track_timer(self.update_period_ts, self._update_callback)

    @typing.override
    def as_diagnostic_dict(self):
        """Used for serialization to debug files or so.
        Returns the configuration."""
        return {
            "source_entity_id": self.source_entity_id,
            "update_period_ts": self.update_period_ts,
        }

    def configure(self, unit: Unit, /):
        self.unit = unit  # type: ignore
        self.input_unit = unit.default  # type: ignore
        self.input_convert = unit.convert_to_default  # type: ignore
        self._source_convert = unit.convert

    def process(self, input: _input_t | None, time_ts: float, /) -> typing.Any:
        self.time_ts = time_ts
        self.input = input

    def update(self, time_ts: float, /):
        self.time_ts = time_ts

    @callback
    def _source_entity_update(
        self, event: "Event[EventStateChangedData] | CallbackTracker.Event", /
    ):
        try:
            state = event.data["new_state"]
            self.process(
                float(state.state) * self.input_convert[state.attributes["unit_of_measurement"]],  # type: ignore
                event.time_fired_timestamp,
            )  # type: ignore
            return
        except AttributeError as e:
            if e.name == "input_convert":
                exception = self.configure_source(state, event.time_fired_timestamp)
                if not exception:
                    return
            else:
                exception = e
        except Exception as e:
            exception = e

        # this is expected and silently managed when state == None or 'unknown'
        self.process(None, event.time_fired_timestamp)
        if state and state.state not in (hac.STATE_UNKNOWN, hac.STATE_UNAVAILABLE):
            self.log_exception(
                self.WARNING,
                exception,
                "_source_entity_update (state:%s)",
                state,
            )

    def configure_source(self, state: "State | None", time_ts: float):
        """Called in an AttributerError handler when (at start) we need to configure unit conversion."""
        from_unit = state.attributes["unit_of_measurement"]  # type: ignore
        for unit in SignalProcessor.Unit:
            if from_unit in unit.units:
                self.configure(unit)
                try:
                    self.process(
                        float(state.state) * self.input_convert[from_unit],  # type: ignore
                        time_ts,
                    )
                    return None
                except KeyError:
                    return ValueError(
                        f"No conversion available from '{from_unit}' to '{self.input_unit}'"
                    )
        else:
            return ValueError(f"Unit '{from_unit}' not supported")

    def _source_convert(
        self, value: float, from_unit: str | None, to_unit: str | None, /
    ) -> float:
        """Installed as _source_convert at init time this will detect the type of source entity
        by inspecting the unit and install the proper converter.
        This is an helping method used in conjunction with tracking an HA entity"""
        for unit in SignalProcessor.Unit:
            if from_unit in unit.units:
                self.configure(unit)
                # TODO: setup a local dict of coefficients to map conversion without
                # having to invoke the HA lib convert function every time.
                # This would allow us to invert signals by inverting the map in place
                # and not going through an additional multiplication
                return self._source_convert(value, from_unit, self.input_unit)

        raise ValueError(f"Unsupported unit of measurement '{from_unit}'")

    @callback
    def _update_callback(self, time_ts: float, /):
        self.update(time_ts)


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
            maximum_latency: NotRequired[float]
            """Maximum time between source pv power/energy samples before considering an error in data sampling."""
            input_max: NotRequired[float]
            """Maximum power expected at the input used to filter out outliers from processing. If not set disables the check."""
            input_min: NotRequired[float]
            """Minimum power expected at the input used to filter out outliers from processing. If not set disables the check."""

        class Args(SignalProcessor.Args):
            config: "SignalEnergyProcessor.Config"

        ENERGY_UNITS: Final

        # built and set automatically on super().__init__
        warning_no_signal: ProcessorWarning
        warning_no_signal_on: bool
        warning_no_signal_activate: ProcessorWarning.ACTIVATE_TYPE
        warning_no_signal_deactivate: ProcessorWarning.DEACTIVATE_TYPE
        warning_maximum_latency: ProcessorWarning
        warning_maximum_latency_on: bool
        warning_maximum_latency_activate: ProcessorWarning.ACTIVATE_TYPE
        warning_maximum_latency_deactivate: ProcessorWarning.DEACTIVATE_TYPE
        warning_out_of_range: ProcessorWarning
        warning_out_of_range_on: bool
        warning_out_of_range_activate: ProcessorWarning.ACTIVATE_TYPE
        warning_out_of_range_deactivate: ProcessorWarning.DEACTIVATE_TYPE

        _differential_mode: bool
        """_differential_mode is not implicitly initialized since it must be configured by calling
        configure() before process()."""

    # these defaults are applied when the corresponding config option is not set or is 0
    # they means by default we're measuring positive energies with (almost) no latency checks
    # In code they're checked anyway but the set limits should be high enough to not pose any real issue
    # preferring this behavior over spending time against checking for 'disabled'
    MAXIMUM_LATENCY_DISABLED = 1e6
    SAFE_MAXIMUM_POWER_DISABLED = 1e6
    SAFE_MINIMUM_POWER_DISABLED = -1e6

    ENERGY_UNITS = (SignalProcessor.Unit.ENERGY, SignalProcessor.Unit.CHARGE)

    class WARNINGS(enum.StrEnum):
        no_signal = enum.auto()
        maximum_latency = enum.auto()
        out_of_range = enum.auto()

    _SLOTS_ = (
        "maximum_latency_ts",
        "input_max",
        "energy_max",
        "input_min",
        "energy_min",
        "_differential_mode",
        "warning_no_signal",
        "warning_maximum_latency",
        "warning_out_of_range",
    )
    # TODO: move warnings related slots to automatic class post_init in BaseProcessor

    @classmethod
    @typing.override
    def get_config_schema(cls, config: "Config | None", /) -> "pmc.ConfigSchema":
        if config is None:
            config = {}
        return {
            hv.opt_config("source_entity_id", config): hv.sensor_selector(
                device_class=[Sensor.DeviceClass.POWER, Sensor.DeviceClass.ENERGY]
            ),
            hv.opt_config("update_period", config): hv.time_period_selector(
                unit_of_measurement=hac.UnitOfTime.SECONDS
            ),
            hv.opt_config("maximum_latency", config): hv.time_period_selector(
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
            config.get("maximum_latency", 0)
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
        super().__init__(id, **kwargs)
        self.warning_no_signal_on = True

    @typing.override
    def as_diagnostic_dict(self):
        return {
            "maximum_latency": (
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

    @typing.override
    def configure(self, unit: SignalProcessor.Unit, /):
        super().configure(unit)
        self._differential_mode = unit in SignalEnergyProcessor.ENERGY_UNITS

    @typing.override
    def process(self, input: float | None, time_ts: float, /) -> float | None:

        try:
            d_ts = time_ts - self.time_ts
            if 0 <= d_ts < self.maximum_latency_ts:

                if self.warning_maximum_latency_on:
                    self.warning_maximum_latency_deactivate()

                if self._differential_mode:
                    energy = input - self.input  # type: ignore
                    if self.energy_min <= energy / d_ts <= self.energy_max:
                        if self.warning_out_of_range_on:
                            self.warning_out_of_range_deactivate()
                        for energy_listener in self.energy_listeners:
                            energy_listener(energy, time_ts)
                    else:
                        # assume an energy reset or out of range
                        energy = None
                        if not self.warning_out_of_range_on:
                            self.warning_out_of_range_activate()
                else:
                    # power left rect integration
                    energy = self.input * d_ts / 3600  # type: ignore
                    for energy_listener in self.energy_listeners:
                        energy_listener(energy, time_ts)
                    if self.input_min <= input <= self.input_max:  # type: ignore
                        if self.warning_out_of_range_on:
                            self.warning_out_of_range_deactivate()
                    else:
                        # discard the out of range observation
                        input = None
                        if not self.warning_out_of_range_on:
                            self.warning_out_of_range_activate()

            else:
                energy = None
                if not self.warning_maximum_latency_on:
                    self.warning_maximum_latency_activate()

            self.input = input
            self.time_ts = time_ts
            return energy

        except TypeError as error:
            # This code path 'must' be executed whenever input or self.input are 'None'
            # in order to correctly manage the warning
            # The context should be like:
            # - self.time_ts == None -> start of processing (equivalent to returning from loss of signal)
            # - input == None -> loss of signal
            # - self.input == None -> previous reading was unavailable due to
            # recovering from loss of signal or any other condition
            if input is None:
                if not self.warning_no_signal_on:
                    self.warning_no_signal_activate()
            else:
                if self.warning_no_signal_on:
                    self.warning_no_signal_deactivate()
            self.input = input
            self.time_ts = time_ts
            return None
        except AttributeError as error:
            if error.name == "_differential_mode":
                raise Exception("configure() need to be called before process()")
            raise error

    @typing.override
    def update(self, time_ts: float, /):
        if self._differential_mode:
            # TODO: interpolate energy
            pass
        else:
            self.process(self.input, time_ts)

    # interface: self
    def reset(self, /):
        """Called to stop accumulation up until time_ts (see energy estimators) without raising any warning or
        dispatching any current accumulated energy."""
        # This should do the job:
        # Now the state is almost like at start and the first sample being processed will
        # start a new accumulation (provided it is a valid value though)
        self.input = None


class Estimator(BaseProcessor):

    if typing.TYPE_CHECKING:

        class Config(BaseProcessor.Config):
            pass

        class Args(BaseProcessor.Args):
            pass

        UPDATE_LISTENER_TYPE = Callable[["Estimator"], None]
        _update_listeners: Final[set[UPDATE_LISTENER_TYPE]]

    estimation_time_ts: DataAttr[int] = 0

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

    # interface: self
    def listen_update(self, callback_func: "UPDATE_LISTENER_TYPE", /):
        self._update_listeners.add(callback_func)

        def _unsub():
            try:
                self._update_listeners.remove(callback_func)
            except KeyError:
                pass

        return _unsub

    def update_estimate(self):
        """Called (internally) whenever we need to update the estimate
        based on new data entering the model."""
        for listener in self._update_listeners:
            listener(self)
