import abc
import enum
import typing


class ProcessorWarning:

    CALLBACK_TYPE = typing.Callable[[bool], None]

    id: str
    on: bool
    listeners: set[CALLBACK_TYPE]

    __slots__ = (
        "id",
        "on",
        "listeners",
    )

    def __init__(self, processor: "BaseProcessor", id: str):
        self.id = id
        self.on = False
        self.listeners = set()
        setattr(processor, f"warning_{id}", self)

    def shutdown(self):
        self.listeners.clear()

    def listen(self, callback_func: CALLBACK_TYPE):
        self.listeners.add(callback_func)
        callback_func(self.on)

        def remove():
            self.listeners.remove(callback_func)

        return remove

    def toggle(self):
        self.on = not self.on
        for listener in self.listeners:
            listener(self.on)


class BaseProcessor[_input_t, _output_t](abc.ABC):

    time_ts: float
    input: _input_t
    output: _output_t

    WARNINGS: typing.ClassVar[typing.Iterable[str]] = ()
    warnings: typing.Final[set[ProcessorWarning]]

    __slots__ = (
        "time_ts",
        "input",
        "output",
        "warnings",
    )

    def __init__(self):
        self.time_ts = None # type: ignore
        self.warnings = {ProcessorWarning(self, id) for id in self.WARNINGS}

    def configure(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def process(self, input: _input_t, time_ts: float) -> typing.Any:
        self.time_ts = time_ts
        self.input = input

    def shutdown(self):
        """Used to remove references when wanting to shutdown resources usage."""
        for warning in self.warnings:
            warning.shutdown()

    def as_dict(self):
        """Used for serialization to debug files or so."""
        return {}

    def get_state_dict(self):
        """Returns a synthetic state dict for the estimator.
        Used for debugging purposes."""
        return {
            "warnings": { warning.id: warning.on for warning in self.warnings},
        }


class EnergyInputMode(enum.Enum):
    POWER = (False, "W")
    ENERGY = (True, "Wh")


class EnergyProcessorWarningId(enum.StrEnum):
    maximum_latency = enum.auto()
    out_of_range = enum.auto()


MAXIMUM_LATENCY_DISABLED = 1e6
SAFE_MAXIMUM_POWER_DISABLED = 1e6


class EnergyProcessorConfig(typing.TypedDict):
    maximum_latency_minutes: typing.NotRequired[float]
    """Maximum time between source pv power/energy samples before considering an error in data sampling."""
    safe_maximum_power_w: typing.NotRequired[float]
    """Maximum power expected at the input used to filter out outliers from processing. If not set disables the chcek."""


class BaseEnergyProcessor(BaseProcessor[float, float]):

    input_mode: typing.Final[bool]
    input_unit: typing.Final[str | None]
    """input_mode is not implicitly initialized since it must be configured by calling
    configure() before process()."""

    WARNINGS = EnergyProcessorWarningId

    # built and set automatically on super().__init__
    warning_maximum_latency: ProcessorWarning
    warning_out_of_range: ProcessorWarning

    ENERGY_LISTENER_TYPE = typing.Callable[[float, float], None]
    energy_listeners: typing.Final[set[ENERGY_LISTENER_TYPE]]

    __slots__ = (
        "maximum_latency_ts",
        "safe_maximum_power",
        "safe_maximum_power_cal",
        "input_mode",
        "input_unit",
        "warning_maximum_latency",
        "warning_out_of_range",
        "energy_listeners",
    )

    def __init__(
        self,
        **kwargs: typing.Unpack[EnergyProcessorConfig],
    ):
        self.maximum_latency_ts = (
            kwargs.get("maximum_latency_minutes", 0) * 60 or MAXIMUM_LATENCY_DISABLED
        )
        self.safe_maximum_power = (
            kwargs.get("safe_maximum_power_w") or SAFE_MAXIMUM_POWER_DISABLED
        )
        self.safe_maximum_power_cal = self.safe_maximum_power / 3600
        self.input_unit = None
        self.energy_listeners = set()
        BaseProcessor.__init__(self)
        self.output = 0

    @typing.override
    def shutdown(self):
        """Used to remove references when wanting to shutdown resources usage."""
        self.energy_listeners.clear()
        BaseProcessor.shutdown(self)

    @typing.override
    def configure(
        self,
        input_mode: EnergyInputMode,
    ):
        self.input_mode, self.input_unit = input_mode.value # type: ignore

    @typing.override
    def process(self, input: float, time_ts: float) -> float | None:

        try:
            d_ts = time_ts - self.time_ts
            if 0 <= d_ts < self.maximum_latency_ts:

                if self.warning_maximum_latency.on:
                    self.warning_maximum_latency.toggle()

                if self.input_mode:
                    energy = input - self.input
                    if 0 <= energy <= self.safe_maximum_power_cal * d_ts:
                        if self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                        self.output += energy
                        for energy_listener in self.energy_listeners:
                            energy_listener(energy, time_ts)
                    else:
                        # assume an energy reset or out of range
                        energy = None
                        if not self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                else:
                    # power left rect integration
                    if 0 <= self.input <= self.safe_maximum_power:  # type: ignore
                        if self.warning_out_of_range.on:
                            self.warning_out_of_range.toggle()
                        energy = self.input * d_ts / 3600
                        self.output += energy
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
            # expected right at the first call..use this to initialize the state
            self.input = input
            self.time_ts = time_ts
            return None
        except AttributeError as error:
            if error.name in ("time_ts", "input"):
                # expected right at the first call..use this to initialize the state
                self.input = input
                self.time_ts = time_ts
                return None
            if error.name == "input_mode":
                raise Exception("configure() need to be called before process()")
            raise error

    def interpolate(self, time_ts: float):
        if self.input_mode:
            # TODO: interpolate energy
            pass
        else:
            self.process(self.input, time_ts)

    @typing.override
    def as_dict(self):
        """Returns the full state info of the estimator as a dictionary.
        Used for serialization to debug logs or so."""
        return {
            "maximum_latency_minutes": (
                None
                if self.maximum_latency_ts is MAXIMUM_LATENCY_DISABLED
                else self.maximum_latency_ts / 60
            ),
            "safe_maximum_power_w": (
                None
                if self.safe_maximum_power is SAFE_MAXIMUM_POWER_DISABLED
                else self.safe_maximum_power
            ),
        }
