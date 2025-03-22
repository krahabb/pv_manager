"""
Helpers!
"""

import abc
import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
import importlib
import logging
from time import gmtime, time
import typing

from homeassistant import const as hac
from homeassistant.core import callback

from .. import const as pmc

if typing.TYPE_CHECKING:
    from datetime import tzinfo
    from types import MappingProxyType
    from typing import Any, Callable, Coroutine

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant


def clamp(_value, _min, _max):
    """
    saturate _value between _min and _max
    """
    if _value >= _max:
        return _max
    elif _value <= _min:
        return _min
    else:
        return _value


def reverse_lookup(_dict: dict, value):
    """
    lookup the values in map (dict) and return
    the corresponding key
    """
    for _key, _value in _dict.items():
        if _value == value:
            return _key
    return None


def versiontuple(version: str):
    """
    helper for version checking, comparisons, etc
    """
    return tuple(map(int, (version.split("."))))


def datetime_from_epoch(epoch, tz: "tzinfo | None" = UTC):
    """
    converts an epoch (UTC seconds) in a datetime.
    Faster than datetime.fromtimestamp with less checks
    and no care for milliseconds.
    If tz is None it'll return a naive datetime in UTC coordinates
    """
    y, m, d, hh, mm, ss, weekday, jday, dst = gmtime(epoch)
    utcdt = datetime(y, m, d, hh, mm, min(ss, 59), 0, UTC)
    if tz is UTC:
        return utcdt
    elif tz is None:
        return utcdt.replace(tzinfo=None)
    else:
        return utcdt.astimezone(tz)


_import_module_lock = asyncio.Lock()
_import_module_cache = {}


async def async_import_module(hass: "HomeAssistant", name: str):

    try:
        return _import_module_cache[name]
    except KeyError:
        async with _import_module_lock:
            # check (again) the module was not asyncronously loaded when waiting the lock
            try:
                return _import_module_cache[name]
            except KeyError:
                module = await hass.async_add_executor_job(
                    importlib.import_module,
                    name,
                    "custom_components.pv_manager",
                )
                _import_module_cache[name] = module
                return module


def typed_dict_keys(td)-> tuple:
    """If td is a TypedDict class, return a dictionary mapping the typed keys to types.
    Otherwise, return None. Examples::

        class TD(TypedDict):
            x: int
            y: int
        class Other(dict):
            x: int
            y: int

        typed_dict_keys(TD) == {'x': int, 'y': int}
        typed_dict_keys(dict) == None
        typed_dict_keys(Other) == None
    """

    #if isinstance(td, typing._TypedDict):
    if hasattr(td, "__annotations__"):
        # return td.__annotations__.copy() returns the full dict
        return tuple(td.__annotations__)

    return ()

def apply_config(obj, config: "MappingProxyType[str, object]", td_class):
    if td_class:
        for key in typed_dict_keys(td_class):
            if key in config:
                setattr(obj, key, config[key])
    else:
        for key, value in config.items():
            setattr(obj, key, value)


try:
    import json
    import os

    class DEBUG:
        """Define a DEBUG symbol which will be None in case the debug conf is missing so
        that the code can rely on this to enable special behaviors."""
        # this will raise an OSError on non-dev machines missing the
        # debug configuration file so the DEBUG symbol will be invalidated
        data = json.load(
            open(
                file="./custom_components/pv_manager/debug.secret.json",
                mode="r",
                encoding="utf-8",
            )
        )

        @staticmethod
        def get_debug_output_filename(hass: "HomeAssistant", filename):
            path = hass.config.path(
                "custom_components", pmc.DOMAIN, "debug"
            )
            os.makedirs(path, exist_ok=True)
            return os.path.join(path, filename)


except Exception:
    DEBUG = None  # type: ignore


def getLogger(name):
    """
    Replaces the default Logger with our wrapped implementation:
    replace your logging.getLogger with helpers.getLogger et voilà
    """
    logger = logging.getLogger(name)
    # watchout: getLogger could return an instance already
    # subclassed if we previously asked for the same name
    # for example when we reload a config entry
    _class = logger.__class__
    if _class not in _Logger._CLASS_HOOKS.values():
        # getLogger returned a 'virgin' class
        if _class in _Logger._CLASS_HOOKS.keys():
            # we've alread subclassed this type, so we reuse it
            logger.__class__ = _Logger._CLASS_HOOKS[_class]
        else:
            logger.__class__ = _Logger._CLASS_HOOKS[_class] = type(
                "Logger",
                (
                    _Logger,
                    logger.__class__,
                ),
                {},
            )

    return logger


class _Logger(logging.Logger if typing.TYPE_CHECKING else object):
    """
    This wrapper will 'filter' log messages and avoid
    verbose over-logging for the same message by using a timeout
    to prevent repeating the very same log before the timeout expires.
    The implementation 'hacks' a standard Logger instance by mixin-ing
    """

    # default timeout: these can be overriden at the log call level
    # by passing in the 'timeout=' param
    # for example: LOGGER.error("This error will %s be logged again", "soon", timeout=5)
    # it can also be overriden at the 'Logger' instance level
    default_timeout = 60 * 60 * 8
    # cache of logged messages with relative last-thrown-epoch
    _LOGGER_TIMEOUTS = {}
    # cache of subclassing types: see getLogger
    _CLASS_HOOKS = {}

    def _log(self, level, msg, args, **kwargs):
        if "timeout" in kwargs:
            timeout = kwargs.pop("timeout")
            epoch = time()
            trap_key = (msg, args)
            if trap_key in _Logger._LOGGER_TIMEOUTS:
                if (epoch - _Logger._LOGGER_TIMEOUTS[trap_key]) < timeout:
                    if self.isEnabledFor(pmc.CONF_LOGGING_VERBOSE):
                        super()._log(
                            pmc.CONF_LOGGING_VERBOSE,
                            f"dropped log message for {msg}",
                            args,
                            **kwargs,
                        )
                    return
            _Logger._LOGGER_TIMEOUTS[trap_key] = epoch

        super()._log(level, msg, args, **kwargs)


LOGGER = getLogger(__name__[:-8])  # get base custom_component name for logging
"""Root logger"""


class Loggable(abc.ABC):
    """
    Helper base class for logging instance name/id related info.
    Derived classes can customize this in different flavours:
    - basic way is to set 'logtag' to provide a custom name.
    - custom way by overriding 'log'.
    """

    hac = hac

    VERBOSE = pmc.CONF_LOGGING_VERBOSE
    DEBUG = pmc.CONF_LOGGING_DEBUG
    INFO = pmc.CONF_LOGGING_INFO
    WARNING = pmc.CONF_LOGGING_WARNING
    CRITICAL = pmc.CONF_LOGGING_CRITICAL

    __slots__ = (
        "id",
        "logtag",
        "logger",
        "__dict__",
    )

    def __init__(
        self,
        id,
        *,
        logger: "Loggable | logging.Logger" = LOGGER,
    ):
        self.id: typing.Final = id
        self.logger = logger
        self.configure_logger()
        self.log(self.DEBUG, "init")

    def configure_logger(self):
        self.logtag = f"{self.__class__.__name__}({self.id})"

    def isEnabledFor(self, level: int):
        return self.logger.isEnabledFor(level)

    def log(self, level: int, msg: str, *args, **kwargs):
        self.logger.log(level, f"{self.logtag}: {msg}", *args, **kwargs)

    def log_exception(
        self, level: int, exception: Exception, msg: str, *args, **kwargs
    ):
        self.log(
            level,
            f"{exception.__class__.__name__}({str(exception)}) in {msg}",
            *args,
            **kwargs,
        )

    @contextmanager
    def exception_warning(self, msg: str, *args, **kwargs):
        try:
            yield
        except Exception as exception:
            self.log_exception(self.WARNING, exception, msg, *args, **kwargs)

    def __del__(self):
        self.log(self.DEBUG, "destroy")
