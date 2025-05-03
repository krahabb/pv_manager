import asyncio
import base64
import contextlib
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import re
import time
import typing
from unittest.mock import ANY, MagicMock, patch
import uuid

import aiohttp
from freezegun.api import (
    FrozenDateTimeFactory,
    StepTickTimeFactory,
    TickingDateTimeFactory,
    freeze_time,
)
from homeassistant import config_entries, const as hac
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry  # type: ignore
from pytest_homeassistant_custom_component.common import async_fire_time_changed_exact

from custom_components.pv_manager import const as pmc
from custom_components.pv_manager.config_flow import ConfigFlow
from custom_components.pv_manager.helpers import Loggable

from . import const as tc

if typing.TYPE_CHECKING:
    from typing import Any, Callable, Coroutine, Final

    from homeassistant.config_entries import (
        ConfigEntriesFlowManager,
        ConfigFlowResult,
        OptionsFlowManager,
    )
    from homeassistant.core import HomeAssistant

    from custom_components.pv_manager.controller import Controller
    from custom_components.pv_manager.manager import ManagerClass

_TimeFactory = FrozenDateTimeFactory | StepTickTimeFactory | TickingDateTimeFactory


async def async_assert_flow_menu_to_step(
    flow: "ConfigEntriesFlowManager | OptionsFlowManager",
    result: "ConfigFlowResult",
    menu_step_id: str,
    next_step_id: str,
    next_step_type: FlowResultType = FlowResultType.FORM,
):
    """
    Checks we've entered the menu 'menu_step_id' and chooses 'next_step_id' asserting it works
    Returns the FlowResult at the start of 'next_step_id'.
    """
    assert result["type"] == FlowResultType.MENU  # type: ignore
    assert result["step_id"] == menu_step_id  # type: ignore
    result = await flow.async_configure(
        result["flow_id"],
        user_input={"next_step_id": next_step_id},
    )
    assert result["type"] == next_step_type  # type: ignore
    if next_step_type == FlowResultType.FORM:
        assert result["step_id"] == next_step_id  # type: ignore
    return result


def ensure_registry_entries(hass: "HomeAssistant"):
    """Preloads the entity registry with some default entities needed to configure/run our controllers."""
    ent_reg = er.async_get(hass)

    for entity_id_enum, kwargs in tc.ENTITY_REGISTRY_PRELOAD.items():
        platform, entity_id = entity_id_enum.split(".")
        ent_reg.async_get_or_create(
            platform, platform, entity_id_enum, suggested_object_id=entity_id, **kwargs
        )


class DictMatcher(dict):
    """
    customize dictionary matching by checking if
    only the keys defined in this object are matched in the
    compared one. It works following the same assumptions as for the ANY
    symbol in the mock library
    """

    def __eq__(self, other):
        for key, value in self.items():
            if value != other.get(key):
                return False
        return True


class TimeMocker(contextlib.AbstractContextManager):
    """
    time mocker helper using freeztime and providing some helpers
    to integrate time changes with HA core mechanics.
    At the time, don't use it together with DeviceContext which
    mocks its own time
    """

    time: _TimeFactory

    __slots__ = (
        "hass",
        "time",
        "_freeze_time",
        "_warp_task",
        "_warp_run",
    )

    def __init__(self, hass: "HomeAssistant", time_to_freeze=None):
        super().__init__()
        self.hass = hass
        self._freeze_time = freeze_time(time_to_freeze)
        self._warp_task: asyncio.Future | None = None
        self._warp_run = False
        hass.loop.slow_callback_duration = 2.1

    def __enter__(self):
        self.time = self._freeze_time.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._freeze_time.stop()

    def tick(self, tick: timedelta | float | int):
        self.time.tick(tick if isinstance(tick, timedelta) else timedelta(seconds=tick))
        async_fire_time_changed_exact(self.hass)

    async def async_tick(self, tick: timedelta | float | int):
        self.time.tick(tick if isinstance(tick, timedelta) else timedelta(seconds=tick))
        async_fire_time_changed_exact(self.hass)
        await self.hass.async_block_till_done()

    async def async_move_to(self, target_datetime: datetime):
        self.time.move_to(target_datetime)
        async_fire_time_changed_exact(self.hass)
        await self.hass.async_block_till_done()

    async def async_warp(
        self,
        timeout: float | int | timedelta | datetime,
        tick: float | int | timedelta = 1,
    ):
        if not isinstance(timeout, datetime):
            if isinstance(timeout, timedelta):
                timeout = self.time() + timeout
            else:
                timeout = self.time() + timedelta(seconds=timeout)
        if not isinstance(tick, timedelta):
            tick = timedelta(seconds=tick)

        """
        This basic time ticking doesn't produce fixed time steps increase
        since the time mocker might also be externally manipulated
        (for example our HTTP mocker introduces a delay)
        This introduces a sort of 'drift' in our time sampling
        while we'd better prefer having stable fixed time steps

        while self.time() < timeout:
            await self.async_tick(tick)

        The following solution instead creates it's own time ramp
        and forces the time mocker to follow our prefixed sampling
        even if it was 'ticked' in other parts of the code

        beware though that the additional ticks don't overflow
        our sampling step tick...as the time could then jump back
        according to this algorithm and might be undesirable
        (to say the least - dunno if freezegun allows this)
        """
        time_current = self.time()
        time_next = time_current + tick
        tick_next = tick
        while time_current < timeout:
            await self.async_tick(tick_next)
            # here self.time() might have been advanced more than tick
            time_current = time_next
            time_next = time_current + tick
            tick_next = time_next - self.time()

    def warp(self, tick: float | int | timedelta = 0.5):
        """
        starts an asynchronous task in an executor which manipulates our
        freze_time so the time passes and get advanced to
        time.time() + timeout.
        While passing it tries to perform HA events rollout
        every tick seconds
        """
        assert self._warp_task is None

        if not isinstance(tick, timedelta):
            tick = timedelta(seconds=tick)

        def _warp():
            print("TimeMocker.warp: entering executor")
            count = 0
            while self._warp_run:
                _time = self.time()
                asyncio.run_coroutine_threadsafe(self.async_tick(tick), self.hass.loop)
                while _time == self.time():
                    time.sleep(0.01)
                count += 1
            print(f"TimeMocker.warp: exiting executor (_warp count={count})")

        self._warp_run = True
        self._warp_task = self.hass.async_add_executor_job(_warp)

    async def async_stopwarp(self):
        print("TimeMocker.warp: stopping executor")
        assert self._warp_task
        self._warp_run = False
        await self._warp_task
        self._warp_task = None


class ConfigEntryMocker[_controllerT: "Controller"](
    contextlib.AbstractAsyncContextManager
):

    __slots__ = (
        "hass",
        "config_entry",
        "config_entry_id",
        "auto_setup",
    )

    def __init__(
        self,
        hass: "HomeAssistant",
        data: tc.ConfigEntriesItem,
        *,
        auto_add: bool = True,
        auto_setup: bool = True,
    ) -> None:
        super().__init__()
        self.hass: "Final" = hass
        self.config_entry: "Final" = MockConfigEntry(
            domain=pmc.DOMAIN,
            data=data["data"],
            options=data.get("options"),
            subentries_data=data.get("subentries_data"),
            version=ConfigFlow.VERSION,
            minor_version=ConfigFlow.MINOR_VERSION,
            unique_id=f"{data["type"]}.{uuid.uuid4().hex}",
        )
        self.config_entry_id: Final = self.config_entry.entry_id
        self.auto_setup = auto_setup
        if auto_add:
            self.config_entry.add_to_hass(hass)

    @property
    def manager(self) -> "ManagerClass":
        return self.hass.data[pmc.DOMAIN]

    @property
    def controller(self) -> _controllerT:
        return self.config_entry.runtime_data

    @property
    def config_entry_loaded(self):
        return self.config_entry.state == config_entries.ConfigEntryState.LOADED

    async def async_setup(self):
        result = await self.hass.config_entries.async_setup(self.config_entry_id)
        await self.hass.async_block_till_done()
        return result

    async def async_unload(self):
        result = await self.hass.config_entries.async_unload(self.config_entry_id)
        await self.hass.async_block_till_done()
        return result

    async def async_test_config_entry_diagnostics(self):
        return
        assert self.config_entry_loaded
        diagnostic = await async_get_config_entry_diagnostics(
            self.hass, self.config_entry
        )
        assert diagnostic

    async def __aenter__(self):
        if self.auto_setup:
            assert await self.async_setup(), self.config_entry.reason
        return self

    async def __aexit__(self, exc_type, exc_value: BaseException | None, traceback):
        if self.config_entry.state.recoverable:
            assert await self.async_unload(), self.config_entry.reason
        return None
