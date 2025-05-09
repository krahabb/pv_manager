"""Global fixtures for integration_blueprint integration."""

# Fixtures allow you to replace functions with a Mock object. You can perform
# many options via the Mock to reflect a particular behavior from the original
# function that you want to see without going through the function's actual logic.
# Fixtures can either be passed into tests as parameters, or if autouse=True, they
# will automatically be used across all tests.
#
# Fixtures that are defined in conftest.py are available across all tests. You can also
# define fixtures within a particular test file to scope them locally.
#
# pytest_homeassistant_custom_component provides some fixtures that are provided by
# Home Assistant core. You can find those fixture definitions here:
# https://github.com/MatthewFlamm/pytest-homeassistant-custom-component/blob/master/pytest_homeassistant_custom_component/common.py
#
# See here for more info: https://docs.pytest.org/en/latest/fixture.html (note that
# pytest includes fixtures OOB which you can use as defined on this page)
import logging
from unittest.mock import patch

import pytest

from custom_components.pv_manager.helpers import Loggable

from . import helpers

pytest_plugins = "pytest_homeassistant_custom_component"

# This is the only damn way I can get to disable the sqlalchemy engine logs...
logging.disable()

# Test initialization must ensure custom_components are enabled
# but we can't autouse a simple fixture for that since the recorder
# need to be initialized first
@pytest.fixture(autouse=True)
def auto_enable(recorder_mock, hass, enable_custom_integrations):
    """
    Special initialization fixture managing recorder mocking.
    For some tests we need a working recorder but recorder_mock
    needs to be init before hass.
    When we don't need it, we'd also want our helpers.get_entity_last_states
    to not return an exception (since the recorder instance is missing then)
    """

    """
    has_recorder = "recorder_mock" in request.fixturenames
    if has_recorder:
        request.getfixturevalue("recorder_mock")

    hass = request.getfixturevalue("hass")
    hass.data.pop("custom_components")
    """


# This fixture is used to prevent HomeAssistant from attempting to create and dismiss persistent
# notifications. These calls would fail without this fixture since the persistent_notification
# integration is never loaded during a test.
@pytest.fixture(name="skip_notifications", autouse=True)
def skip_notifications_fixture():
    """Skip notification calls."""
    with (
        patch("homeassistant.components.persistent_notification.async_create"),
        patch("homeassistant.components.persistent_notification.async_dismiss"),
    ):
        yield


@pytest.fixture(name="disable_debug", autouse=False)
def disable_debug_fixture():
    """Disable development debug code so to test in a production env."""
    with patch("custom_components.pv_manager.const.DEBUG", None):
        yield


@pytest.fixture()
def time_mock(hass):
    with helpers.TimeMocker(hass) as _time_mock:
        yield _time_mock


@pytest.fixture()
def log_exception(autouse=True):
    """Intercepts any code managed exception sent to logging."""

    def _patch_loggable_log_exception(
        level: int, exception: Exception, msg: str, *args, **kwargs
    ):
        raise Exception(
            f'log_exception called with msg="{msg}" args=({args})'
        ) from exception

    with patch.object(
        Loggable,
        "log_exception",
        side_effect=_patch_loggable_log_exception,
    ):
        yield
