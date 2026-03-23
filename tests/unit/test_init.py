"""Tests for flexium.init() API."""

import pytest
import flexium


class TestFlexiumInit:
    """Tests for the simplified init() API."""

    def setup_method(self):
        """Reset state before each test."""
        # Force reset the module state
        flexium._initialized = False
        flexium._init_context = None

    def test_is_initialized_false_by_default(self):
        """Check is_initialized returns False before init."""
        assert flexium.is_initialized() is False

    def test_init_sets_initialized(self):
        """Check init() sets initialized flag."""
        flexium.init()
        assert flexium.is_initialized() is True
        flexium.shutdown()

    def test_double_init_ignored(self, capsys):
        """Check duplicate init() calls are ignored."""
        flexium.init()
        flexium.init()  # Should print warning
        captured = capsys.readouterr()
        assert "Already initialized" in captured.out
        flexium.shutdown()

    def test_shutdown_clears_initialized(self):
        """Check shutdown() clears initialized flag."""
        flexium.init()
        assert flexium.is_initialized() is True
        flexium.shutdown()
        assert flexium.is_initialized() is False

    def test_init_with_server_parameter(self):
        """Check init() accepts server parameter."""
        # This will fail to connect but should not raise
        flexium.init(server="localhost:9999/test")
        assert flexium.is_initialized() is True
        flexium.shutdown()

    def test_init_with_device_parameter(self):
        """Check init() accepts device parameter."""
        flexium.init(device="cuda:0")
        assert flexium.is_initialized() is True
        flexium.shutdown()

    def test_init_with_disabled_parameter(self):
        """Check init() accepts disabled parameter."""
        flexium.init(disabled=True)
        assert flexium.is_initialized() is True
        flexium.shutdown()

    def test_shutdown_when_not_initialized(self):
        """Check shutdown() does nothing when not initialized."""
        # Should not raise when called before init
        flexium._initialized = False
        flexium._init_context = None
        flexium.shutdown()
        assert flexium.is_initialized() is False

    def test_shutdown_handles_exception(self):
        """Check shutdown() handles context manager exceptions gracefully."""
        from unittest.mock import MagicMock

        flexium._initialized = True
        mock_context = MagicMock()
        mock_context.__exit__.side_effect = Exception("cleanup failed")
        flexium._init_context = mock_context

        # Should not raise despite exception
        flexium.shutdown()
        assert flexium.is_initialized() is False
