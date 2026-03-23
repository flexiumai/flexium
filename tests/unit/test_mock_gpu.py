"""Tests for Mock GPU implementation."""

from __future__ import annotations

import pytest


class TestMockGPUInit:
    """Tests for MockGPU initialization."""

    def test_init_default_values(self):
        """Test MockGPU with default values."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU()
        assert gpu._num_devices == 1
        assert gpu._memory_per_device == 16 * 1024**3
        assert gpu._gpu_name == "Mock GPU"
        assert gpu._current_device == 0

    def test_init_custom_values(self):
        """Test MockGPU with custom values."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=4, memory_per_device=32 * 1024**3, gpu_name="Test GPU")
        assert gpu._num_devices == 4
        assert gpu._memory_per_device == 32 * 1024**3
        assert gpu._gpu_name == "Test GPU"

    def test_init_allocated_memory_tracking(self):
        """Test that allocated memory tracking is initialized."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=3)
        assert len(gpu._allocated) == 3
        assert all(v == 0 for v in gpu._allocated.values())


class TestMockGPUGetDeviceCount:
    """Tests for MockGPU.get_device_count()."""

    def test_get_device_count(self):
        """Test get_device_count returns correct count."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=4)
        assert gpu.get_device_count() == 4

    def test_get_device_count_zero(self):
        """Test get_device_count with zero devices."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=0)
        assert gpu.get_device_count() == 0


class TestMockGPUGetDeviceInfo:
    """Tests for MockGPU.get_device_info()."""

    def test_get_device_info_valid_index(self):
        """Test get_device_info with valid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        info = gpu.get_device_info(0)

        assert info is not None
        assert info.device_index == 0
        assert info.uuid == "GPU-MOCK-0000-0000-0000-000000000000"
        assert "Mock GPU 0" in info.name
        assert info.memory_total == 16 * 1024**3
        assert info.memory_used == 0
        assert info.memory_free == 16 * 1024**3

    def test_get_device_info_invalid_index_negative(self):
        """Test get_device_info with negative index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        info = gpu.get_device_info(-1)

        assert info is None

    def test_get_device_info_invalid_index_too_high(self):
        """Test get_device_info with index too high."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        info = gpu.get_device_info(5)

        assert info is None

    def test_get_device_info_with_allocated_memory(self):
        """Test get_device_info reflects allocated memory."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=1, memory_per_device=10000)
        gpu.set_memory_allocated(0, 5000)

        info = gpu.get_device_info(0)

        assert info.memory_used == 5000
        assert info.memory_free == 5000
        assert info.utilization == 50  # 50% used


class TestMockGPUGetAllDeviceReports:
    """Tests for MockGPU.get_all_device_reports()."""

    def test_get_all_device_reports(self):
        """Test get_all_device_reports returns all reports."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=3)
        reports = gpu.get_all_device_reports("testhost")

        assert len(reports) == 3
        assert all(r.hostname == "testhost" for r in reports)

    def test_get_all_device_reports_with_process_count(self):
        """Test process count reflects allocated memory."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        gpu.set_memory_allocated(0, 1000)  # Device 0 has allocation

        reports = gpu.get_all_device_reports("testhost")

        assert reports[0].process_count == 1  # Has allocation
        assert reports[1].process_count == 0  # No allocation


class TestMockGPUGetMemoryInfo:
    """Tests for MockGPU.get_memory_info()."""

    def test_get_memory_info(self):
        """Test get_memory_info returns correct values."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=1)
        gpu.set_memory_allocated(0, 4000)
        gpu.set_memory_reserved(0, 8000)

        allocated, reserved = gpu.get_memory_info(0)

        assert allocated == 4000
        assert reserved == 8000

    def test_get_memory_info_invalid_index(self):
        """Test get_memory_info with invalid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=1)

        allocated, reserved = gpu.get_memory_info(-1)
        assert allocated == 0
        assert reserved == 0

        allocated, reserved = gpu.get_memory_info(5)
        assert allocated == 0
        assert reserved == 0


class TestMockGPUGetGpuUuid:
    """Tests for MockGPU.get_gpu_uuid()."""

    def test_get_gpu_uuid(self):
        """Test get_gpu_uuid returns correct UUID."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)

        uuid0 = gpu.get_gpu_uuid(0)
        uuid1 = gpu.get_gpu_uuid(1)

        assert uuid0 == "GPU-MOCK-0000-0000-0000-000000000000"
        assert uuid1 == "GPU-MOCK-0001-0000-0000-000000000000"

    def test_get_gpu_uuid_invalid_index(self):
        """Test get_gpu_uuid with invalid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=1)
        uuid = gpu.get_gpu_uuid(5)

        assert uuid == ""


class TestMockGPUGetGpuName:
    """Tests for MockGPU.get_gpu_name()."""

    def test_get_gpu_name(self):
        """Test get_gpu_name returns correct name."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2, gpu_name="TestGPU")

        name0 = gpu.get_gpu_name(0)
        name1 = gpu.get_gpu_name(1)

        assert name0 == "TestGPU 0"
        assert name1 == "TestGPU 1"

    def test_get_gpu_name_invalid_index(self):
        """Test get_gpu_name with invalid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=1)
        name = gpu.get_gpu_name(5)

        assert name == ""


class TestMockGPUCurrentDevice:
    """Tests for MockGPU current device methods."""

    def test_get_current_device(self):
        """Test get_current_device returns current device."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=4)
        assert gpu.get_current_device() == 0

    def test_set_current_device(self):
        """Test set_current_device changes device."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=4)
        gpu.set_current_device(2)
        assert gpu.get_current_device() == 2

    def test_set_current_device_invalid_index(self):
        """Test set_current_device ignores invalid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=4)
        gpu.set_current_device(0)
        gpu.set_current_device(10)  # Invalid, should be ignored
        assert gpu.get_current_device() == 0


class TestMockGPUIsAvailable:
    """Tests for MockGPU.is_available()."""

    def test_is_available_true(self):
        """Test is_available returns True with devices."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        assert gpu.is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False with no devices."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=0)
        assert gpu.is_available() is False


class TestMockGPUSetMemory:
    """Tests for MockGPU memory setting methods."""

    def test_set_memory_allocated(self):
        """Test set_memory_allocated sets memory."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        gpu.set_memory_allocated(1, 5000)

        assert gpu._allocated[1] == 5000
        assert gpu._allocated[0] == 0  # Unchanged

    def test_set_memory_allocated_invalid_index(self):
        """Test set_memory_allocated ignores invalid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        gpu.set_memory_allocated(5, 5000)  # Invalid index

        # Should not crash, and no change to valid devices
        assert gpu._allocated[0] == 0
        assert gpu._allocated[1] == 0

    def test_set_memory_reserved(self):
        """Test set_memory_reserved sets memory."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        gpu.set_memory_reserved(0, 8000)

        assert gpu._reserved[0] == 8000

    def test_set_memory_reserved_invalid_index(self):
        """Test set_memory_reserved ignores invalid index."""
        from flexium.gpu.mock import MockGPU

        gpu = MockGPU(num_devices=2)
        gpu.set_memory_reserved(10, 8000)  # Invalid index

        # Should not crash
        assert gpu._reserved[0] == 0
