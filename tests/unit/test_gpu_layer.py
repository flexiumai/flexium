"""Tests for the GPU abstraction layer."""

from __future__ import annotations

import pytest

from flexium.gpu import GPUInterface, GPUInfo, DeviceReport, MockGPU, NvidiaGPU
from flexium.utils.gpu_info import set_gpu_backend, get_gpu_backend, get_all_device_reports


class TestGPUInterface:
    """Tests for GPUInterface contract."""

    def test_mock_gpu_implements_interface(self) -> None:
        """Test MockGPU implements GPUInterface."""
        mock = MockGPU()
        assert isinstance(mock, GPUInterface)

    def test_nvidia_gpu_implements_interface(self) -> None:
        """Test NvidiaGPU implements GPUInterface."""
        nvidia = NvidiaGPU()
        assert isinstance(nvidia, GPUInterface)


class TestMockGPU:
    """Tests for MockGPU implementation."""

    def test_default_device_count(self) -> None:
        """Test default device count is 1."""
        mock = MockGPU()
        assert mock.get_device_count() == 1

    def test_custom_device_count(self) -> None:
        """Test custom device count."""
        mock = MockGPU(num_devices=4)
        assert mock.get_device_count() == 4

    def test_get_device_info(self) -> None:
        """Test get_device_info returns correct info."""
        mock = MockGPU(num_devices=2, gpu_name="Test GPU")

        info = mock.get_device_info(0)
        assert info is not None
        assert info.device_index == 0
        assert "Test GPU" in info.name
        assert info.uuid.startswith("GPU-MOCK-")

    def test_get_device_info_invalid_index(self) -> None:
        """Test get_device_info returns None for invalid index."""
        mock = MockGPU(num_devices=2)
        assert mock.get_device_info(-1) is None
        assert mock.get_device_info(2) is None

    def test_get_all_device_reports(self) -> None:
        """Test get_all_device_reports returns all devices."""
        mock = MockGPU(num_devices=3)
        reports = mock.get_all_device_reports("test-host")

        assert len(reports) == 3
        for i, report in enumerate(reports):
            assert report.hostname == "test-host"
            assert report.gpu_uuid.startswith("GPU-MOCK-")

    def test_memory_tracking(self) -> None:
        """Test memory can be set and retrieved."""
        mock = MockGPU(num_devices=2, memory_per_device=16 * 1024**3)

        # Set memory on device 0
        mock.set_memory_allocated(0, 4 * 1024**3)

        allocated, reserved = mock.get_memory_info(0)
        assert allocated == 4 * 1024**3

    def test_is_available(self) -> None:
        """Test is_available returns True for mock GPUs."""
        mock = MockGPU(num_devices=2)
        assert mock.is_available() is True

        mock_empty = MockGPU(num_devices=0)
        assert mock_empty.is_available() is False


class TestGPUBackend:
    """Tests for GPU backend integration with gpu_info module."""

    def test_set_and_get_backend(self) -> None:
        """Test setting and getting GPU backend."""
        original = get_gpu_backend()

        try:
            mock = MockGPU(num_devices=2)
            set_gpu_backend(mock)
            assert get_gpu_backend() is mock

            set_gpu_backend(None)
            assert get_gpu_backend() is None
        finally:
            set_gpu_backend(original)

    def test_get_all_device_reports_uses_backend(self) -> None:
        """Test get_all_device_reports uses custom backend."""
        original = get_gpu_backend()

        try:
            mock = MockGPU(num_devices=2, gpu_name="Custom Mock")
            set_gpu_backend(mock)

            reports = get_all_device_reports("test-host")

            # Should have CPU + 2 mock GPUs
            assert len(reports) == 3
            assert reports[0]["gpu_uuid"] == "CPU"
            assert "Custom Mock" in reports[1]["gpu_name"]
            assert "Custom Mock" in reports[2]["gpu_name"]
        finally:
            set_gpu_backend(original)

    def test_backend_none_uses_pynvml(self) -> None:
        """Test that None backend falls back to pynvml."""
        original = get_gpu_backend()

        try:
            set_gpu_backend(None)

            # This should not raise, even if pynvml isn't available
            reports = get_all_device_reports("test-host")

            # Should at least have CPU
            assert len(reports) >= 1
            assert reports[0]["gpu_uuid"] == "CPU"
        finally:
            set_gpu_backend(original)


class TestDeviceReport:
    """Tests for DeviceReport dataclass."""

    def test_to_dict(self) -> None:
        """Test DeviceReport.to_dict() returns correct dict."""
        report = DeviceReport(
            gpu_uuid="GPU-TEST-1234",
            gpu_name="Test GPU",
            hostname="test-host",
            memory_total=16 * 1024**3,
            memory_used=4 * 1024**3,
            memory_free=12 * 1024**3,
            gpu_utilization=50,
            temperature=65,
            power_usage=200,
            process_count=2,
        )

        d = report.to_dict()

        assert d["gpu_uuid"] == "GPU-TEST-1234"
        assert d["gpu_name"] == "Test GPU"
        assert d["hostname"] == "test-host"
        assert d["memory_total"] == 16 * 1024**3
        assert d["memory_used"] == 4 * 1024**3
        assert d["gpu_utilization"] == 50
        assert d["process_count"] == 2


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self) -> None:
        """Test GPUInfo can be created with required fields."""
        info = GPUInfo(
            device_index=0,
            uuid="GPU-1234",
            name="Test GPU",
        )

        assert info.device_index == 0
        assert info.uuid == "GPU-1234"
        assert info.name == "Test GPU"
        assert info.memory_total == 0  # default
        assert info.utilization == 0  # default
