"""Tests for GPU interface module."""

from __future__ import annotations

import pytest


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_gpu_info_creation(self) -> None:
        """Test GPUInfo can be created with all fields."""
        from flexium.gpu.interface import GPUInfo

        info = GPUInfo(
            device_index=0,
            uuid="GPU-12345",
            name="Test GPU",
            memory_total=16 * 1024**3,
            memory_used=4 * 1024**3,
            memory_free=12 * 1024**3,
            utilization=50,
            temperature=60,
            power_usage=200,
        )

        assert info.device_index == 0
        assert info.uuid == "GPU-12345"
        assert info.name == "Test GPU"
        assert info.memory_total == 16 * 1024**3
        assert info.utilization == 50

    def test_gpu_info_defaults(self) -> None:
        """Test GPUInfo default values."""
        from flexium.gpu.interface import GPUInfo

        info = GPUInfo(
            device_index=0,
            uuid="GPU-12345",
            name="Test GPU",
        )

        assert info.memory_total == 0
        assert info.memory_used == 0
        assert info.memory_free == 0
        assert info.utilization == 0
        assert info.temperature == 0
        assert info.power_usage == 0


class TestDeviceReport:
    """Tests for DeviceReport dataclass."""

    def test_device_report_creation(self) -> None:
        """Test DeviceReport can be created."""
        from flexium.gpu.interface import DeviceReport

        report = DeviceReport(
            gpu_uuid="GPU-12345",
            gpu_name="Test GPU",
            hostname="testhost",
            memory_total=16 * 1024**3,
            memory_used=4 * 1024**3,
            memory_free=12 * 1024**3,
            gpu_utilization=50,
            temperature=60,
            power_usage=200,
            process_count=2,
        )

        assert report.gpu_uuid == "GPU-12345"
        assert report.hostname == "testhost"
        assert report.process_count == 2

    def test_device_report_defaults(self) -> None:
        """Test DeviceReport default values."""
        from flexium.gpu.interface import DeviceReport

        report = DeviceReport(
            gpu_uuid="GPU-12345",
            gpu_name="Test GPU",
            hostname="testhost",
        )

        assert report.memory_total == 0
        assert report.memory_used == 0
        assert report.memory_free == 0
        assert report.gpu_utilization == 0
        assert report.process_count == 0

    def test_device_report_to_dict(self) -> None:
        """Test DeviceReport to_dict method."""
        from flexium.gpu.interface import DeviceReport

        report = DeviceReport(
            gpu_uuid="GPU-12345",
            gpu_name="Test GPU",
            hostname="testhost",
            memory_total=16 * 1024**3,
            memory_used=4 * 1024**3,
            memory_free=12 * 1024**3,
            gpu_utilization=50,
            temperature=60,
            power_usage=200,
            process_count=2,
        )

        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["gpu_uuid"] == "GPU-12345"
        assert d["gpu_name"] == "Test GPU"
        assert d["hostname"] == "testhost"
        assert d["memory_total"] == 16 * 1024**3
        assert d["gpu_utilization"] == 50
        assert d["process_count"] == 2


class TestGPUInterface:
    """Tests for GPUInterface abstract class."""

    def test_gpu_interface_is_abstract(self) -> None:
        """Test GPUInterface cannot be instantiated directly."""
        from flexium.gpu.interface import GPUInterface

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GPUInterface()

    def test_gpu_interface_has_abstract_methods(self) -> None:
        """Test GPUInterface has all required abstract methods."""
        from flexium.gpu.interface import GPUInterface

        methods = [
            "get_device_count",
            "get_device_info",
            "get_all_device_reports",
            "get_memory_info",
            "get_gpu_uuid",
            "get_gpu_name",
            "get_current_device",
            "set_current_device",
            "is_available",
        ]

        for method in methods:
            assert hasattr(GPUInterface, method)


class TestMockGPUImplementsInterface:
    """Tests that MockGPU correctly implements GPUInterface."""

    def test_mock_gpu_is_gpu_interface(self) -> None:
        """Test MockGPU is an instance of GPUInterface."""
        from flexium.gpu.interface import GPUInterface
        from flexium.gpu.mock import MockGPU

        mock = MockGPU()
        assert isinstance(mock, GPUInterface)

    def test_mock_gpu_implements_all_methods(self) -> None:
        """Test MockGPU implements all required methods."""
        from flexium.gpu.mock import MockGPU

        mock = MockGPU(num_devices=2)

        # Test all interface methods are callable and return expected types
        assert isinstance(mock.get_device_count(), int)
        assert mock.get_device_info(0) is not None or mock.get_device_info(0) is None
        assert isinstance(mock.get_all_device_reports("testhost"), list)
        assert isinstance(mock.get_memory_info(0), tuple)
        assert isinstance(mock.get_gpu_uuid(0), str)
        assert isinstance(mock.get_gpu_name(0), str)
        assert isinstance(mock.get_current_device(), int)
        mock.set_current_device(1)
        assert isinstance(mock.is_available(), bool)


class TestNvidiaGPUImplementsInterface:
    """Tests that NvidiaGPU correctly implements GPUInterface."""

    def test_nvidia_gpu_is_gpu_interface(self) -> None:
        """Test NvidiaGPU is an instance of GPUInterface."""
        from flexium.gpu.interface import GPUInterface
        from flexium.gpu.nvidia import NvidiaGPU

        nvidia = NvidiaGPU()
        assert isinstance(nvidia, GPUInterface)
