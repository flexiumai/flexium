"""Tests for NVIDIA GPU implementation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestNvidiaGPUInit:
    """Tests for NvidiaGPU initialization."""

    def test_init_sets_torch_none(self):
        """Test that init sets torch to None."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        assert gpu._torch is None

    def test_init_sets_pynvml_not_initialized(self):
        """Test that init sets pynvml_initialized to False."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        assert gpu._pynvml_initialized is False


class TestNvidiaGPUEnsureTorch:
    """Tests for NvidiaGPU._ensure_torch()."""

    def test_ensure_torch_imports_torch(self):
        """Test that _ensure_torch imports torch."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        assert gpu._torch is None

        with patch.dict("sys.modules", {"torch": MagicMock()}):
            gpu._ensure_torch()

        assert gpu._torch is not None

    def test_ensure_torch_only_imports_once(self):
        """Test that _ensure_torch only imports once."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        gpu._torch = mock_torch

        gpu._ensure_torch()

        assert gpu._torch is mock_torch  # Should not have changed


class TestNvidiaGPUEnsurePynvml:
    """Tests for NvidiaGPU._ensure_pynvml()."""

    def test_ensure_pynvml_returns_true_if_already_initialized(self):
        """Test _ensure_pynvml returns True if already initialized."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        result = gpu._ensure_pynvml()

        assert result is True

    def test_ensure_pynvml_initializes_pynvml(self):
        """Test _ensure_pynvml initializes pynvml."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()

        mock_pynvml = MagicMock()
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu._ensure_pynvml()

        assert result is True
        assert gpu._pynvml_initialized is True
        mock_pynvml.nvmlInit.assert_called_once()

    def test_ensure_pynvml_returns_false_on_error(self):
        """Test _ensure_pynvml returns False on error."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("Init failed")
        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu._ensure_pynvml()

        assert result is False
        assert gpu._pynvml_initialized is False


class TestNvidiaGPUGetDeviceCount:
    """Tests for NvidiaGPU.get_device_count()."""

    def test_get_device_count_with_cuda(self):
        """Test get_device_count with CUDA available."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        gpu._torch = mock_torch

        result = gpu.get_device_count()

        assert result == 4

    def test_get_device_count_without_cuda(self):
        """Test get_device_count without CUDA."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        gpu._torch = mock_torch

        result = gpu.get_device_count()

        assert result == 0


class TestNvidiaGPUGetDeviceInfo:
    """Tests for NvidiaGPU.get_device_info()."""

    def test_get_device_info_returns_none_without_pynvml(self):
        """Test get_device_info returns None without pynvml."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()

        with patch.object(gpu, "_ensure_pynvml", return_value=False):
            result = gpu.get_device_info(0)

        assert result is None

    def test_get_device_info_returns_gpu_info(self):
        """Test get_device_info returns GPUInfo."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-12345"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"

        mock_memory = MagicMock()
        mock_memory.total = 16000000000
        mock_memory.used = 8000000000
        mock_memory.free = 8000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        mock_util = MagicMock()
        mock_util.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        mock_pynvml.nvmlDeviceGetTemperature.return_value = 65
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # 150W in mW

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_device_info(0)

        assert result is not None
        assert result.uuid == "GPU-12345"
        assert result.name == "Tesla V100"
        assert result.memory_total == 16000000000
        assert result.utilization == 75
        assert result.temperature == 65
        assert result.power_usage == 150

    def test_get_device_info_handles_bytes_name(self):
        """Test get_device_info handles bytes name."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-12345"
        mock_pynvml.nvmlDeviceGetName.return_value = b"Tesla V100"  # bytes

        mock_memory = MagicMock()
        mock_memory.total = 16000000000
        mock_memory.used = 8000000000
        mock_memory.free = 8000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_device_info(0)

        assert result.name == "Tesla V100"  # Should be decoded

    def test_get_device_info_handles_utilization_error(self):
        """Test get_device_info handles utilization error."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-12345"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"

        mock_memory = MagicMock()
        mock_memory.total = 16000000000
        mock_memory.used = 8000000000
        mock_memory.free = 8000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = Exception("Error")
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_device_info(0)

        assert result.utilization == 0  # Should default to 0

    def test_get_device_info_handles_temperature_error(self):
        """Test get_device_info handles temperature error."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-12345"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"

        mock_memory = MagicMock()
        mock_memory.total = 16000000000
        mock_memory.used = 8000000000
        mock_memory.free = 8000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = Exception("Error")
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_device_info(0)

        assert result.temperature == 0  # Should default to 0

    def test_get_device_info_handles_power_error(self):
        """Test get_device_info handles power error."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetUUID.return_value = "GPU-12345"
        mock_pynvml.nvmlDeviceGetName.return_value = "Tesla V100"

        mock_memory = MagicMock()
        mock_memory.total = 16000000000
        mock_memory.used = 8000000000
        mock_memory.free = 8000000000
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory

        mock_util = MagicMock()
        mock_util.gpu = 50
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 60
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("Error")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_device_info(0)

        assert result.power_usage == 0  # Should default to 0

    def test_get_device_info_returns_none_on_exception(self):
        """Test get_device_info returns None on exception."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("Device not found")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_device_info(99)

        assert result is None


class TestNvidiaGPUGetAllDeviceReports:
    """Tests for NvidiaGPU.get_all_device_reports()."""

    def test_get_all_device_reports_returns_empty_without_pynvml(self):
        """Test get_all_device_reports returns empty list without pynvml."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()

        with patch.object(gpu, "_ensure_pynvml", return_value=False):
            result = gpu.get_all_device_reports("localhost")

        assert result == []

    def test_get_all_device_reports_returns_reports(self):
        """Test get_all_device_reports returns DeviceReports."""
        from flexium.gpu.nvidia import NvidiaGPU
        from flexium.gpu.interface import GPUInfo

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [MagicMock()]

        mock_gpu_info = GPUInfo(
            device_index=0,
            uuid="GPU-12345",
            name="Tesla V100",
            memory_total=16000000000,
            memory_used=8000000000,
            memory_free=8000000000,
            utilization=50,
            temperature=60,
            power_usage=150,
        )

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(gpu, "get_device_info", return_value=mock_gpu_info):
                result = gpu.get_all_device_reports("myhost")

        assert len(result) == 2
        assert result[0].hostname == "myhost"
        assert result[0].gpu_uuid == "GPU-12345"
        assert result[0].process_count == 1

    def test_get_all_device_reports_handles_process_count_error(self):
        """Test get_all_device_reports handles process count error."""
        from flexium.gpu.nvidia import NvidiaGPU
        from flexium.gpu.interface import GPUInfo

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.side_effect = Exception("Error")

        mock_gpu_info = GPUInfo(
            device_index=0,
            uuid="GPU-12345",
            name="Tesla V100",
            memory_total=16000000000,
            memory_used=8000000000,
            memory_free=8000000000,
            utilization=50,
            temperature=60,
            power_usage=150,
        )

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch.object(gpu, "get_device_info", return_value=mock_gpu_info):
                result = gpu.get_all_device_reports("myhost")

        assert len(result) == 1
        assert result[0].process_count == 0  # Should default to 0

    def test_get_all_device_reports_handles_exception(self):
        """Test get_all_device_reports handles exception."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        gpu._pynvml_initialized = True

        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.side_effect = Exception("Error")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = gpu.get_all_device_reports("myhost")

        assert result == []


class TestNvidiaGPUGetMemoryInfo:
    """Tests for NvidiaGPU.get_memory_info()."""

    def test_get_memory_info_with_cuda(self):
        """Test get_memory_info with CUDA."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 4000000000
        mock_torch.cuda.memory_reserved.return_value = 8000000000
        gpu._torch = mock_torch

        result = gpu.get_memory_info(0)

        assert result == (4000000000, 8000000000)

    def test_get_memory_info_without_cuda(self):
        """Test get_memory_info without CUDA."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        gpu._torch = mock_torch

        result = gpu.get_memory_info(0)

        assert result == (0, 0)

    def test_get_memory_info_handles_exception(self):
        """Test get_memory_info handles exception."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.side_effect = Exception("Error")
        gpu._torch = mock_torch

        result = gpu.get_memory_info(0)

        assert result == (0, 0)


class TestNvidiaGPUGetGpuUuid:
    """Tests for NvidiaGPU.get_gpu_uuid()."""

    def test_get_gpu_uuid_returns_uuid(self):
        """Test get_gpu_uuid returns UUID."""
        from flexium.gpu.nvidia import NvidiaGPU
        from flexium.gpu.interface import GPUInfo

        gpu = NvidiaGPU()

        mock_info = GPUInfo(
            device_index=0,
            uuid="GPU-ABC123",
            name="Tesla",
            memory_total=1000,
            memory_used=500,
            memory_free=500,
            utilization=0,
            temperature=0,
            power_usage=0,
        )

        with patch.object(gpu, "get_device_info", return_value=mock_info):
            result = gpu.get_gpu_uuid(0)

        assert result == "GPU-ABC123"

    def test_get_gpu_uuid_returns_empty_on_none(self):
        """Test get_gpu_uuid returns empty string on None."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()

        with patch.object(gpu, "get_device_info", return_value=None):
            result = gpu.get_gpu_uuid(0)

        assert result == ""


class TestNvidiaGPUGetGpuName:
    """Tests for NvidiaGPU.get_gpu_name()."""

    def test_get_gpu_name_returns_name(self):
        """Test get_gpu_name returns name."""
        from flexium.gpu.nvidia import NvidiaGPU
        from flexium.gpu.interface import GPUInfo

        gpu = NvidiaGPU()

        mock_info = GPUInfo(
            device_index=0,
            uuid="GPU-ABC123",
            name="Tesla V100",
            memory_total=1000,
            memory_used=500,
            memory_free=500,
            utilization=0,
            temperature=0,
            power_usage=0,
        )

        with patch.object(gpu, "get_device_info", return_value=mock_info):
            result = gpu.get_gpu_name(0)

        assert result == "Tesla V100"

    def test_get_gpu_name_returns_empty_on_none(self):
        """Test get_gpu_name returns empty string on None."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()

        with patch.object(gpu, "get_device_info", return_value=None):
            result = gpu.get_gpu_name(0)

        assert result == ""


class TestNvidiaGPUGetCurrentDevice:
    """Tests for NvidiaGPU.get_current_device()."""

    def test_get_current_device_with_cuda(self):
        """Test get_current_device with CUDA."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 2
        gpu._torch = mock_torch

        result = gpu.get_current_device()

        assert result == 2

    def test_get_current_device_without_cuda(self):
        """Test get_current_device without CUDA."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        gpu._torch = mock_torch

        result = gpu.get_current_device()

        assert result == 0


class TestNvidiaGPUSetCurrentDevice:
    """Tests for NvidiaGPU.set_current_device()."""

    def test_set_current_device_with_cuda(self):
        """Test set_current_device with CUDA."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        gpu._torch = mock_torch

        gpu.set_current_device(2)

        mock_torch.cuda.set_device.assert_called_once_with(2)

    def test_set_current_device_without_cuda(self):
        """Test set_current_device without CUDA does nothing."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        gpu._torch = mock_torch

        gpu.set_current_device(2)

        mock_torch.cuda.set_device.assert_not_called()


class TestNvidiaGPUIsAvailable:
    """Tests for NvidiaGPU.is_available()."""

    def test_is_available_true(self):
        """Test is_available returns True when CUDA available."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        gpu._torch = mock_torch

        assert gpu.is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False when CUDA not available."""
        from flexium.gpu.nvidia import NvidiaGPU

        gpu = NvidiaGPU()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        gpu._torch = mock_torch

        assert gpu.is_available() is False
