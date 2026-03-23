"""Additional tests for GPU information utilities to improve coverage."""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestGetVisibleDeviceIndices:
    """Tests for _get_visible_device_indices function."""

    def test_cuda_visible_devices_with_indices(self) -> None:
        """Test when CUDA_VISIBLE_DEVICES contains indices."""
        from flexium.utils.gpu_info import _get_visible_device_indices

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2,3"}):
            result = _get_visible_device_indices()

            assert result == [0, 2, 3]

    def test_cuda_visible_devices_invalid_entry(self) -> None:
        """Test when CUDA_VISIBLE_DEVICES contains invalid entry."""
        from flexium.utils.gpu_info import _get_visible_device_indices

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,invalid,2"}):
            result = _get_visible_device_indices()

            # Should include valid entries and skip invalid
            assert 0 in result
            assert 2 in result


class TestGetGpuCountViaSmi:
    """Tests for _get_gpu_count_via_smi function."""

    def test_nvidia_smi_success(self) -> None:
        """Test successful nvidia-smi query."""
        from flexium.utils.gpu_info import _get_gpu_count_via_smi

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0\n1\n2\n"

        with patch("subprocess.run", return_value=mock_result):
            result = _get_gpu_count_via_smi()

            assert result == [0, 1, 2]

    def test_nvidia_smi_failure(self) -> None:
        """Test nvidia-smi failure."""
        from flexium.utils.gpu_info import _get_gpu_count_via_smi

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = _get_gpu_count_via_smi()

            assert result == []

    def test_nvidia_smi_exception(self) -> None:
        """Test nvidia-smi exception."""
        from flexium.utils.gpu_info import _get_gpu_count_via_smi

        with patch("subprocess.run", side_effect=Exception("Command failed")):
            result = _get_gpu_count_via_smi()

            assert result == []


class TestUuidToPhysicalIndex:
    """Tests for _uuid_to_physical_index function."""

    def test_uuid_found(self) -> None:
        """Test UUID is found in device list."""
        from flexium.utils.gpu_info import _uuid_to_physical_index

        mock_handle = MagicMock()

        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetCount", return_value=3):
                with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                    with patch("pynvml.nvmlDeviceGetUUID", side_effect=["GPU-0", "GPU-TARGET", "GPU-2"]):
                        with patch("pynvml.nvmlShutdown"):
                            result = _uuid_to_physical_index("GPU-TARGET")

                            assert result == 1

    def test_uuid_not_found(self) -> None:
        """Test UUID not found in device list."""
        from flexium.utils.gpu_info import _uuid_to_physical_index

        mock_handle = MagicMock()

        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetCount", return_value=2):
                with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                    with patch("pynvml.nvmlDeviceGetUUID", return_value="GPU-OTHER"):
                        with patch("pynvml.nvmlShutdown"):
                            result = _uuid_to_physical_index("GPU-NOTFOUND")

                            assert result is None

    def test_uuid_pynvml_exception(self) -> None:
        """Test pynvml exception during UUID lookup."""
        from flexium.utils.gpu_info import _uuid_to_physical_index

        with patch("pynvml.nvmlInit", side_effect=Exception("pynvml failed")):
            result = _uuid_to_physical_index("GPU-TEST")

            assert result is None


class TestMigUuidToPhysicalIndex:
    """Tests for _mig_uuid_to_physical_index function."""

    def test_mig_uuid_exception(self) -> None:
        """Test exception during MIG UUID lookup."""
        from flexium.utils.gpu_info import _mig_uuid_to_physical_index

        with patch("pynvml.nvmlInit", side_effect=Exception("pynvml failed")):
            result = _mig_uuid_to_physical_index("MIG-TEST-UUID")

            assert result is None

    def test_mig_uuid_no_mig_devices(self) -> None:
        """Test when no MIG devices are present."""
        from flexium.utils.gpu_info import _mig_uuid_to_physical_index

        mock_handle = MagicMock()

        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetCount", return_value=1):
                with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                    # MIG mode not enabled
                    with patch("pynvml.nvmlDeviceGetMigMode", return_value=(0, 0)):  # Disabled
                        with patch("pynvml.nvmlShutdown"):
                            with patch("pynvml.NVML_DEVICE_MIG_ENABLE", 1):
                                result = _mig_uuid_to_physical_index("MIG-TEST-UUID")

                                assert result is None


class TestGetGpuInfoPynvml:
    """Tests for get_gpu_info_pynvml function."""

    def test_get_gpu_info_success(self) -> None:
        """Test successful GPU info retrieval."""
        from flexium.utils.gpu_info import get_gpu_info_pynvml

        mock_handle = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.total = 16 * 1024**3

        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                with patch("pynvml.nvmlDeviceGetUUID", return_value="GPU-TEST-UUID"):
                    with patch("pynvml.nvmlDeviceGetName", return_value="Test GPU"):
                        with patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info):
                            # Mock MIG mode as disabled
                            with patch("pynvml.nvmlDeviceGetMigMode", side_effect=AttributeError("No MIG")):
                                with patch("pynvml.nvmlShutdown"):
                                    result = get_gpu_info_pynvml(0)

                                    assert result is not None
                                    assert result["uuid"] == "GPU-TEST-UUID"
                                    assert result["name"] == "Test GPU"


class TestGetEstimatedGpuMemory:
    """Tests for get_estimated_gpu_memory function."""

    def test_get_estimated_memory_returns_int(self) -> None:
        """Test estimated memory returns an integer."""
        from flexium.utils.gpu_info import get_estimated_gpu_memory

        result = get_estimated_gpu_memory("cuda:0")
        assert isinstance(result, int)
        assert result >= 0

    def test_get_estimated_memory_cpu_or_zero(self) -> None:
        """Test estimated memory for CPU device returns 0 or valid int."""
        from flexium.utils.gpu_info import get_estimated_gpu_memory

        result = get_estimated_gpu_memory("cpu")
        # Should return 0 (no GPU) or non-negative int if CUDA is initialized elsewhere
        assert isinstance(result, int)
        assert result >= 0


class TestGetGpuInfo:
    """Tests for get_gpu_info function."""

    def test_get_gpu_info_cpu_device(self) -> None:
        """Test get_gpu_info with CPU device returns None."""
        from flexium.utils.gpu_info import get_gpu_info

        result = get_gpu_info("cpu")

        assert result is None

    def test_get_gpu_info_cuda_device(self) -> None:
        """Test get_gpu_info with CUDA device."""
        from flexium.utils.gpu_info import get_gpu_info

        mock_handle = MagicMock()
        mock_memory_info = MagicMock()
        mock_memory_info.total = 16 * 1024**3

        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                with patch("pynvml.nvmlDeviceGetUUID", return_value="GPU-TEST"):
                    with patch("pynvml.nvmlDeviceGetName", return_value="Test GPU"):
                        with patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_memory_info):
                            with patch("pynvml.nvmlDeviceGetMigMode", side_effect=AttributeError("No MIG")):
                                with patch("pynvml.nvmlShutdown"):
                                    result = get_gpu_info("cuda:0")

                                    assert result is not None


class TestParseMigUuid:
    """Tests for _parse_mig_uuid function."""

    def test_parse_standard_mig_uuid(self) -> None:
        """Test parsing standard MIG UUID format."""
        from flexium.utils.gpu_info import _parse_mig_uuid

        uuid = "MIG-GPU-12345678-1234-1234-1234-123456789012/0/0"
        parent, gi, ci = _parse_mig_uuid(uuid)

        assert parent == "GPU-12345678-1234-1234-1234-123456789012"
        assert gi == 0
        assert ci == 0

    def test_parse_mig_uuid_no_indices(self) -> None:
        """Test parsing MIG UUID without indices."""
        from flexium.utils.gpu_info import _parse_mig_uuid

        uuid = "MIG-12345678-1234-1234-1234-123456789012"
        parent, gi, ci = _parse_mig_uuid(uuid)

        assert parent is None
        assert gi is None
        assert ci is None

    def test_parse_mig_uuid_invalid(self) -> None:
        """Test parsing invalid MIG UUID."""
        from flexium.utils.gpu_info import _parse_mig_uuid

        uuid = "NOT-A-MIG-UUID"
        parent, gi, ci = _parse_mig_uuid(uuid)

        assert parent is None


class TestDiscoverGpuPid:
    """Tests for discover_gpu_pid function."""

    def test_discover_gpu_pid_returns_int_or_none(self) -> None:
        """Test discover_gpu_pid returns int or None."""
        from flexium.utils.gpu_info import discover_gpu_pid

        result = discover_gpu_pid("cuda:0")

        # Result should be int (possibly 0) or None
        assert result is None or isinstance(result, int)


class TestGetAllDeviceReports:
    """Tests for get_all_device_reports function."""

    def test_get_all_device_reports_returns_list(self) -> None:
        """Test get_all_device_reports returns a list."""
        from flexium.utils.gpu_info import get_all_device_reports

        # The function should return a list (possibly empty if no GPUs)
        result = get_all_device_reports("testhost")
        assert isinstance(result, list)


class TestGetGpuMemoryByPhysicalIndex:
    """Tests for get_gpu_memory_by_physical_index function."""

    def test_get_memory_by_index_returns_int(self) -> None:
        """Test getting memory returns an integer."""
        from flexium.utils.gpu_info import get_gpu_memory_by_physical_index

        # The function should return an integer (0 or more)
        result = get_gpu_memory_by_physical_index(0, None)
        assert isinstance(result, int)
        assert result >= 0


class TestGPUInfoDataclass:
    """Tests for GPUInfo dataclass extended functionality."""

    def test_gpu_info_with_memory_total(self) -> None:
        """Test GPUInfo with memory_total field."""
        from flexium.utils.gpu_info import GPUInfo, GPUType

        info = GPUInfo(
            logical_index=0,
            physical_index=0,
            uuid="GPU-TEST-UUID",
            name="Test GPU",
            memory_total=16 * 1024**3,
        )

        assert info.memory_total == 16 * 1024**3
        assert info.gpu_type == GPUType.PHYSICAL
