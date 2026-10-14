"""Unit tests for GPU-accelerated cuML RBF SVM bridge with safe CPU fallback."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.svm import SVC

from project.features.gpu_svm_bridge import (
    GpuSvmBridge,
    is_gpu_available,
)
from project.features.selection import _SparseFrameSVC


def test_gpu_availability_detection():
    """is_gpu_available detects the isolated Python 3.11 environment and NVIDIA GPU."""
    available = is_gpu_available()
    assert isinstance(available, bool)
    # On this machine with RTX 4070 and .venv_gpu_cuml installed, it should be True
    assert available is True


def test_gpu_bridge_fit_and_predict_roundtrip(tmp_path):
    """GpuSvmBridge runs fit and batch predict through the worker process."""
    if not is_gpu_available():
        pytest.skip("GPU not available")

    bridge = GpuSvmBridge(work_dir=tmp_path)
    X = np.random.randn(6000, 8).astype(np.float32)
    y = np.random.randint(0, 2, 6000).astype(np.int32)
    X_val = np.random.randn(100, 8).astype(np.float32)

    try:
        bridge.fit(X, y, C=1.0, gamma="scale", tol=0.001)
        assert bridge.is_fitted
        assert bridge.n_support_ > 0

        preds = bridge.decision_function(X_val)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (100,)
        assert np.isfinite(preds).all()
    finally:
        bridge.close()


def test_sparse_frame_svc_uses_gpu_for_large_data_and_cpu_for_small(monkeypatch):
    """_SparseFrameSVC delegates to GPU for large datasets and retains CPU for small."""
    # 1. Small data (< 5000 rows) should use CPU without launching GPU
    small_X = pd.DataFrame(np.random.randn(50, 4), columns=[f"c{i}" for i in range(4)])
    small_y = np.random.randint(0, 2, 50)
    svc_small = _SparseFrameSVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)
    svc_small.fit(small_X, small_y)
    assert not hasattr(svc_small, "_gpu_bridge") or svc_small._gpu_bridge is None
    preds_small = svc_small.decision_function(small_X)
    assert preds_small.shape == (50,)

    # 2. Large data (>= 5000 rows) should use GPU if available
    if not is_gpu_available():
        pytest.skip("GPU not available")

    large_X = pd.DataFrame(np.random.randn(6000, 4).astype(np.float32), columns=[f"c{i}" for i in range(4)])
    large_y = np.random.randint(0, 2, 6000).astype(np.int32)
    svc_large = _SparseFrameSVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)
    svc_large.fit(large_X, large_y)
    assert hasattr(svc_large, "_gpu_bridge") and svc_large._gpu_bridge is not None
    preds_large = svc_large.decision_function(large_X.iloc[:50])
    assert preds_large.shape == (50,)
    assert np.isfinite(preds_large).all()
    svc_large.close()


def test_sparse_frame_svc_safe_cpu_fallback(monkeypatch):
    """When GPU is disabled via environment or error, _SparseFrameSVC safely falls back to CPU."""
    monkeypatch.setenv("DISABLE_GPU_SVM", "1")
    large_X = pd.DataFrame(np.random.randn(100, 4), columns=[f"c{i}" for i in range(4)])
    large_y = np.random.randint(0, 2, 100)
    svc = _SparseFrameSVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)
    svc.fit(large_X, large_y)
    assert getattr(svc, "_gpu_bridge", None) is None
    preds = svc.decision_function(large_X)
    assert preds.shape == (100,)
