"""Persistent GPU-accelerated cuML RBF SVM bridge with IPC worker daemon and CPU fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GPU_PYTHON = _REPO_ROOT / ".venv_gpu_cuml/bin/python"

_WORKER_SCRIPT = """
import sys, os, numpy as np, scipy.sparse as sp
import cuml
from cuml.svm import SVC

clf = None

for line in sys.stdin:
    parts = line.strip().split()
    if not parts:
        continue
    cmd = parts[0]
    try:
        if cmd == 'FIT':
            x_path, y_path, c_val, gamma_val, tol_val = parts[1:6]
            X = np.load(x_path)
            y = np.load(y_path)
            gamma = float(gamma_val) if gamma_val not in ('scale', 'auto') else gamma_val
            clf = SVC(kernel='rbf', C=float(c_val), gamma=gamma, tol=float(tol_val))
            clf.fit(X, y)
            sys.stdout.write(f'OK {len(clf.support_)}\\n')
            sys.stdout.flush()
        elif cmd == 'PREDICT':
            x_path, out_path = parts[1:3]
            X = np.load(x_path)
            preds = clf.decision_function(X)
            np.save(out_path, preds)
            sys.stdout.write('OK\\n')
            sys.stdout.flush()
        elif cmd == 'QUIT':
            sys.stdout.write('BYE\\n')
            sys.stdout.flush()
            break
        else:
            sys.stdout.write(f'ERROR unknown command {cmd}\\n')
            sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f'ERROR {type(e).__name__}: {e}\\n')
        sys.stdout.flush()
"""


def _clean_gpu_env() -> dict[str, str]:
    """Sanitize environment so isolated Python 3.11 does not inherit Python 3.13 site-packages."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def is_gpu_available() -> bool:
    """Return True if the isolated Python 3.11 environment with cuML and GPU is usable."""
    if os.environ.get("DISABLE_GPU_SVM") == "1":
        return False
    if not _GPU_PYTHON.is_file() or not os.access(_GPU_PYTHON, os.X_OK):
        return False
    try:
        res = subprocess.run(
            [str(_GPU_PYTHON), "-c", "import cuml; from cuml.svm import SVC; import cupy as cp; assert cp.cuda.runtime.getDeviceCount() >= 1"],
            env=_clean_gpu_env(),
            capture_output=True,
            timeout=10,
            check=False,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


class GpuSvmBridge:
    """Persistent bridge to an isolated cuML worker subprocess for fast RBF SVM fit and predict."""

    def __init__(self, work_dir: Path | str | None = None) -> None:
        self.work_dir = Path(tempfile.mkdtemp(prefix="gpu_svm_")) if work_dir is None else Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._owns_work_dir = work_dir is None
        self._proc: subprocess.Popen[str] | None = None
        self._worker_file = self.work_dir / "_cuml_worker.py"
        self._worker_file.write_text(_WORKER_SCRIPT)
        self.is_fitted = False
        self.n_support_ = 0

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [str(_GPU_PYTHON), str(self._worker_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_clean_gpu_env(),
                text=True,
                cwd=str(self.work_dir),
            )
        return self._proc

    def fit(self, X: Any, y: Any, C: float = 1.0, gamma: str | float = "scale", tol: float = 0.001) -> None:
        """Serialize data to disk and execute fit in the cuML worker process."""
        proc = self._ensure_worker()
        assert proc.stdin is not None and proc.stdout is not None

        # Save X and y
        if sp.issparse(X):
            x_arr = X.toarray().astype(np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)
        x_path = self.work_dir / "X_train.npy"
        np.save(x_path, x_arr)

        y_path = self.work_dir / "y_train.npy"
        y_arr = np.asarray(y, dtype=np.int32)
        np.save(y_path, y_arr)

        cmd = f"FIT {x_path} {y_path} {C} {gamma} {tol}\n"
        proc.stdin.write(cmd)
        proc.stdin.flush()

        resp = proc.stdout.readline().strip()
        if not resp.startswith("OK"):
            raise RuntimeError(f"GPU SVM FIT failed: {resp}")

        parts = resp.split()
        self.n_support_ = int(parts[1]) if len(parts) > 1 else 0
        self.is_fitted = True

        # Clean up training files to save disk
        try:
            x_path.unlink(missing_ok=True)
            y_path.unlink(missing_ok=True)
        except OSError:
            pass

    def decision_function(self, X: Any) -> np.ndarray:
        """Evaluate decision_function in the cuML worker process on GPU."""
        if not self.is_fitted or self._proc is None:
            raise RuntimeError("GPU SVM bridge is not fitted")
        proc = self._proc
        assert proc.stdin is not None and proc.stdout is not None

        # Unique batch filenames
        batch_id = os.urandom(8).hex()
        if sp.issparse(X):
            x_arr = X.toarray().astype(np.float32)
        else:
            x_arr = np.asarray(X, dtype=np.float32)
        x_path = self.work_dir / f"X_val_{batch_id}.npy"
        np.save(x_path, x_arr)

        out_path = self.work_dir / f"preds_{batch_id}.npy"
        cmd = f"PREDICT {x_path} {out_path}\n"
        proc.stdin.write(cmd)
        proc.stdin.flush()

        resp = proc.stdout.readline().strip()
        if not resp.startswith("OK"):
            raise RuntimeError(f"GPU SVM PREDICT failed: {resp}")

        preds = np.load(out_path)
        try:
            x_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return preds

    def close(self) -> None:
        """Terminate the worker subprocess and remove temporary files."""
        if self._proc is not None:
            try:
                if self._proc.stdin is not None and not self._proc.stdin.closed:
                    self._proc.stdin.write("QUIT\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except (subprocess.SubprocessError, OSError):
                self._proc.kill()
            finally:
                self._proc = None

        if self._owns_work_dir:
            shutil.rmtree(self.work_dir, ignore_errors=True)
