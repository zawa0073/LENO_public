%%writefile  GMS_LENO_FEM.py 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a LENO/FEM-style neural operator for GMS disk teacher transitions.

Canonical teacher/model input format:
    torch.save({
        "x": (Ntransition, Ntrain, 4),
        "dt": (Ntransition,),
        "y": (Ntransition, Ntrain, 2),
        "sample_id": (Ntransition,),
        "time_index": (Ntransition,),
        "points": (Ntrain, 2),
        ...
    })

The model input is the pair
    ({[u_k(x_j,y_j), v_k(x_j,y_j), x_j, y_j]}_{j=0}^{Ntrain-1}, dt_k).

By default, the five features [u_k, v_k, x, y, dt_k] are standardized with
mean and standard deviation computed only from the training split. The raw dt
used by the time-dependent spectral decay is kept separate from this input
standardization. The original channel-wise spectral decay is preserved;
reporting follows GM_LENO_FDM2dim.py without changing the model architecture.

The four node channels are:
    x[..., 0] = u_k
    x[..., 1] = v_k
    x[..., 2] = coordinate x
    x[..., 3] = coordinate y

Legacy teacher files with x=[u,v,x,y,dt,d1,d2,sigma] are accepted and
converted to this canonical input automatically. By default dt units are read
from metadata or inferred from teacher times and converted to dt / T with T=50.
Set --dt_scale to the teacher T. The raw dt is recovered inside time-dependent layers by
multiplying by --dt_scale.

For --basis_type disk, the script builds Neumann Laplacian eigenfunction
candidates on the disk:
    phi_nm(r,theta) = J_n(alpha_nm r / R) cos(n theta) or sin(n theta),
    J_n'(alpha_nm) = 0,
    mu_nm = (alpha_nm / R)^2,
then mass-orthonormalizes their sampled values on the teacher points.

For --basis_type cartesian, the script builds Kx * Ky Cartesian tensor-product
cosine candidates on the enclosing square and restricts them to the disk
teacher points before mass-orthonormalization.

For --basis_type fourier, the script uses the FNO baseline: disk point data are
zero-extended onto an enclosing square grid, Fourier layers act on that square,
and the training/evaluation loss is taken only on the original disk points.
"""

from __future__ import annotations

import argparse
import atexit
from collections import OrderedDict
from contextlib import nullcontext
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import psutil
except ImportError:
    psutil = None
import csv
import glob
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_WEIGHT_MODE = "cumulative_time_from_dt_train_only"


# ------------------------------------------------------------------------
# Runtime logging and shared reporting
# ------------------------------------------------------------------------

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")
        self.errors = getattr(streams[0], "errors", "replace")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_LOG_FILE_HANDLE = None


def _close_console_log():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _LOG_FILE_HANDLE
    if _LOG_FILE_HANDLE is None:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        _LOG_FILE_HANDLE.close()
        _ORIGINAL_STDOUT = None
        _ORIGINAL_STDERR = None
        _LOG_FILE_HANDLE = None


def setup_console_log(log_file, save_dir, out_prefix):
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _LOG_FILE_HANDLE
    raw_log_file = str(log_file or "").strip()
    if raw_log_file.lower() in {"none", "off", "0"}:
        return ""
    if _LOG_FILE_HANDLE is not None:
        return os.path.abspath(_LOG_FILE_HANDLE.name)

    if raw_log_file == "":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.basename(str(out_prefix).strip() or "run")
        raw_log_file = os.path.join(save_dir, f"{prefix}_{stamp}_log.txt")

    log_path = os.path.abspath(os.path.expanduser(raw_log_file))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _LOG_FILE_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_FILE_HANDLE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_FILE_HANDLE)
    atexit.register(_close_console_log)
    print("log file:", log_path, flush=True)
    return log_path


def _fmt_gb(x):
    return "n/a" if x is None else f"{float(x):.2f}GB"


def _get_ram_stats():
    if psutil is None:
        return None
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return {
        "proc_rss_gb": float(proc.memory_info().rss) / (1024 ** 3),
        "sys_used_gb": float(vm.used) / (1024 ** 3),
        "sys_avail_gb": float(vm.available) / (1024 ** 3),
        "sys_total_gb": float(vm.total) / (1024 ** 3),
    }


def _get_gpu_stats():
    if not torch.cuda.is_available():
        return None
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    return {
        "alloc_gb": float(torch.cuda.memory_allocated(dev)) / (1024 ** 3),
        "reserved_gb": float(torch.cuda.memory_reserved(dev)) / (1024 ** 3),
        "total_gb": float(props.total_memory) / (1024 ** 3),
    }


def log_runtime_status(prefix):
    parts = [str(prefix)]
    ram = _get_ram_stats()
    if ram is not None:
        parts.append(
            "RAM proc={} sys={}/{} avail={}".format(
                _fmt_gb(ram["proc_rss_gb"]),
                _fmt_gb(ram["sys_used_gb"]),
                _fmt_gb(ram["sys_total_gb"]),
                _fmt_gb(ram["sys_avail_gb"]),
            )
        )
    gpu = _get_gpu_stats()
    if gpu is not None:
        parts.append(
            "GPU alloc={} reserved={} total={}".format(
                _fmt_gb(gpu["alloc_gb"]),
                _fmt_gb(gpu["reserved_gb"]),
                _fmt_gb(gpu["total_gb"]),
            )
        )
    print(" | ".join(parts), flush=True)


def _fmt_millions(n):
    return "{:.3f}M".format(float(n) / 1.0e6)


def _fmt_bytes(n):
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0 or unit == "TB":
            return "{:.2f}{}".format(n, unit)
        n /= 1024.0


def print_model_parameter_summary(model):
    rows = OrderedDict()
    total_elems = 0
    total_bytes = 0
    spectral_elems = 0
    spectral_bytes = 0
    for name, p in model.named_parameters():
        elems = int(p.numel())
        bytes_ = int(elems * p.element_size())
        key = str(p.dtype).replace("torch.", "")
        if key not in rows:
            rows[key] = [0, 0]
        rows[key][0] += elems
        rows[key][1] += bytes_
        total_elems += elems
        total_bytes += bytes_
        if str(name).startswith("spec."):
            spectral_elems += elems
            spectral_bytes += bytes_

    print(
        "model params | total={} bytes={} | spectral={} bytes={}".format(
            _fmt_millions(total_elems),
            _fmt_bytes(total_bytes),
            _fmt_millions(spectral_elems),
            _fmt_bytes(spectral_bytes),
        ),
        flush=True,
    )
    for dtype, (elems, bytes_) in rows.items():
        print(
            "  dtype={} params={} bytes={}".format(dtype, _fmt_millions(elems), _fmt_bytes(bytes_)),
            flush=True,
        )
    print(
        "  Adam memory rough lower bound: params+grads+2*moments ~= {}".format(
            _fmt_bytes(total_bytes * 4)
        ),
        flush=True,
    )




def _resolve_amp_dtype(name):
    name = str(name).lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unknown amp dtype: {name}")


def _make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=bool(enabled))
        except TypeError:
            return torch.amp.GradScaler(enabled=bool(enabled))
    return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _model_has_complex_parameters(model):
    return any(torch.is_complex(p) for p in model.parameters())


def _restore_scheduler_epoch(scheduler, completed_epochs):
    if scheduler is None:
        return
    completed_epochs = max(0, int(completed_epochs))
    scheduler.last_epoch = completed_epochs
    if not hasattr(scheduler, "_get_closed_form_lr"):
        raise RuntimeError("the selected learning-rate scheduler cannot resume from a best-only save")
    resumed_lrs = [float(value) for value in scheduler._get_closed_form_lr()]
    for group, resumed_lr in zip(scheduler.optimizer.param_groups, resumed_lrs):
        group["lr"] = resumed_lr
    scheduler._last_lr = resumed_lrs
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = completed_epochs + 1


def _cuda_autocast(enabled, dtype):
    if not bool(enabled):
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype)
    return torch.cuda.amp.autocast(dtype=dtype)


def _plot_relL2_time(times_used, rel_u, rel_v, out_path, title="rollout relative L2"):
    plt.figure(figsize=(7, 4))
    plt.plot(times_used, rel_u, label="u")
    plt.plot(times_used, rel_v, label="v")
    plt.xlabel("t")
    plt.ylabel("Relative L2 error")
    plt.title(str(title))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)


_FULL_TIME_METRIC_NAMES = ["l2_space_time", "relative_l2_space_time", "normal_derivative_rms", "neumann_zero_flux_pred_rms", "neumann_zero_flux_teacher_rms"]

def _full_time_metric_values(e, duration, boundary_length, boundary):
    """Boundary RMS uses physical duration * perimeter; combined sums u/v squares.

    Field L2 remains a space-time norm. Zero-flux metrics measure derivatives,
    without diffusion coefficients, using the same stencil for prediction/teacher.
    """
    if not np.isfinite(duration) or not np.isfinite(boundary_length) or min(duration, boundary_length) <= 0:
        raise ValueError("Boundary RMS requires finite positive duration and boundary length")
    relative = np.sqrt(e[0] / e[1]) if e[1] > 0 else (0.0 if e[0] == 0 else float("inf"))
    measure = duration * boundary_length
    return [np.sqrt(e[0]), relative, np.sqrt(e[2] / measure),
            np.sqrt(e[3] / measure) if boundary == "neumann" else float("nan"),
            np.sqrt(e[4] / measure) if boundary == "neumann" else float("nan")]


def _save_full_time_errors(energies, duration, out_prefix, boundary, evaluation, boundary_length):
    """Time-integrated field/reference/derivative error/prediction flux/teacher flux."""
    energies = np.asarray(energies, dtype=np.float64)
    path = out_prefix + "_full_time_errors.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "channel", "duration_sum", "boundary_length"] + _FULL_TIME_METRIC_NAMES)
        for channel, column in (("u", 0), ("v", 1), ("combined", None)):
            e = energies.sum(axis=1) if column is None else energies[:, column]
            values = _full_time_metric_values(e, duration, boundary_length, boundary)
            writer.writerow([evaluation, channel, duration, boundary_length] + values)
            text = " | ".join(f"{name}={value:.8g}" for name, value in zip(_FULL_TIME_METRIC_NAMES, values))
            print(f"{evaluation} full-time | {channel}: {text}", flush=True)
    print("saved full-time errors:", path, flush=True)


def _save_trajectory_error_statistics(trajectories, out_prefix, boundary):
    """Compute each trajectory's norm/ratio first, then mean and population std (ddof=0)."""
    if not trajectories:
        raise ValueError("No test trajectories available for full-time evaluation")
    metrics = _FULL_TIME_METRIC_NAMES
    values = {channel: [] for channel in ("u", "v", "combined")}
    individual_path = out_prefix + "_full_time_errors_per_trajectory.csv"
    with open(individual_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "source_id", "sample_id", "channel", "duration", "steps", "boundary_length"] + metrics)
        for (source_id, sample_id), item in sorted(trajectories.items()):
            if item["duration"] <= 0:
                raise ValueError(f"Trajectory {(source_id, sample_id)} has no positive physical duration")
            for channel, column in (("u", 0), ("v", 1), ("combined", None)):
                e = item["energies"].sum(axis=1) if column is None else item["energies"][:, column]
                row = _full_time_metric_values(e, item["duration"], item["boundary_length"], boundary)
                values[channel].append(row)
                writer.writerow(["test_one_step_right_endpoint", source_id, sample_id, channel,
                                 item["duration"], item["steps"], item["boundary_length"]] + row)
    summary_path = out_prefix + "_full_time_errors.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evaluation", "channel", "num_trajectories", "std_ddof"] +
                        [f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")])
        for channel, rows in values.items():
            array = np.asarray(rows, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                means, stds = array.mean(axis=0), array.std(axis=0, ddof=0)
            writer.writerow(["test_one_step_right_endpoint", channel, len(rows), 0] +
                            [value for pair in zip(means, stds) for value in pair])
            text = " | ".join(f"{metric}={mean:.8g} +/- {std:.8g}"
                              for metric, mean, std in zip(metrics, means, stds))
            print(f"test full-time | {channel} | trajectories={len(rows)} std_ddof=0 | {text}", flush=True)
    print(f"saved trajectory errors: {individual_path}\nsaved mean/std: {summary_path}", flush=True)




# ------------------------------------------------------------------------
# Reproducibility and checkpoint utilities
# ------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def safe_torch_load(path: str, map_location: str | torch.device = "cpu", retries: int = 6, mmap: bool = False):
    last_exc: BaseException | None = None
    for i in range(int(retries)):
        try:
            try:
                return torch.load(path, map_location=map_location, weights_only=False, mmap=mmap)
            except TypeError:
                return torch.load(path, map_location=map_location)
        except OSError as exc:
            last_exc = exc
            time.sleep(0.2 * float(1 << i))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"failed to load torch file: {path}")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _first_key(dct: dict[str, Any], keys: list[str], required: bool = True):
    for key in keys:
        if key in dct:
            return dct[key]
    if required:
        raise KeyError(f"none of these keys were found: {keys}")
    return None


def parse_int_csv(value: str) -> list[int]:
    value = str(value).strip()
    if value == "":
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip() != ""]


def maybe_make_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def atomic_torch_save(obj: Any, path: str) -> None:
    maybe_make_dir(os.path.dirname(path))
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def optimizer_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _get_rng_state_for_resume() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state_for_resume(state: Any) -> None:
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def copy_history_for_checkpoint(history: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {key: list(value) for key, value in history.items()}


# ------------------------------------------------------------------------
# Teacher data loading and trajectory splitting
# ------------------------------------------------------------------------

def parse_teacher_paths(value: Any) -> list[str]:
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    paths: list[str] = []
    for item in raw_items:
        for part in str(item).split(","):
            path = part.strip()
            if path:
                expanded = os.path.abspath(os.path.expanduser(path))
                if glob.has_magic(expanded):
                    matches = sorted(glob.glob(expanded))
                    if not matches:
                        raise FileNotFoundError(f"teacher path pattern matched no files: {path}")
                    paths.extend(matches)
                else:
                    paths.append(expanded)
    if not paths:
        raise ValueError("--teacher_path must contain at least one path")
    paths = list(dict.fromkeys(paths))
    for path in paths:
        if os.path.splitext(path)[1].lower() != ".pt":
            raise ValueError(f"direct teacher path must be a .pt file: {path}")
    return paths


class TeacherFieldCollection:
    """Index source tensors without concatenating the full teacher fields in RAM."""

    def __init__(self, fields):
        self.fields = list(fields)
        self.offsets = np.cumsum([0] + [len(field) for field in self.fields])
        self.shape = (int(self.offsets[-1]), *self.fields[0].shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            first, *rest = index
            result = self[first]
            return result[(slice(None), *rest)] if result.ndim == len(self.shape) else result[tuple(rest)]
        if isinstance(index, (int, np.integer)) or (torch.is_tensor(index) and index.ndim == 0):
            index = int(index)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            source = int(np.searchsorted(self.offsets, index, side="right") - 1)
            return self.fields[source][index - int(self.offsets[source])].float()
        indices = np.arange(len(self))[index] if isinstance(index, slice) else _to_numpy(index)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not len(indices):
            return torch.empty((0, *self.shape[1:]), dtype=torch.float32)
        return torch.stack([self[int(i)] for i in indices])

    def index_select(self, dim, index):
        if dim != 0:
            raise ValueError("teacher fields support index_select only along transitions")
        return self[index]


def infer_points_from_blob(blob: dict[str, Any], x: torch.Tensor) -> np.ndarray:
    if "points" in blob:
        pts = _to_numpy(blob["points"]).astype(np.float32)
    else:
        pts = x[0, :, 2:4].detach().cpu().numpy().astype(np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (Ntrain, 2), got {pts.shape}")
    return pts


def _is_transition_sized(value: Any, n_transition: int) -> bool:
    if value is None:
        return False
    try:
        return int(_to_numpy(value).shape[0]) == int(n_transition)
    except Exception:
        return False


def _concat_transition_values(values: list[Any]) -> Any:
    if all(torch.is_tensor(v) for v in values):
        return torch.cat([v.contiguous() for v in values], dim=0)
    arrays = [_to_numpy(v) for v in values]
    return np.concatenate(arrays, axis=0)


def _same_array(a: Any, b: Any) -> bool:
    arr_a = _to_numpy(a)
    arr_b = _to_numpy(b)
    return arr_a.shape == arr_b.shape and np.allclose(arr_a, arr_b, rtol=1.0e-6, atol=1.0e-7)


def _safe_filename_tag(value: str) -> str:
    tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    tag = tag.strip("_")
    return tag if tag else "source"


def _transition_parameter_values(
    blob: dict[str, Any],
    x: torch.Tensor,
    channel_idx: int,
    params_col: int,
) -> torch.Tensor | None:
    params = blob.get("params")
    if params is not None:
        params_t = torch.as_tensor(_to_numpy(params), dtype=torch.float32)
        if params_t.ndim == 2 and params_t.shape[0] == x.shape[0] and params_t.shape[1] > int(params_col):
            return params_t[:, int(params_col)].contiguous()

    idx = int(channel_idx)
    if 0 <= idx < x.shape[-1]:
        return x[..., idx].mean(dim=1).contiguous()
    return None


def _load_single_teacher_blob(path: str, args: argparse.Namespace) -> dict[str, Any]:
    use_mmap = bool(getattr(args, "teacher_mmap", 1))
    try:
        blob = safe_torch_load(path, map_location="cpu", mmap=use_mmap)
    except RuntimeError as exc:
        if not use_mmap or "mmap" not in str(exc).lower():
            raise
        print(f"WARNING: memory mapping unavailable for {path}; loading normally on CPU", flush=True)
        blob = safe_torch_load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError(f"teacher file must contain a dict: {path}")
    if "x" not in blob or "y" not in blob:
        raise KeyError(f"teacher file must contain keys 'x' and 'y': {path}")

    x = blob["x"]
    y = blob["y"]
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError(f"'x' and 'y' must be torch tensors: {path}")
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"expected x/y to be 3-D in {path}, got x={tuple(x.shape)}, y={tuple(y.shape)}")
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1] or y.shape[2] != 2:
        raise ValueError(f"incompatible x/y shapes in {path}: x={tuple(x.shape)}, y={tuple(y.shape)}")

    if len(x) == 0:
        raise ValueError(f"teacher file contains no transitions: {path}")
    if x.shape[-1] < 4:
        raise ValueError(
            f"teacher x must contain at least [u_k,v_k,x,y], got Cin={x.shape[-1]} in {path}"
        )

    points = torch.as_tensor(infer_points_from_blob(blob, x), dtype=torch.float32)
    spectral_d1_values = _transition_parameter_values(
        blob, x, int(args.d1_channel_idx), params_col=0
    )
    spectral_d2_values = _transition_parameter_values(
        blob, x, int(args.d2_channel_idx), params_col=1
    )

    blob = dict(blob)
    metadata = blob.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("sample_id", "time_index", "teacher_times", "times", "dt", "dt_physical",
                    "dt_normalized", "dt_unit", "dt_is_normalized", "T", "t_cur", "t_next"):
            if key not in blob and key in metadata:
                blob[key] = metadata[key]
    times_raw = blob.get("teacher_times", blob.get("times"))
    times = None
    if times_raw is not None:
        times = np.asarray(_to_numpy(times_raw), dtype=np.float64).reshape(-1)
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"teacher times must be finite and strictly increasing: {path}")
        n_step = len(times) - 1
        if "sample_id" not in blob or "time_index" not in blob:
            if len(x) % n_step:
                raise ValueError("sample_id/time_index required for non-trajectory-major data")
            blob.setdefault("sample_id", torch.arange(len(x)) // n_step)
            blob.setdefault("time_index", torch.arange(len(x)) % n_step)
        blob["times"] = times
    for key in ("sample_id", "time_index"):
        if key not in blob:
            raise ValueError(f"teacher requires {key} or teacher_times for trajectory-major inference: {path}")
        values = torch.as_tensor(_to_numpy(blob[key])).reshape(-1)
        if values.numel() != len(x) or not torch.isfinite(values).all() or not torch.equal(values, values.long()):
            raise ValueError(f"teacher {key} must contain one integer per transition: {path}")
        blob[key] = values.long()
    for sid in torch.unique(blob["sample_id"]):
        order = blob["time_index"][blob["sample_id"] == sid]
        expected_steps = len(times) - 1 if times is not None else len(order)
        if not torch.equal(order, torch.arange(expected_steps)):
            raise ValueError(f"teacher trajectory {int(sid)} must contain each time_index once in increasing order: {path}")

    scale = float(args.dt_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("dt_scale must be finite and positive")
    if "T" in blob and not math.isclose(float(blob["T"]), scale, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"teacher T={blob['T']} contradicts --dt_scale={scale}")
    explicit_units = None
    if "dt_physical" in blob:
        dt = torch.as_tensor(_to_numpy(blob["dt_physical"]), dtype=torch.float32).reshape(-1).clone()
        explicit_units = "physical"
        dt_source = "explicit dt_physical metadata"
    elif "dt_normalized" in blob:
        dt = torch.as_tensor(_to_numpy(blob["dt_normalized"]), dtype=torch.float32).reshape(-1).clone()
        explicit_units = "normalized"
        dt_source = "explicit dt_normalized metadata"
    elif "dt" in blob:
        dt = torch.as_tensor(_to_numpy(blob["dt"]), dtype=torch.float32).reshape(-1)
        if dt.numel() != x.shape[0]:
            raise ValueError(
                f"top-level dt must have one value per transition in {path}; "
                f"got {dt.numel()} values for {x.shape[0]} transitions"
            )
        dt_source = "top-level dt"
    else:
        dt_idx = int(args.dt_channel_idx)
        if dt_idx < 0 or dt_idx >= x.shape[-1]:
            raise ValueError(
                f"teacher input needs top-level 'dt' or a valid --dt_channel_idx; "
                f"got dt_channel_idx={dt_idx}, Cin={x.shape[-1]} in {path}"
            )
        dt_nodes = x[..., dt_idx]
        dt = dt_nodes.mean(dim=1).contiguous()
        if not torch.allclose(dt_nodes, dt[:, None], rtol=1.0e-5, atol=1.0e-7):
            raise ValueError(f"dt must be constant over the Ntrain nodes of each transition in {path}")
        dt_source = f"legacy x[..., {dt_idx}]"

    if dt.numel() != len(x) or not torch.isfinite(dt).all() or torch.any(dt <= 0):
        raise ValueError(f"teacher dt must contain one finite positive value per transition: {path}")
    mode = str(args.dt_channel_mode)
    units = explicit_units
    if units is None:
        requested = getattr(args, "teacher_dt_units", "auto")
        legacy_units = {"raw": "physical", "normalized": "normalized", "keep_raw": "physical"}.get(mode)
        if requested != "auto" and legacy_units is not None and requested != legacy_units:
            raise ValueError("teacher_dt_units contradicts dt_channel_mode")
        requested = legacy_units or requested
        token = str(blob.get("dt_unit", "")).lower()
        unit_aliases = {"physical": "physical", "raw": "physical", "seconds": "physical",
                        "normalized": "normalized", "dimensionless": "normalized", "dt_norm": "normalized"}
        if token and token not in unit_aliases:
            raise ValueError(f"unrecognized teacher dt_unit: {token}")
        units = unit_aliases.get(token)
        if units is None and "dt_is_normalized" in blob:
            units = "normalized" if blob["dt_is_normalized"] else "physical"
        if requested != "auto":
            if units is not None and units != requested:
                raise ValueError("teacher dt units contradict file metadata")
            units = requested
        if units is None and times is not None:
            expected = torch.as_tensor(np.diff(times), dtype=torch.float32)[blob["time_index"]]
            raw_match = torch.allclose(dt, expected, rtol=1e-6, atol=1e-5)
            norm_match = torch.allclose(dt * scale, expected, rtol=1e-6, atol=1e-5)
            if raw_match != norm_match:
                units = "physical" if raw_match else "normalized"
        if units is None:
            raise ValueError("cannot determine teacher dt units; add dt_unit or pass --teacher_dt_units")
    physical_dt = dt * scale if units == "normalized" else dt.clone()
    normalized_dt = physical_dt / scale
    for key, expected in (("dt_physical", physical_dt), ("dt_normalized", normalized_dt)):
        if key in blob:
            stored = torch.as_tensor(_to_numpy(blob[key]), dtype=torch.float32).reshape(-1)
            if stored.shape != expected.shape or not torch.allclose(stored, expected, rtol=1e-6, atol=1e-8):
                raise ValueError(f"teacher {key} metadata contradicts dt")
    if explicit_units is not None and "dt" in blob:
        stored = torch.as_tensor(_to_numpy(blob["dt"]), dtype=torch.float32).reshape(-1)
        if stored.shape != normalized_dt.shape or not torch.allclose(stored, normalized_dt, rtol=1e-6, atol=1e-8):
            raise ValueError("top-level teacher dt must equal normalized metadata dt")
    if times is not None:
        expected = torch.as_tensor(np.diff(times), dtype=torch.float32)[blob["time_index"]]
        if not torch.allclose(physical_dt, expected, rtol=1e-6, atol=1e-5):
            raise ValueError(f"teacher dt/time_index/times contradiction: {path}")
        for key, offset in (("t_cur", 0), ("t_next", 1)):
            expected = torch.as_tensor(times, dtype=torch.float32)[blob["time_index"] + offset]
            if key in blob:
                stored = torch.as_tensor(_to_numpy(blob[key]), dtype=torch.float32).reshape(-1)
                if stored.shape != expected.shape or not torch.allclose(stored, expected, rtol=0, atol=1e-5):
                    raise ValueError(f"teacher {key} contradicts teacher_times/time_index")
            blob[key] = expected
    effective_dt_scale = 1.0 if mode == "keep_raw" else scale
    dt = physical_dt if mode == "keep_raw" else normalized_dt
    dt_desc = f"transition dt = physical dt / {effective_dt_scale:g}"
    if points.shape != (x.shape[1], 2) or not torch.isfinite(points).all():
        raise ValueError(f"invalid teacher points: {path}")
    if bool(getattr(args, "teacher_full_finite_check", 1)):
        for start in range(0, len(x), 8):
            xc, yc = x[start:start + 8], y[start:start + 8]
            if not torch.isfinite(xc[..., :2]).all() or not torch.isfinite(yc).all():
                raise ValueError(f"teacher fields contain NaN or Inf: {path}")
            if not torch.allclose(xc[..., 2:4].float(), points.expand(len(xc), -1, -1), rtol=1e-5, atol=1e-6):
                raise ValueError(f"teacher coordinates change across transitions: {path}")

    out = dict(blob)
    out["x"] = x[..., :4]
    out["dt"] = dt.contiguous()
    out["y"] = y
    out["points"] = points
    if spectral_d1_values is not None:
        out["_spectral_d1_values"] = spectral_d1_values
    if spectral_d2_values is not None:
        out["_spectral_d2_values"] = spectral_d2_values
    out["_effective_dt_scale"] = float(effective_dt_scale)
    out["_dt_desc"] = dt_desc
    out["_dt_source"] = dt_source
    out["_source_path"] = path
    if "sample_id" in blob:
        out["_original_sample_id"] = torch.as_tensor(_to_numpy(blob["sample_id"]), dtype=torch.long).reshape(-1)
    return out


def _merge_teacher_blobs(blobs: list[dict[str, Any]], paths: list[str]) -> dict[str, Any]:
    if len(blobs) == 1:
        out = dict(blobs[0])
        out["_teacher_paths"] = list(paths)
        out["_source_transition_counts"] = [int(out["x"].shape[0])]
        out["_teacher_source_id"] = torch.zeros((int(out["x"].shape[0]),), dtype=torch.long)
        return out

    first = blobs[0]
    n_points = int(first["x"].shape[1])
    cin = int(first["x"].shape[2])
    points0 = first["points"]
    dt_desc = str(first["_dt_desc"])
    effective_dt_scale = float(first["_effective_dt_scale"])
    counts = [int(blob["x"].shape[0]) for blob in blobs]

    for path, blob in zip(paths, blobs):
        if int(blob["x"].shape[1]) != n_points or int(blob["x"].shape[2]) != cin:
            raise ValueError(
                "all teacher files must share Ntrain and Cin; "
                f"first has {(n_points, cin)}, {path} has {(int(blob['x'].shape[1]), int(blob['x'].shape[2]))}"
            )
        if int(blob["y"].shape[1]) != n_points or int(blob["y"].shape[2]) != 2:
            raise ValueError(f"teacher y shape is incompatible in {path}: {tuple(blob['y'].shape)}")
        if not _same_array(points0, blob["points"]):
            raise ValueError(f"teacher points do not match the first file: {path}")
        if str(blob["_dt_desc"]) != dt_desc or float(blob["_effective_dt_scale"]) != effective_dt_scale:
            raise ValueError(f"teacher dt handling metadata differs in {path}")

    out: dict[str, Any] = {
        "x": TeacherFieldCollection([blob["x"] for blob in blobs]),
        "dt": torch.cat([blob["dt"] for blob in blobs], dim=0).contiguous(),
        "y": TeacherFieldCollection([blob["y"] for blob in blobs]),
        "points": points0,
        "_effective_dt_scale": effective_dt_scale,
        "_dt_desc": dt_desc,
        "_teacher_paths": list(paths),
        "_source_transition_counts": counts,
        "_teacher_source_id": torch.cat(
            [torch.full((count,), source_idx, dtype=torch.long) for source_idx, count in enumerate(counts)],
            dim=0,
        ),
    }

    if all("times" in blob and _same_array(first["times"], blob["times"]) for blob in blobs):
        out["times"] = first["times"]

    common_keys = set(blobs[0].keys())
    for blob in blobs[1:]:
        common_keys.intersection_update(blob.keys())

    skip_keys = {
        "x",
        "dt",
        "y",
        "points",
        "times",
        "teacher_times",
        "metadata",
        "_effective_dt_scale",
        "_dt_desc",
        "_dt_source",
        "_source_path",
        "_teacher_paths",
        "_source_transition_counts",
        "_teacher_source_id",
    }
    for key in sorted(common_keys - skip_keys):
        if all(_is_transition_sized(blob.get(key), count) for blob, count in zip(blobs, counts)):
            if key == "sample_id":
                offset = 0
                shifted: list[torch.Tensor] = []
                for blob in blobs:
                    sample_id = torch.as_tensor(_to_numpy(blob[key]), dtype=torch.long).reshape(-1)
                    unique_ids, sample_id = torch.unique(sample_id, sorted=True, return_inverse=True)
                    sample_id = sample_id + int(offset)
                    shifted.append(sample_id)
                    offset += int(unique_ids.numel())
                out[key] = torch.cat(shifted, dim=0)
            else:
                out[key] = _concat_transition_values([blob[key] for blob in blobs])

    metadata = dict(first.get("metadata", {})) if isinstance(first.get("metadata"), dict) else {}
    metadata.update(
        {
            "source_teacher_paths": list(paths),
            "num_source_files": int(len(paths)),
            "source_transition_counts": [int(c) for c in counts],
            "num_transitions": int(out["x"].shape[0]),
            "index_order": "source-major concatenation; each source keeps its original transition order",
        }
    )
    if "sample_id" in out:
        sample_ids = np.asarray(_to_numpy(out["sample_id"])).reshape(-1)
        metadata["Nsamp"] = int(np.unique(sample_ids.astype(np.int64)).size)
    out["metadata"] = metadata
    return out


def load_teacher_blob(args: argparse.Namespace) -> dict[str, Any]:
    paths = parse_teacher_paths(args.teacher_paths)
    blobs = [_load_single_teacher_blob(path, args) for path in paths]
    return _merge_teacher_blobs(blobs, paths)


def split_counts_by_teacher_source(blob: dict[str, Any], indices: np.ndarray) -> list[tuple[str, int]]:
    source_ids = _tensor_1d_to_numpy(blob.get("_teacher_source_id"))
    paths = blob.get("_teacher_paths")
    if (
        source_ids is None
        or source_ids.shape[0] != int(blob["x"].shape[0])
        or not isinstance(paths, (list, tuple))
        or len(paths) <= 1
    ):
        return []
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    source_ids = source_ids.astype(np.int64)
    return [
        (_teacher_source_label(blob, int(source_id)), int(np.sum(source_ids[idx] == int(source_id))))
        for source_id in np.unique(source_ids)
    ]


def print_teacher_data_summary(blob, train_idx, test_idx):
    """Use the FDM reporting labels while describing FEM node inputs and separate dt."""
    x, dt, y = blob["x"], blob["dt"], blob["y"]
    paths = blob["_teacher_paths"]
    source_ids = _tensor_1d_to_numpy(blob["_teacher_source_id"])
    sample_ids = _tensor_1d_to_numpy(blob.get("_original_sample_id", blob["sample_id"]))
    keys = np.column_stack([source_ids, sample_ids]).astype(np.int64)
    trajectory_keys, transition_counts = np.unique(keys, axis=0, return_counts=True)
    train_keys = np.unique(keys[train_idx], axis=0)
    test_keys = np.unique(keys[test_idx], axis=0)
    effective_dt_scale = float(blob["_effective_dt_scale"])
    dt_values = _tensor_1d_to_numpy(dt)
    n_points, node_channels = int(x.shape[1]), int(x.shape[2])

    print("Teacher data summary", flush=True)
    print(f"  files: {len(paths)}", flush=True)
    print(f"  points: {n_points}", flush=True)
    print(f"  trajectories: {len(trajectory_keys)}", flush=True)
    print(f"  frames/trajectory: {sorted(set((transition_counts + 1).tolist()))}", flush=True)
    print(f"  transitions/trajectory: {sorted(set(transition_counts.tolist()))}", flush=True)
    print(f"  total transitions: {int(x.shape[0])}", flush=True)
    if "times" in blob:
        times = _tensor_1d_to_numpy(blob["times"])
        print(f"  time range: {float(times[0]):g} -> {float(times[-1]):g}", flush=True)
    print(f"  physical dt unique: {np.unique(dt_values * effective_dt_scale).tolist()}", flush=True)
    print(f"  input dt unique: {np.unique(dt_values).tolist()}", flush=True)
    print("  input channels: [u,v,x,y] with separate dt", flush=True)
    print(f"  input shape example: ([B,{n_points},{node_channels}], [B])", flush=True)
    print(f"  target shape example: [B,{n_points},2]", flush=True)
    print(f"  train trajectories: {len(train_keys)}", flush=True)
    print(f"  test trajectories: {len(test_keys)}", flush=True)
    print(f"  train transitions: {len(train_idx)}", flush=True)
    print(f"  test transitions: {len(test_idx)}", flush=True)
    print(f"  dt_scale: {effective_dt_scale:g}", flush=True)
    print(f"  teacher node x shape: {tuple(x.shape)}", flush=True)
    print(f"  teacher dt shape: {tuple(dt.shape)}", flush=True)
    print(f"  teacher y shape: {tuple(y.shape)}", flush=True)
    print(f"  dt channel: {blob['_dt_desc']}", flush=True)
    if len(paths) == 1:
        print(f"  dt source: {blob['_dt_source']}", flush=True)
    print(
        "  model input: ({[u_k(x_j,y_j), v_k(x_j,y_j), x_j, y_j]}_j, dt_k) "
        "-> [u_{k+1}, v_{k+1}]",
        flush=True,
    )
    for source_id, (path, count) in enumerate(zip(paths, blob["_source_transition_counts"])):
        n_trajectories = int(np.count_nonzero(trajectory_keys[:, 0] == source_id))
        print(
            f"  file[{source_id}]: {path} | trajectories={n_trajectories} transitions={count}",
            flush=True,
        )
    for split, split_keys in (("train", train_keys), ("test", test_keys)):
        print(f"  {split} trajectory IDs:", flush=True)
        for source_id in np.unique(split_keys[:, 0]):
            ids = split_keys[split_keys[:, 0] == source_id, 1].tolist()
            print(f"    source {int(source_id)}: {ids}", flush=True)
    source_test_counts = split_counts_by_teacher_source(blob, test_idx)
    if source_test_counts:
        print(
            "  test transitions by teacher source: "
            + ", ".join(f"{label}={count}" for label, count in source_test_counts),
            flush=True,
        )


class TeacherTransitionDataset(Dataset):
    def __init__(self, blob: dict[str, Any], indices: np.ndarray, t_cumulative: torch.Tensor):
        super().__init__()
        self.x = blob["x"]
        self.dt = blob["dt"]
        self.y = blob["y"]
        self.t_cumulative = t_cumulative
        self.indices = torch.as_tensor(indices.astype(np.int64), dtype=torch.long)

        n_transition = int(self.x.shape[0])
        if int(self.t_cumulative.numel()) != n_transition:
            raise ValueError(
                f"t_cumulative must have one value per transition; got {int(self.t_cumulative.numel())} "
                f"values for {n_transition} transitions"
            )

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int):
        j = int(self.indices[int(idx)])
        return self.x[j].float(), self.dt[j], self.t_cumulative[j], self.y[j].float()


def build_split_indices(blob: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    n_total = int(blob["x"].shape[0])
    sample_ids = _tensor_1d_to_numpy(blob.get("_original_sample_id", blob.get("sample_id")))
    source_ids = _tensor_1d_to_numpy(blob.get("_teacher_source_id", np.zeros(n_total, dtype=np.int64)))
    if sample_ids is None or len(sample_ids) != n_total or len(source_ids) != n_total:
        raise ValueError("trajectory split requires one sample_id/source_id per transition")
    records = [(int(src), int(sid)) for src, sid in zip(source_ids, sample_ids)]
    ordered = list(dict.fromkeys(records))
    all_keys = set(ordered)

    def select(value, name):
        value = str(value or "").strip()
        if value.lower() in {"", "none", "-"}:
            return None
        if value.lower() == "last" and name == "test_sample_ids":
            return {max(key for key in ordered if key[0] == src) for src in set(source_ids)}
        selected = set()
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            matches = ({tuple(map(int, token.split(":")))} & all_keys if ":" in token
                       else {key for key in ordered if key[1] == int(token)})
            if not matches:
                raise ValueError(f"--{name} references unknown trajectory: {token}")
            selected.update(matches)
        return selected

    train = select(getattr(args, "train_sample_ids", ""), "train_sample_ids")
    test = select(args.test_sample_ids, "test_sample_ids")
    split_seed = int(getattr(args, "split_seed", 0))
    if train is not None or test is not None:
        train = all_keys - test if train is None else train
        test = all_keys - train if test is None else test
        if train & test:
            raise ValueError("explicit teacher split overlaps on trajectories")
        if train | test != all_keys:
            raise ValueError("explicit teacher split leaves trajectories unassigned")
    else:
        ratio = getattr(args, "test_ratio", None)
        fraction = float(args.train_fraction) if ratio is None else 1.0 - float(ratio)
        if not 0 < fraction < 1:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        permutation = np.random.default_rng(split_seed).permutation(len(ordered))
        n_train = max(1, min(len(ordered) - 1, int(round(fraction * len(ordered)))))
        train = {ordered[int(i)] for i in permutation[:n_train]}
        test = all_keys - train

    def limit(keys, name, offset):
        candidates = [key for key in ordered if key in keys]
        count = int(getattr(args, name, -1))
        if count == -1:
            return keys
        if count <= 0 or count > len(candidates):
            raise ValueError(f"--{name} must be -1 or in [1,{len(candidates)}], got {count}")
        permutation = np.random.default_rng(split_seed + offset).permutation(len(candidates))
        return {candidates[int(i)] for i in permutation[:count]}

    train = limit(train, "num_train_trajectories", 104729)
    test = limit(test, "num_test_trajectories", 130363)
    if not train or not test:
        raise ValueError("teacher split requires nonempty train/test trajectory sets; provide at least two trajectories")
    result = []
    for keys, name in ((train, "max_train_samples"), (test, "max_test_samples")):
        indices = np.asarray([i for i, key in enumerate(records) if key in keys], dtype=np.int64)
        maximum = int(getattr(args, name, 0))
        if maximum > 0 and maximum < len(indices):
            raise ValueError(f"--{name} would truncate trajectories; use --num_train_trajectories/--num_test_trajectories")
        result.append(indices)
    return result[0], result[1]


# ------------------------------------------------------------------------
# Disk geometry, quadrature, and eigenbasis
# ------------------------------------------------------------------------

def build_polar_mass_weights(points: np.ndarray, Nr: int, Ntheta: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])
    if int(Nr) > 0 and int(Ntheta) > 0 and n == int(Nr) * int(Ntheta):
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        dr = 1.0 / float(Nr)
        dtheta = 2.0 * math.pi / float(Ntheta)
        w = r * dr * dtheta
    else:
        w = np.full((n,), math.pi / max(n, 1), dtype=np.float64)
    w = np.maximum(w, 1.0e-14)
    return w.astype(np.float32)


def build_square_grid_embedding(points: np.ndarray, grid_size: int) -> dict[str, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {pts.shape}")
    grid_size = int(grid_size)
    if grid_size < 2:
        raise ValueError(f"fourier grid_size must be >= 2, got {grid_size}")

    x = np.clip(pts[:, 0], -1.0, 1.0)
    y = np.clip(pts[:, 1], -1.0, 1.0)
    ix = np.rint((x + 1.0) * 0.5 * float(grid_size - 1)).astype(np.int64)
    iy = np.rint((y + 1.0) * 0.5 * float(grid_size - 1)).astype(np.int64)
    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)
    node_to_grid = iy * grid_size + ix

    grid_1d = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    gx, gy = np.meshgrid(grid_1d, grid_1d, indexing="xy")
    disk_mask = ((gx * gx + gy * gy) <= 1.0).astype(np.float32)
    disk_mask.reshape(-1)[node_to_grid] = 1.0

    counts = np.bincount(node_to_grid, minlength=grid_size * grid_size).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    return {
        "grid_size": np.asarray([grid_size], dtype=np.int64),
        "node_to_grid": node_to_grid.astype(np.int64),
        "grid_count": counts.reshape(-1, 1).astype(np.float32),
        "disk_mask": disk_mask.astype(np.float32),
        "grid_x": gx.reshape(-1).astype(np.float32),
        "grid_y": gy.reshape(-1).astype(np.float32),
    }


def mass_orthonormalize(B: np.ndarray, mass_w: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    B = np.asarray(B, dtype=np.float64)
    mass_w = np.asarray(mass_w, dtype=np.float64).reshape(-1)
    if B.ndim != 2:
        raise ValueError("basis candidates must be a matrix")
    if B.shape[0] != mass_w.shape[0]:
        raise ValueError("basis candidate and mass sizes do not match")

    sw = np.sqrt(np.maximum(mass_w, 1.0e-14))
    BW = B * sw[:, None]
    Q, R = np.linalg.qr(BW, mode="reduced")
    diag = np.abs(np.diag(R))
    diag_max = float(diag.max()) if diag.size else 0.0
    keep = diag > max(1.0e-10, diag_max * 1.0e-10)
    Q = Q[:, keep]
    if Q.shape[1] < int(K):
        raise ValueError(f"basis rank {Q.shape[1]} is smaller than requested K={K}")
    Q = Q[:, :int(K)]
    Phi = Q / sw[:, None]
    P = mass_w[:, None] * Phi
    return Phi.astype(np.float32), P.astype(np.float32)


def normalized_xy_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    xhat = (x - x.min()) / max(float(x.max() - x.min()), 1.0e-12)
    yhat = (y - y.min()) / max(float(y.max() - y.min()), 1.0e-12)
    return xhat, yhat


def build_cartesian_basis(points: np.ndarray, mass_w: np.ndarray, Kx: int, Ky: int) -> dict[str, np.ndarray]:
    Kx = int(Kx)
    Ky = int(Ky)
    if Kx <= 0 or Ky <= 0:
        raise ValueError(f"Kx and Ky must be positive for cartesian basis, got Kx={Kx}, Ky={Ky}")

    xhat, yhat = normalized_xy_from_points(points)
    cols: list[np.ndarray] = []
    mu: list[float] = []
    kx_list: list[int] = []
    ky_list: list[int] = []

    for ky in range(Ky):
        cy = np.cos(math.pi * float(ky) * yhat)
        for kx in range(Kx):
            cx = np.cos(math.pi * float(kx) * xhat)
            cols.append(cx * cy)
            mu.append(float(kx * kx + ky * ky))
            kx_list.append(int(kx))
            ky_list.append(int(ky))

    B = np.stack(cols, axis=1)
    K_total = int(Kx * Ky)
    Phi, P = mass_orthonormalize(B, mass_w, K_total)
    return {
        "Phi": Phi,
        "P": P,
        "mu": np.asarray(mu, dtype=np.float32),
        "kx": np.asarray(kx_list, dtype=np.int64),
        "ky": np.asarray(ky_list, dtype=np.int64),
        "Kx": np.asarray([Kx], dtype=np.int64),
        "Ky": np.asarray([Ky], dtype=np.int64),
    }


def build_fourier_like_basis(points: np.ndarray, mass_w: np.ndarray, K: int) -> dict[str, np.ndarray]:
    xhat, yhat = normalized_xy_from_points(points)

    cols: list[np.ndarray] = [np.ones_like(xhat)]
    mu: list[float] = [0.0]
    shell = 0
    while len(cols) < int(K) + 8:
        shell += 1
        for kx in range(0, shell + 1):
            for ky in range(-shell, shell + 1):
                if kx == 0 and ky <= 0:
                    continue
                if max(abs(kx), abs(ky)) != shell:
                    continue
                phase = 2.0 * math.pi * (float(kx) * xhat + float(ky) * yhat)
                eig = float(kx * kx + ky * ky)
                cols.append(np.cos(phase))
                mu.append(eig)
                if len(cols) >= int(K) + 8:
                    break
                cols.append(np.sin(phase))
                mu.append(eig)
                if len(cols) >= int(K) + 8:
                    break
            if len(cols) >= int(K) + 8:
                break

    B = np.stack(cols, axis=1)
    Phi, P = mass_orthonormalize(B, mass_w, int(K))
    return {
        "Phi": Phi,
        "P": P,
        "mu": np.asarray(mu[: int(K)], dtype=np.float32),
    }


def disk_radius_from_points(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    r_max = float(np.max(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)))
    if not math.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("disk radius inferred from points must be positive")
    if r_max <= 1.0 + 1.0e-6:
        return 1.0
    return r_max


def neumann_bessel_mode_specs(K: int, radius: float) -> list[dict[str, Any]]:
    try:
        from scipy.special import jnp_zeros
    except ImportError as exc:
        raise ImportError(
            "--basis_type disk requires scipy.special.jnp_zeros to build "
            "Neumann-Bessel disk eigenmodes."
        ) from exc

    K = int(K)
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    radius = float(radius)
    root_count = max(8, int(math.ceil(math.sqrt(float(K)))) * 2)
    n_limit = max(8, int(math.ceil(math.sqrt(float(K)))) * 3)

    for _ in range(8):
        specs: list[dict[str, Any]] = [
            {"n": 0, "m": 0, "alpha": 0.0, "trig": "const", "mu": 0.0}
        ]
        roots_by_n: dict[int, np.ndarray] = {}
        for n in range(n_limit + 1):
            roots = np.asarray(jnp_zeros(n, root_count), dtype=np.float64)
            roots_by_n[n] = roots
            for m, alpha in enumerate(roots, start=1):
                mu = float((float(alpha) / radius) ** 2)
                if n == 0:
                    specs.append({"n": n, "m": m, "alpha": float(alpha), "trig": "cos", "mu": mu})
                else:
                    specs.append({"n": n, "m": m, "alpha": float(alpha), "trig": "cos", "mu": mu})
                    specs.append({"n": n, "m": m, "alpha": float(alpha), "trig": "sin", "mu": mu})

        specs.sort(key=lambda s: (float(s["mu"]), int(s["n"]), int(s["m"]), str(s["trig"])))
        if len(specs) >= K:
            cutoff_mu = float(specs[K - 1]["mu"])
            cutoff_alpha = math.sqrt(max(cutoff_mu, 0.0)) * radius
            mu_tol = max(1.0e-10, abs(cutoff_mu) * 1.0e-10)
            chosen = [s for s in specs if float(s["mu"]) <= cutoff_mu + mu_tol]
            roots_cover_cutoff = all(
                float(roots_by_n[n][-1]) > cutoff_alpha + 1.0e-10
                for n in range(n_limit + 1)
            )
            orders_cover_cutoff = float(n_limit) > cutoff_alpha + 2.0
            if roots_cover_cutoff and orders_cover_cutoff:
                return chosen

        root_count *= 2
        n_limit *= 2

    raise RuntimeError("failed to enumerate enough Neumann-Bessel disk modes")


def build_disk_polar_basis(points: np.ndarray, mass_w: np.ndarray, K: int) -> dict[str, np.ndarray]:
    try:
        from scipy.special import jv
    except ImportError as exc:
        raise ImportError("--basis_type disk requires scipy.special.jv to evaluate Bessel functions.") from exc

    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    radius = disk_radius_from_points(pts)
    r = np.clip(np.sqrt(x * x + y * y) / radius, 0.0, 1.0)
    theta = np.arctan2(y, x)

    specs = neumann_bessel_mode_specs(int(K), radius)
    cols: list[np.ndarray] = []
    for spec in specs:
        n = int(spec["n"])
        alpha = float(spec["alpha"])
        trig = str(spec["trig"])
        if trig == "const":
            col = np.ones_like(r)
        else:
            radial = jv(n, alpha * r)
            if trig == "cos":
                col = radial * np.cos(float(n) * theta)
            elif trig == "sin":
                col = radial * np.sin(float(n) * theta)
            else:
                raise ValueError(f"unknown Bessel angular component: {trig}")
        cols.append(np.asarray(col, dtype=np.float64))

    B = np.stack(cols, axis=1)
    K_eff = len(specs)
    Phi, P = mass_orthonormalize(B, mass_w, K_eff)
    return {
        "Phi": Phi,
        "P": P,
        "mu": np.asarray([float(s["mu"]) for s in specs], dtype=np.float32),
        "bessel_n": np.asarray([int(s["n"]) for s in specs], dtype=np.int64),
        "bessel_m": np.asarray([int(s["m"]) for s in specs], dtype=np.int64),
        "bessel_alpha": np.asarray([float(s["alpha"]) for s in specs], dtype=np.float32),
        "bessel_trig": np.asarray([str(s["trig"]) for s in specs], dtype="<U8"),
        "disk_radius": np.asarray([float(radius)], dtype=np.float32),
        "requested_K": np.asarray([int(K)], dtype=np.int64),
        "K_eff": np.asarray([int(K_eff)], dtype=np.int64),
    }


def load_basis_from_path(path: str, K: int) -> dict[str, np.ndarray]:
    raw_obj: dict[str, Any]
    if path.endswith(".npz"):
        raw_obj = dict(np.load(path, allow_pickle=False))
    else:
        raw_loaded = safe_torch_load(path, map_location="cpu")
        if not isinstance(raw_loaded, dict):
            raise TypeError("basis file must contain a dict")
        raw_obj = raw_loaded

    points = _to_numpy(_first_key(raw_obj, ["points", "nodes", "coords", "x_nodes"], required=True)).astype(np.float32)
    mass_w = _to_numpy(_first_key(raw_obj, ["mass_w", "mass_lumped", "node_mass", "m_lumped", "mass_diag"], required=True)).astype(np.float32).reshape(-1)
    Phi_all = _to_numpy(_first_key(raw_obj, ["Phi", "phi", "eigvecs", "evecs"], required=True)).astype(np.float32)
    P_all = _first_key(raw_obj, ["P", "proj", "mass_proj"], required=False)
    if P_all is None:
        P_all = mass_w[:, None] * Phi_all
    else:
        P_all = _to_numpy(P_all).astype(np.float32)
    mu = _to_numpy(_first_key(raw_obj, ["mu", "eigvals", "evals"], required=False))
    if mu is None:
        mu = np.arange(Phi_all.shape[1], dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32).reshape(-1)

    K_eff = min(int(K), int(Phi_all.shape[1]), int(P_all.shape[1]), int(mu.shape[0]))
    return {
        "points": points,
        "mass_w": mass_w,
        "Phi": Phi_all[:, :K_eff].astype(np.float32),
        "P": P_all[:, :K_eff].astype(np.float32),
        "mu": mu[:K_eff].astype(np.float32),
    }


def build_basis(points: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    if str(args.basis_path).strip():
        basis = load_basis_from_path(args.basis_path, int(args.K))
        if basis["Phi"].shape[0] != points.shape[0]:
            raise ValueError(
                f"basis node count {basis['Phi'].shape[0]} does not match teacher Ntrain {points.shape[0]}"
            )
        return basis

    mass_w = build_polar_mass_weights(points, int(args.Nr), int(args.Ntheta))
    if args.basis_type == "disk":
        basis = build_disk_polar_basis(points, mass_w, int(args.K))
    elif args.basis_type == "cartesian":
        basis = build_cartesian_basis(points, mass_w, int(args.Kx), int(args.Ky))
    elif args.basis_type == "fourier":
        basis = build_fourier_like_basis(points, mass_w, int(args.K))
    else:
        raise ValueError(f"unknown basis_type: {args.basis_type}")

    basis["points"] = np.asarray(points, dtype=np.float32)
    basis["mass_w"] = mass_w.astype(np.float32)
    basis["basis_type"] = np.asarray([str(args.basis_type)], dtype="<U32")
    return basis



def _basis_prefix_relative_errors(coeff, residual_energy, target_energy):
    """Relative spatial L2 for every prefix; coeff is (batch, channel, K).

    Add omitted orthonormal-mode energy to the explicitly computed full-basis
    residual. This avoids cancellation in ||u||^2 - sum(|coeff|^2).
    A zero teacher field has error 0 if reconstructed exactly, otherwise inf.
    """
    power = coeff.square()
    tail = torch.flip(torch.cumsum(torch.flip(power, [-1]), -1), [-1])
    omitted = torch.cat([tail[..., 1:], torch.zeros_like(tail[..., :1])], -1)
    error_energy = (residual_energy[..., None] + omitted).clamp_min(0)
    denominator = target_energy[..., None]
    relative = torch.sqrt(error_energy / denominator.clamp_min(torch.finfo(coeff.dtype).tiny))
    return torch.where(denominator > 0, relative,
                       torch.where(error_energy == 0, torch.zeros_like(relative),
                                   torch.full_like(relative, float("inf"))))


def _update_basis_reconstruction_stats(stats, key, relative):
    # Combine u/v by energy before calling this for the combined channel.
    values = relative.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("basis reconstruction produced non-finite relative errors")
    entry = stats.setdefault(key, {
        "count": 0, "sum": np.zeros(values.shape[1:], dtype=np.float64),
        "max": np.zeros(values.shape[1:], dtype=np.float64),
    })
    entry["count"] += len(values)
    entry["sum"] += values.sum(axis=0)
    entry["max"] = np.maximum(entry["max"], values.max(axis=0))


def save_basis_reconstruction_report(stats, out_prefix, tolerance=0.01):
    """Write all prefix errors and minimum K for mean/all-snapshot criteria."""
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("reconstruction_tol must be finite and positive")
    if not stats:
        raise ValueError("no teacher fields were evaluated for basis reconstruction")
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    curve_path = out_prefix + "_basis_reconstruction_curve.csv"
    summary_path = out_prefix + "_basis_reconstruction_summary.csv"
    common = ["split", "field", "resolution", "channel", "snapshots"]
    with open(curve_path, "w", newline="", encoding="utf-8") as curve_file, open(
        summary_path, "w", newline="", encoding="utf-8"
    ) as summary_file:
        curve = csv.writer(curve_file)
        summary = csv.writer(summary_file)
        curve.writerow(common + ["K", "relative_l2_mean", "relative_l2_max"])
        summary.writerow(common + [
            "configured_K", "relative_l2_mean", "relative_l2_max", "tolerance",
            "min_K_mean", "min_K_all", "all_within_tolerance",
        ])
        for (split, field, resolution), entry in sorted(stats.items()):
            means = entry["sum"] / entry["count"]
            maxima = entry["max"]
            for channel, name in enumerate(("u", "v", "combined")):
                mean, maximum = means[channel], maxima[channel]
                base = [split, field, resolution, name, entry["count"]]
                for k in range(len(mean)):
                    curve.writerow(base + [k + 1, float(mean[k]), float(maximum[k])])
                mean_hits = np.flatnonzero(mean <= tolerance)
                all_hits = np.flatnonzero(maximum <= tolerance)
                min_mean = int(mean_hits[0] + 1) if len(mean_hits) else ""
                min_all = int(all_hits[0] + 1) if len(all_hits) else ""
                summary.writerow(base + [
                    len(mean), float(mean[-1]), float(maximum[-1]), tolerance,
                    min_mean, min_all, bool(maximum[-1] <= tolerance),
                ])
                print(
                    f"basis reconstruction | {split}/{field}/{resolution}/{name} "
                    f"K={len(mean)} mean={mean[-1]:.8g} max={maximum[-1]:.8g} "
                    f"min_K_all(tol={tolerance:g})={min_all or 'not reached within configured basis'}",
                    flush=True,
                )
    print(f"saved: {curve_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)
    return stats


@torch.no_grad()
def evaluate_basis_reconstruction(blob, split_indices, basis_np, args):
    """Project raw input/target u,v onto the configured mass-weighted basis.

    Prefixes use stable ascending mu order, including individual degenerate
    modes. Weighted QR in float64 preserves each prefix span and corrects
    roundoff/non-unit scaling in imported basis files. Model buffers are untouched.
    """
    tolerance = float(args.reconstruction_tol)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("reconstruction_tol must be finite and positive")
    phi = torch.as_tensor(basis_np["Phi"], dtype=torch.float64, device=DEVICE)
    mass = torch.as_tensor(basis_np["mass_w"], dtype=torch.float64, device=DEVICE)
    if phi.ndim != 2 or phi.shape[1] == 0 or phi.shape[1] > phi.shape[0]:
        raise ValueError("basis reconstruction requires a nonempty, independent node basis")
    if mass.shape != (phi.shape[0],) or not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("basis reconstruction requires positive finite node mass weights")
    mu = np.asarray(basis_np["mu"]).reshape(-1)
    if len(mu) != phi.shape[1] or not np.isfinite(mu).all() or not torch.isfinite(phi).all():
        raise ValueError("invalid eigenvalues or basis values for reconstruction")
    order = torch.as_tensor(np.argsort(mu, kind="stable"), device=DEVICE)
    sqrt_mass = mass.sqrt()
    q, r = torch.linalg.qr(sqrt_mass[:, None] * phi[:, order], mode="reduced")
    if (r.diag().abs() <= r.abs().max() * 1.e-12).any():
        raise ValueError("configured reconstruction basis is rank deficient")
    stats = {}
    batch_size = max(1, int(args.batch_size))
    for split, indices in split_indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            for field, storage in (("input", blob["x"]), ("target", blob["y"])):
                values = storage[batch_indices][..., :2].to(device=DEVICE, dtype=torch.float64)
                weighted = values.transpose(1, 2) * sqrt_mass
                coeff = weighted @ q
                residual = weighted - coeff @ q.T
                residual_energy = residual.square().sum(-1)
                target_energy = weighted.square().sum(-1)
                relative = _basis_prefix_relative_errors(coeff, residual_energy, target_energy)
                # Concatenating coefficient vectors would change K; sum modal energies instead.
                combined = _basis_prefix_relative_errors(
                    coeff.square().sum(1, keepdim=True).sqrt(),
                    residual_energy.sum(1, keepdim=True), target_energy.sum(1, keepdim=True),
                )
                _update_basis_reconstruction_stats(
                    stats, (split, field, f"nodes{phi.shape[0]}"),
                    torch.cat([relative, combined], dim=1),
                )
            if start == 0 or (start // batch_size + 1) % 100 == 0:
                print(f"basis reconstruction | {split}: {min(start + batch_size, len(indices))}/{len(indices)} transitions", flush=True)
    return save_basis_reconstruction_report(
        stats, os.path.join(args.save_dir, args.out_prefix), tolerance,
    )


def tensor_basis_to_device(basis: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "points": torch.as_tensor(basis["points"], dtype=torch.float32, device=device),
        "mass_w": torch.as_tensor(basis["mass_w"], dtype=torch.float32, device=device),
        "Phi": torch.as_tensor(basis["Phi"], dtype=torch.float32, device=device),
        "P": torch.as_tensor(basis["P"], dtype=torch.float32, device=device),
        "mu": torch.as_tensor(basis["mu"], dtype=torch.float32, device=device),
    }


# ------------------------------------------------------------------------
# Input features and optional normalization
# ------------------------------------------------------------------------

def append_transition_dt(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"node input must have shape (B,N,C), got {tuple(x.shape)}")
    dt = dt.reshape(-1).to(device=x.device, dtype=x.dtype)
    if dt.shape[0] != x.shape[0]:
        raise ValueError(f"dt batch size {dt.shape[0]} does not match node-input batch size {x.shape[0]}")
    dt_nodes = dt[:, None, None].expand(-1, x.shape[1], 1)
    return torch.cat((x, dt_nodes), dim=-1)


def compute_input_normalization(
    blob: dict[str, Any],
    train_idx: np.ndarray,
    mode: str,
    eps: float,
    chunk_size: int = 16,
    return_feature_std: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalization and, optionally, raw std for [u,v,x,y,dt]."""
    x = blob["x"]
    dt = torch.as_tensor(blob["dt"], dtype=torch.float32).reshape(-1)
    feature_count = int(x.shape[-1]) + 1
    if str(mode) not in {"none", "zscore"}:
        raise ValueError(f"unknown input_normalization: {mode}")
    if str(mode) == "none" and not bool(return_feature_std):
        return torch.zeros(feature_count, dtype=torch.float32), torch.ones(feature_count, dtype=torch.float32)

    indices = np.asarray(train_idx, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("cannot fit input normalization because the training split is empty")
    if float(eps) <= 0.0:
        raise ValueError(f"input_norm_eps must be positive, got {eps}")

    x_sum = torch.zeros(int(x.shape[-1]), dtype=torch.float64)
    x_sum_sq = torch.zeros_like(x_sum)
    dt_sum = torch.zeros(1, dtype=torch.float64)
    dt_sum_sq = torch.zeros_like(dt_sum)
    x_count = 0
    dt_count = 0
    for start in range(0, int(indices.size), max(1, int(chunk_size))):
        chunk_np = indices[start : start + max(1, int(chunk_size))]
        chunk_idx = torch.as_tensor(chunk_np, dtype=torch.long)
        x_chunk = x.index_select(0, chunk_idx).to(dtype=torch.float64)
        dt_chunk = dt.index_select(0, chunk_idx).to(dtype=torch.float64)
        x_sum += x_chunk.sum(dim=(0, 1))
        x_sum_sq += (x_chunk * x_chunk).sum(dim=(0, 1))
        dt_sum += dt_chunk.sum()
        dt_sum_sq += (dt_chunk * dt_chunk).sum()
        x_count += int(x_chunk.shape[0] * x_chunk.shape[1])
        dt_count += int(dt_chunk.numel())

    x_mean = x_sum / float(x_count)
    dt_mean = dt_sum / float(dt_count)
    x_var = torch.clamp(x_sum_sq / float(x_count) - x_mean * x_mean, min=0.0)
    dt_var = torch.clamp(dt_sum_sq / float(dt_count) - dt_mean * dt_mean, min=0.0)
    mean = torch.cat((x_mean, dt_mean), dim=0)
    feature_std = torch.sqrt(torch.cat((x_var, dt_var), dim=0))
    if str(mode) == "zscore":
        scale = torch.where(feature_std >= float(eps), feature_std, torch.ones_like(feature_std))
    else:
        mean = torch.zeros_like(mean)
        scale = torch.ones_like(feature_std)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not torch.isfinite(feature_std).all():
        raise ValueError("non-finite value found while fitting input normalization")
    result = mean.to(dtype=torch.float32), scale.to(dtype=torch.float32)
    if bool(return_feature_std):
        return result + (feature_std.to(dtype=torch.float32),)
    return result


class InputFeatureNormalizer(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        scale_t = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if mean_t.shape != scale_t.shape:
            raise ValueError(f"input normalization mean/scale mismatch: {mean_t.shape} vs {scale_t.shape}")
        if torch.any(scale_t <= 0.0):
            raise ValueError("input normalization scale must be positive")
        self.register_buffer("mean", mean_t, persistent=True)
        self.register_buffer("scale", scale_t, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != int(self.mean.numel()):
            raise ValueError(
                f"input feature count {int(x.shape[-1])} does not match normalizer size {int(self.mean.numel())}"
            )
        shape = (1,) * (x.ndim - 1) + (int(self.mean.numel()),)
        mean = self.mean.to(device=x.device, dtype=x.dtype).reshape(shape)
        scale = self.scale.to(device=x.device, dtype=x.dtype).reshape(shape)
        return (x - mean) / scale


def mean_transition_value(
    values: torch.Tensor,
    indices: np.ndarray | None = None,
) -> float:
    src = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if indices is not None:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size > 0:
            src = src[torch.as_tensor(arr, dtype=torch.long)]
    return float(src.mean().item())


def resolve_spectral_diffusion_init(
    blob: dict[str, Any],
    args: argparse.Namespace,
    train_idx: np.ndarray,
) -> tuple[float, float, str]:
    defaults = {"d1": 0.02 ** 2, "d2": 1.0}
    resolved: dict[str, float] = {}
    sources: list[str] = []
    for name in ("d1", "d2"):
        explicit = getattr(args, f"spectral_{name}_init")
        values = blob.get(f"_spectral_{name}_values")
        if explicit is not None:
            resolved[name] = float(explicit)
            sources.append(f"{name}=--spectral_{name}_init")
        elif values is not None:
            resolved[name] = mean_transition_value(torch.as_tensor(values), train_idx)
            sources.append(f"{name}=teacher params/legacy channel mean")
        else:
            resolved[name] = float(defaults[name])
            sources.append(f"{name}=fallback {defaults[name]:g}")
    return resolved["d1"], resolved["d2"], "; ".join(sources)


# ------------------------------------------------------------------------
# Spectral layers and neural operator models
# ------------------------------------------------------------------------

class SpectralConvFFT2d(nn.Module):
    def __init__(self, channels, Kx, Ky):
        super().__init__()
        self.channels = int(channels)
        self.Kx = int(Kx)
        self.Ky = int(Ky)

        scale = 1.0 / max(1, self.channels * self.channels)
        w1 = scale * torch.randn(self.channels, self.channels, self.Ky, self.Kx, dtype=torch.cfloat)
        w2 = scale * torch.randn(self.channels, self.channels, self.Ky, self.Kx, dtype=torch.cfloat)
        self.weight_pos = nn.Parameter(w1)
        self.weight_neg = nn.Parameter(w2)

    @staticmethod
    def _compl_mul2d(inp, weight):
        return torch.einsum("bcyx,ocyx->boyx", inp, weight)

    @staticmethod
    def _autocast_disabled():
        if torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", enabled=False)
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def forward(self, x, dt_scalar=None):
        # x: (B,C,Ny,Nx)
        batch, _, ny, nx = x.shape
        out_dtype = x.dtype
        with self._autocast_disabled():
            x_ft = torch.fft.rfft2(x.float(), norm="ortho")
            out_ft = torch.zeros(batch, self.channels, ny, nx // 2 + 1, dtype=torch.cfloat, device=x.device)

            modes_y = min(int(self.Ky), int(ny))
            modes_x = min(int(self.Kx), int(nx // 2 + 1))
            if modes_y > 0 and modes_x > 0:
                # Preserve the original assignment order: negative y modes
                # overwrite positive y modes when the two slices overlap.
                pos_rows = min(modes_y, max(0, ny - modes_y))
                if pos_rows > 0:
                    out_ft[:, :, :pos_rows, :modes_x] = self._compl_mul2d(
                        x_ft[:, :, :pos_rows, :modes_x],
                        self.weight_pos[:, :, :pos_rows, :modes_x],
                    )
                out_ft[:, :, -modes_y:, :modes_x] = self._compl_mul2d(
                    x_ft[:, :, -modes_y:, :modes_x],
                    self.weight_neg[:, :, :modes_y, :modes_x],
                )

            y = torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")
        return y.to(dtype=out_dtype)


class SpectralConvNodes(nn.Module):
    """Nonseparable disk counterpart of FDM SpectralConvSeparable.

    Phi: (N,K) eigenfunctions, P: (N,K) mass-weighted projection,
    mu: (K,) physical Laplace eigenvalues, x: (B,C,N) hidden features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Phi: torch.Tensor,
        P: torch.Tensor,
        mu: torch.Tensor,
        use_layer_scaling: bool,
        d1_init: float,
        d2_init: float,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.K = int(Phi.shape[1])
        self.use_layer_scaling = bool(use_layer_scaling)
        self.register_buffer("Phi", Phi, persistent=False)
        self.register_buffer("P", P, persistent=False)
        self.register_buffer("mu", mu.reshape(1, 1, -1), persistent=False)
        self.weight = nn.Parameter(
            0.02 * torch.randn(self.K, self.out_channels, self.in_channels, dtype=torch.float32)
        )
        d_init = torch.full((self.out_channels,), float(d2_init), dtype=torch.float32)
        d_init[: self.out_channels // 2] = float(d1_init)
        self.d_c = nn.Parameter(self._inverse_softplus(d_init))
        self.lambda_c = nn.Parameter(self._inverse_softplus(torch.ones(self.out_channels, dtype=torch.float32)))

    @staticmethod
    def _inverse_softplus(value: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        value = torch.clamp(value.to(dtype=torch.float32), min=float(eps))
        return value + torch.log(-torch.expm1(-value))

    @staticmethod
    def _positive(raw: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        return torch.nn.functional.softplus(raw.float()) + float(eps)

    def set_basis(self, Phi: torch.Tensor, P: torch.Tensor, mu: torch.Tensor) -> None:
        if int(Phi.shape[1]) != int(self.weight.shape[0]):
            raise ValueError(
                f"cannot change spectral K from {int(self.weight.shape[0])} to {int(Phi.shape[1])}"
            )
        self.Phi = Phi
        self.P = P
        self.mu = mu.reshape(1, 1, -1)
        self.K = int(Phi.shape[1])

    def _mode_decay(self, dt_scalar: torch.Tensor) -> torch.Tensor:
        mode_mu = self.mu.to(device=dt_scalar.device, dtype=torch.float32)
        d_c = self._positive(self.d_c.to(device=dt_scalar.device)).view(1, -1, 1)
        lambda_c = self._positive(self.lambda_c.to(device=dt_scalar.device)).view(1, -1, 1)
        return torch.exp(-dt_scalar.float().view(-1, 1, 1) * (d_c * mode_mu + lambda_c))

    def forward(self, x: torch.Tensor, dt_scalar: torch.Tensor | None = None, *, dt=None):
        # dt_scalar is physical time, matching SpectralConvSeparable in FDM.
        # dt is the legacy keyword used by older FEM callers.
        if dt is not None:
            if dt_scalar is not None:
                raise ValueError("Pass only dt_scalar (physical time), not both dt_scalar and dt")
            dt_scalar = dt
        # x: (B, C, N)
        coeff = torch.einsum("bcn,nk->bck", x, self.P)
        coeff = torch.einsum("bik,koi->bok", coeff, self.weight)
        if self.use_layer_scaling:
            if dt_scalar is None:
                raise ValueError("Laplacian spectral kernel requires dt")
            coeff = coeff * self._mode_decay(dt_scalar)
        return torch.einsum("bck,nk->bcn", coeff, self.Phi)


def _canonical_model_state(state):
    """Map pre-alignment pointwise.* weights to the shared FDM name w1.*."""
    if not isinstance(state, dict):
        return state
    result = OrderedDict()
    for key, value in state.items():
        canonical_key = "w1." + key[len("pointwise."):] if key.startswith("pointwise.") else key
        if canonical_key in result:
            raise ValueError(f"Checkpoint contains duplicate weights for {canonical_key}")
        result[canonical_key] = value
    return result


class _LegacyPointwiseState:
    """Preserve old checkpoints without registering duplicate parameter modules."""

    @property
    def pointwise(self):
        return self.w1

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in list(state_dict):
            old_prefix = prefix + "pointwise."
            if key.startswith(old_prefix):
                new_key = prefix + "w1." + key[len(old_prefix):]
                if new_key in state_dict:
                    raise ValueError(f"Checkpoint contains duplicate weights for {new_key}")
                state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class LENO2d(_LegacyPointwiseState, nn.Module):
    """LENO on disk nodes (B,N,C), corresponding to FDM's grid LENO2d."""
    def __init__(
        self,
        node_channels: int,
        out_channels: int,
        width: int,
        depth: int,
        Phi: torch.Tensor,
        P: torch.Tensor,
        mu: torch.Tensor,
        use_layer_scaling: bool,
        dt_scale: float,
        spectral_d1_init: float,
        spectral_d2_init: float,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        super().__init__()
        self.node_channels = int(node_channels)
        self.in_channels = self.node_channels + 1
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.use_layer_scaling = bool(use_layer_scaling)
        self.dt_scale = float(dt_scale)
        self.spectral_d1_init = float(spectral_d1_init)
        self.spectral_d2_init = float(spectral_d2_init)

        self.input_normalizer = InputFeatureNormalizer(input_mean, input_scale)
        self.fc0 = nn.Linear(self.in_channels, self.width)
        self.spec = nn.ModuleList(
            [
                SpectralConvNodes(
                    self.width,
                    self.width,
                    Phi,
                    P,
                    mu,
                    use_layer_scaling=self.use_layer_scaling,
                    d1_init=self.spectral_d1_init,
                    d2_init=self.spectral_d2_init,
                )
                for _ in range(self.depth)
            ]
        )
        self.w1 = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.out_channels)

    def set_basis(self, Phi: torch.Tensor, P: torch.Tensor, mu: torch.Tensor) -> None:
        for layer in self.spec:
            layer.set_basis(Phi, P, mu)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # x: (B,N,4)=[u_k,v_k,x,y], dt: (B,) transition scalar normalized by dt_scale.
        if x.shape[-1] != self.node_channels:
            raise ValueError(
                f"expected {self.node_channels} node channels [u_k,v_k,x,y], got {x.shape[-1]}"
            )
        dt_normalized = dt.reshape(-1)
        dt_physical = dt_normalized * self.dt_scale if self.use_layer_scaling else None
        x = append_transition_dt(x, dt_normalized)
        x = self.fc0(self.input_normalizer(x))
        x = x.transpose(1, 2)
        for sp, w in zip(self.spec, self.w1):
            x = self.act(sp(x, dt_scalar=dt_physical) + w(x))
        x = x.transpose(1, 2)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class DiskFNO2d(_LegacyPointwiseState, nn.Module):
    """Square-grid FNO with disk embedding and loss on the original nodes."""
    def __init__(
        self,
        node_channels: int,
        out_channels: int,
        width: int,
        depth: int,
        Kx: int,
        Ky: int,
        node_to_grid: torch.Tensor,
        grid_count: torch.Tensor,
        disk_mask: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
    ):
        super().__init__()
        self.node_channels = int(node_channels)
        self.in_channels = self.node_channels + 1
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.grid_size = int(disk_mask.shape[-1])
        self.grid_numel = int(self.grid_size * self.grid_size)

        self.register_buffer("node_to_grid", node_to_grid.to(dtype=torch.long), persistent=True)
        self.register_buffer("grid_count", grid_count.reshape(1, self.grid_numel, 1).to(dtype=torch.float32), persistent=True)
        self.register_buffer("disk_mask", disk_mask.reshape(1, 1, self.grid_size, self.grid_size).to(dtype=torch.float32), persistent=True)
        self.register_buffer("grid_x", grid_x.reshape(1, self.grid_numel).to(dtype=torch.float32), persistent=True)
        self.register_buffer("grid_y", grid_y.reshape(1, self.grid_numel).to(dtype=torch.float32), persistent=True)

        self.input_normalizer = InputFeatureNormalizer(input_mean, input_scale)
        self.fc0 = nn.Linear(self.in_channels, self.width)
        self.spec = nn.ModuleList([SpectralConvFFT2d(self.width, Kx, Ky) for _ in range(self.depth)])
        self.w1 = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.depth)])
        self.act = nn.GELU()
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.out_channels)

    def _nodes_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,Cin), output: (B,H,W,Cin). Channels outside the disk are zero.
        B, N, C = x.shape
        if int(N) != int(self.node_to_grid.numel()):
            raise ValueError(f"input node count {int(N)} does not match mapping size {int(self.node_to_grid.numel())}")
        G = int(self.grid_numel)
        grid = x.new_zeros(B, G, C)
        if C >= 2:
            idx = self.node_to_grid.view(1, N, 1).expand(B, N, 2)
            state = x.new_zeros(B, G, 2)
            state.scatter_add_(1, idx, x[:, :, :2])
            state = state / self.grid_count.to(device=x.device, dtype=x.dtype)
            grid[:, :, :2] = state
        if C >= 4:
            grid[:, :, 2] = self.grid_x.to(device=x.device, dtype=x.dtype)
            grid[:, :, 3] = self.grid_y.to(device=x.device, dtype=x.dtype)
        if C > 4:
            scalars = x[:, :, 4:].mean(dim=1)
            grid[:, :, 4:] = scalars[:, None, :].expand(B, G, C - 4)

        mask = self.disk_mask.reshape(1, G, 1).to(device=x.device, dtype=x.dtype)
        grid = grid * mask
        return grid.reshape(B, self.grid_size, self.grid_size, C)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # x: (B,N,4) on disk points; dt: (B,) transition scalar.
        if x.shape[-1] != self.node_channels:
            raise ValueError(
                f"expected {self.node_channels} node channels [u_k,v_k,x,y], got {x.shape[-1]}"
            )
        x = append_transition_dt(x, dt)
        B, N, _ = x.shape
        x = self._nodes_to_grid(x)
        x = self.input_normalizer(x)
        disk_mask = self.disk_mask.to(device=x.device, dtype=x.dtype)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = x * disk_mask
        for sp, w in zip(self.spec, self.w1):
            x = self.act(sp(x) + w(x))
            x = x * disk_mask
        x = x.permute(0, 2, 3, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        x = x * disk_mask.permute(0, 2, 3, 1)

        x_flat = x.reshape(B, self.grid_numel, self.out_channels)
        gather_idx = self.node_to_grid.view(1, N, 1).expand(B, N, self.out_channels)
        return x_flat.gather(1, gather_idx)


# ------------------------------------------------------------------------
# Loss functions and physical-time evaluation
# ------------------------------------------------------------------------

def mass_rel_l2_nodes_channels_per_sample(pred: torch.Tensor, true: torch.Tensor, mass_w: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    # Return per-sample, per-channel relative L2: (B, 2).
    w = mass_w.view(1, -1, 1)
    err = pred - true
    num = torch.sum(w * err * err, dim=1)
    den = torch.sum(w * true * true, dim=1)
    return torch.sqrt(torch.clamp(num, min=0.0) / (torch.clamp(den, min=0.0) + float(eps)))


class DiskErrorQuadrature:
    """Physical disk integrals using the training mass and a radial derivative.

    On the standard cell-centred polar grid, extrapolate the derivative of the
    quadratic through the last three radii to the actual boundary r=R. This
    does not impose zero flux on either the prediction or the teacher.
    For non-polar point clouds, field errors remain available; boundary errors
    are NaN because there is no validated boundary stencil for that geometry.
    """

    def __init__(self, points, mass_w, Nr, Ntheta):
        points = np.asarray(points, dtype=np.float64)
        self.mass_w = torch.as_tensor(mass_w, dtype=torch.float64)
        if self.mass_w.numel() != len(points) or not torch.isfinite(self.mass_w).all() or (self.mass_w <= 0).any():
            raise ValueError("Full-time evaluation needs one finite positive mass weight per node")
        self.radius = disk_radius_from_points(points)
        self.boundary_length = 2.0 * math.pi * self.radius
        self.Nr, self.Ntheta = int(Nr), int(Ntheta)
        self.derivative_weights = None
        if self.Nr >= 3 and self.Ntheta >= 3 and len(points) == self.Nr * self.Ntheta:
            grid = points.reshape(self.Nr, self.Ntheta, 2)
            radii = np.linalg.norm(grid, axis=-1)
            r = radii.mean(axis=1)
            angles = np.arctan2(grid[..., 1], grid[..., 0])
            expected_angles = angles[-1, 0] + np.arange(self.Ntheta) * 2 * math.pi / self.Ntheta
            angle_error = np.angle(np.exp(1j * (angles - expected_angles[None, :])))
            valid = (np.all(np.diff(r) > 0) and np.all(r > 0)
                     and np.allclose(radii, r[:, None], rtol=1e-5, atol=1e-7)
                     and np.max(np.abs(angle_error)) < 1e-5)
            if valid:
                offset = r[-3:] - self.radius
                # sum_j w_j p(r_j) = p'(R), exact for degree <= 2.
                matrix = np.stack([np.ones(3), offset, offset ** 2])
                self.derivative_weights = torch.as_tensor(
                    np.linalg.solve(matrix, np.array([0., 1., 0.])), dtype=torch.float64)
        if self.derivative_weights is None:
            print("boundary metrics: NaN (requires an ordered polar grid with Nr>=3 and Ntheta>=3); field metrics use the training mass weights", flush=True)
        else:
            print(f"boundary metrics: quadratic radial derivative at r={self.radius:g}; perimeter={self.boundary_length:.8g}", flush=True)

    def energies(self, pred, target):
        pred, target = pred.double(), target.double()
        w = self.mass_w.to(pred.device).view(1, -1, 1)
        field_error = ((pred - target).square() * w).sum(dim=1)
        reference = (target.square() * w).sum(dim=1)
        if self.derivative_weights is None:
            missing = torch.full_like(field_error, float("nan"))
            return torch.stack([field_error, reference, missing, missing, missing], dim=1)
        dw = self.derivative_weights.to(pred.device).view(1, 3, 1, 1)

        def derivative(values):
            grid = values.reshape(-1, self.Nr, self.Ntheta, 2)
            return (grid[:, -3:] * dw).sum(dim=1)

        dp, dt = derivative(pred), derivative(target)
        ds = self.boundary_length / self.Ntheta
        return torch.stack([field_error, reference, (dp - dt).square().sum(dim=1) * ds,
                            dp.square().sum(dim=1) * ds, dt.square().sum(dim=1) * ds], dim=1)


def _evaluation_trajectory_keys(blob):
    """Use original per-file IDs for reports even when loading remaps sample IDs."""
    n = len(blob["x"])
    source = _tensor_1d_to_numpy(blob.get("_teacher_source_id"))
    sample = _tensor_1d_to_numpy(blob.get("_original_sample_id", blob.get("sample_id")))
    if sample is None or len(sample) != n:
        raise ValueError("Full-time trajectory statistics require sample_id metadata")
    if source is None:
        source = np.zeros(n, dtype=np.int64)
    if len(source) != n:
        raise ValueError("source_id must have one value per transition")
    return list(zip(source.astype(np.int64).tolist(), sample.astype(np.int64).tolist()))


@torch.no_grad()
def evaluate_full_time_test(model, blob, test_idx, basis_np, args):
    """Same per-trajectory right-endpoint statistics and CSV schema as FDM."""
    model.eval()
    keys = _evaluation_trajectory_keys(blob)
    quadrature = DiskErrorQuadrature(basis_np["points"], basis_np["mass_w"], args.Nr, args.Ntheta)
    trajectories = {}
    for start in range(0, len(test_idx), int(args.batch_size)):
        indices = np.asarray(test_idx[start:start + int(args.batch_size)], dtype=np.int64)
        xb = blob["x"][indices].to(DEVICE).float()
        dtb = blob["dt"][indices].to(DEVICE).float()
        target = blob["y"][indices].to(DEVICE).float()
        dt = dtb.double() * float(blob["_effective_dt_scale"])
        if not torch.isfinite(dt).all() or (dt < 0).any():
            raise ValueError("Full-time evaluation requires finite nonnegative physical dt")
        energies = quadrature.energies(model(xb, dtb), target)
        contributions = (energies * dt[:, None, None]).cpu().numpy()
        for index, contribution, duration in zip(indices, contributions, dt.cpu().tolist()):
            item = trajectories.setdefault(keys[int(index)], {
                "energies": np.zeros((5, 2)), "duration": 0., "steps": 0,
                "boundary_length": quadrature.boundary_length,
            })
            item["energies"] += contribution
            item["duration"] += duration
            item["steps"] += 1
    _save_trajectory_error_statistics(trajectories, os.path.join(args.save_dir, args.out_prefix) + "_test", "neumann")
    return trajectories


def save_rollout_errors(predicted, target, times, quadrature, out_prefix):
    """FDM output schemas; trapezoidal time integration on the disk."""
    times = np.asarray(times, dtype=np.float64)
    dt = np.diff(times)
    if not np.isfinite(times).all() or len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Rollout times must be finite and strictly increasing")
    weights = np.zeros_like(times)
    weights[:-1] += dt / 2
    weights[1:] += dt / 2
    totals = np.zeros((5, 2))
    relative = []
    for pred, truth, weight in zip(predicted, target, weights):
        pred_t = torch.as_tensor(pred).unsqueeze(0)
        true_t = torch.as_tensor(truth).unsqueeze(0)
        totals += weight * quadrature.energies(pred_t, true_t)[0].numpy()
        relative.append(mass_rel_l2_nodes_channels_per_sample(pred_t.double(), true_t.double(), quadrature.mass_w)[0].numpy())
    relative = np.asarray(relative)
    _save_full_time_errors(totals, times[-1] - times[0], out_prefix, "neumann",
                           "rollout_trapezoidal", quadrature.boundary_length)
    _plot_relL2_time(times, relative[:, 0], relative[:, 1], out_prefix + "_relL2.png",
                     title="teacher-data rollout mass-weighted relative L2 vs physical time")
    physical_dt = np.concatenate([[0.], dt])
    np.savez(out_prefix + "_errors.npz", times=times, dt_physical=physical_dt,
             rel_u=relative[:, 0], rel_v=relative[:, 1])
    with open(out_prefix + "_errors.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["physical_time", "dt_physical", "rel_l2_u", "rel_l2_v"])
        writer.writerows(zip(times, physical_dt, relative[:, 0], relative[:, 1]))
    print(f"teacher rollout final relative L2 | u={relative[-1, 0]:.8g} v={relative[-1, 1]:.8g}", flush=True)


def reconstruct_cumulative_physical_time(
    blob: dict[str, Any], effective_dt_scale: float
) -> torch.Tensor:
    """Precompute target times by accumulating physical dt within each trajectory."""
    n_transition = int(blob["x"].shape[0])
    sample_ids = _tensor_1d_to_numpy(blob.get("sample_id"))
    time_indices = _tensor_1d_to_numpy(blob.get("time_index"))
    if sample_ids is None or sample_ids.shape[0] != n_transition:
        raise ValueError(
            "Cannot reconstruct cumulative time safely: teacher data needs one sample_id per transition."
        )
    if time_indices is None or time_indices.shape[0] != n_transition:
        raise ValueError(
            "Cannot reconstruct cumulative time safely: teacher data needs one time_index per transition."
        )

    source_ids = _tensor_1d_to_numpy(blob.get("_teacher_source_id"))
    if source_ids is None:
        source_ids = np.zeros((n_transition,), dtype=np.int64)
    if source_ids.shape[0] != n_transition:
        raise ValueError("_teacher_source_id must have one value per transition")

    dt_physical = (
        torch.as_tensor(blob["dt"], dtype=torch.float64).reshape(-1)
        * float(effective_dt_scale)
    ).cpu().numpy()
    if dt_physical.shape[0] != n_transition:
        raise ValueError("dt must have one value per transition")
    if not np.all(np.isfinite(dt_physical)):
        raise ValueError("physical dt contains NaN or Inf")
    if np.any(dt_physical < 0.0):
        bad = int(np.nonzero(dt_physical < 0.0)[0][0])
        raise ValueError(f"physical dt must be >= 0; index={bad} dt={dt_physical[bad]}")

    metadata = blob.get("metadata", {}) if isinstance(blob.get("metadata"), dict) else {}
    allow_zero_dt = bool(blob.get("allow_zero_dt", metadata.get("allow_zero_dt", False)))
    if np.any(dt_physical == 0.0) and not allow_zero_dt:
        bad = int(np.nonzero(dt_physical == 0.0)[0][0])
        raise ValueError(
            "physical dt=0 was found but the teacher metadata does not explicitly set "
            f"allow_zero_dt=True; index={bad}"
        )

    trajectory_groups: dict[tuple[int, Any], list[int]] = {}
    for index in range(n_transition):
        sample_value = sample_ids[index].item() if hasattr(sample_ids[index], "item") else sample_ids[index]
        key = (int(source_ids[index]), sample_value)
        trajectory_groups.setdefault(key, []).append(index)

    cumulative = np.empty((n_transition,), dtype=np.float64)
    final_times: list[float] = []
    first_rows: list[tuple[int, float, float]] = []
    for trajectory_number, (_key, indices) in enumerate(trajectory_groups.items()):
        order_values = np.asarray(time_indices[indices])
        if not np.all(np.isfinite(order_values.astype(np.float64))):
            raise ValueError("time_index contains NaN or Inf")
        if np.unique(order_values).size != len(indices):
            raise ValueError(f"duplicate time_index values within trajectory {_key!r}")
        ordered = [indices[pos] for pos in np.argsort(order_values, kind="stable")]

        t_running = 0.0
        trajectory_times: list[float] = []
        for step, index in enumerate(ordered):
            t_running = t_running + float(dt_physical[index])
            cumulative[index] = t_running
            trajectory_times.append(t_running)
            if trajectory_number == 0 and step < 5:
                first_rows.append((step, float(dt_physical[index]), t_running))

        if trajectory_times:
            if not math.isclose(
                trajectory_times[0], float(dt_physical[ordered[0]]), rel_tol=1.0e-10, abs_tol=1.0e-12
            ):
                raise RuntimeError("first cumulative time does not equal the first physical dt")
            if np.any(np.diff(np.asarray(trajectory_times, dtype=np.float64)) < 0.0):
                raise RuntimeError(f"cumulative time is not monotone within trajectory {_key!r}")
            final_times.append(float(trajectory_times[-1]))

    if not np.all(np.isfinite(cumulative)):
        raise ValueError("reconstructed cumulative time contains NaN or Inf")
    if not final_times:
        raise ValueError("no trajectories are available for cumulative-time reconstruction")

    result = torch.as_tensor(cumulative, dtype=torch.float32).contiguous()
    blob["t_cumulative"] = result
    blob["_time_weight_sanity_rows"] = first_rows
    blob["_trajectory_final_cumulative_times"] = np.asarray(final_times, dtype=np.float64)
    return result


def batch_time_weights(
    t_cumulative: torch.Tensor,
    loss_time_beta: float,
    time_weight_floor: float = 0.0,
) -> torch.Tensor | None:
    beta = float(loss_time_beta)
    if beta <= 0.0:
        return None
    t_eff = torch.clamp(t_cumulative.reshape(-1), min=float(time_weight_floor))
    return torch.pow(t_eff, beta)


def print_time_weighting_summary(blob: dict[str, Any], args: argparse.Namespace) -> None:
    beta = float(args.loss_time_beta)
    print("time loss weighting:", flush=True)
    print(f"  beta = {beta:g}", flush=True)
    print("  train weighting = cumulative physical time t_k reconstructed from dt", flush=True)
    print("  cumulative update = t_running += dt", flush=True)
    print("  first transition time = first dt", flush=True)
    print("  train reduction = mean(t_k^beta * per-sample relative L2)", flush=True)
    print("  test weighting = none", flush=True)
    print("  best model metric = unweighted test relative L2", flush=True)
    print("  rollout weighting = none", flush=True)
    print("time-weight sanity:", flush=True)
    for step, dt_value, cumulative_t in blob.get("_time_weight_sanity_rows", []):
        weight = 1.0 if beta <= 0.0 else max(cumulative_t, float(args.time_weight_floor)) ** beta
        print(
            f"  step={step} dt={dt_value:.8g} cumulative_t={cumulative_t:.8g} weight={weight:.8g}",
            flush=True,
        )
    final_times = np.asarray(blob.get("_trajectory_final_cumulative_times", []), dtype=np.float64)
    print("trajectory final cumulative time:", flush=True)
    print(f"  min={float(final_times.min()):.8g}", flush=True)
    print(f"  max={float(final_times.max()):.8g}", flush=True)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, mass_w: torch.Tensor, amp=False, amp_dtype=torch.float16) -> tuple[float, float]:
    model.eval()
    u_sum, v_sum, n_sum = 0.0, 0.0, 0
    for xb, dtb, _t_cumulative_b, yb in loader:
        xb = xb.to(DEVICE, dtype=torch.float32, non_blocking=True)
        dtb = dtb.to(DEVICE, dtype=torch.float32, non_blocking=True)
        yb = yb.to(DEVICE, dtype=torch.float32, non_blocking=True)
        with _cuda_autocast(amp, amp_dtype):
            yh = model(xb, dtb)
        rel = mass_rel_l2_nodes_channels_per_sample(yh.float(), yb.float(), mass_w)
        bs = int(rel.shape[0])
        u_sum += float(rel[:, 0].sum().item())
        v_sum += float(rel[:, 1].sum().item())
        n_sum += bs
    if n_sum == 0:
        return float("nan"), float("nan")
    return float(u_sum / n_sum), float(v_sum / n_sum)


def checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    basis_np: dict[str, np.ndarray],
    epoch: int,
    train_u: float,
    train_v: float,
    test_u: float,
    test_v: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    history: dict[str, list[Any]] | None = None,
    best_score: float | None = None,
    best_path: str = "",
    score: float | None = None,
    kind: str = "",
) -> dict[str, Any]:
    basis_payload = {
        "points": basis_np["points"].astype(np.float32),
        "mass_w": basis_np["mass_w"].astype(np.float32),
        "Phi": basis_np["Phi"].astype(np.float32),
        "P": basis_np["P"].astype(np.float32),
        "mu": basis_np["mu"].astype(np.float32),
    }
    for key in [
        "kx",
        "ky",
        "Kx",
        "Ky",
        "basis_type",
        "fourier_grid_size",
        "bessel_n",
        "bessel_m",
        "bessel_alpha",
        "bessel_trig",
        "disk_radius",
        "requested_K",
        "K_eff",
    ]:
        if key in basis_np:
            basis_payload[key] = basis_np[key]
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "args": vars(args),
        "basis": basis_payload,
        "epoch": int(epoch),
        "train_u": float(train_u),
        "train_v": float(train_v),
        "test_u": float(test_u),
        "test_v": float(test_v),
        "rng_state": _get_rng_state_for_resume(),
        "time_weight_mode": TIME_WEIGHT_MODE,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict() if scaler.is_enabled() else None
    if history is not None:
        payload["history"] = copy_history_for_checkpoint(history)
    if best_score is not None:
        payload["best_score"] = float(best_score)
    if best_path:
        payload["best_path"] = str(best_path)
    if score is not None:
        payload["score"] = float(score)
    if kind:
        payload["checkpoint_kind"] = str(kind)
    return payload


def summarize_state_dict_compatibility(
    model: nn.Module,
    checkpoint_model_state: Any,
    max_items: int = 8,
) -> tuple[bool, str]:
    if not isinstance(checkpoint_model_state, dict):
        return False, "checkpoint has no model state_dict"

    checkpoint_model_state = _canonical_model_state(checkpoint_model_state)
    current_state = model.state_dict()
    checkpoint_keys = set(checkpoint_model_state.keys())
    current_keys = set(current_state.keys())
    missing = sorted(current_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - current_keys)
    shape_mismatch: list[str] = []

    for key in sorted(current_keys & checkpoint_keys):
        current_value = current_state[key]
        checkpoint_value = checkpoint_model_state[key]
        if (
            hasattr(current_value, "shape")
            and hasattr(checkpoint_value, "shape")
            and tuple(current_value.shape) != tuple(checkpoint_value.shape)
        ):
            shape_mismatch.append(
                f"{key}: checkpoint{tuple(checkpoint_value.shape)} != model{tuple(current_value.shape)}"
            )

    if not missing and not unexpected and not shape_mismatch:
        return True, ""

    def sample(items: list[str]) -> str:
        shown = items[:max_items]
        suffix = "" if len(items) <= max_items else f", ... (+{len(items) - max_items} more)"
        return ", ".join(shown) + suffix

    parts: list[str] = []
    if missing:
        parts.append(f"missing={sample(missing)}")
    if unexpected:
        parts.append(f"unexpected={sample(unexpected)}")
    if shape_mismatch:
        parts.append(f"shape_mismatch={sample(shape_mismatch)}")
    return False, "; ".join(parts)


# ------------------------------------------------------------------------
# Training history and rollout visualization
# ------------------------------------------------------------------------

def save_training_history(out_prefix, hist, loss_time_beta=0.0):
    n_epoch = len(hist["train_u"])
    epoch_axis = np.asarray(hist.get("epoch") or list(range(1, n_epoch + 1)), dtype=np.int64)
    csv_path = out_prefix + "_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_u", "train_v", "test_u", "test_v"],
        )
        writer.writeheader()
        for index in range(n_epoch):
            writer.writerow(
                {
                    "epoch": int(epoch_axis[index]),
                    "train_u": hist["train_u"][index],
                    "train_v": hist["train_v"][index],
                    "test_u": hist["test_u"][index],
                    "test_v": hist["test_v"][index],
                }
            )
    np.savez(
        out_prefix + "_history.npz",
        epoch=epoch_axis,
        train_u=np.asarray(hist["train_u"], dtype=np.float64),
        train_v=np.asarray(hist["train_v"], dtype=np.float64),
        test_u=np.asarray(hist["test_u"], dtype=np.float64),
        test_v=np.asarray(hist["test_v"], dtype=np.float64),
    )

    test_u = np.asarray(hist["test_u"], dtype=float)
    test_v = np.asarray(hist["test_v"], dtype=float)
    train_u_label = "train_u weighted" if float(loss_time_beta) > 0.0 else "train_u"
    train_v_label = "train_v weighted" if float(loss_time_beta) > 0.0 else "train_v"
    plt.figure(figsize=(7, 4))
    plt.plot(epoch_axis, hist["train_u"], label=train_u_label)
    plt.plot(epoch_axis, hist["train_v"], label=train_v_label)
    plt.plot(epoch_axis[np.isfinite(test_u)], test_u[np.isfinite(test_u)], label="test_u")
    plt.plot(epoch_axis[np.isfinite(test_v)], test_v[np.isfinite(test_v)], label="test_v")
    ax = plt.gca()
    ax.set_xscale("log", base=10)
    plt.xlabel("epoch")
    plt.ylabel("relative L2")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    fig_path = out_prefix + "_history.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    if ax.get_xscale() != "log":
        raise RuntimeError("loss plot x-axis must be log scale")


def _tensor_1d_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = _to_numpy(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return np.asarray(arr).reshape(-1)


def _format_time_tag(time_value: float) -> str:
    text = f"{float(time_value):.6f}".rstrip("0").rstrip(".")
    if text == "":
        text = "0"
    text = text.replace("-", "m").replace(".", "p")
    return f"t{text}"


def _select_rollout_sample_id(
    blob: dict[str, Any],
    test_idx: np.ndarray,
    requested: str,
) -> int | None:
    sample_ids = _tensor_1d_to_numpy(blob.get("sample_id"))
    if sample_ids is None or sample_ids.shape[0] != int(blob["x"].shape[0]):
        return None

    token = str(requested).strip().lower()
    test_idx = np.asarray(test_idx, dtype=np.int64).reshape(-1)
    if test_idx.size > 0:
        test_sample_ids = sample_ids[test_idx]
        unique_test = np.unique(test_sample_ids.astype(np.int64))
    else:
        unique_test = np.unique(sample_ids.astype(np.int64))

    if token in {"", "test", "test_first", "first"}:
        return int(unique_test[0])
    if token in {"test_last", "last"}:
        return int(unique_test[-1])
    return int(token)


def _teacher_source_label(blob: dict[str, Any], source_id: int) -> str:
    paths = blob.get("_teacher_paths")
    if isinstance(paths, (list, tuple)) and 0 <= int(source_id) < len(paths):
        stem = os.path.splitext(os.path.basename(str(paths[int(source_id)])))[0]
        return _safe_filename_tag(stem)
    return f"source{int(source_id) + 1}"


def _rollout_source_jobs(blob: dict[str, Any], test_idx: np.ndarray) -> list[tuple[str, np.ndarray]]:
    test_idx = np.asarray(test_idx, dtype=np.int64).reshape(-1)
    source_ids = _tensor_1d_to_numpy(blob.get("_teacher_source_id"))
    paths = blob.get("_teacher_paths")
    if (
        source_ids is None
        or source_ids.shape[0] != int(blob["x"].shape[0])
        or not isinstance(paths, (list, tuple))
        or len(paths) <= 1
    ):
        return [("", test_idx)]

    source_ids = source_ids.astype(np.int64)
    candidate_idx = test_idx if test_idx.size > 0 else np.arange(int(blob["x"].shape[0]), dtype=np.int64)
    jobs: list[tuple[str, np.ndarray]] = []
    for source_id in np.unique(source_ids[candidate_idx]):
        source_idx = candidate_idx[source_ids[candidate_idx] == int(source_id)]
        if source_idx.size > 0:
            jobs.append((_teacher_source_label(blob, int(source_id)), source_idx.astype(np.int64)))
    return jobs if jobs else [("", test_idx)]


def _rollout_sequence_indices(
    blob: dict[str, Any],
    sample_id: int | None,
    fallback_idx: np.ndarray,
) -> np.ndarray:
    n_total = int(blob["x"].shape[0])
    sample_ids = _tensor_1d_to_numpy(blob.get("sample_id"))
    if sample_id is not None and sample_ids is not None and sample_ids.shape[0] == n_total:
        seq = np.nonzero(sample_ids.astype(np.int64) == int(sample_id))[0].astype(np.int64)
    else:
        seq = np.asarray(fallback_idx, dtype=np.int64).reshape(-1)

    if seq.size == 0:
        raise ValueError("no transitions are available for rollout visualization")

    time_index = _tensor_1d_to_numpy(blob.get("time_index"))
    t_cur = _tensor_1d_to_numpy(blob.get("t_cur"))
    if time_index is not None and time_index.shape[0] == n_total:
        order = np.argsort(time_index[seq], kind="stable")
    elif t_cur is not None and t_cur.shape[0] == n_total:
        order = np.argsort(t_cur[seq], kind="stable")
    else:
        order = np.argsort(seq, kind="stable")
    return seq[order]


def _time_for_transition_start(blob: dict[str, Any], index: int, default: float = 0.0) -> float:
    t_cur = _tensor_1d_to_numpy(blob.get("t_cur"))
    if t_cur is not None and t_cur.shape[0] == int(blob["x"].shape[0]):
        return float(t_cur[int(index)])
    time_index = _tensor_1d_to_numpy(blob.get("time_index"))
    times = _tensor_1d_to_numpy(blob.get("times"))
    if time_index is not None and times is not None and time_index.shape[0] == int(blob["x"].shape[0]):
        k = int(time_index[int(index)])
        if 0 <= k < times.shape[0]:
            return float(times[k])
    return float(default)


def _time_for_transition_next(blob: dict[str, Any], index: int, default: float) -> float:
    t_next = _tensor_1d_to_numpy(blob.get("t_next"))
    if t_next is not None and t_next.shape[0] == int(blob["x"].shape[0]):
        return float(t_next[int(index)])
    time_index = _tensor_1d_to_numpy(blob.get("time_index"))
    times = _tensor_1d_to_numpy(blob.get("times"))
    if time_index is not None and times is not None and time_index.shape[0] == int(blob["x"].shape[0]):
        k = int(time_index[int(index)]) + 1
        if 0 <= k < times.shape[0]:
            return float(times[k])
    return float(default)


def _parse_rollout_snapshot_targets(times: np.ndarray, spec: str) -> list[float]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        return []
    t0 = float(times[0])
    t_final = float(times[-1])
    targets: list[float] = []
    for raw in str(spec).split(","):
        token = raw.strip()
        if token == "":
            continue
        lower = token.lower()
        if lower in {"t/2", "half", "mid", "middle"}:
            targets.append(t0 + 0.5 * (t_final - t0))
        elif lower in {"t", "final", "end"}:
            targets.append(t_final)
        else:
            targets.append(float(token))
    if not targets:
        targets = [t0 + 0.5 * (t_final - t0), t_final]
    return targets


def _nearest_time_indices(times: np.ndarray, targets: list[float]) -> list[int]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    out: list[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(times - float(target))))
        if idx not in out:
            out.append(idx)
    return out


def _expand_color_limits(vmin: float, vmax: float) -> tuple[float, float]:
    vmin = float(vmin)
    vmax = float(vmax)
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return 0.0, 1.0
    if abs(vmax - vmin) < 1.0e-12:
        pad = max(abs(vmin), 1.0) * 1.0e-6
        return vmin - pad, vmax + pad
    return vmin, vmax


def _plot_gt_pred_err(
    points: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    title_suffix: str,
    Nr: int,
    Ntheta: int,
    dpi: int,
    value_limits: tuple[tuple[float, float], tuple[float, float]],
    err_limits: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    err = pred - truth

    can_polar_mesh = int(Nr) > 0 and int(Ntheta) > 0 and truth.shape[0] == int(Nr) * int(Ntheta)
    if can_polar_mesh:
        r_edges = np.linspace(0.0, 1.0, int(Nr) + 1)
        theta_edges = np.linspace(0.0, 2.0 * math.pi, int(Ntheta) + 1)
        x_edges = r_edges[:, None] * np.cos(theta_edges[None, :])
        y_edges = r_edges[:, None] * np.sin(theta_edges[None, :])

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0), constrained_layout=True)
    rows = [
        ("u", truth[:, 0], pred[:, 0], err[:, 0], value_limits[0], err_limits[0]),
        ("v", truth[:, 1], pred[:, 1], err[:, 1], value_limits[1], err_limits[1]),
    ]
    for row_idx, (name, gt_values, pred_values, err_values, value_lim, err_absmax) in enumerate(rows):
        panels = [
            ("GT", gt_values, "viridis", float(value_lim[0]), float(value_lim[1])),
            ("pred", pred_values, "viridis", float(value_lim[0]), float(value_lim[1])),
            ("pred-GT", err_values, "coolwarm", -float(err_absmax), float(err_absmax)),
        ]
        for col_idx, (label, values, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if can_polar_mesh:
                field = values.reshape(int(Nr), int(Ntheta))
                img = ax.pcolormesh(x_edges, y_edges, field, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                img = ax.scatter(pts[:, 0], pts[:, 1], c=values, s=7, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{name} {label} {title_suffix}")
            fig.colorbar(img, ax=ax, fraction=0.046)

    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


@torch.no_grad()
def visualize_rollout_teacher_gm2d(
    model: nn.Module,
    blob: dict[str, Any],
    points: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    source_tag: str = "",
    quadrature=None,
) -> None:
    sample_id = _select_rollout_sample_id(blob, test_idx, str(args.rollout_png_sample_id))
    seq = _rollout_sequence_indices(blob, sample_id, test_idx)
    if seq.size == 0:
        return

    x_all = blob["x"]
    dt_all = blob["dt"]
    y_all = blob["y"]
    start_idx = int(seq[0])

    start_time = _time_for_transition_start(blob, start_idx, default=0.0)
    times: list[float] = [float(start_time)]
    truth_states: list[np.ndarray] = [x_all[start_idx, :, :2].detach().cpu().numpy().astype(np.float32)]
    pred_states: list[np.ndarray] = [truth_states[0].copy()]
    current = truth_states[0].copy()

    model.eval()
    for j_raw in seq:
        j = int(j_raw)
        x_step = x_all[j].clone()
        x_step[:, 0] = torch.as_tensor(current[:, 0], dtype=torch.float32)
        x_step[:, 1] = torch.as_tensor(current[:, 1], dtype=torch.float32)
        dt_step = dt_all[j].reshape(1).to(DEVICE, dtype=torch.float32)
        y_pred = model(x_step.unsqueeze(0).to(DEVICE, dtype=torch.float32), dt_step).squeeze(0)
        current = y_pred.detach().cpu().numpy().astype(np.float32)

        truth_states.append(y_all[j].detach().cpu().numpy().astype(np.float32))
        pred_states.append(current.copy())
        default_next = times[-1] + float(dt_all[j].item()) * float(blob["_effective_dt_scale"])
        times.append(_time_for_transition_next(blob, j, default=default_next))

    times_np = np.asarray(times, dtype=np.float64)
    truth_arr = np.stack(truth_states, axis=0)
    pred_arr = np.stack(pred_states, axis=0)

    targets = _parse_rollout_snapshot_targets(times_np, str(args.t_snap))
    snap_indices = _nearest_time_indices(times_np, targets)

    u_min = float(min(truth_arr[:, :, 0].min(), pred_arr[:, :, 0].min()))
    u_max = float(max(truth_arr[:, :, 0].max(), pred_arr[:, :, 0].max()))
    v_min = float(min(truth_arr[:, :, 1].min(), pred_arr[:, :, 1].min()))
    v_max = float(max(truth_arr[:, :, 1].max(), pred_arr[:, :, 1].max()))
    u_min, u_max = _expand_color_limits(u_min, u_max)
    v_min, v_max = _expand_color_limits(v_min, v_max)
    err = pred_arr - truth_arr
    u_err_absmax = float(max(np.max(np.abs(err[:, :, 0])), 1.0e-12))
    v_err_absmax = float(max(np.max(np.abs(err[:, :, 1])), 1.0e-12))

    out_prefix = os.path.join(args.save_dir, args.out_prefix)
    source_id, original_sample_id = _evaluation_trajectory_keys(blob)[start_idx]
    rollout_prefix = f"{out_prefix}_source{source_id}_sample{original_sample_id}_teacher_rollout"
    if quadrature is None:
        quadrature = DiskErrorQuadrature(points, build_polar_mass_weights(points, args.Nr, args.Ntheta), args.Nr, args.Ntheta)
    print(
        "teacher rollout | source_id={} sample_id={} steps={} final_time={} dt_unique={}".format(
            source_id, original_sample_id, len(seq), times_np[-1],
            np.unique(_tensor_1d_to_numpy(dt_all)[seq] * float(blob["_effective_dt_scale"])).tolist(),
        ),
        flush=True,
    )
    save_rollout_errors(pred_arr, truth_arr, times_np, quadrature, rollout_prefix)
    for snap_idx in snap_indices:
        time_value = float(times_np[int(snap_idx)])
        time_tag = _format_time_tag(time_value)
        time_tag = f"t{int(time_value):05d}" if float(time_value).is_integer() else time_tag
        out_path = f"{rollout_prefix}_{time_tag}_GT_Pred_Err.png"
        _plot_gt_pred_err(
            points=points,
            truth=truth_arr[int(snap_idx)],
            pred=pred_arr[int(snap_idx)],
            out_path=out_path,
            title_suffix=f"t={time_value:g}",
            Nr=int(args.Nr),
            Ntheta=int(args.Ntheta),
            dpi=int(args.plot_dpi),
            value_limits=((u_min, u_max), (v_min, v_max)),
            err_limits=(u_err_absmax, v_err_absmax),
        )
        print("saved:", out_path, flush=True)


def visualize_teacher_rollouts(
    model: nn.Module,
    blob: dict[str, Any],
    points: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    basis_np=None,
) -> None:
    mass_w = basis_np["mass_w"] if basis_np is not None else build_polar_mass_weights(points, args.Nr, args.Ntheta)
    quadrature = DiskErrorQuadrature(points, mass_w, args.Nr, args.Ntheta)
    rollout_jobs = _rollout_source_jobs(blob, test_idx)
    for number, (source_tag, source_test_idx) in enumerate(rollout_jobs, 1):
        print(f"rollout {number}/{len(rollout_jobs)} | {source_tag or 'source0'}", flush=True)
        visualize_rollout_teacher_gm2d(model, blob, points, source_test_idx, args, source_tag=source_tag, quadrature=quadrature)


# ------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------

def train_leno(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    mass_w: torch.Tensor,
    basis_np: dict[str, np.ndarray],
    args: argparse.Namespace,
    effective_dt_scale: float,
):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    lr_schedule = getattr(args, "lr_schedule", "step50")
    sched = (torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5) if lr_schedule == "step50"
             else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(args.epochs)) if lr_schedule == "cosine"
             else None)
    amp_enabled = bool(args.amp) and DEVICE.type == "cuda"
    amp_torch_dtype = _resolve_amp_dtype(getattr(args, "amp_dtype", "float16"))
    has_complex_params = _model_has_complex_parameters(model)
    scaler = _make_grad_scaler(amp_enabled and amp_torch_dtype == torch.float16 and not has_complex_params)
    if bool(args.amp) and not amp_enabled:
        print("AMP requested but CUDA is not available; running in float32.", flush=True)
    if amp_enabled and amp_torch_dtype == torch.float16 and has_complex_params:
        print(
            "NOTE: GradScaler disabled because this model has complex parameters; "
            "PyTorch AMP unscale does not support ComplexFloat gradients.",
            flush=True,
        )
    print(
        "runtime options | eval_every={} amp={} amp_dtype={} grad_scaler={} lr_schedule={}".format(
            int(args.eval_every),
            int(amp_enabled),
            str(amp_torch_dtype).replace("torch.", ""),
            int(scaler.is_enabled()),
            lr_schedule,
        ),
        flush=True,
    )

    out_prefix = os.path.join(args.save_dir, args.out_prefix)
    total_epochs = int(args.epochs)
    latest_path = out_prefix + "_latest.pt"
    best_path = out_prefix + "_best.pt"
    resume_path = str(getattr(args, "resume_path", "")).strip() or best_path
    resume_loaded_path = ""
    skipped_resume_path_abs = ""
    best_return_path = ""
    best_score = float("inf")
    start_epoch = 1
    n_train_batches = len(train_loader)
    detailed_step_log = int(args.log_every) > 0
    hist: dict[str, list[Any]] = {"epoch": [], "train_u": [], "train_v": [], "test_u": [], "test_v": [], "lr": []}

    if int(args.resume) != 0 and resume_path and os.path.exists(resume_path):
        ckpt = safe_torch_load(resume_path, map_location="cpu")
        checkpoint_args = ckpt.get("args", {}) if isinstance(ckpt.get("args"), dict) else {}
        checkpoint_beta = checkpoint_args.get("loss_time_beta", None)
        if float(args.loss_time_beta) > 0.0 and (
            ckpt.get("time_weight_mode") != TIME_WEIGHT_MODE
            or checkpoint_beta is None
            or float(checkpoint_beta) <= 0.0
        ):
            raise ValueError(
                "Checkpoint does not use cumulative-time training loss weighting. "
                "Start a fresh run when loss_time_beta > 0."
            )
        checkpoint_model_state = ckpt.get("model", None)
        is_compatible, incompatibility_reason = summarize_state_dict_compatibility(model, checkpoint_model_state)
        if isinstance(model, LENO2d) and "use_layer_scaling" in checkpoint_args:
            if bool(checkpoint_args["use_layer_scaling"]) != bool(model.use_layer_scaling):
                is_compatible = False
                incompatibility_reason = "use_layer_scaling differs from the checkpoint; start a fresh run for the new setting"
        if not is_compatible:
            skipped_resume_path_abs = os.path.abspath(resume_path)
            print(
                f"WARNING: skip incompatible resume checkpoint: {resume_path}\n"
                f"  reason: {incompatibility_reason}\n"
                "  starting from a fresh model; use --resume 0 or choose a new --out_prefix to silence this warning.",
                flush=True,
            )
        else:
            model.load_state_dict(checkpoint_model_state)
            last_epoch = int(ckpt.get("epoch", 0))

            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
                optimizer_to_device(opt, DEVICE)
            else:
                print(f"WARNING: checkpoint has no optimizer state; resumed model weights only: {resume_path}", flush=True)

            scheduler_state = ckpt.get("scheduler", None)
            if sched is not None and isinstance(scheduler_state, dict):
                sched.load_state_dict(scheduler_state)
            else:
                _restore_scheduler_epoch(sched, last_epoch)

            scaler_state = ckpt.get("scaler", None)
            if scaler.is_enabled() and isinstance(scaler_state, dict) and len(scaler_state) > 0:
                scaler.load_state_dict(scaler_state)

            loaded_history = ckpt.get("history", ckpt.get("hist", None))
            if isinstance(loaded_history, dict):
                for key in hist:
                    hist[key] = list(loaded_history.get(key, hist[key]))
                if not hist["epoch"] and hist["train_u"]:
                    hist["epoch"] = list(range(1, len(hist["train_u"]) + 1))

            try:
                best_score = float(ckpt.get("best_score", ckpt.get("best", ckpt.get("score", best_score))))
            except (TypeError, ValueError):
                best_score = float("inf")
            if not math.isfinite(best_score):
                ckpt_test_u = float(ckpt.get("test_u", float("nan")))
                ckpt_test_v = float(ckpt.get("test_v", float("nan")))
                ckpt_score = ckpt_test_u + ckpt_test_v
                if math.isfinite(ckpt_score):
                    best_score = float(ckpt_score)

            start_epoch = last_epoch + 1
            _restore_rng_state_for_resume(ckpt.get("rng_state", None))
            resume_loaded_path = resume_path
            best_return_path = resume_path
            print(
                "resumed checkpoint: {} | next_epoch={}/{} | next_lr={:.2e} | best={}".format(
                    resume_path,
                    start_epoch,
                    total_epochs,
                    float(opt.param_groups[0]["lr"]),
                    "{:.6e}".format(best_score) if math.isfinite(best_score) else "nan",
                ),
                flush=True,
            )
            if start_epoch > total_epochs:
                print(
                    "resume checkpoint epoch {} is already >= --epochs {}; no training epochs left".format(
                        last_epoch, total_epochs
                    ),
                    flush=True,
                )

    for ep in range(start_epoch, total_epochs + 1):
        model.train()
        tr_u_sum, tr_v_sum, tr_n = 0.0, 0.0, 0
        ep_start = time.time()
        if detailed_step_log:
            print(f"[epoch {ep}/{total_epochs}] start | train_batches={n_train_batches}", flush=True)
            log_runtime_status(f"[epoch {ep}/{total_epochs}] start")
        for it, (xb, dtb, t_cumulative_b, yb) in enumerate(train_loader, start=1):
            xb = xb.to(DEVICE, dtype=torch.float32, non_blocking=True)
            dtb = dtb.to(DEVICE, dtype=torch.float32, non_blocking=True)
            t_cumulative_b = t_cumulative_b.to(DEVICE, dtype=torch.float32, non_blocking=True)
            yb = yb.to(DEVICE, dtype=torch.float32, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with _cuda_autocast(amp_enabled, amp_torch_dtype):
                yh = model(xb, dtb)
                loss_each = mass_rel_l2_nodes_channels_per_sample(yh.float(), yb.float(), mass_w)
                loss_u_each = loss_each[:, 0]
                loss_v_each = loss_each[:, 1]
                loss_weights = batch_time_weights(
                    t_cumulative_b,
                    loss_time_beta=float(args.loss_time_beta),
                    time_weight_floor=float(args.time_weight_floor),
                )
                if loss_weights is None:
                    weighted_loss_u_each = loss_u_each
                    weighted_loss_v_each = loss_v_each
                else:
                    weighted_loss_u_each = loss_weights * loss_u_each
                    weighted_loss_v_each = loss_weights * loss_v_each
                loss_u = weighted_loss_u_each.mean()
                loss_v = weighted_loss_v_each.mean()
                loss = loss_u + loss_v

            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0.0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(opt)
            scaler.update()

            tr_u_sum += float(weighted_loss_u_each.detach().sum().item())
            tr_v_sum += float(weighted_loss_v_each.detach().sum().item())
            tr_n += int(weighted_loss_u_each.numel())

            if detailed_step_log and (it == 1 or it % int(args.log_every) == 0 or it == n_train_batches):
                elapsed = time.time() - ep_start
                print(
                    "epoch {:4d} step {:5d}/{:5d} | train u {:.6e} v {:.6e} | elapsed {:.1f}s".format(
                        ep, it, n_train_batches, tr_u_sum / max(tr_n, 1), tr_v_sum / max(tr_n, 1), elapsed
                    ),
                    flush=True,
                )
                log_runtime_status(f"epoch {ep:4d} step {it:5d}/{n_train_batches:5d}")

        train_u = tr_u_sum / max(tr_n, 1)
        train_v = tr_v_sum / max(tr_n, 1)
        if detailed_step_log:
            log_runtime_status(f"[epoch {ep}/{total_epochs}] pre-eval")
        do_eval = ep == total_epochs or (int(args.eval_every) > 0 and ep % int(args.eval_every) == 0)
        if do_eval:
            test_u, test_v = eval_epoch(model, test_loader, mass_w, amp=amp_enabled, amp_dtype=amp_torch_dtype)
        else:
            test_u, test_v = float("nan"), float("nan")

        current_lr = float(opt.param_groups[0]["lr"])
        hist["epoch"].append(int(ep))
        hist["train_u"].append(float(train_u))
        hist["train_v"].append(float(train_v))
        hist["test_u"].append(float(test_u))
        hist["test_v"].append(float(test_v))
        hist["lr"].append(float(current_lr))
        save_training_history(out_prefix, hist, loss_time_beta=float(args.loss_time_beta))

        if sched is not None:
            sched.step()
        next_lr = float(opt.param_groups[0]["lr"])
        print(
            "epoch {:4d} | train u {:.6e} v {:.6e} | test u {} v {} | next_lr {:.2e} | epoch_time {:.1f}s".format(
                ep,
                train_u,
                train_v,
                "{:.6e}".format(test_u) if do_eval else "skipped",
                "{:.6e}".format(test_v) if do_eval else "skipped",
                next_lr,
                time.time() - ep_start,
            ),
            flush=True,
        )

        score = test_u + test_v if math.isfinite(test_u + test_v) else float("nan")
        improved = math.isfinite(score) and score < best_score
        if improved:
            best_score = float(score)

        if detailed_step_log:
            log_runtime_status(f"[epoch {ep}/{total_epochs}] end")

        latest_payload = checkpoint_payload(
            model,
            args,
            basis_np,
            ep,
            train_u,
            train_v,
            test_u,
            test_v,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler,
            history=hist,
            best_score=best_score,
            best_path=best_path,
            score=score,
            kind="latest",
        )
        atomic_torch_save(latest_payload, latest_path)

        if improved:
            best_payload = checkpoint_payload(
                model,
                args,
                basis_np,
                ep,
                train_u,
                train_v,
                test_u,
                test_v,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
                history=hist,
                best_score=best_score,
                best_path=best_path,
                score=score,
                kind="best",
            )
            atomic_torch_save(best_payload, best_path)
            print("  saved best:", best_path, flush=True)
            best_return_path = best_path

    existing_best_path = best_path if os.path.exists(best_path) else ""
    if existing_best_path and skipped_resume_path_abs and os.path.abspath(existing_best_path) == skipped_resume_path_abs:
        existing_best_path = ""
    final_best_path = best_return_path or existing_best_path or resume_loaded_path
    return model, final_best_path



def parse_log_every(value: str) -> int:
    v = str(value).strip().lower()
    if v in {"off", "false", "no", "none"}:
        return 0
    try:
        return int(v)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("log_every must be an integer or 'off'") from exc


# ------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GMS LENO/FEM training on disk teacher transitions.")
    p.add_argument(
        "--teacher_paths", "--teacher_path",
        dest="teacher_paths",
        type=str,
        nargs="+",
        default=["/content/drive/MyDrive/gms_disk_outputs/teacher_transitions_1.pt"],
        help="One or more teacher .pt files; comma-separated paths and glob patterns are accepted.",
    )
    p.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/gms_disk_outputs/gms_leno_fem_runs")
    p.add_argument("--out_prefix", type=str, default="gms_leno_fem")
    p.add_argument("--resume", type=int, default=1, help="1: resume from --resume_path or save_dir/out_prefix_best.pt when it exists.")
    p.add_argument("--resume_path", type=str, default="", help="Optional checkpoint path for resume; default is save_dir/out_prefix_best.pt.")
    p.add_argument(
        "--save_best_epoch_checkpoints",
        type=int,
        default=1,
        help="Compatibility flag; no epoch-numbered checkpoints are written. save_dir/out_prefix_best.pt is overwritten only when the best score improves.",
    )

    p.add_argument("--Nr", type=int, default=128)
    p.add_argument("--Ntheta", type=int, default=256)
    p.add_argument("--dt_scale", type=float, default=50.0, help="T used for dt/T normalization and raw-dt recovery.")
    p.add_argument(
        "--dt_channel_mode",
        type=str,
        default="auto",
        choices=["auto", "raw", "normalized", "keep_raw"],
        help="auto: read dt metadata/times; raw/normalized: legacy input-unit overrides; keep_raw: retain physical dt.",
    )
    p.add_argument("--teacher_dt_units", default="auto", choices=["auto", "physical", "normalized"])
    p.add_argument("--teacher_mmap", type=int, default=1, help="Memory-map teacher tensor storage on CPU.")
    p.add_argument("--teacher_full_finite_check", type=int, default=1, help="Validate all fields and coordinates in chunks.")
    p.add_argument(
        "--input_normalization",
        type=str,
        default="zscore",
        choices=["zscore", "none"],
        help="zscore: standardize [u,v,x,y,dt] with train-split statistics; none: keep their current scales.",
    )
    p.add_argument(
        "--input_norm_eps",
        type=float,
        default=1.0e-6,
        help="Standard-deviation threshold; near-constant input features use scale 1.",
    )

    p.add_argument("--basis_path", type=str, default="", help="Optional FEM eigenbasis bundle .npz/.pt.")
    p.add_argument("--basis_type", type=str, default="disk", choices=["disk", "fourier", "cartesian"])
    p.add_argument(
        "--K",
        type=int,
        default=128,
        help="Minimum number of global Neumann-Bessel disk modes; all modes sharing the cutoff eigenvalue are included.",
    )
    p.add_argument("--Kx", type=int, default=16, help="x-direction modes for cartesian basis or fourier FNO.")
    p.add_argument("--Ky", type=int, default=16, help="y-direction modes for cartesian basis or fourier FNO.")
    p.add_argument(
        "--fourier_grid_size",
        type=int,
        default=0,
        help="Square grid size for --basis_type fourier FNO; <=0 uses 2*Nr.",
    )
    p.add_argument("--basis_reconstruction", type=int, choices=[0, 1], default=1,
                   help="Evaluate teacher projection errors before training (fixed Laplace/node bases).")
    p.add_argument("--reconstruction_tol", type=float, default=0.01,
                   help="Relative spatial L2 tolerance for the minimum basis count (default: 1 percent).")
    p.add_argument("--reconstruction_only", type=int, choices=[0, 1], default=0,
                   help="Evaluate basis reconstruction and exit before model creation/training.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument(
        "--use_layer_scaling",
        type=int,
        default=0,
        help="Apply learnable diagonal C_m(dt)=diag(exp(-dt*(d_c*mu_m+lambda_c))); 0 omits this decay.",
    )

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--amp", type=int, default=0)
    p.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--lr_schedule", "--lr_scheduler", choices=["step50", "cosine", "constant"], default="step50")
    p.add_argument("--log_file", default="", help="Default: timestamped console log; none disables logging.")
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--test_sample_ids",
        type=str,
        default="",
        help="Comma-separated sample IDs or source_id:sample_id; legacy 'last' selects the last per file.",
    )
    p.add_argument("--train_sample_ids", default="", help="Comma-separated sample IDs or source_id:sample_id.")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_train_trajectories", type=int, default=-1)
    p.add_argument("--num_test_trajectories", type=int, default=-1)
    p.add_argument("--test_ratio", type=float, default=None, help="Legacy override: train_fraction = 1 - test_ratio.")
    p.add_argument("--max_train_samples", type=int, default=0, help="Deprecated; truncating trajectories is rejected.")
    p.add_argument("--max_test_samples", type=int, default=0, help="Deprecated; truncating trajectories is rejected.")

    p.add_argument(
        "--dt_channel_idx",
        type=int,
        default=4,
        help="Legacy teacher-x dt channel; ignored when the teacher dict has top-level 'dt'.",
    )
    p.add_argument("--d1_channel_idx", type=int, default=5, help="Legacy teacher-x d1 channel used only for spectral initialization.")
    p.add_argument("--d2_channel_idx", type=int, default=6, help="Legacy teacher-x d2 channel used only for spectral initialization.")
    p.add_argument(
        "--spectral_d1_init",
        type=float,
        default=None,
        help="Initial spectral diffusion for u; default infers it from teacher params/legacy d1, then falls back to 0.0004.",
    )
    p.add_argument(
        "--spectral_d2_init",
        type=float,
        default=None,
        help="Initial spectral diffusion for v; default infers it from teacher params/legacy d2, then falls back to 1.",
    )
    p.add_argument(
        "--loss_time_beta",
        type=float,
        default=0.0,
        help=(
            "If beta > 0, reconstruct cumulative physical time t_k from dt for each "
            "trajectory and use (t_k)^beta for training loss only."
        ),
    )
    p.add_argument(
        "--loss_time_reference",
        "--loss_time_key",
        type=str,
        default="t_next",
        choices=["t_cur", "t_next"],
        help=(
            "Deprecated compatibility option. Cumulative time is always the target time: "
            "the first transition has t_k equal to its first dt."
        ),
    )
    p.add_argument("--time_weight_floor", type=float, default=0.0)

    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--log_every", type=parse_log_every, default=0, help="Step log interval; <=0 or 'off' disables step logs.")
    p.add_argument("--teacher_rollout", "--save_rollout_png", dest="teacher_rollout", type=int, default=1)
    p.add_argument("--rollout_png_sample_id", type=str, default="test_first")
    p.add_argument("--t_snap", "--rollout_png_times", dest="t_snap", type=str, default="T/2,T")
    p.add_argument("--teacher_rollout_use_best", "--rollout_png_use_best", dest="teacher_rollout_use_best", type=int, default=1)
    p.add_argument("--plot_dpi", type=int, default=200)
    p.add_argument("--inspect_only", type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------------
# Experiment entry point
# ------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    maybe_make_dir(args.save_dir)
    setup_console_log(args.log_file, args.save_dir, args.out_prefix)
    log_runtime_status("startup")

    print("argv:", " ".join(sys.argv), flush=True)
    print("device:", DEVICE, flush=True)
    teacher_paths = parse_teacher_paths(args.teacher_paths)
    print("loading teacher:", ", ".join(teacher_paths), flush=True)
    blob = load_teacher_blob(args)
    x = blob["x"]
    dt = blob["dt"]
    y = blob["y"]
    points = blob["points"].detach().cpu().numpy().astype(np.float32)
    effective_dt_scale = float(blob["_effective_dt_scale"])
    t_cumulative = reconstruct_cumulative_physical_time(blob, effective_dt_scale)

    train_idx, test_idx = build_split_indices(blob, args)
    print_teacher_data_summary(blob, train_idx, test_idx)
    print_time_weighting_summary(blob, args)

    input_mean, input_scale, input_feature_std = compute_input_normalization(
        blob,
        train_idx,
        mode=str(args.input_normalization),
        eps=float(args.input_norm_eps),
        return_feature_std=True,
    )
    std_check_items = (
        ("sigma_u", float(input_feature_std[0].item())),
        ("sigma_v", float(input_feature_std[1].item())),
        ("sigma_Delta_t", float(input_feature_std[-1].item())),
    )
    print(
        "input std threshold check (train split): input_norm_eps={:.8g} | {}".format(
            float(args.input_norm_eps),
            " | ".join(
                f"{name}={value:.8g} (below_input_norm_eps={value < float(args.input_norm_eps)})"
                for name, value in std_check_items
            ),
        ),
        flush=True,
    )
    print(
        "input normalization: {} on train split | features=[u,v,x,y,dt] mean={} scale={}".format(
            str(args.input_normalization),
            [float(f"{value:.8g}") for value in input_mean.tolist()],
            [float(f"{value:.8g}") for value in input_scale.tolist()],
        ),
        flush=True,
    )

    if int(args.inspect_only) == 1:
        print("inspect_only=1: stop before basis build and training", flush=True)
        return

    use_fourier_fno = (str(args.basis_type) == "fourier") and not str(args.basis_path).strip()
    if use_fourier_fno:
        fourier_grid_size = int(args.fourier_grid_size) if int(args.fourier_grid_size) > 0 else 2 * int(args.Nr)
        mass_w_np = build_polar_mass_weights(points, int(args.Nr), int(args.Ntheta))
        grid_map_np = build_square_grid_embedding(points, fourier_grid_size)
        basis_np = {
            "points": np.asarray(points, dtype=np.float32),
            "mass_w": mass_w_np.astype(np.float32),
            "Phi": np.zeros((points.shape[0], 0), dtype=np.float32),
            "P": np.zeros((points.shape[0], 0), dtype=np.float32),
            "mu": np.zeros((0,), dtype=np.float32),
            "basis_type": np.asarray(["fourier_square_fno"], dtype="<U32"),
            "fourier_grid_size": np.asarray([fourier_grid_size], dtype=np.int64),
            "Kx": np.asarray([int(args.Kx)], dtype=np.int64),
            "Ky": np.asarray([int(args.Ky)], dtype=np.int64),
        }
        basis = {
            "mass_w": torch.as_tensor(mass_w_np, dtype=torch.float32, device=DEVICE),
        }
        assigned = int(np.unique(grid_map_np["node_to_grid"]).size)
        area = float(np.sum(mass_w_np))
        print(
            "fourier baseline: square-grid FNO grid={}x{} modes=({},{}) assigned_cells={}/{} mass_sum={:.8f} expected_disk_area={:.8f}".format(
                fourier_grid_size,
                fourier_grid_size,
                int(args.Kx),
                int(args.Ky),
                assigned,
                int(points.shape[0]),
                area,
                math.pi,
            ),
            flush=True,
        )
        if int(args.use_layer_scaling) != 0:
            print("fourier baseline ignores --use_layer_scaling; use_layer_scaling is treated as 0 for FNO.", flush=True)
    elif str(args.basis_type) == "cartesian" and not str(args.basis_path).strip():
        print(
            f"building/loading basis: type=cartesian, Kx={int(args.Kx)}, Ky={int(args.Ky)}, K={int(args.Kx) * int(args.Ky)}",
            flush=True,
        )
        basis_np = build_basis(points, args)
        basis = tensor_basis_to_device(basis_np, DEVICE)
        area = float(np.sum(basis_np["mass_w"]))
        print(
            "basis: N={} K={} mass_sum={:.8f} expected_disk_area={:.8f}".format(
                int(basis_np["Phi"].shape[0]), int(basis_np["Phi"].shape[1]), area, math.pi
            ),
            flush=True,
        )
    else:
        print(f"building/loading basis: type={args.basis_type}, K={args.K}", flush=True)
        basis_np = build_basis(points, args)
        basis = tensor_basis_to_device(basis_np, DEVICE)
        area = float(np.sum(basis_np["mass_w"]))
        print(
            "basis: N={} K={} mass_sum={:.8f} expected_disk_area={:.8f}".format(
                int(basis_np["Phi"].shape[0]), int(basis_np["Phi"].shape[1]), area, math.pi
            ),
            flush=True,
        )
        if "bessel_alpha" in basis_np:
            first_mu = float(basis_np["mu"][1]) if int(basis_np["mu"].shape[0]) > 1 else 0.0
            print(
                "disk basis: Neumann-Bessel J_n'(alpha_nm)=0 radius={:.8f} requested_K={} K_eff={} first_nonzero_mu={:.8f}".format(
                    float(basis_np["disk_radius"][0]),
                    int(basis_np["requested_K"][0]),
                    int(basis_np["K_eff"][0]),
                    first_mu,
                ),
                flush=True,
            )

    if int(args.basis_reconstruction) or int(args.reconstruction_only):
        if use_fourier_fno:
            if int(args.reconstruction_only):
                raise ValueError("--reconstruction_only requires a fixed node basis; Fourier FNO uses a square-grid embedding")
            print("basis reconstruction skipped: FFT operator has no mass-orthonormal Laplace basis", flush=True)
        else:
            evaluate_basis_reconstruction(
                blob, {"train": train_idx, "test": test_idx}, basis_np, args,
            )
            if int(args.reconstruction_only):
                print("reconstruction_only=1: stop before model creation and training", flush=True)
                return

    train_ds = TeacherTransitionDataset(blob, train_idx, t_cumulative)
    test_ds = TeacherTransitionDataset(blob, test_idx, t_cumulative)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        **({"persistent_workers": bool(args.persistent_workers), "prefetch_factor": int(args.prefetch_factor)} if int(args.num_workers) > 0 else {}),
    )
    print("train loader: trajectory-safe direct teacher dataset with transition shuffle", flush=True)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        **({"persistent_workers": bool(args.persistent_workers), "prefetch_factor": int(args.prefetch_factor)} if int(args.num_workers) > 0 else {}),
    )

    node_channels = int(x.shape[-1])
    print("Cin", node_channels, "train_n", len(train_ds), "test_n", len(test_ds), flush=True)
    if node_channels != 4:
        raise ValueError(f"canonical node input must have 4 channels [u_k,v_k,x,y], got {node_channels}")
    spectral_d1_init, spectral_d2_init, spectral_init_source = resolve_spectral_diffusion_init(
        blob, args, train_idx
    )
    if use_fourier_fno:
        model = DiskFNO2d(
            node_channels=node_channels,
            out_channels=2,
            width=int(args.width),
            depth=int(args.depth),
            Kx=int(args.Kx),
            Ky=int(args.Ky),
            node_to_grid=torch.as_tensor(grid_map_np["node_to_grid"], device=DEVICE),
            grid_count=torch.as_tensor(grid_map_np["grid_count"], device=DEVICE),
            disk_mask=torch.as_tensor(grid_map_np["disk_mask"], device=DEVICE),
            grid_x=torch.as_tensor(grid_map_np["grid_x"], device=DEVICE),
            grid_y=torch.as_tensor(grid_map_np["grid_y"], device=DEVICE),
            input_mean=input_mean,
            input_scale=input_scale,
        )
    else:
        model = LENO2d(
            node_channels=node_channels,
            out_channels=2,
            width=int(args.width),
            depth=int(args.depth),
            Phi=basis["Phi"],
            P=basis["P"],
            mu=basis["mu"],
            use_layer_scaling=bool(args.use_layer_scaling),
            dt_scale=effective_dt_scale,
            spectral_d1_init=spectral_d1_init,
            spectral_d2_init=spectral_d2_init,
            input_mean=input_mean,
            input_scale=input_scale,
        )
        print(
            "learned spectral diffusion initialization | d1={:.8g} ({}) d2={:.8g} ({})".format(
                spectral_d1_init, spectral_init_source, spectral_d2_init, spectral_init_source
            ),
            flush=True,
        )
    print_model_parameter_summary(model)
    model, best_path = train_leno(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        mass_w=basis["mass_w"],
        basis_np=basis_np,
        args=args,
        effective_dt_scale=effective_dt_scale,
    )
    last_state_for_rollout = None
    if int(args.teacher_rollout) != 0 and int(args.teacher_rollout_use_best) == 0:
        last_state_for_rollout = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_path:
        ckpt = safe_torch_load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        print("loaded best:", best_path, "epoch", ckpt.get("epoch", "unknown"), flush=True)
    else:
        print("best model not found; using current model:", os.path.join(args.save_dir, args.out_prefix) + "_best.pt", flush=True)
    print(f"saved training history: {os.path.join(args.save_dir, args.out_prefix)}_history.csv/.npz/.png", flush=True)
    evaluate_full_time_test(model, blob, test_idx, basis_np, args)
    if int(args.teacher_rollout) != 0:
        if last_state_for_rollout is not None:
            model.load_state_dict(last_state_for_rollout)
        visualize_teacher_rollouts(model, blob, points, test_idx, args, basis_np=basis_np)


# Backward-compatible Python names; new code uses the canonical names above.
GMSFEMNO = LENO2d
GMSDiskFNO = DiskFNO2d
GeneralSpectralConv = SpectralConvNodes
rel_l2_nodes = mass_rel_l2_nodes_channels_per_sample
train_model = train_leno
save_history = save_training_history
save_rollout_pngs = visualize_teacher_rollouts
get_rng_state_for_checkpoint = _get_rng_state_for_resume
restore_rng_state_from_checkpoint = _restore_rng_state_for_resume


if __name__ == "__main__":
    main()
