#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Originally used from a notebook cell via: %%writefile train_leno_nomask_T55_mix.py
# ここに LENO の全コードを貼る（#!/usr/bin/env python から最後まで）

import os, time, math, argparse, glob, random, sys, atexit, csv, tempfile
from bisect import bisect_right
from contextlib import nullcontext
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from scipy.sparse import diags
from scipy.linalg import cholesky, solve_triangular

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_WEIGHT_MODE = "cumulative_time_from_dt_train_only"

try:
    import psutil
except ImportError:
    psutil = None


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


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu", retries=8, base_sleep=0.2, mmap=False):
    last = None
    for i in range(int(retries)):
        try:
            try:
                kwargs = {"map_location": map_location, "weights_only": False}
                if bool(mmap):
                    kwargs["mmap"] = True
                return torch.load(path, **kwargs)
            except TypeError as e:
                if "weights_only" not in str(e) and "mmap" not in str(e):
                    raise
                return torch.load(path, map_location=map_location)
        except OSError as e:
            last = e
            time.sleep(float(base_sleep) * float(1 << i))
    raise last


def _get_rng_state_for_resume():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_rng_state(value):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng_state_for_resume(state):
    if not isinstance(state, dict):
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_rng_state(state["torch"]))
    if torch.cuda.is_available() and "cuda_all" in state:
        cuda_states = state["cuda_all"]
        if torch.is_tensor(cuda_states):
            cuda_states = [cuda_states]
        cuda_states = [_as_cpu_rng_state(s) for s in list(cuda_states)]
        cuda_states = cuda_states[: torch.cuda.device_count()]
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


def _set_loader_epoch(loader, epoch):
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None and hasattr(batch_sampler, "_epoch"):
        batch_sampler._epoch = int(epoch)


def make_times_55(T_final=250.0):
    t = list(range(0, 51)) + [100, 150, 200, 250]
    t = np.array(t, dtype=np.float32)
    t = t[t <= float(T_final)]
    if t.size < 2:
        raise ValueError("T_final too small for T55")
    return t


def validate_teacher_times_10000(times, source_name="teacher data"):
    """Validate the time array stored by the teacher; never synthesize it here."""
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if times.size != 110:
        raise ValueError(f"{source_name} must store 110 teacher times, got {times.size}")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{source_name} teacher_times must be finite and strictly increasing")
    if times[0] != 0.0 or times[-1] != 10000.0:
        raise ValueError(f"{source_name} teacher_times must span exactly 0 to 10000")
    expected_dt = np.concatenate([
        np.full(10, 10.0, dtype=np.float64),
        np.full(99, 100.0, dtype=np.float64),
    ])
    if not np.allclose(np.diff(times), expected_dt, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"{source_name} teacher_times must follow 0,10,...,100,200,...,10000"
        )
    return times


def _teacher_meta_value(blob, key, default=None):
    if not isinstance(blob, dict):
        return default
    if key in blob:
        return blob[key]
    for container_name in ("metadata", "meta"):
        container = blob.get(container_name)
        if isinstance(container, dict) and key in container:
            return container[key]
    return default


_BOUNDARY_CONDITION_ALIASES = {
    "periodic": "periodic",
    "pbc": "periodic",
    "wrap": "periodic",
    "wrapped": "periodic",
    "neumann": "neumann",
    "zero_flux": "neumann",
    "zero-flux": "neumann",
    "no_flux": "neumann",
    "no-flux": "neumann",
}


def _metadata_text(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.reshape(-1)[0].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _normalize_boundary_condition(value, allow_auto=False):
    token = _metadata_text(value)
    if token is None:
        return None
    token = token.lower().replace(" ", "_")
    if allow_auto and token == "auto":
        return "auto"
    normalized = _BOUNDARY_CONDITION_ALIASES.get(token)
    if normalized is None:
        choices = "auto, periodic, or neumann" if allow_auto else "periodic or neumann"
        raise ValueError(f"boundary condition must be {choices}, got {value!r}")
    return normalized


def _boundary_condition_from_metadata(blob):
    """Return (condition, source) while supporting new and legacy teacher files."""
    if not isinstance(blob, dict):
        return None, None
    for key in ("boundary_condition", "boundary_conditions", "boundary", "bc"):
        value = _teacher_meta_value(blob, key, None)
        if value is not None:
            return _normalize_boundary_condition(value), f"metadata:{key}"

    format_text = _metadata_text(_teacher_meta_value(blob, "format", "")) or ""
    equation_text = _metadata_text(_teacher_meta_value(blob, "equation", "")) or ""
    identity = f"{format_text} {equation_text}".lower().replace("-", "_")
    if "gray_scott" in identity or "grayscott" in identity:
        return "periodic", "legacy Gray-Scott format/equation"
    if "gierer" in identity or "meinhardt" in identity or "gm2d" in identity:
        return "neumann", "legacy Gierer-Meinhardt format/equation"
    if _teacher_meta_value(blob, "F", None) is not None and _teacher_meta_value(blob, "k", None) is not None:
        return "periodic", "legacy Gray-Scott F/k metadata"
    if _teacher_meta_value(blob, "s", None) is not None:
        return "neumann", "legacy Gierer-Meinhardt s metadata"
    return None, None


def _dataset_boundary_entries(dataset, seen=None):
    if dataset is None:
        return []
    if seen is None:
        seen = set()
    object_id = id(dataset)
    if object_id in seen:
        return []
    seen.add(object_id)

    condition = getattr(dataset, "boundary_condition", None)
    if condition is not None:
        return [(
            _normalize_boundary_condition(condition),
            str(getattr(dataset, "boundary_source", type(dataset).__name__)),
        )]

    entries = []
    for attr in ("base", "dataset", "ds_full", "ds_low"):
        entries.extend(_dataset_boundary_entries(getattr(dataset, attr, None), seen=seen))
    for item in getattr(dataset, "sources", []):
        child = item.get("dataset") if isinstance(item, dict) else item
        entries.extend(_dataset_boundary_entries(child, seen=seen))
    return entries


def resolve_basis_boundary(requested, *datasets):
    requested = _normalize_boundary_condition(requested, allow_auto=True)
    entries = []
    for dataset in datasets:
        entries.extend(_dataset_boundary_entries(dataset))
    detected = sorted({condition for condition, _ in entries})
    if len(detected) > 1:
        details = ", ".join(f"{condition} ({source})" for condition, source in entries)
        raise ValueError(f"mixed teacher boundary conditions are not supported: {details}")
    inferred = detected[0] if detected else None

    if requested == "auto":
        if inferred is None:
            raise ValueError(
                "--basis_boundary auto could not infer the teacher boundary condition. "
                "Add boundary_condition metadata or pass --basis_boundary periodic/neumann."
            )
        sources = sorted({source for _, source in entries})
        return inferred, "auto from " + ", ".join(sources)
    if inferred is not None and inferred != requested:
        print(
            f"WARNING: --basis_boundary={requested} overrides teacher metadata ({inferred}); "
            "this is a boundary-condition mismatch ablation.",
            flush=True,
        )
    return requested, "explicit command line"


def _teacher_tensor_1d(value, name, n=None, allow_scalar=False, dtype=None):
    if value is None:
        return None
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if allow_scalar and tensor.numel() == 1 and n is not None:
        tensor = tensor.expand(int(n))
    if n is not None and tensor.numel() != int(n):
        raise ValueError(f"{name} must contain {int(n)} values, got {tensor.numel()}")
    return tensor


def _teacher_grid_view(tensor, name, Ny=None, Nx=None, min_channels=1):
    if not torch.is_tensor(tensor):
        raise TypeError(f"teacher {name} must be a torch.Tensor")
    if tensor.ndim == 4:
        out = tensor
        if Ny is not None and int(out.shape[1]) != int(Ny):
            raise ValueError(f"teacher {name} Ny mismatch: {out.shape[1]} vs {Ny}")
        if Nx is not None and int(out.shape[2]) != int(Nx):
            raise ValueError(f"teacher {name} Nx mismatch: {out.shape[2]} vs {Nx}")
    elif tensor.ndim == 3:
        n_points = int(tensor.shape[1])
        if Ny is None or Nx is None:
            side = int(round(math.sqrt(n_points)))
            if side * side != n_points:
                raise ValueError(
                    f"flattened teacher {name} has {n_points} points; provide Ny/Nx metadata"
                )
            Ny = Nx = side
        if int(Ny) * int(Nx) != n_points:
            raise ValueError(
                f"flattened teacher {name} point count {n_points} != Ny*Nx={int(Ny) * int(Nx)}"
            )
        out = tensor.reshape(int(tensor.shape[0]), int(Ny), int(Nx), int(tensor.shape[-1]))
    else:
        raise ValueError(
            f"teacher {name} must have shape [N,Ny,Nx,C] or [N,Ny*Nx,C], got {tuple(tensor.shape)}"
        )
    if int(out.shape[-1]) < int(min_channels):
        raise ValueError(
            f"teacher {name} needs at least {int(min_channels)} channels, got {int(out.shape[-1])}"
        )
    return out


def _teacher_dt_units(blob, cli_units):
    cli_units = str(cli_units).strip().lower()
    if cli_units not in {"auto", "physical", "normalized"}:
        raise ValueError("teacher dt units must be auto, physical, or normalized")
    meta_unit = _teacher_meta_value(blob, "dt_unit", None)
    meta_normalized = _teacher_meta_value(blob, "dt_is_normalized", None)
    if meta_unit is not None:
        token = str(meta_unit).strip().lower()
        if token in {"normalized", "dimensionless", "dt_norm"}:
            metadata_units = "normalized"
        elif token in {"physical", "raw", "seconds"}:
            metadata_units = "physical"
        else:
            raise ValueError(f"unrecognized teacher dt_unit metadata: {meta_unit!r}")
    elif meta_normalized is not None:
        metadata_units = "normalized" if bool(meta_normalized) else "physical"
    else:
        metadata_units = None
    if cli_units != "auto" and metadata_units is not None and cli_units != metadata_units:
        raise ValueError(
            f"--teacher_dt_units={cli_units} contradicts file metadata ({metadata_units})"
        )
    return cli_units if cli_units != "auto" else metadata_units


def _convert_teacher_dt(raw_dt, blob, dt_scale, cli_units, name):
    raw = _teacher_tensor_1d(raw_dt, name, dtype=torch.float64)
    if raw is None or raw.numel() == 0:
        raise ValueError(f"{name} is empty")
    if not torch.isfinite(raw).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if torch.any(raw <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    units = _teacher_dt_units(blob, cli_units)
    unique = np.unique(raw.numpy())
    if units is None:
        physical_match = np.all(
            [np.any(np.isclose(value, [10.0, 100.0], rtol=1.0e-7, atol=1.0e-9)) for value in unique]
        )
        normalized_match = np.all(
            [np.any(np.isclose(value, [0.001, 0.01], rtol=1.0e-7, atol=1.0e-9)) for value in unique]
        )
        if physical_match == normalized_match:
            raise ValueError(
                f"cannot determine whether {name} is physical or normalized; add dt_unit metadata "
                "or pass --teacher_dt_units"
            )
        units = "physical" if physical_match else "normalized"
    if units == "normalized":
        dt_norm = raw
        dt_physical = raw * float(dt_scale)
    else:
        dt_physical = raw
        dt_norm = raw / float(dt_scale)
    return dt_physical.to(torch.float32).contiguous(), dt_norm.to(torch.float32).contiguous(), units


def _teacher_parameter_series(blob, x, key, channel_idx, n):
    value = _teacher_meta_value(blob, key, None)
    from_metadata = value is not None
    if from_metadata:
        values = _teacher_tensor_1d(
            value, f"teacher {key}", n=n, allow_scalar=True, dtype=torch.float32
        ).contiguous()
    elif int(x.shape[-1]) > int(channel_idx):
        values = x[:, 0, 0, int(channel_idx)].detach().cpu().float().contiguous()
        for index in sorted(set([0, max(0, n // 2), max(0, n - 1)])):
            field = x[index, :, :, int(channel_idx)].detach().cpu().float()
            if not torch.allclose(field, values[index].expand_as(field), rtol=1.0e-5, atol=1.0e-7):
                raise ValueError(f"legacy teacher {key} channel is not spatially constant at transition {index}")
    else:
        return None
    if not torch.isfinite(values).all():
        raise ValueError(f"teacher {key} contains NaN or Inf")
    if from_metadata and int(x.shape[-1]) > int(channel_idx):
        channel_values = x[:, 0, 0, int(channel_idx)].detach().cpu().float()
        if not torch.allclose(values, channel_values, rtol=1.0e-5, atol=1.0e-7):
            raise ValueError(f"teacher {key} metadata contradicts legacy input channel {channel_idx}")
    return values


class TeacherPTSource:
    """One memory-mapped monolithic teacher file, without per-transition grid copies."""

    def __init__(
        self,
        path,
        source_id,
        dt_scale=10000.0,
        Lx=2.5,
        Ly=2.5,
        dt_units="auto",
        mmap=True,
        strict_standard=True,
        full_finite_check=True,
    ):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.source_id = int(source_id)
        self.dt_scale = float(dt_scale)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"teacher file not found: {self.path}")
        if self.dt_scale <= 0.0 or self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError("dt_scale, Lx, and Ly must be positive")
        try:
            blob = safe_torch_load(self.path, map_location="cpu", mmap=bool(mmap))
        except RuntimeError as exc:
            if not bool(mmap) or "mmap" not in str(exc).lower():
                raise
            print(
                f"WARNING: memory mapping is unavailable for {self.path}; loading it normally on CPU",
                flush=True,
            )
            blob = safe_torch_load(self.path, map_location="cpu", mmap=False)
        if not isinstance(blob, dict):
            raise TypeError(f"teacher file must contain a dictionary: {self.path}")
        self.blob = blob
        self.boundary_condition, boundary_source = _boundary_condition_from_metadata(blob)
        self.boundary_source = f"{self.path} ({boundary_source})" if boundary_source else self.path

        Ny_meta = _teacher_meta_value(blob, "Ny", None)
        Nx_meta = _teacher_meta_value(blob, "Nx", None)
        Ny_meta = None if Ny_meta is None else int(Ny_meta)
        Nx_meta = None if Nx_meta is None else int(Nx_meta)
        self.x = _teacher_grid_view(blob.get("x"), "x", Ny_meta, Nx_meta, min_channels=4)
        self.Ny, self.Nx = int(self.x.shape[1]), int(self.x.shape[2])
        self.y = _teacher_grid_view(blob.get("y"), "y", self.Ny, self.Nx, min_channels=2)
        self.n = int(self.x.shape[0])
        if int(self.y.shape[0]) != self.n:
            raise ValueError(f"teacher x/y transition count mismatch: {self.n} vs {self.y.shape[0]}")

        times_raw = _teacher_meta_value(
            blob, "teacher_times", _teacher_meta_value(blob, "times", None)
        )
        if times_raw is None:
            raise ValueError(
                f"direct teacher file must store teacher_times (or legacy times): {self.path}"
            )
        self.times = np.asarray(
            times_raw.detach().cpu().numpy() if torch.is_tensor(times_raw) else times_raw,
            dtype=np.float64,
        ).reshape(-1)
        if self.times.size < 2 or not np.all(np.isfinite(self.times)) or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("teacher times must be finite, strictly increasing, and contain >=2 frames")
        if bool(strict_standard):
            self.times = validate_teacher_times_10000(self.times, source_name=self.path)
            if self.Ny != 256 or self.Nx != 256:
                raise ValueError(f"strict teacher grid must be 256x256, got {self.Ny}x{self.Nx}")

        self.sample_id = _teacher_tensor_1d(
            _teacher_meta_value(blob, "sample_id", None), "teacher sample_id", n=self.n, dtype=torch.long
        )
        self.time_index = _teacher_tensor_1d(
            _teacher_meta_value(blob, "time_index", None), "teacher time_index", n=self.n, dtype=torch.long
        )
        n_step = int(self.times.size - 1)
        if self.sample_id is None or self.time_index is None:
            if self.n % n_step != 0:
                raise ValueError(
                    "teacher sample_id/time_index are required when transition count is not trajectory-major"
                )
            if self.sample_id is None:
                self.sample_id = torch.arange(self.n, dtype=torch.long) // n_step
                print(f"WARNING: inferred trajectory-major sample_id for {self.path}", flush=True)
            if self.time_index is None:
                self.time_index = torch.arange(self.n, dtype=torch.long) % n_step
                print(f"WARNING: inferred trajectory-major time_index for {self.path}", flush=True)
        if torch.any(self.time_index < 0) or torch.any(self.time_index >= n_step):
            raise ValueError(f"teacher time_index must lie in [0,{n_step - 1}]")

        dt_physical_stored = _teacher_meta_value(blob, "dt_physical", None)
        dt_normalized_stored = _teacher_meta_value(blob, "dt_normalized", None)
        dt_top = _teacher_meta_value(blob, "dt", None)
        dt_channel = None
        if int(self.x.shape[-1]) >= 5:
            dt_channel = self.x[:, 0, 0, 4].detach().cpu()
            for index in sorted(set([0, max(0, self.n // 2), max(0, self.n - 1)])):
                field = self.x[index, :, :, 4].detach().cpu().float()
                if not torch.allclose(
                    field, dt_channel[index].float().expand_as(field), rtol=1.0e-5, atol=1.0e-7
                ):
                    raise ValueError(f"teacher dt input channel is not spatially constant at transition {index}")
        if dt_physical_stored is not None:
            self.dt_physical = _teacher_tensor_1d(
                dt_physical_stored, "teacher dt_physical", n=self.n, dtype=torch.float32
            ).contiguous()
            self.dt_norm = self.dt_physical / self.dt_scale
            self.dt_units = "explicit physical metadata"
        elif dt_normalized_stored is not None:
            self.dt_norm = _teacher_tensor_1d(
                dt_normalized_stored, "teacher dt_normalized", n=self.n, dtype=torch.float32
            ).contiguous()
            self.dt_physical = self.dt_norm * self.dt_scale
            self.dt_units = "explicit normalized metadata"
        else:
            if dt_top is None and dt_channel is None:
                raise ValueError(
                    "teacher file needs dt_physical, dt_normalized, top-level dt, "
                    "or a spatially constant x[...,4] dt channel"
                )
            dt_raw = dt_top if dt_top is not None else dt_channel
            self.dt_physical, self.dt_norm, self.dt_units = _convert_teacher_dt(
                dt_raw, blob, self.dt_scale, dt_units, "teacher dt"
            )
        if self.dt_physical.numel() != self.n:
            raise ValueError(f"teacher dt must contain {self.n} values, got {self.dt_physical.numel()}")
        if dt_normalized_stored is not None:
            stored_norm = _teacher_tensor_1d(
                dt_normalized_stored, "teacher dt_normalized", n=self.n, dtype=torch.float32
            )
            if not torch.allclose(self.dt_norm, stored_norm, rtol=1.0e-6, atol=1.0e-8):
                raise ValueError("teacher dt_physical and dt_normalized metadata contradict each other")
        if dt_top is not None:
            stored_dt = _teacher_tensor_1d(dt_top, "teacher dt", n=self.n, dtype=torch.float32)
            if not torch.allclose(self.dt_norm, stored_dt, rtol=1.0e-6, atol=1.0e-8):
                raise ValueError(
                    "top-level teacher dt must equal the stored normalized dt for this dataset"
                )
        stored_T = _teacher_meta_value(blob, "T", None)
        if stored_T is not None and not math.isclose(
            float(stored_T), self.dt_scale, rel_tol=1.0e-9, abs_tol=1.0e-9
        ):
            raise ValueError(
                f"teacher T={float(stored_T):g} contradicts --dt_scale={self.dt_scale:g}"
            )

        expected_dt = torch.as_tensor(np.diff(self.times), dtype=torch.float32)[self.time_index]
        if not torch.allclose(self.dt_physical, expected_dt, rtol=1.0e-6, atol=1.0e-5):
            bad = int(torch.nonzero(~torch.isclose(
                self.dt_physical, expected_dt, rtol=1.0e-6, atol=1.0e-5
            ), as_tuple=False)[0, 0])
            raise ValueError(
                f"teacher dt/time_index/times contradiction at transition {bad}: "
                f"dt={self.dt_physical[bad].item():g}, expected={expected_dt[bad].item():g}"
            )
        expected_norm = self.dt_physical / self.dt_scale
        if not torch.allclose(self.dt_norm, expected_norm, rtol=1.0e-7, atol=1.0e-9):
            raise ValueError("normalized teacher dt is inconsistent with dt_physical/dt_scale")
        if bool(strict_standard):
            if not np.allclose(np.unique(self.dt_physical.numpy()), [10.0, 100.0]):
                raise ValueError("strict teacher physical dt values must be exactly [10,100]")
            if not np.allclose(np.unique(self.dt_norm.numpy()), [0.001, 0.01]):
                raise ValueError("strict teacher normalized dt values must be exactly [0.001,0.01]")

        self.t_start = torch.as_tensor(self.times, dtype=torch.float32)[self.time_index]
        self.t_end = torch.as_tensor(self.times, dtype=torch.float32)[self.time_index + 1]
        for stored_name, expected_value in (("t_cur", self.t_start), ("t_next", self.t_end)):
            stored_value = _teacher_meta_value(blob, stored_name, None)
            if stored_value is not None:
                stored_tensor = _teacher_tensor_1d(
                    stored_value, f"teacher {stored_name}", n=self.n, dtype=torch.float32
                )
                if not torch.allclose(stored_tensor, expected_value, rtol=0.0, atol=1.0e-5):
                    raise ValueError(f"teacher {stored_name} contradicts teacher_times/time_index")
        self.diffusion_u = _teacher_meta_value(blob, "Du", _teacher_meta_value(blob, "d1", None))
        self.diffusion_v = _teacher_meta_value(blob, "Dv", _teacher_meta_value(blob, "d2", None))
        if self.diffusion_u is None and int(self.x.shape[-1]) > 5:
            legacy_d1 = _teacher_parameter_series(blob, self.x, "d1", 5, self.n)
            if not torch.allclose(legacy_d1, legacy_d1[0].expand_as(legacy_d1), rtol=1.0e-6, atol=1.0e-12):
                raise ValueError("legacy d1 varies by transition and cannot initialize one learned spectral d1")
            self.diffusion_u = float(legacy_d1[0])
        if self.diffusion_v is None and int(self.x.shape[-1]) > 6:
            legacy_d2 = _teacher_parameter_series(blob, self.x, "d2", 6, self.n)
            if not torch.allclose(legacy_d2, legacy_d2[0].expand_as(legacy_d2), rtol=1.0e-6, atol=1.0e-12):
                raise ValueError("legacy d2 varies by transition and cannot initialize one learned spectral d2")
            self.diffusion_v = float(legacy_d2[0])
        self.diffusion_u = None if self.diffusion_u is None else float(np.asarray(self.diffusion_u).reshape(-1)[0])
        self.diffusion_v = None if self.diffusion_v is None else float(np.asarray(self.diffusion_v).reshape(-1)[0])
        if self.diffusion_u is not None and (not math.isfinite(self.diffusion_u) or self.diffusion_u <= 0.0):
            raise ValueError("teacher Du/d1 diffusion initialization must be finite and positive")
        if self.diffusion_v is not None and (not math.isfinite(self.diffusion_v) or self.diffusion_v <= 0.0):
            raise ValueError("teacher Dv/d2 diffusion initialization must be finite and positive")
        self.s = _teacher_parameter_series(blob, self.x, "s", 7, self.n)
        self.condition_on_s = bool(
            _teacher_meta_value(blob, "condition_on_s", _teacher_meta_value(blob, "network_condition_s", False))
        )
        if self.condition_on_s and self.s is None:
            raise ValueError("teacher metadata requests s conditioning, but s is absent")
        self.Cin = 6 if self.condition_on_s else 5

        coord = self.x[0, :, :, 2:4].detach().cpu().float()
        coord_normalized_meta = _teacher_meta_value(
            blob, "coordinates_normalized", _teacher_meta_value(blob, "coords_normalized", None)
        )
        xmin, xmax = float(coord[..., 0].min()), float(coord[..., 0].max())
        ymin, ymax = float(coord[..., 1].min()), float(coord[..., 1].max())
        x_period = (xmax - xmin) * self.Nx / max(1, self.Nx - 1)
        y_period = (ymax - ymin) * self.Ny / max(1, self.Ny - 1)
        looks_physical = (
            math.isclose(xmin, 0.0, abs_tol=1.0e-5)
            and math.isclose(ymin, 0.0, abs_tol=1.0e-5)
            and (
                math.isclose(xmax, self.Lx, rel_tol=1.0e-4, abs_tol=1.0e-5)
                or math.isclose(x_period, self.Lx, rel_tol=1.0e-4, abs_tol=1.0e-5)
            )
            and (
                math.isclose(ymax, self.Ly, rel_tol=1.0e-4, abs_tol=1.0e-5)
                or math.isclose(y_period, self.Ly, rel_tol=1.0e-4, abs_tol=1.0e-5)
            )
        )
        looks_normalized = (
            math.isclose(xmin, 0.0, abs_tol=1.0e-5)
            and math.isclose(ymin, 0.0, abs_tol=1.0e-5)
            and math.isclose(xmax, 1.0, rel_tol=1.0e-4, abs_tol=1.0e-5)
            and math.isclose(ymax, 1.0, rel_tol=1.0e-4, abs_tol=1.0e-5)
        )
        if coord_normalized_meta is True:
            if not looks_normalized:
                raise ValueError("coordinates_normalized=True but coordinate span is not [0,1]^2")
            coord_norm = coord
        elif coord_normalized_meta is False:
            if not looks_physical:
                raise ValueError("coordinates_normalized=False but coordinates do not span the physical domain")
            coord_norm = torch.stack([coord[..., 0] / self.Lx, coord[..., 1] / self.Ly], dim=-1)
        elif looks_physical:
            coord_norm = torch.stack([coord[..., 0] / self.Lx, coord[..., 1] / self.Ly], dim=-1)
        elif looks_normalized:
            coord_norm = coord
            print(
                f"WARNING: inferred normalized coordinates for legacy-compatible teacher file {self.path}",
                flush=True,
            )
        else:
            raise ValueError(
                f"teacher coordinates span x=[{xmin:g},{xmax:g}], y=[{ymin:g},{ymax:g}]; "
                f"expected physical [0,{self.Lx:g}]x[0,{self.Ly:g}] or explicit normalized [0,1]^2"
            )
        if looks_physical:
            coord_physical = coord
        else:
            coord_physical = torch.stack(
                [coord_norm[..., 0] * self.Lx, coord_norm[..., 1] * self.Ly], dim=-1
            )
        self.plot_x_range = (
            float(coord_physical[..., 0].min()), float(coord_physical[..., 0].max())
        )
        self.plot_y_range = (
            float(coord_physical[..., 1].min()), float(coord_physical[..., 1].max())
        )
        x_axis = coord_norm[0, :, 0]
        y_axis = coord_norm[:, 0, 1]
        expected_x = x_axis[None, :].expand(self.Ny, self.Nx)
        expected_y = y_axis[:, None].expand(self.Ny, self.Nx)
        if not torch.allclose(coord_norm[..., 0], expected_x, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("teacher flattened/grid point ordering is not x-fast rectilinear ordering")
        if not torch.allclose(coord_norm[..., 1], expected_y, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("teacher flattened/grid point ordering is not y-slow rectilinear ordering")
        if torch.any(torch.diff(x_axis) <= 0.0) or torch.any(torch.diff(y_axis) <= 0.0):
            raise ValueError("teacher coordinate axes must be strictly increasing")
        self.coord_grid = coord_norm.to(torch.float32).contiguous()

        self.trajectory_indices = OrderedDict()
        for index, sample_value in enumerate(self.sample_id.tolist()):
            self.trajectory_indices.setdefault(int(sample_value), []).append(index)
        for sample_value, indices in self.trajectory_indices.items():
            order = self.time_index[indices]
            if len(indices) != n_step or not torch.equal(torch.sort(order).values, torch.arange(n_step)):
                raise ValueError(
                    f"teacher trajectory {sample_value} must contain each of {n_step} transitions exactly once"
                )
            if not torch.equal(order, torch.arange(n_step)):
                raise ValueError(
                    f"teacher trajectory {sample_value} is not monotone in file order; "
                    "store its transitions in increasing time_index order"
                )
        if bool(strict_standard):
            # A source may be one numbered trajectory shard or a monolithic file.
            # The 100-trajectory/10900-transition requirement is checked over the
            # complete --teacher_paths collection after every source is loaded.
            if self.n != len(self.trajectory_indices) * 109:
                raise ValueError(
                    f"strict teacher source {self.path} has {self.n} transitions for "
                    f"{len(self.trajectory_indices)} trajectories; expected exactly 109 per trajectory"
                )

        if bool(full_finite_check):
            self._validate_full_fields_and_coordinates(coord)

    def _validate_full_fields_and_coordinates(self, raw_coord, chunk_size=8):
        for start in range(0, self.n, int(chunk_size)):
            stop = min(self.n, start + int(chunk_size))
            x_chunk = self.x[start:stop]
            y_chunk = self.y[start:stop]
            if not torch.isfinite(x_chunk[..., :2]).all():
                raise ValueError(f"teacher x u/v contains NaN or Inf in transitions [{start},{stop})")
            if not torch.isfinite(y_chunk[..., :2]).all():
                raise ValueError(f"teacher y u/v contains NaN or Inf in transitions [{start},{stop})")
            coords = x_chunk[..., 2:4].float()
            if not torch.allclose(
                coords, raw_coord.unsqueeze(0).expand_as(coords), rtol=1.0e-5, atol=1.0e-6
            ):
                raise ValueError(f"teacher coordinates change across transitions [{start},{stop})")

    def input_at(self, index, u_override=None, v_override=None):
        index = int(index)
        if u_override is None or v_override is None:
            uv = self.x[index, :, :, :2].detach().cpu().float()
        else:
            uv = torch.stack([
                torch.as_tensor(u_override, dtype=torch.float32),
                torch.as_tensor(v_override, dtype=torch.float32),
            ], dim=-1)
        dt_channel = self.dt_norm[index].expand(self.Ny, self.Nx, 1)
        features = [uv, self.coord_grid, dt_channel]
        if self.condition_on_s:
            features.append(self.s[index].expand(self.Ny, self.Nx, 1))
        return torch.cat(features, dim=-1).contiguous()

    def target_at(self, index):
        return self.y[int(index), :, :, :2].detach().cpu().float()


class TeacherTransitionDataset(Dataset):
    def __init__(self, sources):
        super().__init__()
        if not sources:
            raise ValueError("at least one teacher source is required")
        self.sources = list(sources)
        self.Ny, self.Nx, self.Cin = self.sources[0].Ny, self.sources[0].Nx, self.sources[0].Cin
        source_boundaries = {
            source.boundary_condition for source in self.sources
            if source.boundary_condition is not None
        }
        if len(source_boundaries) > 1:
            raise ValueError(f"teacher files mix boundary conditions: {sorted(source_boundaries)}")
        self.boundary_condition = next(iter(source_boundaries), None)
        self.boundary_source = self.sources[0].boundary_source
        if len(self.sources) > 1:
            self.boundary_source += f"; plus {len(self.sources) - 1} teacher files"
        for source in self.sources[1:]:
            if (source.Ny, source.Nx, source.Cin) != (self.Ny, self.Nx, self.Cin):
                raise ValueError(
                    "teacher files have incompatible grid/input shapes: "
                    f"expected {(self.Ny, self.Nx, self.Cin)}, got {(source.Ny, source.Nx, source.Cin)}"
                )
            if not torch.allclose(source.coord_grid, self.sources[0].coord_grid, rtol=1.0e-5, atol=1.0e-6):
                raise ValueError("teacher files use incompatible spatial grids")
        self.records = [
            (source.source_id, local_index)
            for source in self.sources
            for local_index in range(source.n)
        ]
        self.trajectory_keys = [
            (source.source_id, int(sample_id))
            for source in self.sources
            for sample_id in source.trajectory_indices.keys()
        ]
        self._source_by_id = {source.source_id: source for source in self.sources}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        source_id, local_index = self.records[int(index)]
        source = self._source_by_id[int(source_id)]
        metadata = {
            "t_cumulative": source.t_end[local_index],
            "t_start": source.t_start[local_index],
            "t_end": source.t_end[local_index],
            "dt_physical": source.dt_physical[local_index],
            "dt_norm": source.dt_norm[local_index],
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "sample_id": source.sample_id[local_index],
            "time_index": source.time_index[local_index],
        }
        return source.input_at(local_index), source.target_at(local_index), metadata

    def trajectory_key_for_record(self, index):
        source_id, local_index = self.records[int(index)]
        source = self._source_by_id[int(source_id)]
        return int(source_id), int(source.sample_id[local_index])

    def trajectory(self, key):
        source_id, sample_id = int(key[0]), int(key[1])
        source = self._source_by_id[source_id]
        indices = list(source.trajectory_indices[sample_id])
        indices.sort(key=lambda index: int(source.time_index[index]))
        return source, indices


class TeacherTransitionSubset(Dataset):
    def __init__(self, base, indices, trajectory_keys):
        super().__init__()
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.trajectory_keys = list(trajectory_keys)
        self.Ny, self.Nx, self.Cin = base.Ny, base.Nx, base.Cin
        self.boundary_condition = getattr(base, "boundary_condition", None)
        self.boundary_source = getattr(base, "boundary_source", type(base).__name__)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        return self.base[int(self.indices[int(index)])]


def _teacher_id_tokens(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        pieces = []
        for item in value:
            pieces.extend(str(item).split(","))
    else:
        pieces = str(value).split(",")
    return [piece.strip() for piece in pieces if piece.strip()]


def _select_teacher_keys(all_keys, selection, option_name):
    tokens = _teacher_id_tokens(selection)
    if not tokens:
        return None
    selected = set()
    for token in tokens:
        if ":" in token:
            source_text, sample_text = token.split(":", 1)
            key = (int(source_text), int(sample_text))
            if key not in all_keys:
                raise ValueError(f"{option_name} references unknown source_id:sample_id {token}")
            selected.add(key)
        else:
            sample_id = int(token)
            matches = [key for key in all_keys if int(key[1]) == sample_id]
            if not matches:
                raise ValueError(f"{option_name} references unknown sample_id {sample_id}")
            selected.update(matches)
    return selected


def _limit_teacher_trajectory_keys(
    keys_ordered, requested_count, split_seed, option_name, seed_offset,
):
    requested_count = int(requested_count)
    if requested_count == -1:
        return set(keys_ordered)
    if requested_count <= 0:
        raise ValueError(f"{option_name} must be -1 or a positive integer")
    if requested_count > len(keys_ordered):
        raise ValueError(
            f"{option_name}={requested_count} exceeds the {len(keys_ordered)} trajectories "
            "available after the train/test split"
        )
    rng = np.random.default_rng(int(split_seed) + int(seed_offset))
    permutation = rng.permutation(len(keys_ordered))
    return {keys_ordered[int(i)] for i in permutation[:requested_count]}


def split_teacher_trajectories(
    dataset,
    train_fraction=0.8,
    split_seed=0,
    train_sample_ids=None,
    test_sample_ids=None,
    num_train_trajectories=-1,
    num_test_trajectories=-1,
):
    all_keys_ordered = list(dataset.trajectory_keys)
    all_keys = set(all_keys_ordered)
    explicit_train = _select_teacher_keys(all_keys, train_sample_ids, "--train_sample_ids")
    explicit_test = _select_teacher_keys(all_keys, test_sample_ids, "--test_sample_ids")
    if explicit_train is not None or explicit_test is not None:
        train_keys = (all_keys - explicit_test) if explicit_train is None else explicit_train
        test_keys = (all_keys - explicit_train) if explicit_test is None else explicit_test
        overlap = train_keys & test_keys
        if overlap:
            raise ValueError(f"explicit teacher split overlaps on trajectories: {sorted(overlap)}")
        unassigned = all_keys - train_keys - test_keys
        if unassigned:
            raise ValueError(f"explicit teacher split leaves trajectories unassigned: {sorted(unassigned)}")
    else:
        fraction = float(train_fraction)
        if not (0.0 < fraction < 1.0):
            raise ValueError("--train_fraction must be strictly between 0 and 1")
        rng = np.random.default_rng(int(split_seed))
        permutation = rng.permutation(len(all_keys_ordered))
        n_train = int(round(fraction * len(all_keys_ordered)))
        n_train = max(1, min(len(all_keys_ordered) - 1, n_train))
        train_keys = {all_keys_ordered[int(i)] for i in permutation[:n_train]}
        test_keys = all_keys - train_keys
    train_candidates_ordered = [key for key in all_keys_ordered if key in train_keys]
    test_candidates_ordered = [key for key in all_keys_ordered if key in test_keys]
    train_keys = _limit_teacher_trajectory_keys(
        train_candidates_ordered,
        num_train_trajectories,
        split_seed,
        "--num_train_trajectories",
        seed_offset=104729,
    )
    test_keys = _limit_teacher_trajectory_keys(
        test_candidates_ordered,
        num_test_trajectories,
        split_seed,
        "--num_test_trajectories",
        seed_offset=130363,
    )
    if train_keys & test_keys:
        raise RuntimeError("trajectory-level teacher split leaked between train and test")
    train_indices, test_indices = [], []
    for index in range(len(dataset)):
        key = dataset.trajectory_key_for_record(index)
        if key in train_keys:
            train_indices.append(index)
        elif key in test_keys:
            test_indices.append(index)
        else:
            # Count limits intentionally leave some split trajectories unused.
            if key not in all_keys:
                raise RuntimeError(f"teacher transition has unknown trajectory key {key}")
    train_keys_ordered = [key for key in all_keys_ordered if key in train_keys]
    test_keys_ordered = [key for key in all_keys_ordered if key in test_keys]
    return (
        TeacherTransitionSubset(dataset, train_indices, train_keys_ordered),
        TeacherTransitionSubset(dataset, test_indices, test_keys_ordered),
    )


# -------------------------
# sharded PT dataset cache
# -------------------------
def _is_under_dir(path, directory):
    try:
        path_abs = os.path.abspath(path)
        dir_abs = os.path.abspath(directory)
        return os.path.commonpath([path_abs, dir_abs]) == dir_abs
    except (OSError, ValueError):
        return False


def _normalize_shard_path(raw_path, root_dir):
    p = str(raw_path)
    base_name = os.path.basename(os.path.normpath(p.replace("\\", os.sep)))
    local_p = os.path.join(root_dir, base_name) if base_name else ""
    if os.path.isabs(p):
        if local_p and os.path.exists(local_p) and (
            not _is_under_dir(p, root_dir) or not os.path.exists(p)
        ):
            return local_p, (os.path.abspath(local_p) != os.path.abspath(p))
        return p, False

    joined_p = os.path.join(root_dir, p)
    if os.path.exists(joined_p):
        return joined_p, False
    if local_p and local_p != joined_p and os.path.exists(local_p):
        return local_p, True
    return joined_p, False


def normalize_index(idx_obj, root_dir):
    meta = dict(idx_obj)
    shards = meta.get("shards", None)
    if not isinstance(shards, list):
        raise ValueError("index missing 'shards'")

    shard_paths, shard_ns = [], []
    rebased = 0
    for sh in shards:
        p = sh.get("path", None)
        n = sh.get("n_samples", sh.get("n", sh.get("count", None)))
        if p is None:
            continue
        p, did_rebase = _normalize_shard_path(p, root_dir)
        rebased += int(did_rebase)
        shard_paths.append(p)
        shard_ns.append(int(n) if n is not None else -1)

    if rebased > 0:
        print(
            f"NOTE: rebased {rebased}/{len(shard_paths)} shard path(s) to index directory: {root_dir}",
            flush=True,
        )

    if any(n < 0 for n in shard_ns):
        inferred = []
        for p in shard_paths:
            blob = safe_torch_load(p, map_location="cpu")
            inferred.append(int(blob["x"].shape[0]))
        shard_ns = inferred

    total = int(sum(shard_ns))
    return shard_paths, shard_ns, total


def build_spatial_subsample_indices(Ny_full: int, Nx_full: int, low_hw: int):
    Ny_full, Nx_full, low_hw = int(Ny_full), int(Nx_full), int(low_hw)
    if low_hw < 2:
        raise ValueError(f"low_hw must be >= 2 for source-grid subsampling. got {low_hw}")
    if low_hw > Ny_full or low_hw > Nx_full:
        raise ValueError(
            f"low_hw must not exceed source resolution. low_hw={low_hw}, "
            f"source=({Ny_full},{Nx_full})"
        )
    y_idx = np.rint(np.linspace(0, Ny_full - 1, low_hw)).astype(np.int64)
    x_idx = np.rint(np.linspace(0, Nx_full - 1, low_hw)).astype(np.int64)
    return y_idx, x_idx


def _as_numpy_1d(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value).reshape(-1)


def _nested_metadata_value(obj, key):
    if not isinstance(obj, dict):
        return None, False
    if key in obj:
        return obj[key], True
    for container_key in ("metadata", "meta"):
        container = obj.get(container_key)
        if isinstance(container, dict) and key in container:
            return container[key], True
    samples = obj.get("samples")
    if isinstance(samples, (list, tuple)) and samples and all(isinstance(row, dict) for row in samples):
        if all(key in row for row in samples):
            return [row[key] for row in samples], True
    return None, False


def _time_policy_from_index(index_obj):
    if not isinstance(index_obj, dict):
        return {}
    meta = index_obj.get("meta", index_obj.get("metadata", {}))
    if not isinstance(meta, dict):
        meta = {}
    policy = meta.get("time_policy", {})
    return policy if isinstance(policy, dict) else {}


def _metadata_allows_zero_dt(index_obj):
    if not isinstance(index_obj, dict):
        return False
    meta = index_obj.get("meta", index_obj.get("metadata", {}))
    if not isinstance(meta, dict):
        meta = {}
    policy = _time_policy_from_index(index_obj)
    return bool(
        index_obj.get(
            "allow_zero_dt",
            meta.get("allow_zero_dt", policy.get("allow_zero_dt", False)),
        )
    )


def batch_time_weights(t_cumulative, loss_time_beta, time_weight_floor=0.0):
    beta = float(loss_time_beta)
    if beta <= 0.0:
        return None
    t_now = torch.clamp(t_cumulative.reshape(-1), min=float(time_weight_floor))
    return torch.pow(t_now, beta)


class BudgetCachedShardedPTDataset(Dataset):
    def __init__(self, index_path: str, max_cache_gb: float = 32, to_float32: bool = False):
        super().__init__()
        self.root_dir = os.path.dirname(index_path)
        self.index_path = os.path.abspath(index_path)
        idx = safe_torch_load(index_path, map_location="cpu")
        self.index = idx
        self.boundary_condition, boundary_source = _boundary_condition_from_metadata(idx)
        self.boundary_source = (
            f"{index_path} ({boundary_source})" if boundary_source else str(index_path)
        )
        self.Ny = int(idx.get("Ny", -1)) if isinstance(idx, dict) else -1
        self.Nx = int(idx.get("Nx", -1)) if isinstance(idx, dict) else -1
        self.Cin = int(idx.get("Cin", -1)) if isinstance(idx, dict) else -1
        self.shard_paths, self.shard_ns, _ = normalize_index(idx, self.root_dir)

        self.offsets = []
        s = 0
        for n in self.shard_ns:
            self.offsets.append(s)
            s += int(n)
        self.total = int(s)

        self.to_float32 = bool(to_float32)
        self.max_cache_bytes = int(float(max_cache_gb) * (1024 ** 3))
        self.cache = OrderedDict()
        self.cache_bytes = OrderedDict()
        self.cur_bytes = 0
        self.dt_physical = None
        self.t_cumulative = None
        self.trajectory_final_times = None
        self.time_sanity_rows = []
        self._global_field_cache = None
        self._cached_x_dt = None

    def __len__(self):
        return self.total

    def split_trajectory_ids(self):
        """Read original IDs in the trajectory order used by the legacy builder."""
        split_name, _ = _nested_metadata_value(self.index, "split_name")
        if split_name is None:
            folder = os.path.basename(os.path.dirname(self.index_path))
            split_name = folder if folder in ("train", "test") else None
        ids, _ = _nested_metadata_value(self.index, "split_trajectory_ids")
        split_path = os.path.join(os.path.dirname(self.root_dir), "split_ids.npz")
        if split_name in ("train", "test") and os.path.isfile(split_path):
            with np.load(split_path, allow_pickle=False) as split:
                train_ids = np.asarray(split["train_ids"])
                test_ids = np.asarray(split["test_ids"])
                if np.intersect1d(train_ids, test_ids).size:
                    raise ValueError(f"Train/test trajectory IDs overlap in {split_path}")
                file_ids = train_ids if split_name == "train" else test_ids
                if ids is not None and not np.array_equal(np.asarray(ids), file_ids):
                    raise ValueError(f"Index trajectory IDs disagree with {split_path}")
                ids = file_ids
        if ids is not None:
            raw = np.asarray(ids)
            if (raw.ndim != 1 or raw.dtype.kind not in "iu" or
                    np.any(raw < 0) or np.unique(raw).size != raw.size):
                raise ValueError("split_trajectory_ids must be unique nonnegative integer IDs")
            ids = raw.astype(np.int64)
        return split_name, ids

    def _blob_nbytes(self, blob: dict) -> int:
        nb = 0
        for k in ["x", "y"]:
            t = blob.get(k, None)
            if isinstance(t, torch.Tensor):
                nb += t.numel() * t.element_size()
        return int(nb)

    def _evict_if_needed(self):
        while self.cur_bytes > self.max_cache_bytes and len(self.cache) > 0:
            old_idx, _ = self.cache.popitem(last=False)
            old_nb = self.cache_bytes.pop(old_idx)
            self.cur_bytes -= int(old_nb)

    def _load_shard(self, shard_idx: int):
        if shard_idx in self.cache:
            blob = self.cache.pop(shard_idx)
            nb = self.cache_bytes.pop(shard_idx)
            self.cache[shard_idx] = blob
            self.cache_bytes[shard_idx] = nb
            return blob

        shard_path = self.shard_paths[shard_idx]
        if not os.path.exists(shard_path):
            base_name = os.path.basename(os.path.normpath(str(shard_path).replace("\\", os.sep)))
            local_path = os.path.join(self.root_dir, base_name) if base_name else ""
            if local_path != shard_path and os.path.exists(local_path):
                self.shard_paths[shard_idx] = local_path
                shard_path = local_path
        try:
            blob = safe_torch_load(shard_path, map_location="cpu")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"missing shard {shard_idx} referenced by index directory {self.root_dir}: {shard_path}"
            ) from exc
        if self.to_float32:
            blob["x"] = blob["x"].float()
            blob["y"] = blob["y"].float()

        nb = self._blob_nbytes(blob)
        if nb > self.max_cache_bytes:
            self.cache.clear()
            self.cache_bytes.clear()
            self.cur_bytes = 0

        self.cache[shard_idx] = blob
        self.cache_bytes[shard_idx] = nb
        self.cur_bytes += nb
        self._evict_if_needed()
        return blob

    def _extract_x_dt_from_blob(self, blob, shard_idx, shard_n, dt_channel_idx):
        x = blob.get("x")
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError(f"shard {shard_idx} x must have shape (N,Ny,Nx,Cin)")
        idx = int(dt_channel_idx)
        if idx < 0 or idx >= int(x.shape[-1]):
            raise ValueError(
                f"dt channel index {idx} is out of range for shard {shard_idx} Cin={int(x.shape[-1])}"
            )
        dt_nodes = x[..., idx].float()
        dt_scalar = dt_nodes[:, 0, 0]
        if not torch.allclose(
            dt_nodes,
            dt_scalar[:, None, None].expand_as(dt_nodes),
            rtol=1.0e-4,
            atol=1.0e-6,
        ):
            raise ValueError(f"dt channel is not spatially constant in shard {shard_idx}")
        if int(dt_scalar.numel()) != int(shard_n):
            raise RuntimeError("shard dt extraction count mismatch")
        return dt_scalar.cpu().numpy().astype(np.float64, copy=False)

    def _scan_shard_time_metadata(self, dt_channel_idx=None):
        keys = ("dt", "sample_id", "traj_id", "case_id", "time_index", "step_id")
        pieces = {key: [] for key in keys}
        found_counts = {key: 0 for key in keys}
        errors = {}
        x_dt_pieces = []
        for shard_idx, shard_n in enumerate(self.shard_ns):
            blob = self._load_shard(shard_idx)
            for key in keys:
                value, present = _nested_metadata_value(blob, key)
                if not present:
                    pieces[key].append(None)
                    continue
                found_counts[key] += 1
                array = _as_numpy_1d(value)
                if array is None or int(array.size) != int(shard_n):
                    size = 0 if array is None else int(array.size)
                    errors[key] = ValueError(
                        f"shard {shard_idx} field {key!r} must contain {int(shard_n)} values, got {size}"
                    )
                    pieces[key].append(None)
                else:
                    pieces[key].append(array)
            _dt_value, shard_has_dt = _nested_metadata_value(blob, "dt")
            if dt_channel_idx is not None and not shard_has_dt:
                x_dt_pieces.append(
                    self._extract_x_dt_from_blob(blob, shard_idx, shard_n, dt_channel_idx)
                )

        cache = {}
        for key in keys:
            if key in errors:
                cache[key] = errors[key]
            elif found_counts[key] == 0:
                cache[key] = (None, None)
            elif found_counts[key] != len(self.shard_ns):
                cache[key] = ValueError(f"field {key!r} exists in only some shards")
            else:
                cache[key] = (np.concatenate(pieces[key], axis=0), "shards")
        self._global_field_cache = cache
        if found_counts["dt"] == 0 and len(x_dt_pieces) == len(self.shard_ns):
            self._cached_x_dt = np.concatenate(x_dt_pieces, axis=0)

    def _collect_global_field(self, key, dt_channel_idx=None):
        value, present = _nested_metadata_value(self.index, key)
        if present:
            array = _as_numpy_1d(value)
            if array is None or int(array.size) != self.total:
                size = 0 if array is None else int(array.size)
                raise ValueError(
                    f"index field {key!r} must contain {self.total} values, got {size}"
                )
            return array, "index"
        if self._global_field_cache is None:
            self._scan_shard_time_metadata(dt_channel_idx=dt_channel_idx)
        result = self._global_field_cache.get(key, (None, None))
        if isinstance(result, BaseException):
            raise result
        return result

    def _physical_dt_from_x(self, dt_channel_idx, dt_scale):
        if self._cached_x_dt is not None:
            return np.asarray(self._cached_x_dt, dtype=np.float64) * float(dt_scale)
        dt_parts = []
        for shard_idx, shard_n in enumerate(self.shard_ns):
            blob = self._load_shard(shard_idx)
            dt_parts.append(
                self._extract_x_dt_from_blob(blob, shard_idx, shard_n, dt_channel_idx)
            )
        return np.concatenate(dt_parts, axis=0) * float(dt_scale)

    def _fallback_time_grid(self, T_final):
        policy = _time_policy_from_index(self.index)
        meta = self.index.get("meta", self.index.get("metadata", {}))
        if not isinstance(meta, dict):
            meta = {}
        subset = meta.get("subset", {})
        if (
            isinstance(subset, dict)
            and str(subset.get("kind", "")).lower() == "materialized"
            and str(subset.get("select_unit", "trajectory")).lower() != "trajectory"
        ):
            raise ValueError(
                "Cannot reconstruct cumulative time safely because this materialized subset "
                "selected individual transitions rather than complete trajectories."
            )

        desired = policy.get("desired_times", self.index.get("times"))
        if desired is not None:
            times = np.asarray(desired, dtype=np.float64).reshape(-1)
        else:
            reported = meta.get(
                "transitions_per_trajectory",
                policy.get("transitions_per_trajectory", None),
            )
            if reported is None and policy.get("n_frames") is not None:
                reported = int(policy["n_frames"]) - 1
            if reported is None:
                raise ValueError(
                    "Cannot reconstruct cumulative time safely because trajectory or "
                    "time-order metadata is missing."
                )
            times = make_times_55(T_final=float(T_final)).astype(np.float64)
            if int(reported) != int(times.size - 1):
                raise ValueError(
                    "Cannot reconstruct cumulative time safely: index transition count "
                    "does not match make_times_55(T_final)."
                )

        if times.size < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
            raise ValueError("index time grid must be finite and monotone with at least two frames")
        n_step = int(times.size - 1)
        reported = meta.get("transitions_per_trajectory", None)
        if reported is None and policy.get("n_frames") is not None:
            reported = int(policy["n_frames"]) - 1
        if reported is not None and int(reported) != n_step:
            raise ValueError(
                f"index transitions_per_trajectory={int(reported)} disagrees with time grid ({n_step})"
            )
        if self.total % n_step != 0:
            raise ValueError(
                "Cannot reconstruct cumulative time safely because dataset length is not "
                f"divisible by transitions per trajectory: {self.total} % {n_step} != 0."
            )
        return times

    def prepare_cumulative_time(self, time_channel_idx=-1, dt_channel_idx=-1, dt_scale=250.0, T_final=250.0):
        if self.t_cumulative is not None:
            return

        resolved_dt_idx = _resolve_dt_channel_idx(time_channel_idx, dt_channel_idx)
        dt_values, dt_source = self._collect_global_field(
            "dt", dt_channel_idx=resolved_dt_idx
        )
        if dt_values is None:
            policy = _time_policy_from_index(self.index)
            meta = self.index.get("meta", self.index.get("metadata", {}))
            if not isinstance(meta, dict):
                meta = {}
            stored_scale = policy.get("t_scale", meta.get("t_scale", None))
            effective_scale = float(dt_scale)
            if stored_scale is not None:
                stored_scale = float(stored_scale)
                if not math.isfinite(stored_scale) or stored_scale <= 0.0:
                    raise ValueError(f"index dt normalization scale must be positive, got {stored_scale}")
                if not math.isclose(stored_scale, float(dt_scale), rel_tol=1.0e-9, abs_tol=1.0e-12):
                    raise ValueError(
                        "--dt_scale does not match the dataset dt normalization scale: "
                        f"argument={float(dt_scale):g} index={stored_scale:g}."
                    )
                effective_scale = stored_scale
            dt_physical = self._physical_dt_from_x(resolved_dt_idx, effective_scale)
            dt_source = (
                f"x channel {resolved_dt_idx} multiplied by physical scale={effective_scale:g}"
            )
        else:
            # A top-level/shard dt field is physical by default. Only explicit metadata may mark it normalized.
            policy = _time_policy_from_index(self.index)
            meta = self.index.get("meta", self.index.get("metadata", {}))
            if not isinstance(meta, dict):
                meta = {}
            unit = str(meta.get("dt_unit", policy.get("dt_unit", "physical"))).strip().lower()
            normalized = bool(meta.get("dt_is_normalized", policy.get("dt_is_normalized", False)))
            normalized = normalized or unit in {"normalized", "dimensionless", "dt_norm"}
            dt_physical = np.asarray(dt_values, dtype=np.float64)
            if normalized:
                scale = float(meta.get("dt_scale", policy.get("t_scale", dt_scale)))
                dt_physical = dt_physical * scale
                dt_source = f"{dt_source} top-level dt multiplied by explicit scale={scale:g}"
            else:
                dt_source = f"{dt_source} top-level physical dt"

        if int(dt_physical.size) != self.total:
            raise RuntimeError("physical dt count mismatch")
        if not np.all(np.isfinite(dt_physical)):
            raise ValueError("physical dt contains NaN or Inf")
        if np.any(dt_physical < 0.0):
            bad = int(np.nonzero(dt_physical < 0.0)[0][0])
            raise ValueError(f"physical dt must be >= 0; index={bad} dt={dt_physical[bad]}")
        if np.any(dt_physical == 0.0) and not _metadata_allows_zero_dt(self.index):
            bad = int(np.nonzero(dt_physical == 0.0)[0][0])
            raise ValueError(
                "physical dt=0 was found but index metadata does not explicitly set "
                f"allow_zero_dt=True; index={bad}"
            )

        trajectory_values = trajectory_key = None
        for candidate in ("sample_id", "traj_id", "case_id"):
            values, _source = self._collect_global_field(candidate)
            if values is not None:
                trajectory_values, trajectory_key = values, candidate
                break
        order_values = order_key = None
        for candidate in ("time_index", "step_id"):
            values, _source = self._collect_global_field(candidate)
            if values is not None:
                order_values, order_key = values, candidate
                break

        used_global_fallback = trajectory_values is None or order_values is None
        expected_times = None
        if used_global_fallback:
            expected_times = self._fallback_time_grid(T_final)
            n_step = int(expected_times.size - 1)
            expected_dt = np.diff(expected_times)
            dt_matrix = np.asarray(dt_physical, dtype=np.float64).reshape(-1, n_step)
            n_verify = min(5, int(dt_matrix.shape[0]))
            if not np.allclose(
                dt_matrix[:n_verify],
                expected_dt[None, :],
                rtol=5.0e-3,
                atol=5.0e-4,
            ):
                raise ValueError(
                    "Cannot reconstruct cumulative time safely because the physical dt values "
                    "of the first trajectories do not match np.diff(times)."
                )
            fallback_trajectory = np.arange(self.total, dtype=np.int64) // n_step
            fallback_order = np.arange(self.total, dtype=np.int64) % n_step
            if trajectory_values is None:
                _, split_ids = self.split_trajectory_ids()
                if split_ids is not None:
                    if len(split_ids) != self.total // n_step:
                        raise ValueError("Split trajectory ID count disagrees with legacy transition count")
                    trajectory_values = split_ids[fallback_trajectory]
                    trajectory_key = "split_trajectory_ids trajectory-major mapping"
                else:
                    trajectory_values = fallback_trajectory
                    trajectory_key = "global_index trajectory-major fallback"
            else:
                for trajectory_number in range(int(self.total // n_step)):
                    block = trajectory_values[trajectory_number * n_step:(trajectory_number + 1) * n_step]
                    if np.unique(block).size != 1:
                        raise ValueError(
                            "Cannot reconstruct cumulative time safely: trajectory metadata is not "
                            "trajectory-major in global sample order."
                        )
            if order_values is None:
                order_values = fallback_order
                order_key = "global_index time-major fallback"

        groups = OrderedDict()
        for index in range(self.total):
            raw_key = trajectory_values[index]
            key = raw_key.item() if hasattr(raw_key, "item") else raw_key
            groups.setdefault(key, []).append(index)

        cumulative = np.empty((self.total,), dtype=np.float64)
        self.evaluation_trajectory_ids = np.empty((self.total,), dtype=np.int64)
        final_times = []
        sanity_rows = []
        for trajectory_code, (key, indices) in enumerate(groups.items()):
            if not isinstance(key, (int, np.integer)) or key < 0:
                raise ValueError(f"Trajectory sample_id must be a nonnegative integer, got {key!r}")
            self.evaluation_trajectory_ids[indices] = key
            raw_order = np.asarray(order_values[indices])
            try:
                finite_order = np.asarray(raw_order, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{order_key} must be numeric for trajectory {key!r}") from exc
            if not np.all(np.isfinite(finite_order)):
                raise ValueError(f"{order_key} contains NaN or Inf for trajectory {key!r}")
            if np.unique(raw_order).size != len(indices):
                raise ValueError(f"duplicate {order_key} values within trajectory {key!r}")
            ordered = [indices[pos] for pos in np.argsort(raw_order, kind="stable")]
            t_running = 0.0
            trajectory_times = []
            for step, index in enumerate(ordered):
                t_running = t_running + float(dt_physical[index])
                cumulative[index] = t_running
                trajectory_times.append(t_running)
                if trajectory_code == 0 and step < 5:
                    sanity_rows.append((step, float(dt_physical[index]), t_running))
            if not math.isclose(
                trajectory_times[0], float(dt_physical[ordered[0]]), rel_tol=1.0e-10, abs_tol=1.0e-12
            ):
                raise RuntimeError("first cumulative time does not equal the first physical dt")
            if np.any(np.diff(np.asarray(trajectory_times, dtype=np.float64)) < 0.0):
                raise RuntimeError(f"cumulative time is not monotone for trajectory {key!r}")
            final_times.append(float(trajectory_times[-1]))

        if not np.all(np.isfinite(cumulative)):
            raise ValueError("reconstructed cumulative time contains NaN or Inf")
        self.dt_physical = torch.as_tensor(dt_physical, dtype=torch.float32).contiguous()
        self.t_cumulative = torch.as_tensor(cumulative, dtype=torch.float32).contiguous()
        self.trajectory_final_times = np.asarray(final_times, dtype=np.float64)
        self.time_sanity_rows = sanity_rows
        self.time_metadata_summary = {
            "trajectory_key": trajectory_key,
            "order_key": order_key,
            "dt_source": dt_source,
            "used_global_fallback": bool(used_global_fallback),
        }

    def __getitem__(self, idx: int):
        if self.t_cumulative is None:
            raise RuntimeError("prepare_cumulative_time must be called before reading dataset samples")
        shard_idx = bisect_right(self.offsets, int(idx)) - 1
        if shard_idx < 0:
            shard_idx = 0
        local = int(idx - self.offsets[shard_idx])
        blob = self._load_shard(shard_idx)
        third = self.t_cumulative[int(idx)]
        if getattr(self, "full_time_evaluation_metadata", False):
            third = {
                "t_cumulative": third,
                "dt_physical": self.dt_physical[int(idx)],
                "source_id": self.full_time_evaluation_source_id,
                "sample_id": int(self.evaluation_trajectory_ids[int(idx)]),
            }
        return blob["x"][local], blob["y"][local], third


class SourceGridSubsampledPTDataset(Dataset):
    def __init__(self, base: Dataset, low_hw: int):
        super().__init__()
        self.base = base
        self.low_hw = int(low_hw)
        self.source_Ny = int(getattr(base, "Ny", -1))
        self.source_Nx = int(getattr(base, "Nx", -1))
        self.Cin = int(getattr(base, "Cin", -1))
        if self.source_Ny <= 0 or self.source_Nx <= 0:
            raise ValueError("base dataset must expose positive Ny/Nx")

        y_idx, x_idx = build_spatial_subsample_indices(self.source_Ny, self.source_Nx, self.low_hw)
        self.Ny = int(y_idx.size)
        self.Nx = int(x_idx.size)
        self.y_idx = torch.as_tensor(y_idx, dtype=torch.long)
        self.x_idx = torch.as_tensor(x_idx, dtype=torch.long)
        self.index = getattr(base, "index", {})

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y, t_cumulative = self.base[int(idx)]
        x_low = x.index_select(0, self.y_idx).index_select(1, self.x_idx)
        y_low = y.index_select(0, self.y_idx).index_select(1, self.x_idx)
        return x_low, y_low, t_cumulative


def maybe_source_grid_subsample_dataset(ds: Dataset, low_hw):
    if low_hw is None:
        return ds
    low_hw = int(low_hw)
    Ny = int(getattr(ds, "Ny", -1))
    Nx = int(getattr(ds, "Nx", -1))
    if low_hw == Ny and low_hw == Nx:
        return ds
    return SourceGridSubsampledPTDataset(ds, low_hw)


# -------------------------
# aligned mix dataset (full/low)
# -------------------------
_MASK64 = (1 << 64) - 1


def _u01_from_seed_idx(seed: int, idx: int) -> float:
    # splitmix64 (deterministic, worker-safe)
    z = (int(seed) + 0x9E3779B97F4A7C15 + int(idx)) & _MASK64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK64
    z = (z ^ (z >> 31)) & _MASK64
    return float(z) / float(1 << 64)


class AlignedMixDataset(Dataset):
    # full/low datasets must share the same split and sample ordering.
    def __init__(self, ds_full: Dataset, ds_low: Dataset, full_ratio: float, seed: int = 0):
        super().__init__()
        self.ds_full = ds_full
        self.ds_low = ds_low
        self.full_ratio = float(full_ratio)
        self.seed = int(seed)
        n1 = len(ds_full)
        n2 = len(ds_low)
        if n1 != n2:
            raise ValueError(f"full/low length mismatch: full={n1} low={n2}. "
                             f"Ensure both datasets were built from the same split and identical ordering.")
        self.total = int(n1)

    def __len__(self):
        return self.total

    def _get_from_source(self, source, idx: int):
        if source == "full":
            return self.ds_full[int(idx)]
        if source == "low":
            return self.ds_low[int(idx)]
        raise ValueError(f"unknown mix source: {source}")

    def __getitem__(self, idx: int):
        if isinstance(idx, tuple) and len(idx) == 2:
            source, raw_idx = idx
            return self._get_from_source(source, int(raw_idx))

        u = _u01_from_seed_idx(self.seed, int(idx))
        if u < self.full_ratio:
            return self._get_from_source("full", int(idx))
        else:
            return self._get_from_source("low", int(idx))


class MultiResolutionSubsetDataset(Dataset):
    def __init__(self, datasets, ratios, seed: int = 0, names=None):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("datasets must not be empty")
        if len(datasets) != len(ratios):
            raise ValueError(f"datasets/ratios length mismatch: {len(datasets)} vs {len(ratios)}")
        if names is None:
            names = [f"source{i}" for i in range(len(datasets))]
        if len(names) != len(datasets):
            raise ValueError(f"names/datasets length mismatch: {len(names)} vs {len(datasets)}")

        cin0 = getattr(datasets[0], "Cin", None)
        self.sources = []
        self.cum_counts = []
        total = 0
        for i, (ds, ratio, name) in enumerate(zip(datasets, ratios, names)):
            ratio = float(ratio)
            if not (0.0 <= ratio <= 1.0):
                raise ValueError(f"multi-resolution ratio must be in [0,1]. source={name} ratio={ratio}")
            cin = getattr(ds, "Cin", None)
            if cin0 is not None and cin is not None and int(cin) != int(cin0):
                raise ValueError(f"Cin mismatch between multi-resolution datasets: {cin0} vs {cin} at {name}")

            n = int(len(ds))
            if ratio >= 1.0:
                take = n
            else:
                take = int(math.floor(float(n) * ratio))
            take = max(0, min(n, take))

            if take == n:
                indices = np.arange(n, dtype=np.int64)
            elif take == 0:
                indices = np.empty((0,), dtype=np.int64)
            else:
                rng = np.random.default_rng(int(seed) + 1000003 * (i + 1))
                indices = np.sort(rng.permutation(n)[:take].astype(np.int64))

            self.sources.append({
                "dataset": ds,
                "ratio": ratio,
                "indices": indices,
                "name": str(name),
                "Ny": int(getattr(ds, "Ny", -1)),
                "Nx": int(getattr(ds, "Nx", -1)),
                "Cin": int(getattr(ds, "Cin", -1)),
            })
            total += int(indices.size)
            self.cum_counts.append(total)

        self.total = int(total)
        if self.total <= 0:
            raise ValueError("multi-resolution dataset is empty; all ratios selected zero samples")

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int):
        if isinstance(idx, tuple) and len(idx) == 2:
            source_idx, raw_idx = idx
            src = self.sources[int(source_idx)]
            return src["dataset"][int(raw_idx)]

        gidx = int(idx)
        source_idx = bisect_right(self.cum_counts, gidx)
        prev = 0 if source_idx == 0 else int(self.cum_counts[source_idx - 1])
        local = gidx - prev
        src = self.sources[source_idx]
        raw_idx = int(src["indices"][local])
        return src["dataset"][raw_idx]

    def summary(self):
        out = []
        for src in self.sources:
            ds = src["dataset"]
            selected = int(src["indices"].size)
            total = int(len(ds))
            out.append({
                "name": src["name"],
                "ratio": float(src["ratio"]),
                "selected": selected,
                "total": total,
                "Ny": int(src["Ny"]),
                "Nx": int(src["Nx"]),
                "Cin": int(src["Cin"]),
            })
        return out


def _iter_unique_sharded_bases(dataset, seen=None):
    if seen is None:
        seen = set()
    if isinstance(dataset, BudgetCachedShardedPTDataset):
        if id(dataset) not in seen:
            seen.add(id(dataset))
            yield dataset
        return
    if isinstance(dataset, SourceGridSubsampledPTDataset):
        yield from _iter_unique_sharded_bases(dataset.base, seen)
        return
    if isinstance(dataset, AlignedMixDataset):
        yield from _iter_unique_sharded_bases(dataset.ds_full, seen)
        yield from _iter_unique_sharded_bases(dataset.ds_low, seen)
        return
    if isinstance(dataset, MultiResolutionSubsetDataset):
        for source in dataset.sources:
            yield from _iter_unique_sharded_bases(source["dataset"], seen)
        return
    raise TypeError(f"unsupported dataset wrapper for cumulative time: {type(dataset).__name__}")


def prepare_dataset_cumulative_time(dataset, args):
    bases = list(_iter_unique_sharded_bases(dataset))
    for base in bases:
        base.prepare_cumulative_time(
            time_channel_idx=int(args.time_channel_idx),
            dt_channel_idx=int(args.dt_channel_idx),
            dt_scale=float(args.dt_scale),
            T_final=float(args.T_final),
        )

    def validate_alignment(current):
        if isinstance(current, SourceGridSubsampledPTDataset):
            validate_alignment(current.base)
        elif isinstance(current, AlignedMixDataset):
            validate_alignment(current.ds_full)
            validate_alignment(current.ds_low)
            full_base = next(_iter_unique_sharded_bases(current.ds_full))
            low_base = next(_iter_unique_sharded_bases(current.ds_low))
            if len(full_base) != len(low_base) or not torch.allclose(
                full_base.t_cumulative, low_base.t_cumulative, rtol=1.0e-6, atol=1.0e-7
            ):
                raise ValueError("aligned full/low datasets do not have identical cumulative times")
        elif isinstance(current, MultiResolutionSubsetDataset):
            source_bases = []
            for source in current.sources:
                validate_alignment(source["dataset"])
                source_bases.append(next(_iter_unique_sharded_bases(source["dataset"])))
            reference = source_bases[0]
            for candidate in source_bases[1:]:
                if len(candidate) == len(reference) and not torch.allclose(
                    candidate.t_cumulative,
                    reference.t_cumulative,
                    rtol=1.0e-6,
                    atol=1.0e-7,
                ):
                    raise ValueError(
                        "same-length multi-resolution sources do not have identical cumulative times"
                    )

    validate_alignment(dataset)
    return bases


def print_time_weighting_summary(dataset, loss_time_beta, time_weight_floor):
    bases = list(_iter_unique_sharded_bases(dataset))
    if not bases:
        raise ValueError("no sharded dataset is available for time-weight sanity logging")
    beta = float(loss_time_beta)
    print("time loss weighting:", flush=True)
    print(f"  beta = {beta:g}", flush=True)
    print("  train weighting = cumulative physical time t_k reconstructed from dt", flush=True)
    print("  cumulative update = t_running += dt", flush=True)
    print("  first transition time = first dt", flush=True)
    print("  train reduction = mean(t_k^beta * per-sample relative L2)", flush=True)
    print("  test weighting = none", flush=True)
    print("  best model metric = unweighted test relative L2", flush=True)
    print("  rollout weighting = none", flush=True)
    summary = bases[0].time_metadata_summary
    print(
        "  metadata = trajectory:{} order:{} dt:{} fallback:{}".format(
            summary["trajectory_key"],
            summary["order_key"],
            summary["dt_source"],
            int(summary["used_global_fallback"]),
        ),
        flush=True,
    )
    print("time-weight sanity:", flush=True)
    for step, dt_value, cumulative_t in bases[0].time_sanity_rows:
        weight = 1.0 if beta <= 0.0 else max(cumulative_t, float(time_weight_floor)) ** beta
        print(
            f"  step={step} dt={dt_value:.8g} cumulative_t={cumulative_t:.8g} weight={weight:.8g}",
            flush=True,
        )
    final_times = np.concatenate([base.trajectory_final_times for base in bases])
    print("trajectory final cumulative time:", flush=True)
    print(f"  min={float(final_times.min()):.8g}", flush=True)
    print(f"  max={float(final_times.max()):.8g}", flush=True)



# -------------------------
# shard batch sampler (reduce shard switching)
# -------------------------
from torch.utils.data import Sampler

def _get_sharded_base(ds):
    if isinstance(ds, BudgetCachedShardedPTDataset):
        return ds
    if isinstance(ds, SourceGridSubsampledPTDataset):
        return _get_sharded_base(ds.base)
    if isinstance(ds, AlignedMixDataset):
        return _get_sharded_base(ds.ds_full)
    return None


class ShardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        base = _get_sharded_base(dataset)
        if base is None:
            raise TypeError("ShardBatchSampler requires BudgetCachedShardedPTDataset or AlignedMixDataset over it")
        self.offsets = list(base.offsets)
        self.shard_ns = list(base.shard_ns)
        self._epoch = 0

    def __len__(self):
        bs = int(self.batch_size)
        if bs <= 0:
            return 0
        n = 0
        for sn in self.shard_ns:
            sn = int(sn)
            if self.drop_last:
                n += sn // bs
            else:
                n += (sn + bs - 1) // bs
        return int(n)

    def __iter__(self):
        import random
        bs = int(self.batch_size)
        shard_ids = list(range(len(self.shard_ns)))
        rng = random.Random(int(self.seed) + int(self._epoch))
        mix_ds = self.dataset if isinstance(self.dataset, AlignedMixDataset) else None
        batch_counter = 0
        if self.shuffle:
            rng.shuffle(shard_ids)

        for sid in shard_ids:
            start = int(self.offsets[sid])
            sn = int(self.shard_ns[sid])
            if sn <= 0:
                continue

            nb = (sn // bs) if self.drop_last else ((sn + bs - 1) // bs)
            batch_ids = list(range(nb))
            if self.shuffle:
                rng.shuffle(batch_ids)

            for bid in batch_ids:
                b0 = start + bid * bs
                b1 = min(b0 + bs, start + sn)
                if self.drop_last and (b1 - b0) < bs:
                    continue
                batch = list(range(b0, b1))
                if self.shuffle:
                    rng.shuffle(batch)
                if mix_ds is None:
                    yield batch
                else:
                    if mix_ds.full_ratio <= 0.0:
                        source = "low"
                    elif mix_ds.full_ratio >= 1.0:
                        source = "full"
                    else:
                        source_epoch = int(self._epoch) if self.shuffle else 0
                        u = _u01_from_seed_idx(
                            int(self.seed) + 1000003 * (source_epoch + 1),
                            int(batch_counter),
                        )
                        source = "full" if u < mix_ds.full_ratio else "low"
                    yield [(source, i) for i in batch]
                batch_counter += 1

        self._epoch += 1


class MultiResolutionShardBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=0):
        super().__init__()
        if not isinstance(dataset, MultiResolutionSubsetDataset):
            raise TypeError("MultiResolutionShardBatchSampler requires MultiResolutionSubsetDataset")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        self.groups = []
        for source_idx, src in enumerate(self.dataset.sources):
            base = _get_sharded_base(src["dataset"])
            if base is None:
                raise TypeError("multi-resolution sources must be backed by BudgetCachedShardedPTDataset")
            by_shard = [[] for _ in base.shard_ns]
            for raw_idx in src["indices"]:
                ridx = int(raw_idx)
                shard_idx = bisect_right(base.offsets, ridx) - 1
                if shard_idx < 0:
                    shard_idx = 0
                by_shard[shard_idx].append(ridx)
            for shard_idx, raw_indices in enumerate(by_shard):
                if len(raw_indices) > 0:
                    self.groups.append((int(source_idx), int(shard_idx), raw_indices))

    def __len__(self):
        bs = int(self.batch_size)
        if bs <= 0:
            return 0
        if self.drop_last:
            totals_by_source = {}
            for source_idx, _, raw_indices in self.groups:
                totals_by_source[int(source_idx)] = totals_by_source.get(int(source_idx), 0) + int(len(raw_indices))
            return int(sum(total // bs for total in totals_by_source.values()))

        n = 0
        for _, _, raw_indices in self.groups:
            gn = int(len(raw_indices))
            n += (gn + bs - 1) // bs
        return int(n)

    def __iter__(self):
        import random
        bs = int(self.batch_size)
        rng = random.Random(int(self.seed) + int(self._epoch))
        group_ids = list(range(len(self.groups)))
        pending_by_source = {}
        if self.shuffle:
            rng.shuffle(group_ids)

        for gid in group_ids:
            source_idx, _, raw_indices = self.groups[gid]
            indices = list(raw_indices)
            if self.shuffle:
                rng.shuffle(indices)
            nb = (len(indices) + bs - 1) // bs
            batch_ids = list(range(nb))
            if self.shuffle:
                rng.shuffle(batch_ids)

            for bid in batch_ids:
                b0 = int(bid) * bs
                batch = indices[b0:b0 + bs]
                if self.drop_last and len(batch) < bs:
                    pending = pending_by_source.setdefault(int(source_idx), [])
                    pending.extend(batch)
                    while len(pending) >= bs:
                        out = pending[:bs]
                        del pending[:bs]
                        if self.shuffle:
                            rng.shuffle(out)
                        yield [(source_idx, i) for i in out]
                else:
                    if self.shuffle:
                        rng.shuffle(batch)
                    yield [(source_idx, i) for i in batch]

        self._epoch += 1

# -------------------------
# metrics
# -------------------------
def mean_rel_l2_flat(a, b, eps=1.0e-12):
    num = torch.linalg.vector_norm(a - b, dim=1)
    den = torch.linalg.vector_norm(b, dim=1) + eps
    return (num / den).mean()


def _rel_l2_2d_np(a, b, eps=1e-12):
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return float(num / den)


# -------------------------
# FE mass-weighted (consistent mass) relative L2
# basis is orthonormal under <u,v>_M = u^T M v, so loss/metrics should use the same inner product
# For tensor-product grids: M2D = My ⊗ Mx, and vec(U)^T M2D vec(U) = sum(U * (My @ (U @ Mx)))
# We apply M efficiently using tri-diagonal stencils (no dense matrices / no 2D Kronecker build).
# -------------------------

def _build_mass_coeffs_1d_torch(
    N: int, L: float = 1.0, device=DEVICE, dtype=torch.float32,
    boundary_condition="neumann",
):
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")
    boundary_condition = _normalize_boundary_condition(boundary_condition)
    if boundary_condition == "periodic":
        main = torch.full((N,), float(L) / float(N), dtype=dtype, device=device)
        zeros = torch.zeros((N,), dtype=dtype, device=device)
        return main, zeros.clone(), zeros
    Ne = N - 1
    h = float(L) / float(Ne)
    main = torch.full((N,), 2.0 * h / 3.0, dtype=dtype, device=device)
    main[0] = h / 3.0
    main[-1] = h / 3.0
    off = torch.full((N - 1,), h / 6.0, dtype=dtype, device=device)

    off_l = torch.zeros((N,), dtype=dtype, device=device)  # multiplies u[..., j-1]
    off_r = torch.zeros((N,), dtype=dtype, device=device)  # multiplies u[..., j+1]
    off_l[1:] = off
    off_r[:-1] = off
    return main, off_l, off_r


def _apply_mass_x(u4, main_x, off_l_x, off_r_x):
    # u4: (B,C,Ny,Nx)
    u_pad = F.pad(u4, (1, 1, 0, 0), mode="constant", value=0.0)
    left = u_pad[..., 0:-2]
    right = u_pad[..., 2:]
    return (
        u4 * main_x.view(1, 1, 1, -1)
        + left * off_l_x.view(1, 1, 1, -1)
        + right * off_r_x.view(1, 1, 1, -1)
    )


def _apply_mass_y(u4, main_y, off_l_y, off_r_y):
    # u4: (B,C,Ny,Nx)
    u_pad = F.pad(u4, (0, 0, 1, 1), mode="constant", value=0.0)
    up = u_pad[:, :, 0:-2, :]
    down = u_pad[:, :, 2:, :]
    return (
        u4 * main_y.view(1, 1, -1, 1)
        + up * off_l_y.view(1, 1, -1, 1)
        + down * off_r_y.view(1, 1, -1, 1)
    )


def _mass_quadratic_2d(u4, mass):
    # u4: (B,C,Ny,Nx)
    main_x, off_l_x, off_r_x = mass["mx"]
    main_y, off_l_y, off_r_y = mass["my"]
    tmp = _apply_mass_x(u4, main_x, off_l_x, off_r_x)          # (B,C,Ny,Nx)
    Mu = _apply_mass_y(tmp, main_y, off_l_y, off_r_y)          # (B,C,Ny,Nx) = (My @ (u @ Mx))
    return (u4 * Mu).sum(dim=(-1, -2))                          # (B,C)


def mass_rel_l2_2d_per_sample(u_pred, u_true, mass, eps=1.0e-12):
    # u_pred/u_true: (B,Ny,Nx) or (B,1,Ny,Nx)
    if u_pred.ndim == 3:
        u_pred = u_pred.unsqueeze(1)
    if u_true.ndim == 3:
        u_true = u_true.unsqueeze(1)

    e = u_pred - u_true
    q_num = _mass_quadratic_2d(e, mass).squeeze(1)             # (B,)
    q_den = _mass_quadratic_2d(u_true, mass).squeeze(1)        # (B,)

    # guard against tiny negatives from fp roundoff
    q_num = torch.clamp(q_num, min=0.0)
    q_den = torch.clamp(q_den, min=0.0)

    rel = torch.sqrt(q_num / (q_den + float(eps)))             # (B,)
    return rel


def mean_mass_rel_l2_2d(u_pred, u_true, mass, eps=1.0e-12):
    return mass_rel_l2_2d_per_sample(u_pred, u_true, mass, eps=eps).mean()


def mass_rel_l2_2d_channels_per_sample(u_pred, u_true, mass, eps=1.0e-12):
    # u_pred/u_true: (B,Ny,Nx,C) or (B,C,Ny,Nx)
    if u_pred.ndim != 4 or u_true.ndim != 4:
        raise ValueError("u_pred and u_true must be 4D tensors")
    ny = int(mass["my"][0].numel())
    nx = int(mass["mx"][0].numel())
    if u_pred.shape == u_true.shape and int(u_pred.shape[1]) == ny and int(u_pred.shape[2]) == nx:
        pred = u_pred.permute(0, 3, 1, 2)
        true = u_true.permute(0, 3, 1, 2)
    elif u_pred.shape == u_true.shape and int(u_pred.shape[2]) == ny and int(u_pred.shape[3]) == nx:
        pred = u_pred
        true = u_true
    else:
        raise ValueError("u_pred and u_true shapes must be (B,Ny,Nx,C) or (B,C,Ny,Nx)")

    e = pred - true
    q_num = _mass_quadratic_2d(e, mass)
    q_den = _mass_quadratic_2d(true, mass)
    q_num = torch.clamp(q_num, min=0.0)
    q_den = torch.clamp(q_den, min=0.0)
    return torch.sqrt(q_num / (q_den + float(eps)))


def _build_mass_coeffs_1d_np(N: int, L: float = 1.0, boundary_condition="neumann"):
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")
    boundary_condition = _normalize_boundary_condition(boundary_condition)
    if boundary_condition == "periodic":
        main = np.full((N,), float(L) / float(N), dtype=np.float64)
        zeros = np.zeros((N,), dtype=np.float64)
        return main, zeros.copy(), zeros
    Ne = N - 1
    h = float(L) / float(Ne)
    main = np.full((N,), 2.0 * h / 3.0, dtype=np.float64)
    main[0] = h / 3.0
    main[-1] = h / 3.0
    off = np.full((N - 1,), h / 6.0, dtype=np.float64)
    off_l = np.zeros((N,), dtype=np.float64)
    off_r = np.zeros((N,), dtype=np.float64)
    off_l[1:] = off
    off_r[:-1] = off
    return main, off_l, off_r


def _apply_mass_x_np(U, main_x, off_l_x, off_r_x):
    # U: (Ny,Nx)
    pad = np.pad(U, ((0, 0), (1, 1)), mode="constant", constant_values=0.0)
    left = pad[:, 0:-2]
    right = pad[:, 2:]
    return U * main_x[None, :] + left * off_l_x[None, :] + right * off_r_x[None, :]


def _apply_mass_y_np(U, main_y, off_l_y, off_r_y):
    # U: (Ny,Nx)
    pad = np.pad(U, ((1, 1), (0, 0)), mode="constant", constant_values=0.0)
    up = pad[0:-2, :]
    down = pad[2:, :]
    return U * main_y[:, None] + up * off_l_y[:, None] + down * off_r_y[:, None]


def _rel_l2_2d_mass_np(a, b, mass_np, eps=1e-12):
    # a,b: (Ny,Nx)
    ex = a - b
    (mx_main, mx_l, mx_r), (my_main, my_l, my_r) = mass_np
    num = np.sum(ex * _apply_mass_y_np(_apply_mass_x_np(ex, mx_main, mx_l, mx_r), my_main, my_l, my_r))
    den = np.sum(b * _apply_mass_y_np(_apply_mass_x_np(b, mx_main, mx_l, mx_r), my_main, my_l, my_r))
    num = float(max(num, 0.0))
    den = float(max(den, 0.0))
    return float(math.sqrt(num / (den + float(eps))))


# -------------------------
# LENO basis (1D mass-orthonormal Laplacian eigenfunctions)
# -------------------------
def build_mass_matrix_1d(Nn, L):
    Ne = int(Nn - 1)
    h = float(L) / float(Ne)
    main_M = np.full(Nn, 2.0 * h / 3.0, dtype=np.float64)
    main_M[0] = h / 3.0
    main_M[-1] = h / 3.0
    off_M = np.full(Nn - 1, h / 6.0, dtype=np.float64)
    M = diags([off_M, main_M, off_M], offsets=[-1, 0, 1], format="csr")
    return M


def build_laplacian_basis_1d_cosine(Nn, K, L):
    x = np.linspace(0.0, float(L), int(Nn), dtype=np.float64)
    Phi = np.zeros((Nn, int(K)), dtype=np.float64)
    Phi[:, 0] = 1.0 / math.sqrt(float(L))
    for k in range(1, int(K)):
        Phi[:, k] = math.sqrt(2.0 / float(L)) * np.cos(float(k) * math.pi * x / float(L))
    return Phi


def mass_orthonormalize_1d(Phi, M):
    MPhi = (M @ Phi)
    G = Phi.T @ MPhi
    Lc = cholesky(G, lower=True)
    Phi_ortho = solve_triangular(Lc, Phi.T, lower=True).T
    P = (M @ Phi_ortho)
    return Phi_ortho.astype(np.float32), P.astype(np.float32)


def build_neumann_basis_1d_mass_orthonormal(N, K, L=1.0):
    K = max(1, min(int(K), int(N)))
    M = build_mass_matrix_1d(int(N), float(L))
    Phi_c = build_laplacian_basis_1d_cosine(int(N), K, float(L))
    Phi, P = mass_orthonormalize_1d(Phi_c, M)
    return Phi, P


def build_periodic_basis_1d_mass_orthonormal(N, K, L=1.0):
    """Real Fourier basis: 1, cos(m), sin(m), without a duplicated endpoint."""
    N = int(N)
    K = max(1, int(K))
    x = np.arange(N, dtype=np.float64) * float(L) / float(N)
    columns = [np.ones(N, dtype=np.float64)]
    max_frequency = min(K - 1, N // 2)
    for frequency in range(1, max_frequency + 1):
        angle = 2.0 * math.pi * float(frequency) * x / float(L)
        columns.append(np.cos(angle))
        if not (N % 2 == 0 and frequency == N // 2):
            columns.append(np.sin(angle))
    Phi_c = np.stack(columns, axis=1)
    M = diags(
        [np.full(N, float(L) / float(N), dtype=np.float64)],
        offsets=[0],
        format="csr",
    )
    return mass_orthonormalize_1d(Phi_c, M)


def _requested_basis_columns(K: int, boundary_condition="neumann") -> int:
    K = max(1, int(K))
    boundary_condition = _normalize_boundary_condition(boundary_condition)
    return K if boundary_condition == "neumann" else 2 * K - 1


def _effective_modes_1d(N: int, K: int, boundary_condition="neumann") -> int:
    return max(1, min(int(N), _requested_basis_columns(K, boundary_condition)))


def _load_or_build_laplace_basis(
    Nx: int, Ny: int, Kx: int, Ky: int, cache_dir: str,
    device=DEVICE, boundary_condition="neumann",
):
    Nx = int(Nx)
    Ny = int(Ny)
    boundary_condition = _normalize_boundary_condition(boundary_condition)
    kx_eff = _effective_modes_1d(Nx, Kx, boundary_condition)
    ky_eff = _effective_modes_1d(Ny, Ky, boundary_condition)
    os.makedirs(cache_dir, exist_ok=True)
    basis_cache = os.path.join(
        cache_dir,
        f"leno_basis_sep_{boundary_condition}_Nx{Nx}_Ny{Ny}_Kx{kx_eff}_Ky{ky_eff}.pt",
    )
    if os.path.exists(basis_cache):
        ck = safe_torch_load(basis_cache, map_location="cpu")
        ok = (
            int(ck["Nx"]) == Nx and int(ck["Ny"]) == Ny
            and int(ck["Kx"]) == kx_eff and int(ck["Ky"]) == ky_eff
            and ck.get("boundary_condition") == boundary_condition
        )
        if not ok:
            raise ValueError("basis cache mismatch: change the cache directory or basis settings")
        Phix = ck["Phix"].to(device)
        Px   = ck["Px"].to(device)
        Phiy = ck["Phiy"].to(device)
        Py   = ck["Py"].to(device)
    else:
        if boundary_condition == "periodic":
            builder = build_periodic_basis_1d_mass_orthonormal
            build_kx, build_ky = int(Kx), int(Ky)
        else:
            builder = build_neumann_basis_1d_mass_orthonormal
            build_kx, build_ky = kx_eff, ky_eff
        Phix_np, Px_np = builder(Nx, build_kx, L=1.0)
        Phiy_np, Py_np = builder(Ny, build_ky, L=1.0)
        if Phix_np.shape[1] != kx_eff or Phiy_np.shape[1] != ky_eff:
            raise RuntimeError("constructed Laplace basis has an unexpected mode count")
        Phix = torch.from_numpy(Phix_np).to(device)
        Px   = torch.from_numpy(Px_np).to(device)
        Phiy = torch.from_numpy(Phiy_np).to(device)
        Py   = torch.from_numpy(Py_np).to(device)
        torch.save(
            {
                "Nx": Nx, "Ny": Ny, "Kx": kx_eff, "Ky": ky_eff,
                "boundary_condition": boundary_condition,
                "Phix": Phix.cpu(), "Px": Px.cpu(),
                "Phiy": Phiy.cpu(), "Py": Py.cpu(),
            },
            basis_cache
        )
    return Phix, Px, Phiy, Py, kx_eff, ky_eff



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


def _basis_reconstruction_loader(dataset, args):
    """Visit every selected transition once, without dropping the last batch."""
    options = {"num_workers": 0, "generator": torch.Generator().manual_seed(0)}
    if isinstance(dataset, MultiResolutionSubsetDataset):
        sampler = MultiResolutionShardBatchSampler(
            dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False, seed=0,
        )
        return DataLoader(dataset, batch_sampler=sampler, **options)
    if isinstance(dataset, AlignedMixDataset):
        sampler = ShardBatchSampler(
            dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False, seed=0,
        )
        return DataLoader(dataset, batch_sampler=sampler, **options)
    return DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, drop_last=False, **options)


def _reconstruction_axis_basis(phi, mass_coeffs):
    """Mass-orthonormalize the configured column span in float64."""
    phi = phi.to(dtype=torch.float64)
    main, left, right = mass_coeffs
    padded = F.pad(phi, (0, 0, 1, 1))
    mphi = main[:, None] * phi + left[:, None] * padded[:-2] + right[:, None] * padded[2:]
    gram = phi.T @ mphi
    factor = torch.linalg.cholesky((gram + gram.T) / 2)
    return (torch.linalg.solve_triangular(factor, phi.T, upper=False).T,
            torch.linalg.solve_triangular(factor, mphi.T, upper=False).T)


@torch.no_grad()
def evaluate_basis_reconstruction(datasets, resolution_ops, args):
    """Evaluate raw u/v teacher fields; sort configured tensor modes by physical mu.

    The search covers every prefix of the configured Kx_eff*Ky_eff modes. It
    neither builds missing higher modes nor changes the model's rectangular mask.
    """
    tolerance = float(args.reconstruction_tol)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("reconstruction_tol must be finite and positive")
    stats, cache = {}, {}
    for split, dataset in datasets.items():
        seen = 0
        for batch in _basis_reconstruction_loader(dataset, args):
            xb, yb, _, _ = _unpack_training_batch(batch)
            ny, nx = int(yb.shape[1]), int(yb.shape[2])
            key = (ny, nx)
            if key not in cache:
                phix, _, phiy, _ = resolution_ops.get_basis(ny, nx)
                device = phix.device
                mass = {
                    "mx": _build_mass_coeffs_1d_torch(nx, device=device, dtype=torch.float64,
                                                     boundary_condition=resolution_ops.basis_boundary),
                    "my": _build_mass_coeffs_1d_torch(ny, device=device, dtype=torch.float64,
                                                     boundary_condition=resolution_ops.basis_boundary),
                }
                phix, px = _reconstruction_axis_basis(phix, mass["mx"])
                phiy, py = _reconstruction_axis_basis(phiy, mass["my"])
                mu = _build_mode_mu_2d(
                    phix.shape[1], phiy.shape[1], Lx=args.Lx, Ly=args.Ly,
                    device=device, dtype=torch.float64, boundary_condition=resolution_ops.basis_boundary,
                )
                order = torch.argsort(mu.flatten(), stable=True)
                cache[key] = (phix, px, phiy, py, mass, order)
            phix, px, phiy, py, mass, order = cache[key]
            for field, values in (("input", xb), ("target", yb)):
                values = values[..., :2].to(device=phix.device, dtype=torch.float64).permute(0, 3, 1, 2)
                coeff = torch.einsum("bcyk,yl->bclk", values @ px, py)
                reconstructed = torch.einsum("bclk,yl->bcyk", coeff, phiy) @ phix.T
                residual_energy = _mass_quadratic_2d(values - reconstructed, mass).clamp_min(0)
                target_energy = _mass_quadratic_2d(values, mass).clamp_min(0)
                coeff = coeff.flatten(2)[..., order]
                relative = _basis_prefix_relative_errors(coeff, residual_energy, target_energy)
                combined = _basis_prefix_relative_errors(
                    coeff.square().sum(1, keepdim=True).sqrt(),
                    residual_energy.sum(1, keepdim=True), target_energy.sum(1, keepdim=True),
                )
                _update_basis_reconstruction_stats(
                    stats, (split, field, f"{ny}x{nx}"), torch.cat([relative, combined], dim=1),
                )
            seen += len(yb)
            if seen == len(yb) or seen // max(1, int(args.batch_size)) % 100 == 0:
                print(f"basis reconstruction | {split}: {seen}/{len(dataset)} transitions", flush=True)
    return save_basis_reconstruction_report(
        stats, os.path.join(args.save_dir, args.out_prefix), tolerance,
    )


class ResolutionOpsCache:
    def __init__(
        self, Kx: int, Ky: int, basis_cache_dir: str, device=DEVICE,
        basis_boundary="neumann",
    ):
        self.Kx = int(Kx)
        self.Ky = int(Ky)
        self.basis_cache_dir = str(basis_cache_dir)
        self.device = device
        self.basis_boundary = _normalize_boundary_condition(basis_boundary)
        self.mass_cache = {}
        self.basis_cache = {}
        self.current_basis_key = None

    def get_mass(self, Ny: int, Nx: int):
        key = (int(Ny), int(Nx))
        if key not in self.mass_cache:
            ny, nx = key
            self.mass_cache[key] = {
                "mx": _build_mass_coeffs_1d_torch(
                    nx, L=1.0, device=self.device, dtype=torch.float32,
                    boundary_condition=self.basis_boundary,
                ),
                "my": _build_mass_coeffs_1d_torch(
                    ny, L=1.0, device=self.device, dtype=torch.float32,
                    boundary_condition=self.basis_boundary,
                ),
            }
        return self.mass_cache[key]

    def get_basis(self, Ny: int, Nx: int):
        key = (int(Ny), int(Nx))
        if key not in self.basis_cache:
            ny, nx = key
            self.basis_cache[key] = _load_or_build_laplace_basis(
                nx, ny, self.Kx, self.Ky, self.basis_cache_dir, device=self.device,
                boundary_condition=self.basis_boundary,
            )[:4]
        return self.basis_cache[key]

    def prepare_model(self, model, Ny: int, Nx: int):
        if model is None or not bool(getattr(model, "requires_basis", False)):
            return
        key = (int(Ny), int(Nx))
        if self.current_basis_key == key:
            return
        model.set_basis(*self.get_basis(int(Ny), int(Nx)))
        self.current_basis_key = key


# -------------------------
# LENO model
# -------------------------
def _extract_scalar_channel(x, channel_idx, name, required=False):
    idx = int(channel_idx)
    if idx < 0:
        if required:
            raise ValueError(f"{name} channel is required but channel_idx={idx}")
        return None
    if idx >= x.shape[-1]:
        raise ValueError(f"{name} channel_idx={idx} is out of range for input with Cin={x.shape[-1]}")
    return x[..., idx].mean(dim=(1, 2))


def _basis_mode_frequencies_1d(mode_count, boundary_condition, device=None, dtype=torch.float32):
    indices = torch.arange(int(mode_count), dtype=dtype, device=device)
    if _normalize_boundary_condition(boundary_condition) == "periodic":
        return torch.floor((indices + 1.0) / 2.0)
    return indices


def _build_mode_mu_2d(
    Kx, Ky, Lx=2.5, Ly=2.5, device=None, dtype=torch.float32,
    boundary_condition="neumann",
):
    if float(Lx) <= 0.0 or float(Ly) <= 0.0:
        raise ValueError("Lx and Ly must be positive")
    boundary_condition = _normalize_boundary_condition(boundary_condition)
    kx = _basis_mode_frequencies_1d(Kx, boundary_condition, device=device, dtype=dtype)
    ky = _basis_mode_frequencies_1d(Ky, boundary_condition, device=device, dtype=dtype)
    wave_number = 2.0 * math.pi if boundary_condition == "periodic" else math.pi
    return (
        (wave_number * ky[:, None] / float(Ly)) ** 2
        + (wave_number * kx[None, :] / float(Lx)) ** 2
    )


class ModeTimeScaling2d(nn.Module):
    def __init__(
        self, channels, Kx, Ky, Lx=2.5, Ly=2.5,
        d1_init=2.0e-5, d2_init=1.0e-5, boundary_condition="neumann",
    ):
        super().__init__()
        self.to_state = nn.Conv2d(int(channels), 2, kernel_size=1, bias=False)
        self.from_state = nn.Conv2d(2, int(channels), kernel_size=1, bias=False)
        d_init = torch.tensor([float(d1_init), float(d2_init)], dtype=torch.float32)
        if not torch.isfinite(d_init).all() or torch.any(d_init <= 0.0):
            raise ValueError("spectral d1/d2 initial values must be finite and positive")
        self.d_raw = nn.Parameter(self._inverse_softplus(d_init))
        self.lambda_raw = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.register_buffer(
            "mode_mu",
            _build_mode_mu_2d(
                Kx, Ky, Lx=Lx, Ly=Ly, dtype=torch.float32,
                boundary_condition=boundary_condition,
            ),
            persistent=False,
        )

    @staticmethod
    def _inverse_softplus(value, eps=1.0e-12):
        value = torch.clamp(value.to(dtype=torch.float32), min=float(eps))
        return value + torch.log(-torch.expm1(-value))

    def learned_diffusion(self):
        return F.softplus(self.d_raw.float())

    def forward(self, coeff, dt_scalar):
        # coeff: (B,C,Ky,Kx), dt_scalar: physical time increment (B,)
        z = self.to_state(coeff)
        ky, kx = int(coeff.shape[-2]), int(coeff.shape[-1])
        mu = self.mode_mu[:ky, :kx].unsqueeze(0).unsqueeze(0)                  # (1,1,Ky,Kx)
        lam = F.softplus(self.lambda_raw).view(1, 2, 1, 1)                     # (1,2,1,1)
        d = self.learned_diffusion().to(device=coeff.device).view(1, 2, 1, 1)  # (1,2,1,1)
        dt = dt_scalar.view(-1, 1, 1, 1)                                        # (B,1,1,1)
        scale = torch.exp(-dt * (d * mu + lam))                                 # (B,2,Ky,Kx)
        return coeff + self.from_state(z * scale)


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


class SpectralConvSeparable(nn.Module):
    def __init__(
        self, channels, Kx, Ky, Phix, Px, Phiy, Py,
        use_layer_scaling=False, Lx=2.5, Ly=2.5,
        d1_init=2.0e-5, d2_init=1.0e-5, basis_boundary="neumann",
    ):
        super().__init__()
        self.channels = int(channels)
        self.basis_boundary = _normalize_boundary_condition(basis_boundary)
        self.Kx = _requested_basis_columns(Kx, self.basis_boundary)
        self.Ky = _requested_basis_columns(Ky, self.basis_boundary)
        self.use_layer_scaling = bool(use_layer_scaling)
        self.register_buffer("Phix", Phix, persistent=False)  # (Nx,Kx_eff)
        self.register_buffer("Px",   Px,   persistent=False)  # (Nx,Kx_eff)
        self.register_buffer("Phiy", Phiy, persistent=False)  # (Ny,Ky_eff)
        self.register_buffer("Py",   Py,   persistent=False)  # (Ny,Ky_eff)

        w = torch.empty(self.channels, self.channels, self.Ky, self.Kx, dtype=torch.float32)
        nn.init.xavier_uniform_(w)
        self.weight = nn.Parameter(w)
        self.scaler = (
            ModeTimeScaling2d(
                self.channels, self.Kx, self.Ky, Lx=Lx, Ly=Ly,
                d1_init=d1_init, d2_init=d2_init,
                boundary_condition=self.basis_boundary,
            )
            if self.use_layer_scaling else None
        )

    def set_basis(self, Phix, Px, Phiy, Py):
        self.Phix = Phix
        self.Px = Px
        self.Phiy = Phiy
        self.Py = Py

    def forward(self, x, dt_scalar=None):
        # x: (B,C,Ny,Nx)
        _, _, ny, nx = x.shape
        if self.Px is None or self.Py is None or self.Phix is None or self.Phiy is None:
            raise RuntimeError("Laplace basis is not set")
        if int(self.Px.shape[0]) != int(nx) or int(self.Py.shape[0]) != int(ny):
            raise RuntimeError(
                f"basis resolution mismatch: input Ny,Nx=({int(ny)},{int(nx)}) "
                f"but basis Ny,Nx=({int(self.Py.shape[0])},{int(self.Px.shape[0])})"
            )
        modes_x = min(int(self.Kx), int(self.Px.shape[1]), int(self.Phix.shape[1]))
        modes_y = min(int(self.Ky), int(self.Py.shape[1]), int(self.Phiy.shape[1]))
        Px = self.Px[:, :modes_x]
        Py = self.Py[:, :modes_y]
        Phix = self.Phix[:, :modes_x]
        Phiy = self.Phiy[:, :modes_y]
        w = self.weight[:, :, :modes_y, :modes_x]
        tmp   = torch.einsum("bcyx,xk->bcyk", x, Px)                # (B,C,Ny,Kx_eff)
        coeff = torch.einsum("bcyk,yl->bclk", tmp, Py)              # (B,C,Ky_eff,Kx_eff)
        outc  = torch.einsum("bclk,oclk->bolk", coeff, w)           # (B,C,Ky_eff,Kx_eff)
        if self.scaler is not None:
            if dt_scalar is None:
                raise ValueError("Layer scaling requires physical dt_scalar")
            outc = self.scaler(outc, dt_scalar)
        tmp2  = torch.einsum("bolk,yl->boyk", outc, Phiy)           # (B,C,Ny,Kx_eff)
        y     = torch.einsum("boyk,xk->boyx", tmp2, Phix)           # (B,C,Ny,Nx)
        return y


class LENO2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        depth,
        Kx,
        Ky,
        Phix,
        Px,
        Phiy,
        Py,
        pad=0,
        operator_type="laplace",
        basis_boundary="neumann",
        use_layer_scaling=False,
        time_channel_idx=-1,
        dt_channel_idx=-1,
        d1_channel_idx=5,
        d2_channel_idx=6,
        time_channel_scale=250.0,
        dt_scale=250.0,
        Lx=2.5,
        Ly=2.5,
        spectral_d1_init=2.0e-5,
        spectral_d2_init=1.0e-5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.depth = int(depth)
        self.pad = int(pad)
        self.operator_type = str(operator_type).lower()
        self.basis_boundary = _normalize_boundary_condition(basis_boundary)
        self.use_layer_scaling = bool(use_layer_scaling)
        self.time_channel_idx = int(time_channel_idx)
        self.dt_channel_idx = int(dt_channel_idx)
        self.d1_channel_idx = int(d1_channel_idx)
        self.d2_channel_idx = int(d2_channel_idx)
        self.time_channel_scale = float(time_channel_scale)
        self.dt_scale = float(dt_scale)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.spectral_d1_init = float(spectral_d1_init)
        self.spectral_d2_init = float(spectral_d2_init)
        self.requires_basis = (self.operator_type == "laplace")

        if self.operator_type not in {"laplace", "fft"}:
            raise ValueError("operator_type must be 'laplace' or 'fft'")
        if self.use_layer_scaling and self.operator_type != "laplace":
            raise ValueError("use_layer_scaling=1 is only supported with operator_type='laplace'")

        self.fc0 = nn.Linear(self.in_channels, self.width)
        if self.operator_type == "laplace":
            self.spec = nn.ModuleList([
                SpectralConvSeparable(
                    self.width, Kx, Ky, Phix, Px, Phiy, Py,
                    use_layer_scaling=self.use_layer_scaling,
                    basis_boundary=self.basis_boundary,
                    Lx=self.Lx, Ly=self.Ly,
                    d1_init=self.spectral_d1_init,
                    d2_init=self.spectral_d2_init,
                )
                for _ in range(self.depth)
            ])
        else:
            self.spec = nn.ModuleList([SpectralConvFFT2d(self.width, Kx, Ky) for _ in range(self.depth)])
        self.w1   = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.depth)])
        self.act  = nn.GELU()
        self.fc1  = nn.Linear(self.width, self.width)
        self.fc2  = nn.Linear(self.width, self.out_channels)

    def set_basis(self, Phix, Px, Phiy, Py):
        if not self.requires_basis:
            return
        for sp in self.spec:
            sp.set_basis(Phix, Px, Phiy, Py)

    @staticmethod
    def _validate_explicit_scalar(value, x, name):
        if value is None:
            return None
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=x.dtype, device=x.device)
        value = value.to(device=x.device, dtype=x.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.expand(int(x.shape[0]))
        if value.numel() != int(x.shape[0]):
            raise ValueError(f"{name} must be scalar or batch-sized; got {value.numel()} for B={x.shape[0]}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return value

    def _extract_layer_scalars(self, x, dt_scalar=None):
        if not self.use_layer_scaling:
            return None, None, None

        if dt_scalar is None:
            dt_idx = _resolve_dt_channel_idx(self.time_channel_idx, self.dt_channel_idx)
            dt_scalar = _extract_scalar_channel(x, dt_idx, "dt", required=True)
            dt_scalar = dt_scalar * float(self.dt_scale)
        else:
            dt_scalar = self._validate_explicit_scalar(dt_scalar, x, "dt_scalar")
        dt_scalar = self._validate_explicit_scalar(dt_scalar, x, "dt_scalar")
        return dt_scalar, None, None

    def forward(self, x, dt_scalar=None):
        # x: (B,Ny,Nx,Cin)
        layer_dt, _, _ = self._extract_layer_scalars(x, dt_scalar=dt_scalar)
        x = self.fc0(x)                 # (B,Ny,Nx,width)
        x = x.permute(0, 3, 1, 2)       # (B,width,Ny,Nx)

        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")

        for sp, w in zip(self.spec, self.w1):
            x = self.act(sp(x, dt_scalar=layer_dt) + w(x))

        if self.pad > 0:
            x = x[:, :, self.pad:-self.pad, self.pad:-self.pad]

        x = x.permute(0, 2, 3, 1)       # (B,Ny,Nx,width)
        x = self.act(self.fc1(x))
        x = self.fc2(x)                 # (B,Ny,Nx,2)
        return x


def build_xy_channels_normalized(x, y):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x0, x1 = float(x[0]), float(x[-1])
    y0, y1 = float(y[0]), float(y[-1])
    dx = (x1 - x0) if (x1 - x0) != 0.0 else 1.0
    dy = (y1 - y0) if (y1 - y0) != 0.0 else 1.0
    xn = (x - x0) / dx
    yn = (y - y0) / dy
    Xg, Yg = np.meshgrid(xn, yn, indexing="xy")
    return Xg[..., None].astype(np.float32), Yg[..., None].astype(np.float32)


def make_input_features(u2d, v2d, x_chan, y_chan, dt, d1, d2, s, dt_scale=250.0, t_now=None, time_scale=250.0):
    Ny, Nx = u2d.shape
    dt_norm = float(dt) / float(dt_scale)
    dt_ch = np.full((Ny, Nx, 1), dt_norm, dtype=np.float32)
    d1_ch = np.full((Ny, Nx, 1), float(d1), dtype=np.float32)
    d2_ch = np.full((Ny, Nx, 1), float(d2), dtype=np.float32)
    s_ch  = np.full((Ny, Nx, 1), float(s),  dtype=np.float32)

    # canonical layout: u,v,x_norm,y_norm,(optional) t_norm,dt_norm,d1,d2,s
    feats = [
        np.stack([u2d, v2d], axis=-1).astype(np.float32),
        x_chan, y_chan,
    ]
    if t_now is not None:
        t_ch = np.full((Ny, Nx, 1), float(t_now) / float(time_scale), dtype=np.float32)
        feats.append(t_ch)
    feats.extend([dt_ch, d1_ch, d2_ch, s_ch])
    X = np.concatenate(feats, axis=-1)
    return X


def make_input_cin8(u2d, v2d, x_chan, y_chan, dt, d1, d2, s, dt_scale=250.0):
    return make_input_features(
        u2d, v2d,
        x_chan, y_chan,
        dt=dt, d1=d1, d2=d2, s=s,
        dt_scale=dt_scale,
        t_now=None,
        time_scale=dt_scale,
    )


# -------------------------
# train/eval
# -------------------------
_BASIS_STATE_SUFFIXES = (".Phix", ".Px", ".Phiy", ".Py", ".mode_mu")


def _without_basis_buffers(state):
    return {
        k: v for k, v in state.items()
        if not any(str(k).endswith(suffix) for suffix in _BASIS_STATE_SUFFIXES)
    }


def load_model_state_ignoring_basis(model, state):
    current_state = model.state_dict()
    if "fc0.weight" in state and "fc0.weight" in current_state:
        saved_shape = tuple(state["fc0.weight"].shape)
        current_shape = tuple(current_state["fc0.weight"].shape)
        if saved_shape != current_shape:
            raise ValueError(
                "saved best input dimension is incompatible with this model: "
                f"saved fc0.weight={saved_shape}, current={current_shape}. "
                "Do not truncate or pad learned weights; start a fresh run with --resume 0 "
                "or use a new --out_prefix."
            )
    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        filtered = _without_basis_buffers(state)
        incompatible = model.load_state_dict(filtered, strict=False)
        unexpected = [
            k for k in incompatible.unexpected_keys
            if not any(str(k).endswith(suffix) for suffix in _BASIS_STATE_SUFFIXES)
        ]
        if incompatible.missing_keys or unexpected:
            raise
        print("loaded model parameters; ignored cached basis buffers from saved best", flush=True)


def _batch_ny_nx(x):
    return int(x.shape[1]), int(x.shape[2])


def _unpack_training_batch(batch):
    if not isinstance(batch, (list, tuple)) or len(batch) != 3:
        raise ValueError("dataset batches must be (x, y, time_or_metadata)")
    xb, yb, third = batch
    if isinstance(third, dict):
        metadata = third
        if "t_cumulative" not in metadata:
            raise ValueError("teacher batch metadata is missing t_cumulative")
        t_cumulative = metadata["t_cumulative"]
    else:
        metadata = None
        t_cumulative = third
    return xb, yb, t_cumulative, metadata


def _metadata_to_device(metadata, device):
    if metadata is None:
        return None
    out = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            dtype = torch.float32 if torch.is_floating_point(value) else value.dtype
            out[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            out[key] = value
    return out


def _model_forward_batch(model, xb, metadata=None):
    if metadata is None or not bool(getattr(model, "use_layer_scaling", False)):
        return model(xb)
    if "dt_physical" not in metadata:
        raise ValueError("layer scaling teacher batch is missing physical dt metadata")
    dt_physical = metadata["dt_physical"]
    if not torch.is_tensor(dt_physical) or not torch.isfinite(dt_physical).all():
        raise ValueError("layer scaling requires finite teacher dt_physical values")
    return model(xb, dt_scalar=dt_physical)


@torch.no_grad()
def eval_epoch(model, loader, resolution_ops, amp=False, amp_dtype=torch.float16):
    model.eval()
    amp_enabled = bool(amp) and DEVICE.type == "cuda"
    amp_torch_dtype = amp_dtype if isinstance(amp_dtype, torch.dtype) else _resolve_amp_dtype(amp_dtype)
    u_sum, v_sum, n_sum = 0.0, 0.0, 0
    for batch in loader:
        xb, yb, _t_cumulative_b, metadata = _unpack_training_batch(batch)
        xb = xb.to(DEVICE, non_blocking=True).float()
        yb = yb.to(DEVICE, non_blocking=True).float()
        metadata = _metadata_to_device(metadata, DEVICE)
        Ny, Nx = _batch_ny_nx(xb)
        resolution_ops.prepare_model(model, Ny, Nx)
        mass = resolution_ops.get_mass(Ny, Nx)
        with _cuda_autocast(amp_enabled, amp_torch_dtype):
            yh = _model_forward_batch(model, xb, metadata)

        loss_each = mass_rel_l2_2d_channels_per_sample(yh.float(), yb.float(), mass)
        bs = int(loss_each.shape[0])
        u_sum += float(loss_each[:, 0].sum().item())
        v_sum += float(loss_each[:, 1].sum().item())
        n_sum += bs

    if n_sum == 0:
        return float("nan"), float("nan")
    return float(u_sum / n_sum), float(v_sum / n_sum)


def _full_time_error_energies(pred, target, Lx, Ly, boundary):
    """Per-sample/channel spatial integrals; inputs are (B, Ny, Nx, C)."""
    pred, target = pred.double(), target.double()
    ny, nx = pred.shape[1:3]
    if min(nx, ny) < 3 or min(Lx, Ly) <= 0:
        raise ValueError("Full-time boundary evaluation needs >=3 grid points and positive domain lengths")
    mass = {
        "mx": _build_mass_coeffs_1d_torch(nx, Lx, pred.device, pred.dtype, boundary),
        "my": _build_mass_coeffs_1d_torch(ny, Ly, pred.device, pred.dtype, boundary),
    }
    def volume(field):
        return _mass_quadratic_2d(field.permute(0, 3, 1, 2), mass).clamp_min(0)

    def normals(field):
        if boundary == "periodic":
            # Endpoint-excluded grid: both faces use the derivative at the seam.
            dx, dy = Lx / nx, Ly / ny
            gx = (field[:, :, 1] - field[:, :, -1]) / (2 * dx)
            gy = (field[:, 1] - field[:, -1]) / (2 * dy)
            return (-gx, gx, -gy, gy)
        dx, dy = Lx / (nx - 1), Ly / (ny - 1)
        return (
            (3 * field[:, :, 0] - 4 * field[:, :, 1] + field[:, :, 2]) / (2 * dx),
            (3 * field[:, :, -1] - 4 * field[:, :, -2] + field[:, :, -3]) / (2 * dx),
            (3 * field[:, 0] - 4 * field[:, 1] + field[:, 2]) / (2 * dy),
            (3 * field[:, -1] - 4 * field[:, -2] + field[:, -3]) / (2 * dy),
        )

    def surface(fields):
        total = pred.new_zeros((pred.shape[0], pred.shape[-1]))
        for field, length in zip(fields, (Ly, Ly, Lx, Lx)):
            squared = field.square()
            if boundary == "periodic":
                total += squared.mean(dim=1) * length
            else:
                total += (squared.sum(dim=1) - 0.5 * (squared[:, 0] + squared[:, -1])) * (length / (field.shape[1] - 1))
        return total

    pn, tn = normals(pred), normals(target)
    return torch.stack((volume(pred - target), volume(target),
                        surface(tuple(p - t for p, t in zip(pn, tn))), surface(pn), surface(tn)), dim=1)


_FULL_TIME_METRIC_NAMES = [
    "l2_space_time", "relative_l2_space_time", "normal_derivative_rms",
    "neumann_zero_flux_pred_rms", "neumann_zero_flux_teacher_rms",
]


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


@torch.no_grad()
def evaluate_full_time_test(model, loader, resolution_ops, args, out_prefix):
    """Equal-weight trajectory mean/std of physical-time integrated one-step errors."""
    model.eval()
    entries = _dataset_boundary_entries(loader.dataset)
    boundary = entries[0][0] if entries else args.basis_boundary
    # Reuse the test batch sampler but read in this process so persistent training
    # workers never retain the temporary evaluation metadata setting.
    bases = ([] if isinstance(loader.dataset, (TeacherTransitionDataset, TeacherTransitionSubset))
             else list(_iter_unique_sharded_bases(loader.dataset)))
    previous = [(base, getattr(base, "full_time_evaluation_metadata", False),
                 getattr(base, "full_time_evaluation_source_id", 0)) for base in bases]
    try:
        for source_id, base in enumerate(bases):
            split_name, split_ids = base.split_trajectory_ids()
            if split_name is not None and split_name != "test":
                raise ValueError(f"Full-time test evaluation requires test data: {base.index_path}")
            if split_ids is not None and not np.isin(base.evaluation_trajectory_ids, split_ids).all():
                raise ValueError("Full-time test evaluation contains sample IDs outside the test split")
            base.full_time_evaluation_metadata = True
            base.full_time_evaluation_source_id = source_id
        evaluation_loader = DataLoader(loader.dataset, batch_sampler=loader.batch_sampler,
                                       num_workers=0, collate_fn=loader.collate_fn)
        trajectories = _collect_full_time_test_errors(model, evaluation_loader, resolution_ops, args, boundary)
    finally:
        for base, enabled, source_id in previous:
            base.full_time_evaluation_metadata = enabled
            base.full_time_evaluation_source_id = source_id
    _save_trajectory_error_statistics(trajectories, out_prefix + "_test", boundary)


def _collect_full_time_test_errors(model, loader, resolution_ops, args, boundary):
    trajectories = {}
    for batch in loader:
        xb, yb, _, metadata = _unpack_training_batch(batch)
        if metadata is None or "source_id" not in metadata or "sample_id" not in metadata:
            raise ValueError("Trajectory statistics require source_id and sample_id metadata")
        keys = list(zip(metadata["source_id"].tolist(), metadata["sample_id"].tolist()))
        xb, yb = xb.to(DEVICE).float(), yb.to(DEVICE).float()
        metadata = _metadata_to_device(metadata, DEVICE)
        dt = metadata["dt_physical"].double().reshape(-1)
        if not torch.isfinite(dt).all() or (dt < 0).any():
            raise ValueError("Full-time evaluation requires finite nonnegative physical dt")
        resolution_ops.prepare_model(model, *_batch_ny_nx(xb))
        pred = _model_forward_batch(model, xb, metadata)
        energy = _full_time_error_energies(pred, yb, args.Lx, args.Ly, boundary)
        weighted = (energy * dt[:, None, None]).detach().cpu().numpy()
        for key, contribution, step_dt in zip(keys, weighted, dt.cpu().tolist()):
            item = trajectories.setdefault(key, {"energies": np.zeros((5, 2)), "duration": 0.0,
                                                 "steps": 0, "boundary_length": 2 * (args.Lx + args.Ly)})
            item["energies"] += contribution
            item["duration"] += step_dt
            item["steps"] += 1
    return trajectories


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


def save_rollout_full_time_errors(predicted, target, times, Lx, Ly, boundary, out_prefix):
    """Integrate all rollout snapshots by the trapezoidal rule on the physical time grid."""
    times = np.asarray(times, dtype=np.float64)
    dt = np.diff(times)
    if not np.all(np.isfinite(times)) or len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("Rollout times must be finite and strictly increasing")
    weights = np.zeros_like(times)
    weights[:-1] += dt / 2
    weights[1:] += dt / 2
    totals = np.zeros((5, 2), dtype=np.float64)
    for k, weight in enumerate(weights):
        energy = _full_time_error_energies(torch.as_tensor(predicted[k]).unsqueeze(0),
                                         torch.as_tensor(target[k]).unsqueeze(0), Lx, Ly, boundary)
        totals += weight * energy[0].numpy()
    _save_full_time_errors(totals, times[-1] - times[0], out_prefix, boundary,
                           "rollout_trapezoidal", boundary_length=2 * (Lx + Ly))


def _resolve_dt_channel_idx(time_channel_idx, dt_channel_idx):
    dt_idx = int(dt_channel_idx)
    if dt_idx >= 0:
        return dt_idx
    if int(time_channel_idx) >= 0:
        return int(time_channel_idx) + 1
    return 4


def validate_channel_indices(args, Cin, explicit_parameters=False):
    Cin = int(Cin)
    time_idx = int(args.time_channel_idx)
    dt_idx = int(_resolve_dt_channel_idx(time_idx, int(args.dt_channel_idx)))
    d1_idx = int(args.d1_channel_idx)
    d2_idx = int(args.d2_channel_idx)

    def require_channel(name, idx):
        idx = int(idx)
        if idx < 0 or idx >= Cin:
            raise ValueError(f"{name} channel_idx={idx} is out of range for dataset Cin={Cin}")

    if time_idx >= 0:
        require_channel("current time", time_idx)
        if Cin < 9:
            raise ValueError(
                "time_channel_idx was set, but the loaded dataset has the 8-channel layout. "
                "For dt-only scaling use --time_channel_idx -1 --dt_channel_idx 4 "
                "--d1_channel_idx 5 --d2_channel_idx 6."
            )

    if int(args.use_layer_scaling) == 1 or float(args.loss_time_beta) > 0.0:
        require_channel("dt", dt_idx)

    # d1/d2 are learned positive spectral parameters. Legacy d1/d2 channels may
    # still exist in old inputs, but they are not required as per-sample features.


def _model_channel_config(model, time_channel_idx, dt_channel_idx, dt_scale):
    time_idx = int(time_channel_idx)
    dt_idx = int(_resolve_dt_channel_idx(time_idx, int(dt_channel_idx)))
    return {
        "in_channels": int(getattr(model, "in_channels", -1)),
        "use_layer_scaling": bool(getattr(model, "use_layer_scaling", False)),
        "time_channel_idx": time_idx,
        "dt_channel_idx": dt_idx,
        "d1_channel_idx": int(getattr(model, "d1_channel_idx", -1)),
        "d2_channel_idx": int(getattr(model, "d2_channel_idx", -1)),
        "basis_boundary": str(getattr(model, "basis_boundary", "neumann")),
        "dt_scale": float(dt_scale),
        "Lx": float(getattr(model, "Lx", 1.0)),
        "Ly": float(getattr(model, "Ly", 1.0)),
    }


def _validate_model_channel_config(ck, expected):
    got = ck.get("channel_config", None) if isinstance(ck, dict) else None
    if got is None:
        if bool(expected.get("use_layer_scaling", False)):
            raise ValueError(
                "saved best model has no channel_config metadata, so it may be from the old "
                "current-time scaling layout. Start a fresh dt-only scaling run with "
                "--resume 0 or use a new --out_prefix."
            )
        return
    keys = ["in_channels", "use_layer_scaling", "time_channel_idx", "dt_channel_idx"]
    mismatches = []
    for key in keys:
        if got.get(key, None) != expected.get(key, None):
            mismatches.append(f"{key}: saved={got.get(key, None)} current={expected.get(key, None)}")
    if abs(float(got.get("dt_scale", expected["dt_scale"])) - float(expected["dt_scale"])) > 1.0e-12:
        mismatches.append(f"dt_scale: saved={got.get('dt_scale', None)} current={expected['dt_scale']}")
    for key in ("Lx", "Ly"):
        if abs(float(got.get(key, expected[key])) - float(expected[key])) > 1.0e-12:
            mismatches.append(f"{key}: saved={got.get(key, None)} current={expected[key]}")
    saved_boundary = _normalize_boundary_condition(got.get("basis_boundary", "neumann"))
    if saved_boundary != expected["basis_boundary"]:
        mismatches.append(
            f"basis_boundary: saved={saved_boundary} current={expected['basis_boundary']}"
        )
    if mismatches:
        raise ValueError("saved best model channel_config mismatch; " + "; ".join(mismatches))


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


def _profile_time():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return time.time()


def save_training_history(out_prefix, hist, loss_time_beta=0.0):
    n_epoch = len(hist["train_u"])
    epoch_axis = np.arange(1, n_epoch + 1, dtype=np.int64)
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


def train_leno(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    weight_decay,
    out_prefix,
    resolution_ops,
    log_every=0,
    loss_time_beta=0.0,
    time_weight_floor=0.0,
    time_channel_idx=-1,
    dt_channel_idx=-1,
    dt_scale=250.0,
    resume=True,
    eval_every=1,
    amp=0,
    amp_dtype="float16",
    lr_schedule="cosine",
    profile_steps=0,
):
    model.to(DEVICE)
    total_epochs = int(epochs)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    lr_schedule = str(lr_schedule).strip().lower()
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    elif lr_schedule in {"step50", "step", "half50"}:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    elif lr_schedule in {"constant", "none", "off"}:
        sched = None
    else:
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")
    n_train_batches = len(train_loader)
    eval_every = int(eval_every)
    amp_enabled = bool(int(amp)) and DEVICE.type == "cuda"
    amp_torch_dtype = _resolve_amp_dtype(amp_dtype)
    has_complex_params = _model_has_complex_parameters(model)
    scaler_enabled = amp_enabled and amp_torch_dtype == torch.float16 and not has_complex_params
    scaler = _make_grad_scaler(scaler_enabled)
    if bool(int(amp)) and not amp_enabled:
        print("AMP requested but CUDA is not available; running in float32.", flush=True)
    if amp_enabled and amp_torch_dtype == torch.float16 and has_complex_params:
        print(
            "NOTE: GradScaler disabled because this model has complex parameters; "
            "PyTorch AMP unscale does not support ComplexFloat gradients.",
            flush=True,
        )
    profile_steps = max(0, int(profile_steps))
    print(
        "runtime options | eval_every={} amp={} amp_dtype={} grad_scaler={} lr_schedule={} profile_steps={}".format(
            eval_every,
            int(amp_enabled),
            str(amp_torch_dtype).replace("torch.", ""),
            int(scaler.is_enabled()),
            lr_schedule,
            profile_steps,
        ),
        flush=True,
    )

    best = 1.0e30
    best_path = out_prefix + "_best.pt"
    hist = {"train_u": [], "train_v": [], "test_u": [], "test_v": []}
    start_epoch = 1
    channel_config = _model_channel_config(model, time_channel_idx, dt_channel_idx, dt_scale)

    if bool(resume) and os.path.exists(best_path):
        best_state = safe_torch_load(best_path, map_location="cpu")
        saved_lr_schedule = str(best_state.get("lr_schedule", lr_schedule)).strip().lower()
        if saved_lr_schedule != lr_schedule:
            raise ValueError(
                "saved best learning-rate schedule mismatch: "
                f"saved={saved_lr_schedule}, current={lr_schedule}. "
                "Use --resume 0 or a new --out_prefix for the step50 schedule."
            )
        if float(loss_time_beta) > 0.0 and (
            best_state.get("time_weight_mode") != TIME_WEIGHT_MODE
            or float(best_state.get("loss_time_beta", 0.0)) <= 0.0
        ):
            raise ValueError(
                "Saved best model does not use cumulative-time training loss weighting. "
                "Start a fresh run when loss_time_beta > 0."
            )
        if "channel_config" in best_state:
            _validate_model_channel_config(best_state, channel_config)
        load_model_state_ignoring_basis(model, best_state["model"])
        if "test_u" in best_state and "test_v" in best_state:
            best = float(best_state["test_u"]) + float(best_state["test_v"])
        else:
            best = float(best_state.get("best", best))
        last_epoch = int(best_state.get("epoch", 0))
        loaded_hist = best_state.get("hist", None)
        if isinstance(loaded_hist, dict):
            for key in hist:
                values = list(loaded_hist.get(key, []))[:last_epoch]
                hist[key] = values + [float("nan")] * max(0, last_epoch - len(values))
        else:
            for key in hist:
                hist[key] = [float("nan")] * last_epoch
            if last_epoch > 0:
                for key in hist:
                    if key in best_state:
                        hist[key][-1] = float(best_state[key])
        start_epoch = last_epoch + 1
        _set_loader_epoch(train_loader, last_epoch)
        _restore_rng_state_for_resume(best_state.get("rng_state", None))
        _restore_scheduler_epoch(sched, last_epoch)
        saved_total_epochs = best_state.get("total_epochs", None)
        if saved_total_epochs is not None and int(saved_total_epochs) != total_epochs:
            print(
                "WARNING: best model was created with total_epochs={} but current --epochs={}".format(
                    int(saved_total_epochs), total_epochs
                ),
                flush=True,
            )
        print(
            "resumed best weights: {} | optimizer/scaler reset | next_epoch={}/{} "
            "| next_lr={:.2e} | best={:.6e}".format(
                best_path, start_epoch, total_epochs, float(opt.param_groups[0]["lr"]), best
            ),
            flush=True,
        )
        del best_state

    for ep in range(start_epoch, total_epochs + 1):
        ep_start = time.time()
        model.train()
        tr_u_sum, tr_v_sum, tr_n = 0.0, 0.0, 0
        detailed_step_log = int(log_every) > 0
        if detailed_step_log:
            print(f"[epoch {ep}/{total_epochs}] start | train_batches={n_train_batches}", flush=True)
            log_runtime_status(f"[epoch {ep}/{total_epochs}] start")

        profile_sums = OrderedDict([
            ("data", 0.0),
            ("h2d", 0.0),
            ("prep", 0.0),
            ("zero_grad", 0.0),
            ("forward_loss", 0.0),
            ("backward", 0.0),
            ("optimizer", 0.0),
            ("bookkeeping", 0.0),
            ("total", 0.0),
        ])
        profile_n = 0
        train_iter = iter(train_loader)
        for it in range(1, n_train_batches + 1):
            profile_this = profile_steps > 0 and it <= profile_steps
            if profile_this:
                t_step0 = _profile_time()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            xb, yb, t_cumulative_b, metadata = _unpack_training_batch(batch)
            if profile_this:
                t_data = _profile_time()

            xb = xb.to(DEVICE, dtype=torch.float32, non_blocking=True)  # (B,Ny,Nx,Cin)
            yb = yb.to(DEVICE, dtype=torch.float32, non_blocking=True)  # (B,Ny,Nx,2)
            t_cumulative_b = t_cumulative_b.to(
                DEVICE, dtype=torch.float32, non_blocking=True
            )
            metadata = _metadata_to_device(metadata, DEVICE)
            if profile_this:
                t_h2d = _profile_time()

            Ny, Nx = _batch_ny_nx(xb)
            resolution_ops.prepare_model(model, Ny, Nx)
            mass = resolution_ops.get_mass(Ny, Nx)
            if profile_this:
                t_prep = _profile_time()

            opt.zero_grad(set_to_none=True)
            if profile_this:
                t_zero = _profile_time()

            with _cuda_autocast(amp_enabled, amp_torch_dtype):
                yh = _model_forward_batch(model, xb, metadata)           # (B,Ny,Nx,2)
                loss_each = mass_rel_l2_2d_channels_per_sample(yh.float(), yb.float(), mass)
                loss_u_each = loss_each[:, 0]
                loss_v_each = loss_each[:, 1]

                loss_weights = batch_time_weights(
                    t_cumulative_b,
                    loss_time_beta=loss_time_beta,
                    time_weight_floor=time_weight_floor,
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
            if profile_this:
                t_forward = _profile_time()

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if profile_this:
                    t_backward = _profile_time()
                scaler.step(opt)
                scaler.update()
                if profile_this:
                    t_optimizer = _profile_time()
            else:
                loss.backward()
                if profile_this:
                    t_backward = _profile_time()
                opt.step()
                if profile_this:
                    t_optimizer = _profile_time()

            tr_u_sum += float(weighted_loss_u_each.detach().sum().item())
            tr_v_sum += float(weighted_loss_v_each.detach().sum().item())
            tr_n += int(weighted_loss_u_each.numel())
            if profile_this:
                t_end = _profile_time()
                profile_sums["data"] += t_data - t_step0
                profile_sums["h2d"] += t_h2d - t_data
                profile_sums["prep"] += t_prep - t_h2d
                profile_sums["zero_grad"] += t_zero - t_prep
                profile_sums["forward_loss"] += t_forward - t_zero
                profile_sums["backward"] += t_backward - t_forward
                profile_sums["optimizer"] += t_optimizer - t_backward
                profile_sums["bookkeeping"] += t_end - t_optimizer
                profile_sums["total"] += t_end - t_step0
                profile_n += 1
                if it == profile_steps or it == n_train_batches:
                    parts = []
                    for key, value in profile_sums.items():
                        parts.append("{}={:.3f}s".format(key, value / max(1, profile_n)))
                    print(
                        "profile first {} train steps avg | {}".format(profile_n, " ".join(parts)),
                        flush=True,
                    )
                    log_runtime_status(f"profile epoch {ep} step {it}")

            if detailed_step_log and ((it == 1) or (it % int(log_every) == 0) or (it == n_train_batches)):
                elapsed = time.time() - ep_start
                mean_u = float(tr_u_sum / tr_n) if tr_n > 0 else float("nan")
                mean_v = float(tr_v_sum / tr_n) if tr_n > 0 else float("nan")
                print(
                    "epoch {:4d} step {:5d}/{:5d} | train u {:.6e} v {:.6e} | elapsed {:.1f}s".format(
                        ep, it, n_train_batches, mean_u, mean_v, elapsed
                    ),
                    flush=True,
                )
                log_runtime_status(f"epoch {ep:4d} step {it:5d}/{n_train_batches:5d}")

        if detailed_step_log:
            log_runtime_status(f"[epoch {ep}/{total_epochs}] pre-eval")
        do_eval = (
            (eval_every <= 0 and ep == total_epochs)
            or (eval_every > 0 and ((ep % eval_every) == 0 or ep == total_epochs))
        )
        if do_eval:
            te_u, te_v = eval_epoch(
                model, test_loader, resolution_ops,
                amp=amp_enabled, amp_dtype=amp_torch_dtype,
            )
        else:
            te_u, te_v = float("nan"), float("nan")
        if sched is not None:
            sched.step()
        current_lr = float(opt.param_groups[0]["lr"])

        tr_u_m = float(tr_u_sum / tr_n) if tr_n > 0 else float("nan")
        tr_v_m = float(tr_v_sum / tr_n) if tr_n > 0 else float("nan")
        hist["train_u"].append(tr_u_m)
        hist["train_v"].append(tr_v_m)
        hist["test_u"].append(te_u)
        hist["test_v"].append(te_v)
        save_training_history(out_prefix, hist, loss_time_beta=loss_time_beta)

        print(
            "epoch {:4d} | train u {:.6e} v {:.6e} | test u {} v {} | next_lr {:.2e} | epoch_time {:.1f}s".format(
                ep,
                tr_u_m,
                tr_v_m,
                "{:.6e}".format(te_u) if do_eval else "skipped",
                "{:.6e}".format(te_v) if do_eval else "skipped",
                current_lr,
                time.time() - ep_start,
            ),
            flush=True,
        )
        if detailed_step_log:
            log_runtime_status(f"[epoch {ep}/{total_epochs}] end")

        score = te_u + te_v
        if do_eval and math.isfinite(score) and score < best:
            candidate_best = float(score)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "total_epochs": total_epochs,
                    "best": candidate_best,
                    "hist": hist,
                    "lr_schedule": lr_schedule,
                    "channel_config": channel_config,
                    "train_u": tr_u_m,
                    "train_v": tr_v_m,
                    "test_u": te_u,
                    "test_v": te_v,
                    "time_weight_mode": TIME_WEIGHT_MODE,
                    "loss_time_beta": float(loss_time_beta),
                    "rng_state": _get_rng_state_for_resume(),
                },
                best_path,
            )
            print("  saved best:", best_path, flush=True)
            best = candidate_best

    if os.path.exists(best_path):
        ck = safe_torch_load(best_path, map_location="cpu")
        if float(loss_time_beta) > 0.0 and (
            ck.get("time_weight_mode") != TIME_WEIGHT_MODE
            or float(ck.get("loss_time_beta", 0.0)) <= 0.0
        ):
            raise ValueError(
                "Saved best model does not use cumulative-time training loss weighting. "
                "Start a fresh run when loss_time_beta > 0."
            )
        load_model_state_ignoring_basis(model, ck["model"])
        print("loaded best:", best_path, "epoch", ck["epoch"])
    else:
        print("best model not found; using current model:", best_path, flush=True)

    print(f"saved training history: {out_prefix}_history.csv/.npz/.png", flush=True)

    return model


# -------------------------
# visualization
# -------------------------
def _mesh_xy(x_1d, y_1d):
    Xg, Yg = np.meshgrid(x_1d.astype(np.float32), y_1d.astype(np.float32), indexing="xy")
    return Xg[..., None], Yg[..., None]


def _pick_time_indices(times_np, t_targets, tol=1e-3):
    idx = []
    for tt in t_targets:
        j = int(np.argmin(np.abs(times_np - float(tt))))
        if float(np.abs(times_np[j] - float(tt))) > float(tol):
            raise ValueError(f"target time {tt} not found (closest={times_np[j]})")
        idx.append(j)
    return idx


@torch.no_grad()
def _plot_gt_pred_err(
    u_gt,
    u_pr,
    v_gt,
    v_pr,
    out_path,
    title_suffix="",
    x_range=None,
    y_range=None,
    field_cmap="jet",
):
    def readable_outward_range(values, name):
        array = np.asarray(values)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError(f"{name} contains no finite values for rollout plotting")
        value_min = float(finite.min())
        value_max = float(finite.max())
        if value_min == value_max:
            pad = max(abs(value_min) * 1.0e-6, 1.0e-12)
            value_min -= pad
            value_max += pad
        span = value_max - value_min
        decimals = int(max(0, min(8, 1 - math.floor(math.log10(span)))))
        factor = float(10 ** decimals)

        def stabilize_scaled(value):
            scaled = value * factor
            nearest = round(scaled)
            if math.isclose(scaled, nearest, rel_tol=0.0, abs_tol=1.0e-6):
                return float(nearest)
            return scaled

        rounded_min = math.floor(stabilize_scaled(value_min)) / factor
        rounded_max = math.ceil(stabilize_scaled(value_max)) / factor
        if rounded_min == rounded_max:
            step = 1.0 / factor
            rounded_min -= step
            rounded_max += step
        return float(rounded_min), float(rounded_max)

    eu = (u_pr - u_gt)
    ev = (v_pr - v_gt)
    if x_range is None:
        x_range = (0.0, float(np.asarray(u_gt).shape[1] - 1))
    if y_range is None:
        y_range = (0.0, float(np.asarray(u_gt).shape[0] - 1))
    x_range = readable_outward_range(x_range, "x coordinates")
    y_range = readable_outward_range(y_range, "y coordinates")
    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    u_gt_range = readable_outward_range(u_gt, "u_true")
    u_pr_range = readable_outward_range(u_pr, "u_pred")
    v_gt_range = readable_outward_range(v_gt, "v_true")
    v_pr_range = readable_outward_range(v_pr, "v_pred")
    eu_range = readable_outward_range(eu, "u_error")
    ev_range = readable_outward_range(ev, "v_error")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

    im = axes[0, 0].imshow(
        u_gt, origin="lower", interpolation="bilinear", extent=extent,
        cmap=field_cmap, vmin=u_gt_range[0], vmax=u_gt_range[1],
    )
    axes[0, 0].set_title("u_true" + title_suffix)
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(
        u_pr, origin="lower", interpolation="bilinear", extent=extent,
        cmap=field_cmap, vmin=u_pr_range[0], vmax=u_pr_range[1],
    )
    axes[0, 1].set_title("u_pred" + title_suffix)
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(
        eu, origin="lower", interpolation="bilinear", extent=extent,
        cmap="RdBu_r", vmin=eu_range[0], vmax=eu_range[1],
    )
    axes[0, 2].set_title("u_error(pred-true)" + title_suffix)
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(
        v_gt, origin="lower", interpolation="bilinear", extent=extent,
        cmap=field_cmap, vmin=v_gt_range[0], vmax=v_gt_range[1],
    )
    axes[1, 0].set_title("v_true" + title_suffix)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(
        v_pr, origin="lower", interpolation="bilinear", extent=extent,
        cmap=field_cmap, vmin=v_pr_range[0], vmax=v_pr_range[1],
    )
    axes[1, 1].set_title("v_pred" + title_suffix)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(
        ev, origin="lower", interpolation="bilinear", extent=extent,
        cmap="RdBu_r", vmin=ev_range[0], vmax=ev_range[1],
    )
    axes[1, 2].set_title("v_error(pred-true)" + title_suffix)
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

    plt.savefig(out_path, dpi=200)
    plt.close()
    print("saved:", out_path)
    return {
        "u_true": u_gt_range,
        "u_pred": u_pr_range,
        "v_true": v_gt_range,
        "v_pred": v_pr_range,
        "u_error": eu_range,
        "v_error": ev_range,
        "x": x_range,
        "y": y_range,
    }


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


def _load_npz_traj(npz_dir, sample_id):
    p1 = os.path.join(npz_dir, f"sample_{sample_id}.npz")
    p2 = os.path.join(npz_dir, f"sample_{sample_id:03d}.npz")
    npz_path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"trajectory not found: {p1} or {p2}")

    d = np.load(npz_path, allow_pickle=False)
    U = d["U"].astype(np.float32)
    V = d["V"].astype(np.float32)
    times_traj = d["times"].astype(np.float32).reshape(-1) if "times" in d.files else None
    d.close()
    return U, V, times_traj


@torch.no_grad()
def visualize_rollout_leno_gm2d(
    model,
    npz_dir,
    sample_id,
    out_prefix,
    Kx,
    Ky,
    T_final=250.0,
    t_snap=(50.0, 250.0),
    use_teacher_forcing_until=0,
    dt_scale=250.0,
    basis_cache_dir="/content/drive/MyDrive/output_data",
):
    meta_path = os.path.join(npz_dir, "meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.npz not found in: " + npz_dir)

    m = np.load(meta_path, allow_pickle=False)
    x = m["x"].astype(np.float32).reshape(-1)
    y = m["y"].astype(np.float32).reshape(-1)
    times_all = m["times"].astype(np.float32).reshape(-1)
    d1 = float(np.asarray(m["d1"]).reshape(-1)[0]) if "d1" in m.files else 0.0
    d2 = float(np.asarray(m["d2"]).reshape(-1)[0]) if "d2" in m.files else 0.0
    s  = float(np.asarray(m["s"]).reshape(-1)[0])  if "s"  in m.files else 0.0
    m.close()

    x_chan, y_chan = build_xy_channels_normalized(x, y)  # (Ny,Nx,1)

    keep_t = make_times_55(T_final=T_final)

    U, V, times_traj = _load_npz_traj(npz_dir, sample_id)
    if times_traj is None:
        times_traj = times_all

    Ny, Nx = int(U.shape[1]), int(U.shape[2])
    basis_boundary = _normalize_boundary_condition(
        getattr(model, "basis_boundary", "neumann")
    )

    # Mass coefficients on [0,1] to match basis orthonormality
    mass_np = (
        _build_mass_coeffs_1d_np(Nx, L=1.0, boundary_condition=basis_boundary),
        _build_mass_coeffs_1d_np(Ny, L=1.0, boundary_condition=basis_boundary),
    )

    needs_basis = bool(model is None or getattr(model, "requires_basis", True))
    if needs_basis:
        Phix, Px, Phiy, Py, _, _ = _load_or_build_laplace_basis(
            Nx, Ny, int(Kx), int(Ky), basis_cache_dir, device=DEVICE,
            boundary_condition=basis_boundary,
        )

        if (model is not None) and hasattr(model, "set_basis"):
            model.set_basis(Phix, Px, Phiy, Py)

    keep_idx = _pick_time_indices(times_traj, keep_t.tolist(), tol=1e-3)

    U_gt = U[keep_idx]
    V_gt = V[keep_idx]

    # --- rollout (auto-regressive in physical space)
    u_pred = [U_gt[0].copy()]
    v_pred = [V_gt[0].copy()]
    use_input_time = bool(model is not None and int(getattr(model, "time_channel_idx", -1)) >= 0)
    time_scale = float(getattr(model, "time_channel_scale", dt_scale))
    expected_cin = int(getattr(model, "in_channels", -1)) if model is not None else -1

    model.eval()
    for n in range(len(keep_idx) - 1):
        dt = float(keep_t[n + 1] - keep_t[n])

        if n < int(use_teacher_forcing_until):
            u_in = U_gt[n]
            v_in = V_gt[n]
        else:
            u_in = u_pred[-1]
            v_in = v_pred[-1]

        Xin = make_input_features(
            u_in, v_in,
            x_chan, y_chan,
            dt=dt,
            d1=d1, d2=d2, s=s,
            dt_scale=float(dt_scale),
            t_now=float(keep_t[n]) if use_input_time else None,
            time_scale=time_scale,
        ).astype(np.float32)

        if expected_cin > 0 and int(Xin.shape[-1]) != expected_cin:
            raise ValueError(
                "rollout input channel mismatch: built "
                f"Cin={int(Xin.shape[-1])}, but model expects Cin={expected_cin}. "
                "For dt-only scaling use [u,v,x,y,dt,d1,d2,s] with "
                "--time_channel_idx -1 --dt_channel_idx 4 --d1_channel_idx 5 --d2_channel_idx 6."
            )

        x_4d = torch.from_numpy(Xin).unsqueeze(0).to(DEVICE)
        y_4d = model(x_4d)
        y2 = y_4d.squeeze(0).detach().cpu().numpy()

        u_pred.append(y2[..., 0].astype(np.float32))
        v_pred.append(y2[..., 1].astype(np.float32))

    save_rollout_full_time_errors(
        np.stack((u_pred, v_pred), axis=-1), np.stack((U_gt, V_gt), axis=-1),
        keep_t,
        float(x[-1] - x[0] + (x[1] - x[0] if basis_boundary == "periodic" else 0)),
        float(y[-1] - y[0] + (y[1] - y[0] if basis_boundary == "periodic" else 0)),
        basis_boundary, out_prefix + "_rollout",
    )

    # --- rollout error (mass-weighted relL2 in physical space)
    rel_u = [_rel_l2_2d_mass_np(u_pred[k], U_gt[k], mass_np) for k in range(len(keep_idx))]
    rel_v = [_rel_l2_2d_mass_np(v_pred[k], V_gt[k], mass_np) for k in range(len(keep_idx))]
    _plot_relL2_time(
        keep_t, rel_u, rel_v,
        out_prefix + f"_rollout_relL2_T55_t0_{int(T_final)}.png",
        title="rollout mass-weighted relative L2 vs time"
    )

    snap_idx = _pick_time_indices(keep_t, list(t_snap), tol=1e-3)
    for tt, kk in zip(t_snap, snap_idx):
        _plot_gt_pred_err(
            U_gt[kk], u_pred[kk],
            V_gt[kk], v_pred[kk],
            out_prefix + f"_rollout_t{int(tt):04d}_GT_Pred_Err.png",
            title_suffix=f" (t={int(tt)})",
            x_range=(float(np.min(x)), float(np.max(x))),
            y_range=(float(np.min(y)), float(np.max(y))),
        )


def teacher_rollout_schedule(source, indices):
    ordered = sorted(indices, key=lambda index: int(source.time_index[index]))
    time_indices = source.time_index[ordered].numpy()
    if not np.array_equal(time_indices, np.arange(len(source.times) - 1, dtype=np.int64)):
        raise ValueError("teacher rollout requires a complete, ordered trajectory")
    dt_physical = source.dt_physical[ordered].numpy().astype(np.float64)
    times = np.concatenate([[float(source.times[0])], source.t_end[ordered].numpy().astype(np.float64)])
    if not np.allclose(np.diff(times), dt_physical, rtol=1.0e-7, atol=1.0e-7):
        raise ValueError("teacher rollout times do not match per-transition physical dt")
    return ordered, times, dt_physical


@torch.no_grad()
def visualize_rollout_teacher_gm2d(
    model,
    teacher_dataset,
    trajectory_key,
    out_prefix,
    resolution_ops,
    t_snap=(10.0, 100.0, 1000.0, 5000.0, 10000.0),
):
    source, trajectory_indices = teacher_dataset.trajectory(trajectory_key)
    ordered, times_used, dt_physical = teacher_rollout_schedule(source, trajectory_indices)
    initial = source.x[ordered[0], :, :, :2].detach().cpu().float().numpy()
    gt = [initial]
    gt.extend(source.target_at(index).numpy() for index in ordered)
    gt = np.stack(gt, axis=0)

    model.eval()
    resolution_ops.prepare_model(model, source.Ny, source.Nx)
    predicted = [initial.copy()]
    for step, index in enumerate(ordered):
        features = source.input_at(
            index,
            u_override=predicted[-1][..., 0],
            v_override=predicted[-1][..., 1],
        ).unsqueeze(0).to(DEVICE)
        if int(features.shape[-1]) != int(getattr(model, "in_channels", features.shape[-1])):
            raise ValueError(
                f"teacher rollout built Cin={features.shape[-1]}, model expects {model.in_channels}"
            )
        metadata = None
        if bool(getattr(model, "use_layer_scaling", False)):
            metadata = {
                "dt_physical": source.dt_physical[index].reshape(1).to(DEVICE),
            }
        next_state = _model_forward_batch(model, features, metadata)
        predicted.append(next_state.squeeze(0).detach().cpu().float().numpy())
    predicted = np.stack(predicted, axis=0)

    save_rollout_full_time_errors(
        predicted, gt, times_used, source.Lx, source.Ly,
        source.boundary_condition or resolution_ops.basis_boundary, out_prefix + "_teacher_rollout",
    )

    mass_np = (
        _build_mass_coeffs_1d_np(source.Nx, L=1.0),
        _build_mass_coeffs_1d_np(source.Ny, L=1.0),
    )
    rel_u = [
        _rel_l2_2d_mass_np(predicted[k, ..., 0], gt[k, ..., 0], mass_np)
        for k in range(len(times_used))
    ]
    rel_v = [
        _rel_l2_2d_mass_np(predicted[k, ..., 1], gt[k, ..., 1], mass_np)
        for k in range(len(times_used))
    ]
    _plot_relL2_time(
        times_used,
        rel_u,
        rel_v,
        out_prefix + "_teacher_rollout_relL2.png",
        title="teacher-data rollout mass-weighted relative L2 vs physical time",
    )
    np.savez(
        out_prefix + "_teacher_rollout_errors.npz",
        times=times_used,
        dt_physical=np.concatenate([[0.0], dt_physical]),
        rel_u=np.asarray(rel_u),
        rel_v=np.asarray(rel_v),
    )
    with open(out_prefix + "_teacher_rollout_errors.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["physical_time", "dt_physical", "rel_l2_u", "rel_l2_v"])
        for row in zip(times_used, np.concatenate([[0.0], dt_physical]), rel_u, rel_v):
            writer.writerow(row)
    snap_indices = _pick_time_indices(times_used, list(t_snap), tol=1.0e-6)
    for physical_time, snap_index in zip(t_snap, snap_indices):
        _plot_gt_pred_err(
            gt[snap_index, ..., 0],
            predicted[snap_index, ..., 0],
            gt[snap_index, ..., 1],
            predicted[snap_index, ..., 1],
            out_prefix + f"_teacher_rollout_t{int(physical_time):05d}_GT_Pred_Err.png",
            title_suffix=f" (t={physical_time:g})",
            x_range=source.plot_x_range,
            y_range=source.plot_y_range,
        )
    print(
        "teacher rollout | source_id={} sample_id={} steps={} final_time={} "
        "dt_unique={}".format(
            trajectory_key[0], trajectory_key[1], len(ordered), times_used[-1],
            np.unique(dt_physical).tolist(),
        ),
        flush=True,
    )
    print(
        f"teacher rollout final relative L2 | u={rel_u[-1]:.8g} v={rel_v[-1]:.8g}",
        flush=True,
    )
    return {
        "times": times_used,
        "dt_physical": dt_physical,
        "rel_u": np.asarray(rel_u),
        "rel_v": np.asarray(rel_v),
    }


def legacy_test_rollout_jobs(test_dataset, npz_dirs_text):
    """Pair each legacy test source with its NPZ directory and original test IDs."""
    bases = list(_iter_unique_sharded_bases(test_dataset))
    directories = [part.strip() for part in npz_dirs_text.split(",") if part.strip()]
    if directories and len(directories) != len(bases):
        raise ValueError("--npz_dirs must provide one directory per test source, in test-source order")
    jobs = []
    for source_id, base in enumerate(bases):
        split_name, ids = base.split_trajectory_ids()
        if split_name != "test" or ids is None:
            raise ValueError("All-test rollout requires test split metadata and original split trajectory IDs")
        ddir = directories[source_id] if directories else _nested_metadata_value(
            base.index, "source_input_dir"
        )[0]
        if not ddir:
            raise ValueError("Test index has no source_input_dir; specify --npz_dirs")
        ddir = str(ddir)
        if not os.path.isfile(os.path.join(ddir, "meta.npz")):
            raise FileNotFoundError(f"meta.npz not found in {ddir}; specify the matching --npz_dirs")
        for sample_id in ids.tolist():
            candidates = [os.path.join(ddir, f"sample_{sample_id}.npz"),
                          os.path.join(ddir, f"sample_{sample_id:03d}.npz")]
            if not any(os.path.isfile(path) for path in candidates):
                raise FileNotFoundError(f"Test trajectory {sample_id} not found in {ddir}")
            jobs.append((source_id, ddir, int(sample_id)))
    if not jobs:
        raise ValueError("No test trajectories available for rollout")
    return jobs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher_paths", nargs="*", default=[],
        help=(
            "one or more direct teacher .pt files; comma-separated entries and glob patterns "
            "such as '/path/teacher_*.pt' are also accepted"
        ),
    )
    p.add_argument(
        "--teacher_path", action="append", default=[],
        help="one direct teacher .pt file (repeatable alias for --teacher_paths)",
    )
    p.add_argument(
        "--teacher_dt_units", default="auto", choices=["auto", "physical", "normalized"],
        help="interpretation of top-level teacher dt when metadata does not specify it",
    )
    p.add_argument("--teacher_mmap", type=int, default=1, help="memory-map monolithic teacher tensor storage")
    p.add_argument(
        "--teacher_strict_standard", type=int, default=1,
        help="validate the exact 100-trajectory, 256x256, 0-to-10000 experiment",
    )
    p.add_argument(
        "--teacher_full_finite_check", type=int, default=1,
        help="scan u/v and repeated coordinate fields in bounded CPU chunks before training",
    )
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--train_sample_ids", nargs="*", default=[])
    p.add_argument("--test_sample_ids", nargs="*", default=[])
    p.add_argument(
        "--num_train_trajectories", "--train_trajectory_count",
        dest="num_train_trajectories", type=int, default=-1,
        help=(
            "number of complete train trajectories retained after the trajectory-level split; "
            "-1 uses every trajectory assigned to train"
        ),
    )
    p.add_argument(
        "--num_test_trajectories", "--test_trajectory_count",
        dest="num_test_trajectories", type=int, default=-1,
        help=(
            "number of complete test trajectories retained after the trajectory-level split; "
            "-1 uses every trajectory assigned to test"
        ),
    )
    p.add_argument("--teacher_rollout", type=int, default=1)
    p.add_argument("--rollout_all_test", type=int, choices=[0, 1], default=1,
                   help="1: save rollout plots for every test trajectory; 0: use the single-sample options")
    p.add_argument("--teacher_rollout_source_id", type=int, default=-1)
    p.add_argument("--teacher_rollout_sample_id", type=int, default=-1)
    p.add_argument("--pt_root", type=str, default="/content/drive/MyDrive/pt_data2/gm2d_T55_full")
    p.add_argument("--train_index", type=str, default="")
    p.add_argument("--test_index", type=str, default="")

    p.add_argument("--npz_dirs", type=str, default="")  # comma-separated
    p.add_argument("--sample_id", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument(
        "--lr_schedule", "--lr_scheduler",
        dest="lr_schedule",
        type=str,
        default="step50",
        choices=["step50", "cosine", "constant"],
        help="learning-rate schedule: step50 halves lr every 50 epochs; cosine or constant are legacy options",
    )
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--max_cache_gb", type=float, default=30)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--log_every", type=int, default=0, help="step log interval; <=0 disables step logs")
    p.add_argument("--eval_every", type=int, default=1, help="run test evaluation every N epochs; <=0 means final epoch only")
    p.add_argument("--amp", type=int, default=0, help="1: use CUDA AMP mixed precision during train/eval")
    p.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="AMP autocast dtype")
    p.add_argument("--profile_steps", type=int, default=0, help="profile the first N train steps of each epoch; <=0 disables it")

    p.add_argument("--Kx", type=int, default=16)
    p.add_argument("--Ky", type=int, default=16)
    p.add_argument("--basis_reconstruction", type=int, choices=[0, 1], default=1,
                   help="Evaluate teacher projection errors before training (fixed Laplace/node bases).")
    p.add_argument("--reconstruction_tol", type=float, default=0.01,
                   help="Relative spatial L2 tolerance for the minimum basis count (default: 1 percent).")
    p.add_argument("--reconstruction_only", type=int, choices=[0, 1], default=0,
                   help="Evaluate basis reconstruction and exit before model creation/training.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument(
        "--operator_type", "--basis",
        dest="operator_type",
        type=str,
        default="laplace",
        choices=["laplace", "fft"],
        help=(
            "spectral operator: laplace uses the mass-orthonormal Laplacian eigenbasis "
            "selected by --basis_boundary; fft uses the Fourier basis"
        ),
    )
    p.add_argument(
        "--basis_boundary",
        type=str,
        default="auto",
        choices=["auto", "neumann", "periodic"],
        help=(
            "boundary condition used by the Laplace basis and loss quadrature. "
            "auto reads teacher metadata and supports legacy Gray-Scott F/k and GM s metadata"
        ),
    )
    p.add_argument("--use_layer_scaling", type=int, default=0, help="1: inject exp(-dt(d*mu+lambda)) scaling into each Laplace layer")

    p.add_argument("--dt_scale", type=float, default=None)
    p.add_argument("--T_final", type=float, default=None)
    p.add_argument("--t_snap", type=str, default="")
    p.add_argument("--teacher_forcing", type=int, default=0)
    p.add_argument("--time_channel_idx", type=int, default=-1, help="optional current-time channel index; unused by dt-only layer scaling")
    p.add_argument("--time_channel_scale", type=float, default=None, help="physical scale of the optional time channel")
    p.add_argument("--Lx", type=float, default=2.5, help="physical x-domain length used by spectral decay")
    p.add_argument("--Ly", type=float, default=2.5, help="physical y-domain length used by spectral decay")
    p.add_argument("--d1_channel_idx", type=int, default=5, help="legacy d1 channel used only to initialize learned spectral d1")
    p.add_argument("--d2_channel_idx", type=int, default=6, help="legacy d2 channel used only to initialize learned spectral d2")
    p.add_argument(
        "--spectral_d1_init", type=float, default=None,
        help="initial value of learned spectral d1; default reads teacher Du/d1, then falls back to 2e-5",
    )
    p.add_argument(
        "--spectral_d2_init", type=float, default=None,
        help="initial value of learned spectral d2; default reads teacher Dv/d2, then falls back to 1e-5",
    )
    p.add_argument("--dt_channel_idx", type=int, default=-1, help="dt channel index; default auto-detects 4 for [u,v,x,y,dt,d1,d2,s] or time_channel_idx+1")
    p.add_argument(
        "--loss_time_beta",
        type=float,
        default=0.0,
        help=(
            "if beta > 0, reconstruct cumulative physical time t_k from dt for each "
            "trajectory and use (t_k)^beta for training loss only"
        ),
    )
    p.add_argument("--time_weight_floor", type=float, default=0.0)

    p.add_argument("--out_prefix", type=str, default="leno_nomask_T55")
    p.add_argument("--resume", type=int, default=1, help="1: resume model weights from out_prefix_best.pt when it exists")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--pt_root_low", type=str, default="", help="low-res dataset root. if empty, no mixing")
    p.add_argument("--mix_full_ratio", type=float, default=1.0, help="probability to draw full sample (aligned by idx)")
    p.add_argument("--mix_on_test", type=int, default=0, help="0: test on full only, 1: apply same mixing on test")
    p.add_argument("--pt_roots", type=str, default="", help="comma-separated dataset roots ordered low-to-high; enables multi-resolution subset training")
    p.add_argument("--pt_source_grid_hws", type=str, default="", help="comma-separated output sizes built from --pt_root by source-grid subsampling, e.g. 25,50,75,full")
    p.add_argument("--pt_root_ratios", "--pt_root_fractions", dest="pt_root_ratios", type=str, default="", help="comma-separated fractions for --pt_roots, e.g. 0.9,0.5,0.3,0.2")
    p.add_argument("--multi_res_on_test", type=int, default=0, help="0: test on highest-resolution root, 1: apply --pt_root_ratios to test too")
    p.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/output_data")
    p.add_argument("--log_file", type=str, default="", help="console log path; default save_dir/out_prefix_YYYYMMDD_HHMMSS_log.txt. use 'none' to disable")
    p.add_argument("--smoke_test", action="store_true", help="run lightweight CPU-safe compatibility checks and exit")

    return p.parse_args()


def _split_csv(value):
    return [s.strip() for s in str(value or "").split(",") if s.strip() != ""]


def _resolve_teacher_paths(args):
    raw_paths = []
    for raw in list(args.teacher_paths or []) + list(args.teacher_path or []):
        raw_paths.extend(_split_csv(raw))
    paths = []
    for raw_path in raw_paths:
        expanded = os.path.abspath(os.path.expanduser(raw_path))
        if glob.has_magic(expanded):
            matches = sorted(glob.glob(expanded))
            if not matches:
                raise FileNotFoundError(f"teacher path pattern matched no files: {raw_path}")
            paths.extend(matches)
        else:
            paths.append(expanded)
    normalized = []
    seen = set()
    for path in paths:
        absolute = os.path.abspath(os.path.expanduser(path))
        if absolute in seen:
            continue
        if os.path.splitext(absolute)[1].lower() != ".pt":
            raise ValueError(f"direct teacher path must be a .pt file: {path}")
        seen.add(absolute)
        normalized.append(absolute)
    return normalized


def _apply_data_mode_defaults(args, teacher_mode):
    default_scale = 10000.0 if bool(teacher_mode) else 250.0
    if args.dt_scale is None:
        args.dt_scale = default_scale
    if args.T_final is None:
        args.T_final = 10000.0 if bool(teacher_mode) else 250.0
    if args.time_channel_scale is None:
        args.time_channel_scale = default_scale
    if float(args.dt_scale) <= 0.0 or float(args.T_final) <= 0.0:
        raise ValueError("dt_scale and T_final must be positive")
    if float(args.Lx) <= 0.0 or float(args.Ly) <= 0.0:
        raise ValueError("Lx and Ly must be positive")


def _load_direct_teacher_data(args, paths):
    sources = [
        TeacherPTSource(
            path=path,
            source_id=source_id,
            dt_scale=float(args.dt_scale),
            Lx=float(args.Lx),
            Ly=float(args.Ly),
            dt_units=args.teacher_dt_units,
            mmap=bool(args.teacher_mmap),
            strict_standard=bool(args.teacher_strict_standard),
            full_finite_check=bool(args.teacher_full_finite_check),
        )
        for source_id, path in enumerate(paths)
    ]
    dataset = TeacherTransitionDataset(sources)
    if bool(args.teacher_strict_standard):
        n_trajectories = len(dataset.trajectory_keys)
        if n_trajectories < 100 or n_trajectories % 100 != 0:
            raise ValueError(
                "strict teacher collection must contain 100 trajectories (or a whole-number "
                f"multiple when combining complete teacher sets); got {n_trajectories}. "
                "For numbered shards, pass the complete pattern such as teacher_*.pt."
            )
        if len(dataset) != n_trajectories * 109:
            raise ValueError(
                f"strict teacher collection has {len(dataset)} transitions for {n_trajectories} "
                "trajectories; expected exactly 109 transitions per trajectory"
            )
    train_ds, test_ds = split_teacher_trajectories(
        dataset,
        train_fraction=float(args.train_fraction),
        split_seed=int(args.split_seed),
        train_sample_ids=args.train_sample_ids,
        test_sample_ids=args.test_sample_ids,
        num_train_trajectories=int(args.num_train_trajectories),
        num_test_trajectories=int(args.num_test_trajectories),
    )
    return dataset, train_ds, test_ds


def resolve_spectral_diffusion_init(args, teacher_dataset=None, first_x=None):
    def resolve_one(cli_value, metadata_attr, legacy_channel, fallback, name):
        if cli_value is not None:
            value, source = float(cli_value), f"--spectral_{name}_init"
        elif teacher_dataset is not None:
            values = [
                float(getattr(source_obj, metadata_attr))
                for source_obj in teacher_dataset.sources
                if getattr(source_obj, metadata_attr) is not None
            ]
            if len(values) != len(teacher_dataset.sources):
                value, source = float(fallback), "fallback"
            elif not np.allclose(values, values[0], rtol=1.0e-7, atol=1.0e-12):
                raise ValueError(
                    f"teacher files contain different {metadata_attr} values {sorted(set(values))}; "
                    f"one globally learned {name} needs a single initialization or an explicit "
                    f"--spectral_{name}_init"
                )
            else:
                value, source = float(values[0]), f"teacher {metadata_attr}"
        elif first_x is not None and int(legacy_channel) < int(first_x.shape[-1]):
            channel = first_x[..., int(legacy_channel)].float()
            value, source = float(channel.mean().item()), f"legacy input channel {legacy_channel}"
        else:
            value, source = float(fallback), "fallback"
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"learned spectral {name} initialization must be finite and positive")
        return value, source

    d1, source1 = resolve_one(
        args.spectral_d1_init, "diffusion_u", int(args.d1_channel_idx), 2.0e-5, "d1"
    )
    d2, source2 = resolve_one(
        args.spectral_d2_init, "diffusion_v", int(args.d2_channel_idx), 1.0e-5, "d2"
    )
    return d1, d2, source1, source2


def _teacher_keys_by_source(keys):
    grouped = OrderedDict()
    for source_id, sample_id in keys:
        grouped.setdefault(int(source_id), []).append(int(sample_id))
    return grouped


def print_teacher_data_summary(dataset, train_ds, test_ds, args):
    physical_unique = np.unique(np.concatenate([
        source.dt_physical.numpy() for source in dataset.sources
    ])).tolist()
    normalized_unique = np.unique(np.concatenate([
        source.dt_norm.numpy() for source in dataset.sources
    ])).tolist()
    transitions_per_trajectory = sorted({int(source.times.size - 1) for source in dataset.sources})
    frames_per_trajectory = sorted({int(source.times.size) for source in dataset.sources})
    time_min = min(float(source.times[0]) for source in dataset.sources)
    time_max = max(float(source.times[-1]) for source in dataset.sources)
    mu = _build_mode_mu_2d(
        max(2, int(args.Kx)), max(2, int(args.Ky)),
        Lx=float(args.Lx), Ly=float(args.Ly), dtype=torch.float64,
    )
    print("Teacher data summary", flush=True)
    print(f"  files: {len(dataset.sources)}", flush=True)
    print(f"  grid: {dataset.Ny} x {dataset.Nx}", flush=True)
    print(f"  points: {dataset.Ny * dataset.Nx}", flush=True)
    print(f"  domain: [0,{float(args.Lx):g}] x [0,{float(args.Ly):g}]", flush=True)
    print(f"  trajectories: {len(dataset.trajectory_keys)}", flush=True)
    print(f"  frames/trajectory: {frames_per_trajectory}", flush=True)
    print(f"  transitions/trajectory: {transitions_per_trajectory}", flush=True)
    print(f"  total transitions: {len(dataset)}", flush=True)
    print(f"  time range: {time_min:g} -> {time_max:g}", flush=True)
    print(f"  physical dt unique: {physical_unique}", flush=True)
    print(f"  normalized dt unique: {normalized_unique}", flush=True)
    channels = "[u,v,xhat,yhat,dt_norm,s]" if dataset.Cin == 6 else "[u,v,xhat,yhat,dt_norm]"
    print(f"  input channels: {channels}", flush=True)
    print(f"  input shape example: [B,{dataset.Ny},{dataset.Nx},{dataset.Cin}]", flush=True)
    print(f"  target shape example: [B,{dataset.Ny},{dataset.Nx},2]", flush=True)
    print(f"  train trajectories: {len(train_ds.trajectory_keys)}", flush=True)
    print(f"  test trajectories: {len(test_ds.trajectory_keys)}", flush=True)
    print(f"  train transitions: {len(train_ds)}", flush=True)
    print(f"  test transitions: {len(test_ds)}", flush=True)
    print(f"  dt_scale: {float(args.dt_scale):g}", flush=True)
    print(f"  Lx: {float(args.Lx):g}", flush=True)
    print(f"  Ly: {float(args.Ly):g}", flush=True)
    print(f"  mu(0,0): {float(mu[0, 0]):.10g}", flush=True)
    print(f"  mu(1,0): {float(mu[0, 1]):.10g}", flush=True)
    print(f"  mu(0,1): {float(mu[1, 0]):.10g}", flush=True)
    print(f"  mu(1,1): {float(mu[1, 1]):.10g}", flush=True)
    for source in dataset.sources:
        print(
            "  file[{}]: {} | trajectories={} transitions={} Du={} Dv={}".format(
                source.source_id,
                source.path,
                len(source.trajectory_indices),
                source.n,
                source.diffusion_u,
                source.diffusion_v,
            ),
            flush=True,
        )
    print("  train trajectory IDs:", flush=True)
    for source_id, sample_ids in _teacher_keys_by_source(train_ds.trajectory_keys).items():
        print(f"    source {source_id}: {sample_ids}", flush=True)
    print("  test trajectory IDs:", flush=True)
    for source_id, sample_ids in _teacher_keys_by_source(test_ds.trajectory_keys).items():
        print(f"    source {source_id}: {sample_ids}", flush=True)

    beta = float(args.loss_time_beta)
    source, indices = dataset.trajectory(dataset.trajectory_keys[0])
    ordered, _, _ = teacher_rollout_schedule(source, indices)
    print("teacher time-weight sanity (training convention uses transition end time):", flush=True)
    print("  time_index t_start t_end dt_physical dt_norm weight", flush=True)
    for index in ordered[:6]:
        t_end = float(source.t_end[index])
        weight = 1.0 if beta <= 0.0 else max(t_end, float(args.time_weight_floor)) ** beta
        print(
            "  {:10d} {:7g} {:5g} {:11g} {:7g} {:.8g}".format(
                int(source.time_index[index]),
                float(source.t_start[index]),
                t_end,
                float(source.dt_physical[index]),
                float(source.dt_norm[index]),
                weight,
            ),
            flush=True,
        )
    print("  validation/test/rollout weighting: none", flush=True)


def log_first_batch_layer_scaling(model, xb, metadata, Lx, Ly):
    if not bool(getattr(model, "use_layer_scaling", False)):
        return
    if metadata is None:
        dt_idx = _resolve_dt_channel_idx(model.time_channel_idx, model.dt_channel_idx)
        dt_norm = _extract_scalar_channel(xb, dt_idx, "dt", required=True)
        dt_physical = dt_norm * float(model.dt_scale)
    else:
        dt_norm = metadata["dt_norm"].reshape(-1).to(xb)
        dt_physical = metadata["dt_physical"].reshape(-1).to(xb)
        extracted_dt, _, _ = model._extract_layer_scalars(xb, dt_scalar=dt_physical)
        if not torch.allclose(extracted_dt, dt_physical):
            raise RuntimeError("physical teacher dt did not reach layer scaling unchanged")
    mu = _build_mode_mu_2d(2, 2, Lx=Lx, Ly=Ly, device=xb.device, dtype=xb.dtype)
    print("first-batch layer-scaling sanity (printed once):", flush=True)
    print(f"  dt_norm min/max: {dt_norm.min().item():.8g} / {dt_norm.max().item():.8g}", flush=True)
    print(
        f"  dt_physical min/max: {dt_physical.min().item():.8g} / {dt_physical.max().item():.8g}",
        flush=True,
    )
    for layer_index, spectral_layer in enumerate(model.spec):
        if getattr(spectral_layer, "scaler", None) is None:
            continue
        learned_d = spectral_layer.scaler.learned_diffusion().detach().cpu()
        print(
            f"  layer {layer_index} learned d1/d2: "
            f"{learned_d[0].item():.8g} / {learned_d[1].item():.8g}",
            flush=True,
        )
    print(f"  mu_00: {mu[0, 0].item():.10g}", flush=True)
    print(f"  mu_10: {mu[0, 1].item():.10g}", flush=True)
    print(f"  mu_01: {mu[1, 0].item():.10g}", flush=True)
    print(f"  mu_11: {mu[1, 1].item():.10g}", flush=True)


def _parse_float_csv(value, name):
    items = _split_csv(value)
    try:
        return [float(x) for x in items]
    except ValueError as e:
        raise ValueError(f"{name} must be comma-separated floats: {value}") from e


def _parse_source_grid_hws(value):
    out = []
    for item in _split_csv(value):
        token = item.lower()
        if token in {"full", "max", "source"}:
            out.append(None)
            continue
        try:
            out.append(int(item))
        except ValueError as e:
            raise ValueError(f"--pt_source_grid_hws must contain integers or 'full': {value}") from e
    return out


def _source_grid_hw_name(hw, base_ds):
    if hw is None:
        return f"full{int(base_ds.Ny)}"
    return f"low{int(hw)}"


def _source_grid_hw_value(hw, base_ds):
    if hw is None:
        return min(int(base_ds.Ny), int(base_ds.Nx))
    return int(hw)


def _pick_highest_source_grid_hw(hws, base_ds):
    return max(hws, key=lambda hw: _source_grid_hw_value(hw, base_ds))


def _default_index_path(pt_root, split_name):
    fname = f"gm2d_{split_name}_201_index.pt"
    return os.path.join(str(pt_root), split_name, fname)


def _load_pt_dataset_from_root(pt_root, split_name, max_cache_gb):
    index_path = _default_index_path(pt_root, split_name)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{split_name} index not found: {index_path}")
    return BudgetCachedShardedPTDataset(index_path, max_cache_gb=max_cache_gb, to_float32=False)


def _load_pt_dataset_from_args(args, split_name):
    if split_name == "train":
        index_path = args.train_index.strip() or _default_index_path(args.pt_root, "train")
    elif split_name == "test":
        index_path = args.test_index.strip() or _default_index_path(args.pt_root, "test")
    else:
        raise ValueError(f"unknown split: {split_name}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{split_name} index not found: {index_path}")
    return BudgetCachedShardedPTDataset(index_path, max_cache_gb=args.max_cache_gb, to_float32=False)


def _make_source_grid_multires_sources(base_ds, hws):
    sources = [maybe_source_grid_subsample_dataset(base_ds, hw) for hw in hws]
    names = [
        f"{getattr(base_ds, 'root_dir', 'pt')}@source_grid_{_source_grid_hw_name(hw, base_ds)}"
        for hw in hws
    ]
    return sources, names


def _print_multi_res_summary(label, dataset):
    print(f"{label} multi-resolution summary:", flush=True)
    for i, row in enumerate(dataset.summary()):
        print(
            "  [{}] Ny,Nx=({},{}) ratio={:.3f} selected={}/{} root={}".format(
                i, row["Ny"], row["Nx"], row["ratio"], row["selected"], row["total"], row["name"]
            ),
            flush=True,
        )


def dataloader_worker_kwargs(args):
    num_workers = int(args.num_workers)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def run_smoke_tests():
    results = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        results.append(name)

    times = np.concatenate([
        np.arange(0.0, 101.0, 10.0, dtype=np.float64),
        np.arange(200.0, 10001.0, 100.0, dtype=np.float64),
    ])
    times = validate_teacher_times_10000(times, source_name="synthetic smoke teacher")
    dt_physical = np.diff(times)
    check("1 time grid: 110 frames / 109 transitions", len(times) == 110 and len(dt_physical) == 109)
    dt_norm = dt_physical / 10000.0
    check("2 normalized dt values: 0.001 and 0.01", np.allclose(np.unique(dt_norm), [0.001, 0.01]))
    check("3 physical dt recovery: 10 and 100", np.allclose(np.unique(dt_norm * 10000.0), [10.0, 100.0]))

    class SmokeTrajectoryDataset:
        Ny = Nx = Cin = 1
        boundary_condition = "neumann"
        boundary_source = "synthetic smoke dataset"
        trajectory_keys = [(0, sample_id) for sample_id in range(6)]

        def __len__(self):
            return 12

        @staticmethod
        def trajectory_key_for_record(index):
            return 0, int(index) // 2

    limited_train, limited_test = split_teacher_trajectories(
        SmokeTrajectoryDataset(),
        train_fraction=2.0 / 3.0,
        split_seed=7,
        num_train_trajectories=2,
        num_test_trajectories=1,
    )
    check(
        "13 teacher trajectory count limits retain complete disjoint trajectories",
        len(limited_train.trajectory_keys) == 2
        and len(limited_test.trajectory_keys) == 1
        and len(limited_train) == 4
        and len(limited_test) == 2
        and not (set(limited_train.trajectory_keys) & set(limited_test.trajectory_keys)),
    )

    flattened = torch.zeros(1, 65536, 4)
    reshaped = _teacher_grid_view(flattened, "smoke flattened x", Ny=256, Nx=256, min_channels=4)
    check("5 flattened 65536-point reshape", tuple(reshaped.shape) == (1, 256, 256, 4))

    with tempfile.TemporaryDirectory(prefix="gm_leno_smoke_") as temp_dir:
        Ny = Nx = 4
        x_axis = torch.linspace(0.0, 2.5, Nx)
        y_axis = torch.linspace(0.0, 2.5, Ny)
        Y, X = torch.meshgrid(y_axis, x_axis, indexing="ij")
        coord = torch.stack([X, Y], dim=-1)
        n = 4
        legacy_x = torch.zeros(n, Ny * Nx, 8)
        legacy_y = torch.zeros(n, Ny * Nx, 2)
        for index in range(n):
            grid = torch.zeros(Ny, Nx, 8)
            grid[..., 0] = 0.1 * index
            grid[..., 1] = 0.2 * index
            grid[..., 2:4] = coord
            grid[..., 4] = 0.001
            grid[..., 5] = 0.01
            grid[..., 6] = 0.02
            grid[..., 7] = 3.0
            legacy_x[index] = grid.reshape(Ny * Nx, 8)
            legacy_y[index] = grid[..., :2].reshape(Ny * Nx, 2) + 0.05
        teacher_path = os.path.join(temp_dir, "legacy_teacher.pt")
        torch.save(
            {
                "x": legacy_x,
                "y": legacy_y,
                "sample_id": torch.tensor([0, 0, 1, 1]),
                "time_index": torch.tensor([0, 1, 0, 1]),
                "times": torch.tensor([0.0, 10.0, 20.0]),
            },
            teacher_path,
        )
        source = TeacherPTSource(
            teacher_path,
            source_id=0,
            dt_scale=10000.0,
            Lx=2.5,
            Ly=2.5,
            strict_standard=False,
            full_finite_check=True,
        )
        direct_dataset = TeacherTransitionDataset([source])
        train_subset, test_subset = split_teacher_trajectories(
            direct_dataset, train_fraction=0.5, split_seed=0
        )
        check(
            "4 trajectory split has no overlap",
            not (set(train_subset.trajectory_keys) & set(test_subset.trajectory_keys)),
        )
        sample_x, sample_y, sample_meta = direct_dataset[0]
        check(
            "12 legacy eight-channel teacher loads as five-channel input",
            tuple(sample_x.shape) == (Ny, Nx, 5)
            and tuple(sample_y.shape) == (Ny, Nx, 2)
            and "d1" not in sample_meta
            and "d2" not in sample_meta,
        )

        fno = LENO2d(
            in_channels=5, out_channels=2, width=4, depth=1, Kx=2, Ky=2,
            Phix=None, Px=None, Phiy=None, Py=None,
            operator_type="fft", use_layer_scaling=False, dt_scale=10000.0,
            Lx=2.5, Ly=2.5,
        )
        tiny_x = torch.randn(2, Ny, Nx, 5)
        fno_out = fno(tiny_x)
        check("6 five-channel model forward shape", tuple(fno_out.shape) == (2, Ny, Nx, 2))
        check("9 FNO mode needs no d1/d2", torch.isfinite(fno_out).all())

        Phix_np, Px_np = build_neumann_basis_1d_mass_orthonormal(Nx, 2, L=1.0)
        Phiy_np, Py_np = build_neumann_basis_1d_mass_orthonormal(Ny, 2, L=1.0)
        scaling_model = LENO2d(
            in_channels=5, out_channels=2, width=4, depth=1, Kx=2, Ky=2,
            Phix=torch.from_numpy(Phix_np), Px=torch.from_numpy(Px_np),
            Phiy=torch.from_numpy(Phiy_np), Py=torch.from_numpy(Py_np),
            operator_type="laplace", use_layer_scaling=True, dt_scale=10000.0,
            d1_channel_idx=5, d2_channel_idx=6, Lx=2.5, Ly=2.5,
            spectral_d1_init=2.0e-5, spectral_d2_init=1.0e-5,
        )
        captured = []

        def capture_scalars(_module, inputs):
            captured.append(inputs[1].detach().clone())

        hook = scaling_model.spec[0].scaler.register_forward_pre_hook(capture_scalars)
        explicit_dt = torch.tensor([10.0, 100.0])
        scaled_out = scaling_model(tiny_x, dt_scalar=explicit_dt)
        hook.remove()
        learned_d = scaling_model.spec[0].scaler.learned_diffusion()
        check(
            "7 physical dt reaches scaling and d1/d2 are learned parameters",
            tuple(scaled_out.shape) == (2, Ny, Nx, 2)
            and len(captured) == 1
            and torch.equal(captured[0], explicit_dt)
            and scaling_model.spec[0].scaler.d_raw.requires_grad
            and torch.allclose(learned_d, torch.tensor([2.0e-5, 1.0e-5]), rtol=1.0e-5),
        )

        mu = _build_mode_mu_2d(2, 2, Lx=2.5, Ly=2.5, dtype=torch.float64)
        check(
            "8 physical-domain mu(1,0)",
            math.isclose(float(mu[0, 1]), (math.pi / 2.5) ** 2, rel_tol=1.0e-12),
        )

        class ZeroModel(nn.Module):
            requires_basis = False
            use_layer_scaling = False

            def forward(self, value):
                return torch.zeros(value.shape[0], value.shape[1], value.shape[2], 2, device=value.device)

        eval_x = torch.zeros(2, Ny, Nx, 5)
        eval_y = torch.ones(2, Ny, Nx, 2)
        eval_t = torch.tensor([10.0, 10000.0])
        eval_loader = DataLoader(torch.utils.data.TensorDataset(eval_x, eval_y, eval_t), batch_size=2)
        eval_ops = ResolutionOpsCache(2, 2, temp_dir, device=DEVICE)
        eval_u, eval_v = eval_epoch(ZeroModel().to(DEVICE), eval_loader, eval_ops)
        weights = batch_time_weights(eval_t, loss_time_beta=0.75)
        check(
            "10 evaluation loss remains unweighted",
            math.isclose(eval_u, 1.0, rel_tol=1.0e-6)
            and math.isclose(eval_v, 1.0, rel_tol=1.0e-6)
            and not torch.allclose(weights, torch.ones_like(weights)),
        )

        plot_path = os.path.join(temp_dir, "rollout_dynamic_ranges.png")
        plot_ranges = _plot_gt_pred_err(
            np.asarray([[0.203, 0.8], [0.4, 0.997]], dtype=np.float32),
            np.asarray([[0.101, 0.6], [0.3, 0.899]], dtype=np.float32),
            np.asarray([[0.021, 0.20], [0.10, 0.392]], dtype=np.float32),
            np.asarray([[0.011, 0.15], [0.08, 0.348]], dtype=np.float32),
            plot_path,
            x_range=(0.25, 2.25),
            y_range=(-0.5, 1.5),
        )
        check(
            "14 rollout panels and coordinates use independent data min/max",
            os.path.isfile(plot_path)
            and np.allclose(plot_ranges["u_true"], (0.2, 1.0))
            and np.allclose(plot_ranges["u_pred"], (0.1, 0.9))
            and np.allclose(plot_ranges["v_true"], (0.02, 0.4))
            and np.allclose(plot_ranges["v_pred"], (0.01, 0.35))
            and plot_ranges["x"] == (0.2, 2.3)
            and plot_ranges["y"] == (-0.5, 1.5),
        )

    class ScheduleSource:
        pass

    schedule_source = ScheduleSource()
    schedule_source.times = times
    schedule_source.time_index = torch.arange(109)
    schedule_source.dt_physical = torch.as_tensor(dt_physical, dtype=torch.float32)
    schedule_source.t_end = torch.as_tensor(times[1:], dtype=torch.float32)
    _, rollout_times, rollout_dt = teacher_rollout_schedule(schedule_source, list(range(109)))
    check(
        "11 rollout exact variable-step grid",
        np.array_equal(rollout_times, times) and np.allclose(rollout_dt, dt_physical),
    )

    lr_parameter = nn.Parameter(torch.tensor(0.0))
    lr_optimizer = torch.optim.SGD([lr_parameter], lr=1.0e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(lr_optimizer, step_size=50, gamma=0.5)
    learning_rates = []
    for _ in range(100):
        lr_optimizer.step()
        lr_scheduler.step()
        learning_rates.append(float(lr_optimizer.param_groups[0]["lr"]))
    check(
        "13 learning rate halves every 50 epochs",
        math.isclose(learning_rates[48], 1.0e-3)
        and math.isclose(learning_rates[49], 5.0e-4)
        and math.isclose(learning_rates[98], 5.0e-4)
        and math.isclose(learning_rates[99], 2.5e-4),
    )

    resume_parameter = nn.Parameter(torch.tensor(0.0))
    resume_optimizer = torch.optim.SGD([resume_parameter], lr=1.0e-3)
    resume_scheduler = torch.optim.lr_scheduler.StepLR(
        resume_optimizer, step_size=50, gamma=0.5
    )
    _restore_scheduler_epoch(resume_scheduler, completed_epochs=49)
    lr_before_epoch_50 = float(resume_optimizer.param_groups[0]["lr"])
    resume_optimizer.step()
    resume_scheduler.step()
    check(
        "15 best-only resume preserves the global step50 schedule",
        math.isclose(lr_before_epoch_50, 1.0e-3)
        and resume_scheduler.last_epoch == 50
        and math.isclose(float(resume_optimizer.param_groups[0]["lr"]), 5.0e-4),
    )

    print("Smoke tests passed", flush=True)
    for result in results:
        print(f"  PASS: {result}", flush=True)


def main():
    args = parse_args()
    teacher_paths = _resolve_teacher_paths(args)
    teacher_mode = len(teacher_paths) > 0
    _apply_data_mode_defaults(args, teacher_mode=teacher_mode)
    if bool(args.smoke_test):
        run_smoke_tests()
        return
    log_path = setup_console_log(args.log_file, args.save_dir, args.out_prefix)
    if log_path:
        print("argv:", " ".join(sys.argv), flush=True)
    set_seed(args.seed)

    source_grid_hws = _parse_source_grid_hws(args.pt_source_grid_hws)
    pt_roots = _split_csv(args.pt_roots)
    teacher_dataset = None
    if teacher_mode:
        if source_grid_hws or pt_roots or args.train_index.strip() or args.test_index.strip():
            print(
                "NOTE: direct --teacher_path(s) are active; old sharded/multi-resolution dataset "
                "arguments are ignored.",
                flush=True,
            )
        teacher_dataset, train_ds, test_ds = _load_direct_teacher_data(args, teacher_paths)
        print_teacher_data_summary(teacher_dataset, train_ds, test_ds, args)
    elif len(source_grid_hws) > 0:
        ratios = _parse_float_csv(args.pt_root_ratios, "--pt_root_ratios")
        if len(ratios) == 0:
            raise ValueError("--pt_root_ratios is required when --pt_source_grid_hws is set")
        if len(source_grid_hws) != len(ratios):
            raise ValueError(
                f"--pt_source_grid_hws and --pt_root_ratios must have the same length: "
                f"{len(source_grid_hws)} vs {len(ratios)}"
            )
        if len(pt_roots) > 0:
            print("WARNING: --pt_roots is ignored when --pt_source_grid_hws is set.", flush=True)

        train_base = _load_pt_dataset_from_args(args, "train")
        train_sources, train_names = _make_source_grid_multires_sources(train_base, source_grid_hws)
        train_ds = MultiResolutionSubsetDataset(
            train_sources, ratios, seed=int(args.seed), names=train_names
        )
        _print_multi_res_summary("train", train_ds)

        test_base = _load_pt_dataset_from_args(args, "test")
        if int(args.multi_res_on_test) == 1 or int(args.mix_on_test) == 1:
            test_sources, test_names = _make_source_grid_multires_sources(test_base, source_grid_hws)
            test_ds = MultiResolutionSubsetDataset(
                test_sources, ratios, seed=int(args.seed) + 7919, names=test_names
            )
            _print_multi_res_summary("test", test_ds)
        else:
            test_hw = _pick_highest_source_grid_hw(source_grid_hws, test_base)
            test_ds = maybe_source_grid_subsample_dataset(test_base, test_hw)
            print(
                "source-grid test: highest-resolution source only: {}".format(
                    _source_grid_hw_name(test_hw, test_base)
                ),
                flush=True,
            )
    elif len(pt_roots) > 0:
        ratios = _parse_float_csv(args.pt_root_ratios, "--pt_root_ratios")
        if len(ratios) == 0:
            raise ValueError("--pt_root_ratios is required when --pt_roots is set")
        if len(pt_roots) != len(ratios):
            raise ValueError(f"--pt_roots and --pt_root_ratios must have the same length: {len(pt_roots)} vs {len(ratios)}")
        if args.train_index.strip() or args.test_index.strip():
            print("WARNING: --train_index/--test_index are ignored when --pt_roots is set.", flush=True)

        train_sources = [
            _load_pt_dataset_from_root(root, "train", args.max_cache_gb)
            for root in pt_roots
        ]
        train_ds = MultiResolutionSubsetDataset(
            train_sources, ratios, seed=int(args.seed), names=pt_roots
        )
        _print_multi_res_summary("train", train_ds)

        if int(args.multi_res_on_test) == 1 or int(args.mix_on_test) == 1:
            test_sources = [
                _load_pt_dataset_from_root(root, "test", args.max_cache_gb)
                for root in pt_roots
            ]
            test_ds = MultiResolutionSubsetDataset(
                test_sources, ratios, seed=int(args.seed) + 7919, names=pt_roots
            )
            _print_multi_res_summary("test", test_ds)
        else:
            test_root = pt_roots[-1]
            test_ds = _load_pt_dataset_from_root(test_root, "test", args.max_cache_gb)
            print(f"multi-resolution test: highest-resolution root only: {test_root}", flush=True)
    else:
        train_index = args.train_index.strip() or os.path.join(args.pt_root, "train", "gm2d_train_201_index.pt")
        test_index  = args.test_index.strip()  or os.path.join(args.pt_root, "test",  "gm2d_test_201_index.pt")

        train_ds_full = BudgetCachedShardedPTDataset(train_index, max_cache_gb=args.max_cache_gb, to_float32=False)
        test_ds_full  = BudgetCachedShardedPTDataset(test_index,  max_cache_gb=args.max_cache_gb, to_float32=False)

        train_ds = train_ds_full
        test_ds  = test_ds_full

        full_ratio = float(args.mix_full_ratio)
        pt_root_low = args.pt_root_low.strip()
        if not (0.0 <= full_ratio <= 1.0):
            raise ValueError(f"mix_full_ratio must be in [0,1]. got {full_ratio}")

        do_mix = (pt_root_low != "") and (full_ratio < 1.0)
        if (pt_root_low == "") and (full_ratio < 1.0):
            print("WARNING: mix_full_ratio < 1 but pt_root_low is empty. Mixing is disabled; train/test use full only.", flush=True)
        if (pt_root_low != "") and (full_ratio >= 1.0):
            print("NOTE: pt_root_low is provided but mix_full_ratio=1.0, so low dataset will not be loaded.", flush=True)

        if do_mix:
           train_index_low = os.path.join(pt_root_low, "train", "gm2d_train_201_index.pt")
           if not os.path.exists(train_index_low):
               raise FileNotFoundError("low-res train index not found: " + train_index_low)
           train_ds_low = BudgetCachedShardedPTDataset(train_index_low, max_cache_gb=args.max_cache_gb, to_float32=False)
           train_ds = AlignedMixDataset(train_ds_full, train_ds_low, full_ratio=full_ratio, seed=args.seed)

           if int(args.mix_on_test) == 1:
               test_index_low  = os.path.join(pt_root_low, "test",  "gm2d_test_201_index.pt")
               if not os.path.exists(test_index_low):
                   raise FileNotFoundError("low-res test index not found: " + test_index_low)
               test_ds_low  = BudgetCachedShardedPTDataset(test_index_low,  max_cache_gb=args.max_cache_gb, to_float32=False)
               test_ds  = AlignedMixDataset(test_ds_full, test_ds_low, full_ratio=full_ratio, seed=args.seed)
           else:
               test_ds = test_ds_full
           print(f"mix enabled: full_ratio={full_ratio:.3f} pt_root_low={pt_root_low}", flush=True)
        else:
           train_ds = train_ds_full
           test_ds = test_ds_full
           print("mix disabled (train/test use full only)", flush=True)

    if not teacher_mode:
        prepare_dataset_cumulative_time(train_ds, args)
        prepare_dataset_cumulative_time(test_ds, args)
        print_time_weighting_summary(
            train_ds,
            loss_time_beta=float(args.loss_time_beta),
            time_weight_floor=float(args.time_weight_floor),
        )

    loader_kwargs = dataloader_worker_kwargs(args)
    if teacher_mode:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(args.batch_size),
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        print("train loader: trajectory-safe direct teacher dataset with transition shuffle", flush=True)
    elif int(args.batch_size) == 1:
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            **loader_kwargs
        )
        print("train loader: batch_size=1 with global shuffle=True", flush=True)
    else:
        if isinstance(train_ds, MultiResolutionSubsetDataset):
            train_batch_sampler = MultiResolutionShardBatchSampler(
                train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True, seed=args.seed
            )
        else:
            train_batch_sampler = ShardBatchSampler(
                train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=True, seed=args.seed
            )
        train_loader = DataLoader(
            train_ds, batch_sampler=train_batch_sampler,
            **loader_kwargs
        )
    if teacher_mode:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    elif isinstance(test_ds, MultiResolutionSubsetDataset):
        test_batch_sampler = MultiResolutionShardBatchSampler(
            test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False, seed=args.seed
        )
        test_loader = DataLoader(
            test_ds, batch_sampler=test_batch_sampler,
            **loader_kwargs
        )
    elif isinstance(test_ds, AlignedMixDataset):
        test_batch_sampler = ShardBatchSampler(
            test_ds, batch_size=int(args.batch_size), shuffle=False, drop_last=False, seed=args.seed
        )
        test_loader = DataLoader(
            test_ds, batch_sampler=test_batch_sampler,
            **loader_kwargs
        )
    else:
        test_loader = DataLoader(
            test_ds, batch_size=int(args.batch_size), shuffle=False,
            drop_last=False, **loader_kwargs
        )

    first_batch = next(iter(train_loader))
    xb0, _, _, metadata0 = _unpack_training_batch(first_batch)
    Cin = int(xb0.shape[-1])
    Ny  = int(xb0.shape[1])
    Nx  = int(xb0.shape[2])
    print("Cin", Cin,  "train_n", len(train_ds), "test_n", len(test_ds), flush=True)
    validate_channel_indices(args, Cin, explicit_parameters=teacher_mode)
    args.basis_boundary, boundary_source = resolve_basis_boundary(
        args.basis_boundary, train_ds, test_ds
    )
    print(
        "ablation | operator={} boundary={} laplace_scaling={} loss_time_beta={}".format(
            args.operator_type, args.basis_boundary,
            int(args.use_layer_scaling), float(args.loss_time_beta)
        ),
        flush=True,
    )
    print(f"boundary selection | {boundary_source}", flush=True)
    log_runtime_status("after dataloader warmup")

    os.makedirs(args.save_dir, exist_ok=True)
    resolution_ops = ResolutionOpsCache(
        Kx=int(args.Kx), Ky=int(args.Ky), basis_cache_dir=args.save_dir, device=DEVICE,
        basis_boundary=args.basis_boundary,
    )
    Phix = Px = Phiy = Py = None
    if str(args.operator_type).lower() == "laplace":
        Phix, Px, Phiy, Py = resolution_ops.get_basis(Ny, Nx)
        kx_eff = _effective_modes_1d(Nx, int(args.Kx), args.basis_boundary)
        ky_eff = _effective_modes_1d(Ny, int(args.Ky), args.basis_boundary)
        kx_requested = _requested_basis_columns(int(args.Kx), args.basis_boundary)
        ky_requested = _requested_basis_columns(int(args.Ky), args.basis_boundary)
        if kx_eff < kx_requested or ky_eff < ky_requested:
            print(
                f"initial resolution caps basis columns: requested=({kx_requested},{ky_requested}) "
                f"using ({kx_eff},{ky_eff}) for Ny,Nx=({Ny},{Nx})",
                flush=True,
            )

    if int(args.basis_reconstruction) or int(args.reconstruction_only):
        if str(args.operator_type).lower() != "laplace":
            if int(args.reconstruction_only):
                raise ValueError("--reconstruction_only requires --operator_type laplace")
            print("basis reconstruction skipped: FFT operator has no mass-orthonormal Laplace basis", flush=True)
        else:
            evaluate_basis_reconstruction(
                {"train": train_ds, "test": test_ds}, resolution_ops, args,
            )
            if int(args.reconstruction_only):
                print("reconstruction_only=1: stop before model creation and training", flush=True)
                return

    spectral_d1_init, spectral_d2_init, d1_init_source, d2_init_source = (
        resolve_spectral_diffusion_init(
            args,
            teacher_dataset=teacher_dataset if teacher_mode else None,
            first_x=xb0,
        )
    )
    print(
        "learned spectral diffusion initialization | d1={:.8g} ({}) d2={:.8g} ({})".format(
            spectral_d1_init, d1_init_source, spectral_d2_init, d2_init_source
        ),
        flush=True,
    )

    out_prefix_base = os.path.join(args.save_dir, args.out_prefix)
    model = LENO2d(
        in_channels=Cin, out_channels=2,
        width=int(args.width), depth=int(args.depth),
        Kx=int(args.Kx), Ky=int(args.Ky),
        Phix=Phix, Px=Px, Phiy=Phiy, Py=Py,
        pad=int(args.pad),
        operator_type=args.operator_type,
        basis_boundary=args.basis_boundary,
        use_layer_scaling=bool(args.use_layer_scaling),
        time_channel_idx=int(args.time_channel_idx),
        dt_channel_idx=int(args.dt_channel_idx),
        d1_channel_idx=int(args.d1_channel_idx),
        d2_channel_idx=int(args.d2_channel_idx),
        time_channel_scale=float(args.time_channel_scale),
        dt_scale=float(args.dt_scale),
        Lx=float(args.Lx),
        Ly=float(args.Ly),
        spectral_d1_init=spectral_d1_init,
        spectral_d2_init=spectral_d2_init,
    )
    print_model_parameter_summary(model)
    log_first_batch_layer_scaling(
        model,
        xb0.to(DEVICE, dtype=torch.float32),
        _metadata_to_device(metadata0, DEVICE),
        Lx=float(args.Lx),
        Ly=float(args.Ly),
    )

    model = train_leno(
        model, train_loader, test_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        out_prefix=out_prefix_base,
        resolution_ops=resolution_ops,
        log_every=int(args.log_every),
        loss_time_beta=float(args.loss_time_beta),
        time_weight_floor=float(args.time_weight_floor),
        time_channel_idx=int(args.time_channel_idx),
        dt_channel_idx=int(args.dt_channel_idx),
        dt_scale=float(args.dt_scale),
        resume=bool(args.resume),
        eval_every=int(args.eval_every),
        amp=int(args.amp),
        amp_dtype=args.amp_dtype,
        lr_schedule=args.lr_schedule,
        profile_steps=int(args.profile_steps),
    )

    evaluate_full_time_test(model, test_loader, resolution_ops, args, out_prefix_base)

    if teacher_mode:
        if int(args.teacher_rollout) == 1:
            if int(args.rollout_all_test) == 1:
                rollout_keys = list(test_ds.trajectory_keys)
            else:
                requested_key = (int(args.teacher_rollout_source_id), int(args.teacher_rollout_sample_id))
                if requested_key[0] < 0 or requested_key[1] < 0:
                    requested_key = tuple(test_ds.trajectory_keys[0])
                if requested_key not in set(test_ds.trajectory_keys):
                    raise ValueError(f"teacher rollout trajectory must belong to the test split: {requested_key}")
                rollout_keys = [requested_key]
            t_snap_text = args.t_snap.strip() or "10,100,1000,5000,10000"
            t_snap = tuple(float(value) for value in t_snap_text.split(",") if value.strip())
            for number, requested_key in enumerate(rollout_keys, 1):
                source_tag = f"source{requested_key[0]}_sample{requested_key[1]}"
                print(f"rollout {number}/{len(rollout_keys)} | {source_tag}", flush=True)
                visualize_rollout_teacher_gm2d(
                    model=model,
                    teacher_dataset=teacher_dataset,
                    trajectory_key=requested_key,
                    out_prefix=out_prefix_base + "_" + source_tag,
                    resolution_ops=resolution_ops,
                    t_snap=t_snap,
                )
        if args.npz_dirs.strip():
            print("NOTE: --npz_dirs rollout is skipped in direct-teacher mode.", flush=True)
        return

    if int(args.rollout_all_test) == 1:
        rollout_jobs = legacy_test_rollout_jobs(test_ds, args.npz_dirs)
    else:
        if args.npz_dirs.strip() == "":
            print("npz_dirs is empty: skip visualization")
            return
        rollout_jobs = [(i, ddir.strip(), int(args.sample_id))
                        for i, ddir in enumerate(args.npz_dirs.split(",")) if ddir.strip()]

    t_snap_text = args.t_snap.strip() or "50,250"
    t_snap = tuple(float(x) for x in t_snap_text.split(","))
    for number, (source_id, ddir, sample_id) in enumerate(rollout_jobs, 1):
        tag = os.path.basename(ddir.rstrip("/"))
        outp = out_prefix_base + f"_{tag}_source{source_id}_sample{sample_id}"
        print(f"rollout {number}/{len(rollout_jobs)} | source={source_id} sample_id={sample_id}", flush=True)
        visualize_rollout_leno_gm2d(
            model=model,
            npz_dir=ddir,
            sample_id=sample_id,
            out_prefix=outp,
            Kx=int(args.Kx),
            Ky=int(args.Ky),
            T_final=float(args.T_final),
            t_snap=t_snap,
            use_teacher_forcing_until=int(args.teacher_forcing),
            dt_scale=float(args.dt_scale),
            basis_cache_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()


# Example command for the new direct-teacher experiment. Choose one teacher root:
#   /content/drive/MyDrive/gray_scott/F_0.01240_k_0.05000shards
#   /content/drive/MyDrive/gray_scott/F_0.0240_k_0.05000shards
# Numbered shards are selected with teacher_*.pt (quote it so this program expands
# and sorts the paths deterministically):
# python GM_LENO_FDM2dim.py \
#   --teacher_paths "/content/drive/MyDrive/gray_scott/F_0.01240_k_0.05000shards/teacher_*.pt" \
#   --save_dir "/content/drive/MyDrive/gray_scott/LENO_output/F_0.01240_k_0.05000" \
#   --out_prefix "GM_LENO_F_0.01240_k_0.05000" \
#   --num_train_trajectories 80 --num_test_trajectories 20 --split_seed 0 \
#   --T_final 10000 --dt_scale 10000 --time_channel_scale 10000 \
#   --basis laplace --Lx 2.5 --Ly 2.5 --Kx 32 --Ky 32 \
#   --width 64 --depth 4 \
#   --epochs 200 --batch_size 2 --lr_schedule step50 --loss_time_beta 0.75 \
#   --use_layer_scaling 1 --resume 0

# Historical notebook launcher: examples/fdm_colab_legacy.md
