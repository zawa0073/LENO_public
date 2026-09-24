"""Disk-node LENO/FNO models, embeddings, and mass-orthonormal bases.

Read the basis builders, spectral layers, LENO2d, and DiskFNO2d here.
Data loading, training, diagnostics, and plots live in circledomain.py.
The training entry point also re-exports these definitions for compatibility.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------------
# Basis-file loading
# ------------------------------------------------------------------------


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


# ------------------------------------------------------------------------
# Disk geometry, quadrature weights, and eigenbases
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


def tensor_basis_to_device(basis: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "points": torch.as_tensor(basis["points"], dtype=torch.float32, device=device),
        "mass_w": torch.as_tensor(basis["mass_w"], dtype=torch.float32, device=device),
        "Phi": torch.as_tensor(basis["Phi"], dtype=torch.float32, device=device),
        "P": torch.as_tensor(basis["P"], dtype=torch.float32, device=device),
        "mu": torch.as_tensor(basis["mu"], dtype=torch.float32, device=device),
    }


# ------------------------------------------------------------------------
# Model input features
# ------------------------------------------------------------------------


def append_transition_dt(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"node input must have shape (B,N,C), got {tuple(x.shape)}")
    dt = dt.reshape(-1).to(device=x.device, dtype=x.dtype)
    if dt.shape[0] != x.shape[0]:
        raise ValueError(f"dt batch size {dt.shape[0]} does not match node-input batch size {x.shape[0]}")
    dt_nodes = dt[:, None, None].expand(-1, x.shape[1], 1)
    return torch.cat((x, dt_nodes), dim=-1)


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


# ------------------------------------------------------------------------
# Spectral layers and disk neural operators
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
# Legacy model names
# ------------------------------------------------------------------------


GMSFEMNO = LENO2d


GMSDiskFNO = DiskFNO2d


GeneralSpectralConv = SpectralConvNodes
