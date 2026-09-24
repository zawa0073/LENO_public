"""Rectangular-grid LENO/FNO models and mass-orthonormal Laplace bases.

Read the basis builders, spectral layers, and LENO2d here. Data loading,
training, reconstruction diagnostics, and plots live in squaredomain.py.
The training entry point also re-exports these definitions for compatibility.
"""
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import diags
from scipy.linalg import cholesky, solve_triangular


# ------------------------------------------------------------------------
# Device and basis-file loading
# ------------------------------------------------------------------------


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ------------------------------------------------------------------------
# Boundary conventions
# ------------------------------------------------------------------------


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


# ------------------------------------------------------------------------
# Mass matrices and one-dimensional eigenbases
# ------------------------------------------------------------------------


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


# ------------------------------------------------------------------------
# Spectral layers and rectangular-grid neural operator
# ------------------------------------------------------------------------


def _resolve_dt_channel_idx(time_channel_idx, dt_channel_idx):
    dt_idx = int(dt_channel_idx)
    if dt_idx >= 0:
        return dt_idx
    if int(time_channel_idx) >= 0:
        return int(time_channel_idx) + 1
    return 4


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


# ------------------------------------------------------------------------
# Checkpoint compatibility
# ------------------------------------------------------------------------


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
