# LENO training: rectangular grids and disk nodes

This repository provides code for learning one-step time evolution from reaction-diffusion data on rectangular grids and disks.
Shared operations use consistent names, while operations specific to a domain or data layout use the `Disk` or `Nodes` suffix.
The entry points are `squaredomain.py` and `circledomain.py`.
Model definitions and basis operations are in `model_squaredomain.py` and `model_circledomain.py`, respectively.
**Keep all four Python files in the same directory.**

## File structure and suggested reading order

| File | Contents |
|---|---|
| [squaredomain.py](squaredomain.py) | Data loading, training, evaluation, plotting, and CLI for square and rectangular grids |
| [model_squaredomain.py](model_squaredomain.py) | Grid mass matrices, Neumann and periodic bases, spectral layers, and `LENO2d` |
| [circledomain.py](circledomain.py) | Data loading, training, evaluation, plotting, and CLI for disks |
| [model_circledomain.py](model_circledomain.py) | Disk bases, input normalization layers, spectral layers, `LENO2d`, and `DiskFNO2d` |

To review the mathematics and model architecture, start with basis construction in `model_*.py`,
then read the spectral layers and the model's `forward` method.
To follow the experiment workflow, start with `main` in each entry point.
The model modules do not import the training scripts, so you can use the models independently in other experiments.

```python
from model_squaredomain import LENO2d as SquareLENO2d
from model_circledomain import LENO2d as CircleLENO2d
```

The same model classes and basis functions can also be imported from the entry point modules.
The file split and renaming preserve the mathematical operations, default training settings, and `state_dict` keys.
Checkpoints produced by the previous training scripts remain usable.
Update any commands or import statements that still use the old filenames.

## Features by domain

| | `squaredomain.py` | `circledomain.py` |
|---|---|---|
| Domain | Rectangle | Disk |
| Input state | `(B, Ny, Nx, Cin)` | `(B, N, 4)` with a separate time step `(B,)` |
| Output state | `(B, Ny, Nx, 2)` | `(B, N, 2)` |
| Laplace model | `LENO2d` | `LENO2d` |
| Laplace spectral layer | `SpectralConvSeparable` | `SpectralConvNodes` |
| Fourier spectral layer | `SpectralConvFFT2d` | `SpectralConvFFT2d` |
| Fourier model for disks | — | `DiskFNO2d` |
| Training and evaluation | `train_leno`, `eval_epoch` | Same names |
| Training history export | `save_training_history` | Same name |
| Full-time test evaluation | `evaluate_full_time_test` | Same name |
| Single-trajectory visualization | `visualize_rollout_teacher_gm2d` | Same name |

Functions with the same name use different bases, mass weights, and data arguments depending on the domain.
Models and checkpoints are therefore specific to their domain and cannot be used interchangeably between the two scripts.

## Running the code

Python 3.10 or later, NumPy, SciPy, PyTorch, and Matplotlib are required.
The optional psutil package is used to report RAM usage.

```bash
python -m pip install numpy scipy torch matplotlib psutil
```

The following example uses disk data. Replace the path with the location of your teacher data files.
`--teacher_paths` accepts multiple paths, comma-separated paths, and wildcards.
The FEM pipeline also uses CPU memory mapping and retrieves only the required transitions,
without concatenating all files' x/y arrays.
Use `--teacher_mmap 0` to switch to regular loading.

```bash
python circledomain.py \
  --teacher_paths "data/disk/teacher_[0-9]*.pt" \
  --dt_scale 50 --train_fraction 0.8 --split_seed 0 \
  --save_dir outputs/disk --out_prefix leno_disk \
  --Nr 128 --Ntheta 256 --basis_type disk --K 128 \
  --width 64 --depth 4 --epochs 200 --batch_size 1 \
  --lr_schedule step50 --use_layer_scaling 0 --resume 0 \
  --teacher_rollout 1 --t_snap "T/2,T"
```

Set `--dt_scale` to match `T` in the teacher data.
The FEM pipeline reads the time step in this order of precedence:
`dt_physical`, `dt_normalized`, `dt`, and the legacy input channel.
Units are inferred from metadata or differences between saved times.
If the units cannot be determined, specify `--teacher_dt_units physical` or `normalized`.
Disk coordinates and the four-channel input with a separate dt are preserved.
The FDM checks for the strict 256×256 rectangular grid and fixed saved-time specification do not apply to disk data.

By default, both scripts split the trajectories across all files into 80% training and 20% test data.
Use `--num_train_trajectories` / `--num_test_trajectories` to limit the number of trajectories selected after splitting
(`-1` selects all candidates). The same `--split_seed` reproduces the selection.
`--train_sample_ids` / `--test_sample_ids` accept comma-separated original sample_id values or `source_id:sample_id` pairs.
Using `--max_train_samples` / `--max_test_samples` in a way that truncates a trajectory raises an error.
Legacy data without sample_id/time_index is supported when trajectory order can be reconstructed from saved times.
Individual transitions are not randomly split between training and test sets.

The following example uses direct teacher data for a rectangular domain.
`--teacher_strict_standard 0` disables checks against the fixed experiment specification.
Set the times, boundary conditions, and domain size to match your teacher data.

```bash
python squaredomain.py \
  --teacher_paths data/rectangle/teacher.pt \
  --teacher_strict_standard 0 \
  --save_dir outputs/rectangle --out_prefix leno_rectangle \
  --operator_type laplace --Kx 16 --Ky 16 \
  --width 64 --depth 4 --epochs 200 --batch_size 1 \
  --lr_schedule step50 --use_layer_scaling 0 --resume 0 \
  --teacher_rollout 1 --t_snap "50,250"
```

The line continuations in these code blocks use bash syntax. In PowerShell, run each command on a single line.
In Colab, place all four Python files in the same working directory and run
`!python circledomain.py ...` or `!python squaredomain.py ...`.
Do not include `%%writefile` inside the `.py` files themselves.
To list the available arguments, run:

```bash
python squaredomain.py --help
python circledomain.py --help
```

Teacher data is not included in this repository. Prepare the data for your experiment separately
and replace `--teacher_paths` in the examples with the actual location.

## Basis reconstruction error for teacher data

By default, both scripts project the teacher u/v fields onto the configured basis before training
and compute the relative spatial L2 error `||u-u_K||_M / ||u||_M`.
FEM uses nodal mass weights; FDM uses the same mass matrix as the basis.
The evaluation covers every selected training and test transition.
Input states `x[..., :2]` and next-step targets `y[..., :2]` are aggregated separately,
without standardization or time weighting.
Each transition receives equal weight, so the same state may appear as both the target of one transition
and the input of the next.

| Option | Default | Description |
|---|---|---|
| `--basis_reconstruction` | `1` | Set to `0` to skip reconstruction evaluation before training |
| `--reconstruction_tol` | `0.01` | Relative L2 error tolerance; 0.01 means 1% |
| `--reconstruction_only` | `0` | Set to `1` to evaluate reconstruction only and exit before model creation and training |

Add `--reconstruction_only 1` to your usual arguments to inspect the basis without training.
Evaluation still runs when this option is combined with `--basis_reconstruction 0`.
Do not use the FEM option `--inspect_only 1` for this evaluation, as it exits before basis construction.

Results are saved as CSV files using `save_dir/out_prefix` with the following suffixes:

- `_basis_reconstruction_curve.csv`: Mean and maximum relative L2 errors for K=1 through the configured effective basis size.
- `_basis_reconstruction_summary.csv`: Errors using the full configured basis, the tolerance,
  `min_K_mean` (the smallest K satisfying the mean-error tolerance), and
  `min_K_all` (the smallest K satisfying the tolerance for every evaluated state).
  Minimum-K fields are left blank if the tolerance is not reached within the configured range.

CSV results are grouped by split, input/target, resolution, and u/v/combined.
`combined` is computed from the sum of the squared u/v norms.
A zero field with zero reconstruction error is assigned a relative error of 0.
To keep both components of every teacher state within 1%, check `min_K_all` in the u/v rows
for both input and target in each split, and use the largest value.
If any of these fields is blank, increase the basis size and evaluate again.

K denotes **the first K modes in the configured basis, sorted by ascending eigenvalue**.
Modes with equal eigenvalues retain their original column order and are evaluated one column at a time.
For FDM, K is the total number of modes, up to `Kx_eff * Ky_eff`;
the minimum K should not be used directly as `--Kx` or `--Ky`.
The FEM disk basis includes degenerate modes, so its effective size may exceed `--K`.
The evaluation does not add modes outside the configured range or automatically change the model basis or training settings.

Supported bases are FEM disk, Cartesian, and loaded nodal bases, and FDM
`--operator_type laplace` with Neumann or periodic boundary conditions.
For Fourier FNO models, regular training skips this evaluation and prints the reason;
`--reconstruction_only 1` raises an error because reconstruction evaluation is unsupported.
Only the diagnostic basis is orthogonalized with respect to the mass inner product in float64.
The full-basis residual is computed directly, then the squared coefficients of omitted modes are added.
This reduces cancellation when errors are small.

Basis reconstruction has been validated against an independent least-squares solution using a dense mass matrix.

## Shared variable names

| Name | Meaning |
|---|---|
| `xb`, `yb`, `yh` | Batches of inputs, teacher outputs, and model predictions |
| `hist` | Training history for each epoch |
| `out_prefix` | Output filename prefix including the directory; the CLI argument `args.out_prefix` contains only the name portion |
| `opt`, `sched`, `scaler` | Optimizer, learning rate scheduler, and AMP GradScaler |
| `amp_torch_dtype` | PyTorch dtype used for AMP |
| `tr_u_sum`, `tr_v_sum`, `tr_n` | Accumulated training errors and sample count |
| `loss_each`, `loss_u_each`, `loss_v_each` | Per-sample relative L2 errors for u/v |
| `loss_weights` | Training loss weights computed from cumulative physical time |
| `t_cumulative_b` | Cumulative physical time at the target of each transition |
| `dt_scale` | Scale used to normalize and recover input time steps |
| `dt_physical` / layer variable `dt_scalar` | Time step in physical units |
| `spec`, `w1`, `fc0`, `fc1`, `fc2` | Spectral layers, pointwise linear layers, and input/output linear layers |
| `Phi`, `P`, `mu` | Disk eigenfunctions, mass-weighted projection, and Laplace eigenvalues |

Time-dependent layers are disabled by default in both scripts.
The FEM formulation for time-dependent layers is unchanged.
The grid and disk implementations use different bases, mass weights, and input standardization.
Each domain has its own model module and entry point, and functions with the same name
may accept different arguments and data shapes.

## Validation

```bash
python squaredomain.py --smoke_test
```

This smoke test runs without teacher data.

The file reorganization was checked using 34 existing tests in the development environment.
These covered small CPU training runs, checkpoint resumption, Fourier models, physical integration,
output formats, and compatibility with legacy CLI options and model weights.
Across 10 model configurations, the bases, predictions, gradients, Adam updates,
and results resumed from checkpoints created before the split were also confirmed to match
before and after the reorganization.
The validation environment used Python with PyTorch 2.8.0+cpu, NumPy 2.2.6, and SciPy 1.15.3.
