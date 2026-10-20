# cymetric performance benchmarks

A reproducible harness for the parts of `cymetric` that dominate a metric
training run, with a recorded baseline for upstream commit `be772af`.

It exists so that a performance claim about a fork can be checked by someone
who did not write the fork: one entry point, machine-readable output, full
provenance in every record, and compile time reported separately from
steady-state time.

## Quick start

```bash
# create the environment (conda; the harness needs jax, equinox, optax; the
# torch reference cases additionally need torch)
conda create -n cymfork python=3.12
conda activate cymfork
pip install "jax[cuda13]==0.12.0" equinox optax     # or jax[cuda12] / plain jax
pip install -r requirements-core.txt

# list the cases
python benchmarks/bench.py --list

# smoke test (small sizes, few repeats: NOT a baseline)
python benchmarks/bench.py --quick --cases 'train_jax.quintic'

# the full baseline, pinned to one GPU
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmarks/bench.py --out benchmarks/my-baseline.json

# re-print the table of a result file
python benchmarks/bench.py --table-from benchmarks/baseline-rtx5060ti.json
```

If jax reports `INTERNAL: libdevice not found at ./libdevice.10.bc` or
`An NVIDIA GPU may be present ... but a CUDA-enabled jaxlib is not installed`,
the CUDA plugin is incomplete: `pip install "jax[cuda13]==0.12.0"` (or
`jax-cuda13-plugin jax-cuda13-pjrt nvidia-cuda-nvcc nvidia-nvvm` for a
CUDA-13 wheel set that already has the other nvidia packages). The harness
does not fail in that case -- it falls back to CPU and labels every record
accordingly -- which is exactly the situation in which a baseline would
silently be worthless, so check the `device` line the harness prints before
trusting a number.

`bench.py` prepends its own repository root to `sys.path` and records the
resolved `cymetric.__file__` in the provenance block, so an editable install
somewhere else cannot silently be the thing that got measured. Point sets are
generated once and cached under `benchmarks/cache/` (gitignored); dataset
generation is setup and is never inside a measured window.

## What the harness guarantees

* **Compile time is separated from steady state.** Every timed quantity is
  reported as `first_call_s` (XLA compilation plus one execution) and then the
  median and the p25/p75 spread of `repeats` further calls, each one
  synchronised with `block_until_ready`. On the quintic the first training
  step is ~4000x the steady-state step, so a harness that reported only a mean
  over a few calls would be measuring the compiler, not the model.
* **Median, never mean**, plus min/max/p25/p75 and all raw samples in the JSON.
* **One subprocess per case**, so peak device memory, compile time and crashes
  are per-case. A case that cannot run is recorded with `status: blocked` or
  `error` and the exact exception text, and stays in the case list.
* **Peak device memory per case** from `jax.local_devices()[0].memory_stats()`
  (null on CPU, where the backend does not implement it).
* **CPU fallback**: with no GPU visible, jax falls back to CPU, the harness
  says so in the header and in every record's `device` block, and keeps going.

## The cases

The toric workload (Kreuzer-Skarke hypersurfaces over a range of h11) is the
target workload, so toric cases are present even where upstream cannot run
them; the CICY cases are the ones that currently produce numbers along a
growing-h11 axis.

### Point generation (`--groups pointgen`)

| case | what |
|---|---|
| `pointgen.quintic.points` | `PointGenerator.generate_points` on the Fermat quintic in P^4 (h11=1, 5 monomials): the line-intersection sampler, points only |
| `pointgen.quintic.weights` | `generate_point_weights(omega=True)` on the same geometry: what `prepare_dataset` actually calls (points + integration weights + Omega) |
| `pointgen.quintic_dense.*` | the same ambient space with the dense quintic (126 monomials, fixed pseudo-random complex coefficients, i.e. a generic complex structure). Isolates the monomial count from h11 |
| `pointgen.bicubic.*` | (3,3) in P^2xP^2, h11=2, 100 monomials |
| `pointgen.tetraquadric.*` | (2,2,2,2) in (P^1)^4, h11=4, 81 monomials |
| `pointgen.toric.h11_{1,5,10}` | `ToricPointGeneratorMC.generate_points` on KS hypersurfaces from `benchmarks/data/` |
| `pointgen.toric.h11_5.n500` | the same geometry at n_p=500: exposes the fixed per-call auto-tuning cost of the Metropolis sampler |

The toric sampler re-tunes its step size and thinning on *every* call (the
tuned values are not cached on the generator unless passed to the
constructor), and that cost is inside the measurement, exactly as a user pays
it.

### Metric training, JAX (`--groups train_jax`)

`PhiFSModel` (`g = g_FS + ddbar phi`) with the standard cymetric network from
`notebooks/0.GettingStarted.ipynb`: 3 hidden layers of width 64, gelu, scalar
output. The timed call is the jitted step built by
`cymetric.jax.models.helper._make_train_step` -- the function `train_model`
runs in its inner loop -- on batches that are already on the device, cycling
over distinct batches so that nothing is served from a single cached slice.

* `train_jax.quintic.bs{64,1024,10000}`: batch 64 is the `train_model` phase-1
  default, 10000 the phase-2 default.
* `train_jax.bicubic.bs1024`, `train_jax.tetraquadric.bs1024`: the h11 axis
  (h11 = 1, 2, 4; 5, 6, 8 ambient coordinates; 1, 2, 4 projective factors).

Each record carries both `ms_per_step_median` (individually synchronised
steps) and `ms_per_step_pipelined` (a contiguous block of steps with a single
synchronisation at the end, which is what an epoch actually does), plus
`s_per_epoch_est = ms_per_step_pipelined * (n_train / batch_size)`.

### det g (`--groups detg_jax`)

`det g` over a point set in fixed-size chunks, jitted as one function, for the
quintic and the tetraquadric. This is the hot spot every downstream measure
(sigma, Ricci, volumes) is built on: `PhiFSModel.__call__` builds a per-point
Hessian of the network with `vmap(jacobian(grad(phi)))` before contracting
with the pullbacks, so its memory grows with batch x (2*ncoords)^2.

### Toric metric models (`--groups train_jax_toric,metric_torch_toric`)

These are the target workload and they are **blocked upstream**; see the next
section. They are kept in the harness so that the day the JAX toric model can
be constructed, `python benchmarks/bench.py --groups train_jax_toric` produces
numbers instead of errors.

## Geometry inputs

`benchmarks/data/toric_h11_{1,5,10}.json` hold cymetric's `toric_data` dicts
(anticanonical hypersurface, sections of the Kaehler-cone generators, GLSM
charges, patch masks of the maximal cones, volume normalisation) plus the
matching `kmoduli` and intersection numbers, together with a provenance block
naming the Kreuzer-Skarke polytope they come from. Upstream produces that dict
inside SageMath (`cymetric.sage.sagelib.prepare_toric_cy_data`); these files
were produced with CYTools from the recorded polytope vertices, because this
box has no SageMath. `benchmarks/make_toric_data.py --check` re-validates them
by building cymetric's own toric point generator from each one;
`make_toric_data.py --from-pickle ...` converts a sage- or cytools-produced
pickle into the same format.

One harness-side adjustment is needed and is called out in the code
(`cases.toric_basis`): cymetric's toric point generators do not produce an
`AMBIENT` entry satisfying the base model's `sum(AMBIENT + 1) == ncoords`
requirement, since `ToricPointGenerator` overwrites `self.ambient` with a
sampling helper, so the harness supplies a partition of `ncoords` into
`len(sections)` blocks. The toric metric reads only the number of blocks. No
library code is modified by the harness.

## Cases that could not be made to work, and why

Every one of these is a defect in upstream `be772af`, reproduced from an
unmodified library (`provenance.git.library_is_pristine` is true in the
baseline file). They are recorded, not worked around.

1. **The JAX toric metric model cannot be constructed at all.**
   `ToricModel._generate_helpers` is defined in
   `cymetric/jax/models/models.py:559` and that line is the *only* occurrence
   of the name in the whole JAX backend -- nothing ever calls it, while
   `cymetric/torch/models/fubinistudy.py:62` and
   `cymetric/tensorflow/models/fubinistudy.py:45` both call theirs. So
   `patch_degrees`, `transition_coefficients` and `transition_degrees` are
   never assigned, and equinox refuses the half-built module:

   ```
   TypeError: Field 'patch_degrees' was not initialized.
   ```

   (case `train_jax.toric.h11_1`, status `blocked`).

2. **At h11 > 1 the toric construction fails earlier, in every backend that
   calls `_generate_helpers` through `super()`.** The base class'
   `__init__`/`_generate_helpers` builds `fixed_patches` by calling the
   *toric* `_generate_all_patches` override while `self.nTransitions` still
   holds the CICY value derived from `AMBIENT`, instead of the toric
   `max_i #{patches with z_i = 0}`:

   ```
   ValueError: shape mismatch: value array of shape (12,12) could not be
   broadcast to indexing result of shape (12,4)
   ```

   This is what `train_jax.toric.h11_5`, `train_jax.toric.h11_10` and
   `metric_torch.toric.h11_5` report. The TensorFlow backend does not have
   the defect, because its `ToricModel._generate_helpers`
   (`cymetric/tensorflow/models/models.py:874`) overrides the base method
   *without* calling `super()` and sets `nTransitions` before building
   `fixed_patches`. That ordering is the reference for a JAX fix: calling a
   JAX `_generate_helpers` that delegates to the base one first would
   reproduce defect 2 instead of fixing defect 1.

3. **The torch toric model cannot train even at h11 = 1.** The forward pass
   (and hence `det g`) works, but `compute_transition_loss`, and therefore
   `compute_loss`, raises

   ```
   IndexError: The shape of the mask [10240, 5] at index 0 does not match the
   shape of the indexed tensor [2048, 5] at index 0
   ```

   (batch 512 x nTransitions 4 = 2048 rows, mask built at x nPatches 5).
   `metric_torch.toric.h11_1` therefore times the forward `det g` only and
   stores that error in `info.compute_loss_error`.

4. **The torch toric model is CPU-only here.** Constructed with
   `device='cuda'` it keeps index buffers on the host and the forward raises
   `RuntimeError: indices should be either on cpu or on the same device as
   the indexed tensor (cpu)`; the case falls back to CPU and records the GPU
   error in `info.device_errors`. Its timings are therefore CPU timings and
   are *not* comparable to the JAX GPU numbers.

5. **No toric JAX det g / h11-scaling curve.** This was the measurement most
   wanted (cost versus h11 on the real workload) and it is unobtainable on
   this commit: defects 1 and 2 block the only class that evaluates a toric
   Fubini-Study metric. The CICY cases (h11 = 1, 2, 4) are the stand-in, and
   they grow the ambient coordinate count and the number of projective
   factors, but not the toric section structure, which is what actually
   explodes at high h11 (the h11 = 10 geometry here has 39 section monomials
   over 12 Kaehler-cone generators; at h11 = 15 the same family has 389).

6. **Not benchmarked: TensorFlow.** It is the one backend whose toric model
   constructs at every h11 tested, so it would give the h11 curve -- but the
   fork's work is on the JAX and torch paths, and adding a TF case family
   would have meant a second set of timing conventions (`tf.function`
   retracing) for numbers nobody will compare against. Stated here so the
   omission is a choice, not an oversight.

7. **Not benchmarked: the Mathematica and Sage point generators**
   (`ToricPointGeneratorMathematica`, `pointgenIPS_mathematica`,
   `sagelib`). They need external kernels that are not installed on the
   reference box. `ToricPointGeneratorMC` is the only toric generator that
   runs from a plain Python environment, and it is the one benchmarked --
   note that its own docstring warns its integration weights are known to be
   wrong, so it is a *performance* reference only.

## How to read the numbers, and what is confounded

* **Shared box.** The baseline was taken on a workstation that another job
  was using at the same time: the GPU cases were serialised against it with a
  `flock` and were alone on the reference card, but the *host* side (point
  generation, which is pure numpy/joblib on the CPU, and the Python dispatch
  in every GPU case) competed with it for 24 CPU threads at a load average
  that rose from 3 to 7 during the run. The p25-p75 columns are the honest
  measure of how much that matters, and it is not uniform: the large-batch GPU
  steps and the point-generation rates are reproducible to a few percent
  (`bs10000`: 15.1-15.6 ms; `pointgen.quintic.points`: 164.0k-165.7k
  points/s), but the *small*-batch GPU steps are host-latency bound and spread
  by +-25% (`bs64`: 0.59-0.95 ms), and the torch CPU case spreads by 5x. Read
  the small-batch step times as "under a millisecond", not as 0.79.
* **Point generation is dominated by the monomial count, not by h11.** The
  Fermat quintic (5 monomials) and the dense quintic (126 monomials) live in
  the same ambient space with the same h11 and differ by ~40x. Any
  "points/s versus h11" plot that mixes Fermat with generic hypersurfaces is
  measuring the complex structure, not the geometry's h11.
* **`s_per_epoch_est` is an extrapolation** from a 32-step block, and it
  covers phase 1 only. `train_model_jax.*` is the end-to-end number, with
  both phases, host-side shuffling and compilation included; prefer it when
  quoting a per-epoch cost.
* **`first_call_s` is compile + execute, not compile.** Subtract the median to
  get a compile estimate. For the training step on the quintic that is ~5 s
  against ~1 ms of steady-state work, so any benchmark that averages over
  "a few" steps is reporting the compiler.
* **Peak device memory is per process and monotone.** It is the peak over
  everything the case did, including the dataset upload and the pullbacks, not
  only the timed call. It is the right number for "will this fit", not for
  attributing bytes to one op.
* **det g throughput is measured on data already on the device**, in chunks,
  with the pullbacks precomputed (as `train_model` does). Add host transfer
  and pullback construction for an end-to-end figure.
* **The toric sampler numbers include its per-call auto-tuning**, which is why
  `pointgen.toric.h11_5.n500` has a much worse per-point rate than the same
  geometry at n_p = 2000.

## Baseline

Recorded on 2026-10-20 (UTC 16:17) on `compute02`: **NVIDIA GeForce RTX 5060 Ti**
(16 GB, driver 595.99.02, `CUDA_VISIBLE_DEVICES=1`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`), 24 CPU threads, Linux 7.0.0,
python 3.12.15, jax/jaxlib 0.12.0, equinox 0.13.8, optax 0.2.8, numpy 2.5.4,
torch 2.14.1+cu130, library commit `be772af` with the harness commit on top
(`library_is_pristine: true` -- no file under `cymetric/` was modified).
Full records, including every raw sample, in
[`baseline-rtx5060ti.json`](baseline-rtx5060ti.json); `python
benchmarks/bench.py --table-from benchmarks/baseline-rtx5060ti.json` reprints
the table below. 25 of 29 cases ran; 4 are blocked upstream.

### 1. Point generation (CPU, 4 joblib workers for the CICY sampler)

| case | points/s | p25-p75 | first call | median | ncoords | monomials |
|---|---|---|---|---|---|---|
| `pointgen.quintic.points` | **165 553** | 164 024 - 165 685 | 0.46 s | 0.121 s | 5 | 5 |
| `pointgen.quintic.weights` | **51 137** | 50 212 - 51 750 | 0.83 s | 0.391 s | 5 | 5 |
| `pointgen.quintic_dense.points` | 4 351 | 4 347 - 4 353 | 5.48 s | 4.60 s | 5 | 126 |
| `pointgen.quintic_dense.weights` | 710 | 709 - 711 | 28.5 s | 28.2 s | 5 | 126 |
| `pointgen.bicubic.points` | 11 378 | 11 342 - 11 513 | 2.68 s | 1.76 s | 6 | 100 |
| `pointgen.bicubic.weights` | 3 522 | 3 513 - 3 550 | 6.73 s | 5.68 s | 6 | 100 |
| `pointgen.tetraquadric.points` | 9 969 | 9 805 - 9 970 | 3.21 s | 2.01 s | 8 | 81 |
| `pointgen.tetraquadric.weights` | 2 807 | 2 806 - 2 835 | 8.23 s | 7.12 s | 8 | 81 |
| `pointgen.toric.h11_1` | **105** | 104 - 105 | 19.4 s | 19.1 s | 5 | 126 |
| `pointgen.toric.h11_5` | **205** | 204 - 205 | 9.62 s | 9.78 s | 9 | 50 |
| `pointgen.toric.h11_5.n500` | 131 | 131 - 131 | 3.83 s | 3.82 s | 9 | 50 |
| `pointgen.toric.h11_10` | **90** | 90 - 91 | 22.1 s | 22.1 s | 14 | 29 |

Point generation, not the metric network, is what sets the cost of a toric
training set: 20 000 points on the h11 = 10 geometry is ~3.7 minutes of
single-core Python, against 0.12 s for the Fermat quintic. The honest
apples-to-apples comparison is the h11 = 1 row of the toric sampler against
`quintic_dense`, because that KS geometry *is* a dense quintic in P^4 (5
coordinates, 126 monomials): 105 points/s versus 4 351, i.e. the Metropolis
sampler costs **41x** more per point than the line sampler on the same
hypersurface. Against the Fermat quintic the ratio is 1 577x, but most of that
is the complex structure, not the sampler. Note also that the toric rate does
not fall monotonically with h11 -- h11 = 5 is the fastest of the three --
because the chain's cost follows the hypersurface's monomial count (126, 50,
29 respectively) and the auto-tuned thinning, not h11; comparing
`pointgen.toric.h11_5` at n_p = 2000 (205 points/s) with the same geometry at
n_p = 500 (131 points/s) separates the two: a fixed
~1.8 s per call of initial point search and auto-tuning, and a chain that then
runs at ~250 points/s.

### 2. JAX metric training and det g (GPU 1)

| case | compile (first call) | ms/step median | p25-p75 | ms/step pipelined | s/epoch | peak dev mem |
|---|---|---|---|---|---|---|
| `train_jax.quintic.bs64` | 4.15 s | **0.79** | 0.59 - 0.95 | 0.60 | 0.17 (281 steps) | 199 MB |
| `train_jax.quintic.bs1024` | 4.87 s | **1.28** | 1.20 - 1.51 | 1.79 | 0.03 (17 steps) | 214 MB |
| `train_jax.quintic.bs10000` | 5.46 s | **15.4** | 15.1 - 15.6 | 15.0 | 0.015 (1 step) | 574 MB |
| `train_jax.bicubic.bs1024` | 4.79 s | 1.63 | 1.53 - 1.70 | 1.52 | 0.026 | 216 MB |
| `train_jax.tetraquadric.bs1024` | 5.12 s | 2.60 | 2.44 - 2.69 | 2.51 | 0.043 | 296 MB |
| `train_model_jax.quintic` | 17.5 s / 3 epochs | - | - | - | **2.04** end-to-end | 543 MB |
| `train_model_jax.tetraquadric` | 17.4 s / 3 epochs | - | - | - | **2.10** end-to-end | 731 MB |
| `detg_jax.quintic.n20000` | 1.10 s | 2.65 per 15k points | 2.52 - 2.76 | - | 5.7e6 points/s | 184 MB |
| `detg_jax.tetraquadric.n20000` | 1.12 s | 5.17 per 15k points | 5.13 - 5.22 | - | 2.9e6 points/s | 264 MB |

Three things stand out.

* **Compilation is 5 200x the step.** 4.2 s to compile a step that then takes
  0.79 ms. Anything that retraces -- a new batch shape, a changed
  `learn_*` flag, a rebuilt step function -- pays it again. `train_model`
  switches flags between its two phases every epoch, which is why its first
  call costs 17.5 s against 6.1 s for the next identical call.
* **The default batch size wastes the GPU.** 64 points cost 0.79 ms (12 us per
  point); 10 000 points cost 15.4 ms (1.5 us per point). Phase 1 of
  `train_model` runs at batch 64, so it is running an 8x-underutilised device.
* **Most of an epoch is not the metric.** The measured end-to-end epoch on the
  quintic is 2.04 s, while the steps it contains account for about 0.20 s
  (281 x 0.60 ms phase-1 steps plus two phase-2 steps). Roughly 90% of the
  epoch is host-side: `np.random.permutation` over the training set and a
  numpy fancy-index gather of every batch out of device arrays, per step, plus
  the per-step `eqx.tree_at` rebuilds and `float()` syncs. That is where the
  easy factor of several is, and it is independent of the geometry (the
  tetraquadric epoch is only 3% slower than the quintic epoch despite a 4x
  more expensive step).

### 3. Cost versus h11 on the toric workload (numpy, CPU)

The JAX toric model cannot be built (section 4), so the h11 curve is measured
on `ToricPointGeneratorMC.fubini_study_metrics`, the numpy original that
`cymetric.jax.models.models.ToricModel._fubini_study_n_metrics` mirrors
formula for formula. Same batch of 500 points throughout.

| case | generators | sections per generator | sum of sections^2 | ncoords | FS metric points/s | det g points/s | median |
|---|---|---|---|---|---|---|---|
| `fsmetric_toric_numpy.h11_1` | 1 | 5 | 25 | 5 | **235 225** | 204 079 | 2.13 ms |
| `fsmetric_toric_numpy.h11_5` | 6 | 4,2,2,3,3,3 | 51 | 9 | **30 729** | 29 815 | 16.3 ms |
| `fsmetric_toric_numpy.h11_10` | 12 | 5,2,2,3,4,5,2,3,3,4,4,2 | 141 | 14 | **4 852** | 4 746 | 103 ms |

Cost per point grows **48x from h11 = 1 to h11 = 10** on these geometries.
Most of that is an implementation artefact, not geometry. The routine's two
contractions are written as

```python
np.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees)
- np.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees)
```

and `np.einsum` defaults to `optimize=False`, so both are evaluated as naive
double sums over the sections `a, b` even though both factorise (the first is
`(sum_b mss_b) * (sum_a mss_a d_ai d_aj)`, the second is the outer product of
`sum_a mss_a d_ai` with itself). Measured on the h11 = 10 geometry, for the
largest generator and the same 500 points:

| variant | time | agreement with as-written |
|---|---|---|
| as written (`optimize` default) | 15.5 ms | - |
| `optimize=True` | **0.3 ms** | 1.8e-15 relative |
| hand-factorised | 2.5 ms | 1.4e-15 relative |

That is ~50x on the contraction that dominates the h11 = 10 case, for one
keyword, bit-for-bit equivalent. Two consequences for reading the curve
above: (a) the numpy toric path -- which is what the toric point generator and
its weights use today -- has a cheap and large win available; and (b)
`jax.numpy.einsum` defaults to `optimize='auto'`, so a JAX port of the same
formula probably does *not* inherit the quadratic term, and this curve is an
upper bound on what the JAX toric model should cost, not a prediction of it.
The residual h11 dependence -- sections per generator, number of generators,
and `ncoords^2` in the metric's index pair -- is real and will show up in any
backend.

### 4. Blocked cases, verbatim

| case | status | error |
|---|---|---|
| `train_jax.toric.h11_1` | blocked | `TypeError: Field 'patch_degrees' was not initialized.` |
| `train_jax.toric.h11_5` | blocked | `ValueError: shape mismatch: value array of shape (12,12) could not be broadcast to indexing result of shape (12,4)` |
| `train_jax.toric.h11_10` | blocked | `ValueError: shape mismatch: value array of shape (12,12) could not be broadcast to indexing result of shape (12,2)` |
| `metric_torch.toric.h11_5` | blocked | same shape mismatch, on both cuda and cpu |
| `metric_torch.toric.h11_1` | ok, but | forward only: `compute_loss` raises `IndexError: The shape of the mask [10240, 5] ... does not match ... [2048, 5]`, and the model is CPU-only (`info.device_errors`) |

The `metric_torch.toric.h11_1` timing (96.6 ms per 512-point batch, p25-p75
57-324 ms) is a *CPU* number taken while the box was loaded, and its spread is
5x: treat it as an order of magnitude, nothing more. Its `peak_dev_mem` of
0 MB is an artefact of reading JAX's device counters in a case that uses torch.
