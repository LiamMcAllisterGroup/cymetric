# This fork

A fork of [ruehlef/cymetric](https://github.com/ruehlef/cymetric), branched at
`be772af`, maintained for the McAllister group at Cornell.

Upstream is the reference implementation and is where general bug reports
should go. This fork exists because the group's workload is **toric /
Kreuzer-Skarke geometries across a range of h11**, and on that workload
upstream's JAX backend does not run at all. Upstream has not historically
prioritised performance; that is the other reason this fork exists, and the
reason it carries a benchmark harness.

## What is wrong with upstream, as measured

All three were verified against `be772af`, not inherited from older notes. The
exact errors are recorded in `benchmarks/README.md` as blocked cases, so they
turn into passing cases when fixed rather than being forgotten.

1. **The JAX toric model cannot be constructed.** `_generate_helpers`
   (`cymetric/jax/models/models.py:559`) is the only occurrence of that name in
   the whole JAX backend -- nothing calls it, although torch
   (`torch/models/fubinistudy.py:62`) and TensorFlow
   (`tensorflow/models/fubinistudy.py:45`) both do. So `patch_degrees`,
   `transition_coefficients` and `transition_degrees` are declared fields that
   are never assigned, and h11 = 1 fails with
   `TypeError: Field 'patch_degrees' was not initialized.`

2. **A second, backend-independent defect blocks toric at h11 >= 5.** Both JAX
   and torch raise
   `ValueError: shape mismatch: value array of shape (12,12) could not be broadcast to indexing result of shape (12,4)`
   in `ToricModel._generate_all_patches`, because the base class builds
   `fixed_patches` while `nTransitions` still holds the CICY value. TensorFlow
   is correct only because its `ToricModel._generate_helpers`
   (`tensorflow/models/models.py:874`) overrides *without* `super()` and sets
   `nTransitions` first.

   **This matters for anyone fixing defect 1:** a JAX `_generate_helpers` that
   delegates to the base implementation before setting the toric `nTransitions`
   converts defect 1 into defect 2, and the result passes at h11 = 1 while
   still failing at h11 >= 5. Test at h11 >= 5.

3. **The torch toric model cannot train even at h11 = 1.** The forward pass
   works; `compute_transition_loss` / `compute_loss` raise
   `IndexError: mask [10240, 5] vs indexed tensor [2048, 5]`. With
   `device='cuda'` the forward pass itself raises a host/device index error.

## Where the time actually goes

Measured, RTX 5060 Ti, upstream `be772af`. These are the facts that should
drive any optimisation work, and two of them are not where one would guess.

- **About 90% of a training epoch is not the metric step.** A quintic epoch
  measures 2.04 s; the steps inside it account for about 0.20 s. The remainder
  is `train_model`'s host side: `np.random.permutation` over the training set
  plus a numpy fancy-index gather of each batch *out of device arrays*, every
  step, plus per-step `eqx.tree_at` rebuilds and `float()` synchronisations.
  It is geometry-independent -- the tetraquadric epoch is 3% slower than the
  quintic's despite a step that costs twice as much.
- **A faster GPU barely helps, which is the same finding from the hardware side.**
  The identical sweep on an RTX 5090 against the RTX 5060 Ti:

  | case | 5060 Ti | 5090 | gain |
  |---|---|---|---|
  | step, batch 64 | 0.79 ms | 0.91 ms | 0.87x |
  | step, batch 1024 | 1.28 ms | 1.10 ms | 1.16x |
  | step, batch 10000 | 15.41 ms | 3.97 ms | **3.88x** |
  | step, tetraquadric batch 1024 | 2.60 ms | 0.98 ms | 2.64x |
  | det g, quintic | 2.65 ms | 1.38 ms | 1.91x |
  | `train_model`, quintic | 6.11 s | 5.36 s | **1.14x** |
  | `train_model`, tetraquadric | 6.29 s | 6.03 s | **1.04x** |

  A card roughly 4x faster on the arithmetic buys 4 to 14 per cent end to end,
  and at batch 64 it is *slower*, which is pure kernel-launch latency. So the
  available win is in the host loop, not in the device or in the kernels.

- **Compilation dominates anything that retraces**: 4.2 s to compile a 0.79 ms
  step, a factor of 5200. `train_model` flips its `learn_*` flags between two
  phases every epoch; its first three-epoch call costs 17.5 s against 6.1 s
  for the next.
- **The default batch size wastes the device**: 12 us/point at batch 64 against
  1.5 us/point at batch 10000, and phase 1 runs at 64.
- **Toric cost today is point generation, not the network**: 105 points/s for
  the toric Metropolis chain against 4351 points/s for the line sampler on the
  *same* dense quintic, i.e. about 3.7 minutes of single-core Python for 20k
  points at h11 = 10.
- **One keyword in the numpy toric metric**: both contractions in
  `_fubini_study_n_metrics` factorise and `np.einsum` defaults to
  `optimize=False`. On the h11 = 10 geometry's largest generator, 500 points:
  15.5 ms as written, 0.3 ms with `optimize=True`, agreeing to 1.8e-15.
  (`jnp.einsum` defaults to `optimize='auto'`, so the JAX path may already be
  fine -- check before changing it.)

## Benchmarks

`benchmarks/` carries a harness and a baseline for pristine upstream. It is the
gate for every performance change here: the rule is that a change lands with a
test showing the new route agrees with the old to a stated tolerance, and with
before/after numbers from this harness.

```bash
conda activate cymfork            # or any env with jax and this package
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmarks/bench.py --out benchmarks/mine.json
python benchmarks/bench.py --list          # the 29 cases
python benchmarks/bench.py --quick         # a fast subset
```

Properties worth knowing before trusting a number from it: **first call
(compile plus execute) is always reported separately from the median of
subsequent calls**, each synchronised with `block_until_ready`, because in JAX
the first call otherwise hides exactly the regressions this is meant to catch;
one subprocess per case, so peak device memory is per-case and a crash cannot
take the run down; and every record carries the library commit, a
`library_is_pristine` flag, the resolved `cymetric.__file__`, the GPU name and
all library versions.

Baselines are per card: `baseline-rtx5060ti.json`, `baseline-rtx5090.json`.
Do not compare across cards or across machines; compare a change against the
baseline for the same card.

## Caveats on the baseline

Stated by the person who measured it, and worth keeping:

- The **36-61 GB memory blowups reported at h11 = 16-17 are unconfirmed here.**
  Peak device memory never exceeded 731 MB in any runnable case, because the
  cases that would show it are precisely the blocked ones.
- The **48x h11 cost curve is numpy, not JAX**, and is an upper bound: it
  includes the unoptimised `einsum` above. Do not quote it as a JAX prediction.
- The three toric geometries are a **curated pick**, one favourable polytope per
  h11, not a random KS sample. h11 = 10 here has 39 section monomials while the
  same family at h11 = 15 has 389, so the curve understates the tail.
- Machine load varied during the run. Large-batch GPU steps and point
  generation are reproducible to a few percent; the batch-64 step is
  host-latency bound and varies by +-25%, and CPU cases by +-20% across runs.
- `s_per_epoch_est`, extrapolated from 32 steps, is **12x optimistic** against
  the measured end-to-end epoch. Prefer the `train_model_jax.*` cases. The gap
  between them is itself the headline finding above.

## Related: cypoints

Point sampling and integration weights were split into a separate package,
[LiamMcAllisterGroup/cypoints](https://github.com/LiamMcAllisterGroup/cypoints),
which does not depend on this one. It fixes two defects in upstream's **toric
sampler and measure** that bias integrals silently:

- the Metropolis move accepts on `|Omega|^2` but proposes a tangent Gaussian
  symmetric in a different metric, so the chain samples
  `det(J J^dagger) |Omega ^ Omega-bar|`: about 7% on a chart-sensitive
  observable, 0.5-1.4% on volume integrals away from the normalising ray, and
  20-40% on an h11 = 3 polytope;
- the residue formula omits `|det Q[:, pivots]|`, which is 2 to 4 in orbifold
  charts: 30-50% on Kreuzer-Skarke fans.

`cypoints/benchmarks/bias/` reproduces both. If you take points from upstream's
toric sampler, those biases are in your integrals.

## Maintaining this

- **Rebase onto upstream periodically.** `git remote add upstream
  https://github.com/ruehlef/cymetric.git`, then `git fetch upstream`. Two
  earlier local changes (pullback caching, a per-epoch jit leak) were carried
  here for months and have since been merged upstream as `86c407e` and in
  `jax/models/helper.py`; checking before porting saves that work.
- **Re-run the baseline after any rebase**, on the same card, before trusting
  a comparison.
- The group's own geometry data for the benchmark cases is in
  `benchmarks/data/`, exported with CYTools; `make_toric_data.py --check`
  revalidates it by building cymetric's own toric generator from it.
