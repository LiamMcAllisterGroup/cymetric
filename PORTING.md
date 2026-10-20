# Porting the extensionmetric JAX work into cymetric's JAX backend

Branch `perf-jax-models`, forked from upstream `be772af`.  Target workload:
toric (Kreuzer-Skarke) geometries across a range of h11, for the McAllister
group.  Donor (read-only reference): `/home/nate/github/extensionmetric`,
whose `docs/optimization_summary.md` rows are cited below as "donor row N".

Everything here is in `cymetric/jax`; the TF, torch, pointgen and sage code is
untouched.  API compatibility with upstream was not a constraint and a few
signatures did change, all listed under "Things a reviewer should check by
hand" at the end.

## Test fixtures

`tests/data/*.pickle`, each holding the `BASIS` that cymetric's own
`nphelper.prepare_basis_pickle` writes, 200 sampled points with weights and
`|Omega|^2`, and (toric) the `toric_data` dict:

| fixture | geometry | ncoords | generators | patches | fan nTransitions | projective count from AMBIENT |
|---|---|---|---|---|---|---|
| `tetraquadric` | (P^1)^4 ambient, h11 = 4 | 8 | 4 | 16 | 8 | 192 |
| `h11_3` | favorable KS polytope, h11 = 3 | 7 | 3 | 12 | 8 | 112 |
| `h11_5` | favorable KS polytope, h11 = 5 | 9 | 6 | 20 | 12 | 10 000 |
| `h11_10` | favorable KS polytope, h11 = 10 | 14 | 12 | 48 | 16 | 290 304 000 |
| `quintic` | Fermat quintic in P^4 | 5 | - | - | 625 | 625 |
| `tetraquadric_cicy` | (2,2,2,2) in (P^1)^4 | 8 | - | - | 8 | 8 |

The last column is what upstream's `FSModel._patch_transitions` returns for
that BASIS, and it is the size of the `fixed_patches` table and the factor by
which the transition loss repeats every point.  At h11 = 10 that table would
be (14, 48, 2.9e8) int32, i.e. 780 GB, and the loss would be divided by
2.9e8 instead of 16.  The four KS polytopes are the ones the donor's h11 scan
picked (`benchmarks/h11_scan/h11_scan.json`); `h11_3` and `h11_10` have
smooth charts, `h11_5` has chart order 2.

The toric ones were generated with `cytools` + `cypoints` (the donor's
sampler) and then pushed through cymetric's own `ToricPointGeneratorMC` and
`prepare_basis_pickle`, so the `BASIS` is exactly what a cymetric user has,
including the AMBIENT convention (`len(sections) + 1` per generator) that the
donor calls "off by two".  The two CICY ones come from cymetric's own
`PointGenerator`.  No SageMath, Mathematica or cytools is needed to run the
tests.

98 cases pass with the float32 tolerances and 97 (plus the float32-only
underflow test, skipped) under x64.  Run the suite with

    PYTHONPATH=<worktree> JAX_PLATFORMS=cpu pytest tests/ -q          # float32
    PYTHONPATH=<worktree> JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu pytest tests/ -q

The agreement tolerances follow the precision (`conftest.rtol()`: 1e-5 in
float32, 1e-12 under x64), so both runs are meaningful: float32 is the
"agrees to float32 rounding" claim, x64 shows the routes are algebraically
identical rather than accidentally close.

## 1. The toric JAX model could not be constructed at all

`cymetric/jax/models/fubinistudy.py`, `cymetric/jax/models/models.py`.

Upstream's `FSModel.__init__` inlined the projective-space helper block
instead of calling `self._generate_helpers()` the way the TF
(`tensorflow/models/fubinistudy.py:45`) and torch
(`torch/models/fubinistudy.py:62`) backends do.  Consequences, all fixed:

* `ToricModel._generate_helpers` was dead code, so `patch_degrees`,
  `transition_coefficients` and `transition_degrees` -- declared fields of an
  `eqx.Module` -- were never assigned and construction failed.
* `_generate_proj_matrix` ran for toric models as well.  It requires
  `sum(AMBIENT + 1) == ncoords`; a toric BASIS carries section counts there
  (`[3,3,7]` for 7 coordinates), so it raised a shape error first.  The toric
  override now sets `proj_matrix` and `_proj_indices` to None, as the TF
  version does: the toric routines index patches by the fan.
* `nTransitions` came from `FSModel._patch_transitions` (a product of
  projective dimensions).  `ToricModel._patch_transitions` now returns the
  fan count, the largest number of patches in which one coordinate is not a
  pivot.  Donor row 32 / `extensionbundle/toric.py::PhiModelToric`.
* `fixed_patches_n` (a JAX-only field, None for `nhyper == 1`) was also
  unassigned by the toric helper.

The order inside the toric `_generate_helpers` matters and is deliberate:
`nTransitions` is set from `_patch_transitions` **before**
`_generate_all_patches` runs, and the override does not call `super()` --
the same structure as TF's `ToricModel._generate_helpers`.  Delegating to the
base helper first would build the toric patch table while `nTransitions`
still held the projective value, which raises
`ValueError: shape mismatch: value array of shape (12,12) could not be
broadcast to indexing result of shape (12,4)` for any fan where a coordinate
is free in more patches than that count -- the failure the torch backend
shows at h11 = 5 and 10.  That is a false green waiting to happen, because it
does not fire on a one-generator geometry, so the tests construct and
evaluate models at h11 = 3, 5 and 10, not only at h11 = 1.

Two further JAX-only defects surfaced immediately after construction worked,
both in `ToricModel`, both of which made the toric patch-transition loss
impossible to call rather than merely wrong:

* `_mask_to_patch_index` used `jnp.where(match)`, whose output shape depends
  on the data: `ConcretizationTypeError` under `eqx.filter_jit`.  Now
  `argmax` over the patch axis (each row matches exactly one patch).
* `get_transition_matrix` split the batch with `jnp.where(cond)[0]` twice and
  scattered the two halves back: same problem.  Now the monomial is evaluated
  for every row and the same-patch rows are overwritten with the identity,
  with their coordinates replaced by ones first so that a negative exponent
  on a vanishing coordinate cannot leave an inf (or a nan gradient) in the
  discarded branch.

Donor defect 3 (`ToricPointGeneratorMC.ambient_dims` off by two) was checked
and **not** ported: the donor works around it in its own data layer
(`cypoints.toric.prepare_basis` writes an AMBIENT that is a partition of
ncoords).  Changing `ambient_dims` in cymetric's pointgen would change the
auxiliary-weight code paths that read `point_gen.ambient` for every toric
user, and the JAX fix above makes the toric models independent of AMBIENT
anyway.  It is recorded here as an upstream data-side wart, not fixed.

Tests: `tests/test_jax_toric_model.py` (40 cases over the four toric
fixtures).  The one that matters most is
`test_fs_metric_transitions_between_toric_patches`: with a zero network the
model's metric is exactly the toric Fubini-Study metric, which is a genuine
tensor, so `T^dagger g_i T - g_j` must vanish on every one of the
`nTransitions` transitions of every point.  It does, to 4e-9 (x64), which
exercises `patch_degrees`, `transition_coefficients`, `transition_degrees`,
`fixed_patches` and the transition count together.

Measured: upstream cannot build the model, so there is no before/after time.
The counterfactual cost of the wrong transition count is in the table below.

### Not fixed: upstream's patch-transition data is not always right

The transition monomials come from
`cymetric.pointgen.nphelper.compute_all_w_of_x` (numpy, shared with the TF
and torch backends), which least-squares-fits the exponents of each
transition and rounds them to integers.  Two failures show up in the FS
transition loss, which would be zero if the data were exact:

* `h11_5` has chart order 2 (an orbifold fan).  The fit is not integral,
  upstream prints "WARNING W(X): NO INTEGER COEFFICIENTS" 352 times while
  building the model and rounds anyway; the FS transition loss is then O(1).
* `h11_10` is a smooth fan with every generator base-point free, no warning
  is printed, and the FS transition loss is still 2.4e-4 at 6% of the points
  under x64 (0.028 in float32).  A handful of patch pairs are wrong: the
  least-squares solve is over-determined, so when the target exponent vector
  is not exactly in the span the rounded answer is silently wrong.  The
  tetraquadric and `h11_3` are exact (5e-8 under x64, the floor set by the
  points' own 1e-9 residual on the hypersurface).

Both are numpy-side and affect all three backends, so they are out of scope
for this branch; `tests/test_jax_toric_model.py` has the strict test on the
two exact fans and a loose, documented one on the other two, so a JAX-side
regression would still show.  `PhiFSModelToric`'s default transition loss
(random GLSM scalings) does not touch this data and is fine on any fan.
This is worth fixing upstream: `compute_all_w_of_x` is also
O(ncoords x nPatches^2) with a `lstsq` per entry and allocates
(ncoords, nP, nP, nfold, nfold, ncoords) ints -- 32 MB of
`transition_degrees` already at h11 = 10.

## 2 and 3. ddbar phi without the full Hessian (and det g in chunks)

`cymetric/jax/models/models.py`: `ddbar_phi_pullback` (new),
`ddbar_phi_pullback_full` (the old expression), `phi_metric`,
`choose_hessian_route`, and `metric_low_rank` / `metric_full_hessian` on both
phi models.

`g = g_FS + J (ddbar phi) J^H` contracts the Hessian of the network with the
`nfold` rows of the pullback on each side, so the `(2 ncoords)^2` Hessian
upstream formed per point is `2 nfold` directional second derivatives -- six
for a threefold -- each a forward-over-reverse Hessian-vector product.  With
`T = [[Re J, Im J], [Im J, -Re J]]` and `S = T Hess(phi) T^T`,

    J (ddbar phi) J^H = [(S_RR + S_II) + i (S_IR - S_RI)] / 4

which is the donor's `cy_laplacian_lowrank` / `train._g_pred` identity
(donor row 2), kept in matrix rather than trace form because the metric needs
all nine entries, not just `tr(Omega Hess f)`.  It is geometry-agnostic: the
shapes carry `nfold` and `ncoords`, and the same function serves the CICY and
the toric models.  The products can run in a `lax.map` rather than a `vmap`
(`sequential=True`), because each one carries the intermediates of phi's
input map, which for the toric nets of the McAllister scan are all pairwise
section products (donor row 35); that is the `'map'` route below.

Which route `__call__` takes is a measured choice, not an unconditional
switch (`choose_hessian_route`, resolved once at construction into the
`hessian_route` field, overridable with the `hessian_route=` argument):

* `'map'` -- the low-rank route, products one at a time -- whenever the
  network's input map is wider than 4096 values.  That is every toric net of
  the McAllister kind, and the case where the full Hessian needs tens of GB.
* `'full'` -- the old route -- when the point has at most 16 real inputs
  (ncoords <= 8) and the input map is narrow.  There the full Hessian is both
  small and faster: it needs `2 ncoords` directions against `2 nfold`, but
  gets all of them from one batched reverse pass, while the low-rank route
  pays a separate forward-over-reverse pass per direction.  Measured
  crossover on an RTX 5090: at `2 ncoords = 14` the full Hessian is 1.0-1.4x
  ahead, at 28 the low-rank route is 1.3-2.2x ahead.  This keeps the quintic
  and the CICY models exactly as fast as they were.
* `'vmap'` -- the low-rank route with the products batched -- otherwise,
  which is every toric geometry from about h11 = 5 up with a plain MLP.

`det g` on a validation set is additionally chunked:
`measures.det_g_chunked`, with `sigma_measure` routed through it and a `chunk`
argument added.  The chunk defaults to a size derived from the width of the
network's input map (`_n_features`, via `jax.eval_shape` on the first layer),
between 128 and 4096 points -- the donor's `train.metric_det` / `_det_chunk`
(donor rows 33, 35).

Verification before trusting any of it (`tests/test_jax_metric_routes.py`,
all six fixtures): the metric, its determinant and the gradient of the
Monge-Ampere loss with respect to the network parameters all agree between
the two routes to 2.4e-7 in float32 and 5e-16 under x64; `sequential=True`
and `False` agree; the reference route is pinned to the pre-change
expression character by character; and the chunked determinant equals the
one-shot determinant at chunk sizes 7, 64 and the default.  The gradient test
is the one that matters for training, and it is checked separately from the
value because a wrong tangent structure can still give the right value.

Measured (RTX 5090, batch 128): see the tables at the end.

## 4. A float32 underflow that made det g nan

`ToricModel._section_norms` (new), used by `_fubini_study_n_metrics` and
`_fubini_study_n_potentials`.

Upstream evaluates the section monomials as `prod(z_i ** d_ai)`.  At h11 = 16
the sections reach degree 87, and a coordinate of 0.025 contributes 1e-140:
in float32 every `|s_a|^2` of a generator flushes to zero, `kappa_alpha` is
0, and `1/kappa^2 * 0` makes the toric FS metric -- hence `det g` -- nan at
about 1% of the points (donor row 36; the donor fixed it on its own feature
map, the same arithmetic appears in cymetric's FS part).

`_section_norms` returns `|s_a|^2` shifted so that the largest section of
each point is exactly 1, plus the log of the factor taken out.  Every
expression downstream is homogeneous of degree zero in the section norms
(ratios `mss_a mss_b / kappa^2`), so the shift cancels identically; the
potential adds it back as `shift + log(sum)`.  `log|z|^2` is floored at the
smallest normal float so that a vanishing coordinate gives a finite log and
a zero gradient rather than `-inf` and `0 * -inf`.

Tests: `test_section_norms_match_the_direct_product` (3e-7 float32, 5e-16
x64, on all three toric fixtures) and
`test_high_degree_sections_do_not_underflow`, which scales the `h11_5`
exponents by 60 to reproduce the real failure: the product form loses every
section of a generator and its FS metric comes out non-finite, the log form
stays finite with a positive determinant and still matches a float64 numpy
reference to 1e-4 on the section ratios.

## 5. Retracing: measured, real, fixed

Measured first, as asked (`JAX_LOG_COMPILES` through a logging handler;
`tests/test_jax_retracing.py` keeps the measurement as a test).  With a
jitted program that takes the model (`det g`), on `h11_3`:

| call | upstream | now |
|---|---|---|
| first call | 1 compile | 1 compile |
| same model again | 0 | 0 |
| identical model rebuilt | **1 compile** | 0 |
| same geometry, different KAPPA | **1 compile** | 0 |
| new network weights | 0 | 0 |

Everything on an `eqx.Module` that is not a JAX array is static, so two
causes: `FreeModel.kappa` was a Python float, and `_sigma_loss_fn` held a
closure built per instance (a fresh function object, so never equal).  This
is the donor's row 15, found the same way, and it matters for the Cornell
workload: a scan builds one model per moduli point and per geometry, and
every jitted program that takes the model recompiled for each.

Fixes: `kappa` is a dynamic array field; the closure is gone, replaced by a
`sigma_loss` method over the new module-level `losses.sigma_integrand`, which
takes kappa as an argument; and `FSModel.__init__` now converts plain Python
scalars in BASIS (`KAPPA`, `NFOLD`, `NHYPER` as `prepare_basis_pickle`
writes them) to arrays alongside the numpy ones.  `alpha`, `n`, `gclipping`
and the `learn_*` flags are left static on purpose: they are configuration,
constant across a scan, and `train_model` already flips the flags through
`eqx.tree_at` once per phase.

`losses.sigma_loss` (the closure factory) is unchanged and still exported, and
`test_kappa_is_an_array_and_still_reaches_the_loss` pins the model's loss to
it numerically, in both the matrix and the flat branch.

## 6. The epoch's host side, which is most of an epoch

`cymetric/jax/models/helper.py`: `_shuffled` (new), `train_model`.

Measured on an RTX 5090, quintic, 20 000 points, batch 64 (the benchmarking
agent found the same thing on a 5060 Ti and it is the larger end-to-end win):

| per step | run A, ms | run B, ms |
|---|---|---|
| `train_step` on a batch already on the device | 0.901 | 1.220 |
| plus the per-batch numpy fancy-index gather `train_model` did | 2.516 | 2.593 |
| plus one device-side shuffle per phase, then contiguous slices | 0.948 | 1.181 |

(Two runs of the same script; the absolute step time moves by a third between
them, so the quantity to read is the within-run difference: the gather adds
1.37-1.62 ms, the shuffle-and-slice adds 0.00-0.05 ms.)

The gather copied a numpy index array to the device and launched a gather for
each of X, y and the pullbacks, 313 times an epoch, which was more than half
the step time.  `train_model` now permutes once per phase (same
`np.random.permutation` draw, so a seeded run is reproducible and the batches
are the same points in the same groups) and slices.  That is 2.2x off the
step at batch 64, and the smaller the batch the larger the share.  End to
end, 4 epochs of the quintic at 20 000 points now take 3.2 s including the
compile, i.e. 0.80 s per epoch, of which 0.37 s is the 313 steps themselves.

The same function also precomputes the training set's pullbacks in chunks
(`pullback_chunk=4096`): one program over the whole set holds a
`(n_p, ncoords, n_terms, ncoords)` intermediate of dQ/dz, which at a few
hundred hypersurface monomials and 100k points is tens of GB.

Tests: `test_training_batches_are_the_same_after_the_device_shuffle` (the
batches are element-for-element what the gather produced, for the same seed)
and `test_chunked_pullbacks_precompute_matches_one_program` (exactly equal).

What is left on the host side, and not done here: a warm epoch of 313 steps
still costs more than 313 steps, because `eqx.filter_jit` re-partitions the
model pytree on every call and the optimizer state is rebuilt per step.  The
donor's answer is one `lax.scan` per epoch over device-resident data (donor
row 1, 767 -> 175 s there).  That restructures the loop, the callbacks and
the metrics, so it is the recommended next step rather than part of this
branch.

## 7. The toric Fubini-Study contraction

`ToricModel._fubini_study_n_metrics`.  Both contractions of the section norms
factorise:

    sum_ab mss_a mss_b d_ai d_aj = kappa_alpha * sum_a mss_a d_ai d_aj
    sum_ab mss_a mss_b d_ai d_bj = (sum_a mss_a d_ai) (sum_b mss_b d_bj)

which is `O(n_sections ncoords^2)` instead of `O(n_sections^2 ncoords^2)` --
1e9 flops per point for the 1574-section generator of the h11 = 16 fan.  The
benchmarking agent measured 15.5 ms -> 0.3 ms from `optimize=True` on the
numpy side of the same expression.  On the JAX side `jnp.einsum` already
optimizes the path, so the win is smaller but real: 1.4-1.9x at 512-2048
synthetic sections on 200 points (CPU), and the code no longer depends on
what path the einsum optimizer picks.  Test:
`test_factorised_fs_contraction_matches_the_double_sum`, against the literal
double sum on every toric fixture (9e-8 float32, 5e-16 x64).

## Rejected, or deliberately not done

* **The donor's fused forward-over-reverse kernel** (`fused.py`,
  `fused_fwdrev.py`, donor rows 8 and 39), worth 2.5-3x per training epoch
  there.  It is a hand-written recursion over a specific network layout
  (`make_nn`'s four Linear layers with a `gelu3` nonlinearity, weights
  extracted by position) plus a Pallas variant.  cymetric's JAX models take
  an arbitrary `eqx.Module` backbone, so this cannot be ported without
  either restricting the backbone or shipping a second network class.  Out of
  scope for this branch; the generic low-rank route in item 2 is the part
  that generalises, and the donor's own numbers put the extra kernel fusion
  at 1.9x on top of it for its own net only.
* **`ToricPointGeneratorMC.ambient_dims`**: see item 1.
* **The Ricci measure and the Ricci loss** (`measures.ricci_measure`,
  `FSModel.compute_ricci_loss`) still take the full Hessian of `log det g`
  over the `2 ncoords` inputs, and they contract it with the pullbacks on
  both sides exactly as the metric does -- so `ddbar_phi_pullback` applies
  verbatim and would cut the same factor.  Not done here: `learn_ricci` is
  False by default, the Ricci measure is a diagnostic rather than part of the
  training loop, and each of its Hessian-vector products would nest a
  forward-mode pass through the whole metric (itself six products), which
  needs its own agreement test.  Recommended as the next item.
* **Padding the last `det g` chunk to a fixed shape** (donor row 10).  A
  validation set whose size is not a multiple of the chunk compiles a second
  program for the tail.  That is one extra compile per run, not per call, so
  it was not worth the padding and masking.

## Cross-check against the benchmark harness's geometries

`/home/nate/github/cymetric-bench` has three JAX toric cases (h11 = 1, 5, 10)
that its baseline records as blocked, on toric data of its own and with the
other AMBIENT convention -- a partition of ncoords, so for it the projective
transition count comes out too *small* (4 against the fan's 12 at h11 = 5),
where the fixtures here carry upstream's section counts and it comes out far
too large.  Both directions are the same bug and both are fixed.

Running that harness with `PYTHONPATH` pointed here does not test this tree:
`bench.py` and `cases.py` each insert their own repo root at the front of
`sys.path`, so the run reports its own copy of the library (its output names
the file it resolved).  Driving the same three geometries directly -- their data files and their
basis construction, this library, in a 70-line script kept outside the repo
at `/home/nate/.cache/tmp/claude-1000/-home-nate-github-extensionmetric/a9730421-6ade-43cb-87cd-27511d216fe1/scratchpad/investigate/cross_check.py`
-- gives, on an RTX 5090 at batch 512:

| geometry | ncoords | nTransitions | route | step |
|---|---|---|---|---|
| h11 = 1 | 5 | 4 | full | 1.14 ms |
| h11 = 5 | 9 | 12 | vmap | 1.14 ms |
| h11 = 10 | 14 | 16 | vmap | 1.35 ms |

with a finite metric and a positive determinant at every point in each case.
Before: `TypeError: Field 'patch_degrees' was not initialized` at h11 = 1 and
`ValueError: shape mismatch: value array of shape (12,12) could not be
broadcast to indexing result of shape (12,4)` at h11 = 5 and 10.

## Guidance for the high-h11 workload (not code changes)

* `helper.train_model`'s default `batch_sizes=(64, 10000)` sets the memory
  peak of a training run, and the second phase evaluates the metric on 10000
  points at once.  With a wide input map that is still tens of GB after the
  changes above, because it is proportional to points x feature width.  The
  donor trains its toric metrics at `(256, 512)`; use something of that order
  at large h11.  `det g`/`sigma_measure` on a validation set is now chunked
  and needs no such care.
* `measures.ricci_measure` and `FSModel.compute_ricci_loss` are the remaining
  full-Hessian consumers; see "Rejected".
* The toric patch-transition loss needs a fan whose charts are all smooth
  (see item 1).  `PhiFSModelToric`'s default random-scaling transition loss
  has no such restriction.

## Things a reviewer should check by hand

Signature and attribute changes (upstream compatibility was not a
constraint, but these are what a downstream script would notice):

* `FreeModel.kappa` is a 0-d JAX array rather than a Python float.
  `float(model.kappa)` and arithmetic still work; an `isinstance(..., float)`
  or a dict key would not.
* `FreeModel.sigma_loss` is a method `(y_true, y_pred, flat=False)` instead
  of a property returning a closure.  `model.sigma_loss(y, g)` is unchanged,
  and the bound method can still be passed around; `sigma_loss(kappa, nfold)`
  in `losses.py` is untouched for anyone who built the closure themselves.
* `FreeModel.__init__` gained `hessian_route=None`, `train_model` gained
  `pullback_chunk=4096`, `sigma_measure` gained `chunk=None`.
* `ToricModel._mask_to_patch_index` returns an `(n_p,)` int32 array of patch
  indices instead of a tuple of arrays from `jnp.where`.

Judgement calls worth a second opinion:

* **The transition-loss normalisation.** The toric patch table pads unused
  slots with "transition to the same patch", which contributes nothing, and
  the loss divides by `nTransitions` regardless.  So a coordinate that is
  free in fewer patches than the maximum has its transition loss scaled
  down.  That is upstream's behaviour in all three backends and this branch
  preserves it, but it is arguably wrong, and it is why the projective count
  changed the loss value by exactly the ratio of the counts.
* **The same-patch substitution** in `ToricModel.get_transition_matrix`:
  rows that stay in their patch get their coordinates replaced by ones
  before the transition monomial is evaluated, and the identity is written
  over the result.  The values are right (the test on the FS metric covers
  it) and it avoids an inf in a discarded branch, but it is a behaviour
  change relative to the old index-select version for any row whose
  exponents are negative and whose coordinate vanishes.
* **The route thresholds** (16 real inputs, 4096 feature values) come from
  measurements on one GPU with one network shape.  They change speed, never
  results, and `hessian_route=` overrides them.
* **The fixtures** in `tests/data` were generated with `cytools` + the
  donor's `cypoints` in the `cypoints` conda env and then pushed through
  cymetric's own `ToricPointGeneratorMC`/`prepare_basis_pickle`; the
  generator script is in this session's scratchpad, not the repo, so the
  pickles are the record.  Their points satisfy the hypersurface to a
  relative 1e-9, which is the floor of the exact transition tests.
* **`det_g_chunked`** compiles a second program for a ragged last chunk (one
  extra compile per run, not per call).  Padding would avoid it.
* The `h11_10` transition residual and the orbifold-chart warnings are
  upstream numpy-side defects left in place; see item 1.

## Measured: before and after

RTX 5090 (GPU 0), float32, `h11_3` (7 coordinates, 3 generators).  "full"
is upstream's route (`metric_full_hessian`), "low-rank" is `__call__` after
this branch.  The input maps stand in for the toric nets of the McAllister
scan: `pairwise(n)` feeds `n^2` pairwise products of `n` values to the net,
so `pairwise(300)` is 90k inputs, the size of the h11 = 16 feature map that
needed 36-61 GB.

### g on 128 points, `h11_3` (ncoords 7, so 14 real inputs against 6 tangents)

| input map | n_in | ddbar phi, vmap | ddbar phi, lax.map | full g, low-rank | full g, full Hessian | peak GB low-rank | peak GB full |
|---|---|---|---|---|---|---|---|
| MLP(2 ncoords) | 14 | 0.4 ms | 0.3 ms | 0.6 ms | 0.3 ms | 0.001 | 0.032 |
| pairwise(48) | 2 304 | 0.6 ms | 0.6 ms | 0.6 ms | 0.4 ms | 0.033 | 0.033 |
| pairwise(140) | 19 600 | 0.8 ms | 0.7 ms | 1.3 ms | 1.1 ms | 0.077 | 0.205 |
| pairwise(300) | 90 000 | 2.4 ms | 2.8 ms | 2.9 ms | 3.2 ms | 0.212 | 0.776 |

Agreement in the same runs: 2.0e-5 to 2.9e-5 relative (float32, `max|dg| /
mean|g|`), 5e-16 under x64.

So at the smallest geometry in the set the low-rank route is **not** a time
win -- 14 input directions against 6 tangents, and the full Hessian gets all
14 from one batched reverse pass -- which is why `choose_hessian_route` keeps
the full Hessian there.  What the low-rank route buys even at ncoords = 7 is
memory: 0.21 GB against 0.78 GB at a 90k-wide input map, which is why the
policy switches to it as soon as the input map is wide, whatever ncoords is.

### g on 128 points, `h11_10` (ncoords 14, so 28 real inputs against 6 tangents)

| input map | n_in | ddbar phi, vmap | ddbar phi, lax.map | full g, low-rank | full g, full Hessian | speedup | peak GB low-rank | peak GB full |
|---|---|---|---|---|---|---|---|---|
| MLP(2 ncoords) | 28 | 0.4 ms | 0.3 ms | 0.4 ms | 0.6 ms | 1.33 | 0.001 | 0.001 |
| pairwise(48) | 2 304 | 0.5 ms | 0.3 ms | 0.6 ms | 0.6 ms | 0.99 | 0.012 | 0.041 |
| pairwise(140) | 19 600 | 1.1 ms | 0.9 ms | 1.3 ms | 1.8 ms | 1.36 | 0.077 | 0.326 |
| pairwise(300) | 90 000 | 2.2 ms | 2.6 ms | 2.7 ms | 5.8 ms | 2.17 | 0.212 | 1.333 |

Loss + gradient of the whole batch, which is what a training step costs:
1.34x at 28 inputs with an MLP, 1.48x at a 19.6k-wide input map.  Agreement
3.3e-6 to 3.8e-6 relative in float32.  `det g` on 20 000 points: 0.075 s
chunked against 0.068 s in one program, the latter peaking at 3.05 GB.

Both tables are the two routes called explicitly (`metric_low_rank` and
`metric_full_hessian`); what `__call__` picks is the policy above.

### det g on 20 000 points

| input map | chunked | one program | peak GB, one program |
|---|---|---|---|
| MLP(2 ncoords) | 5 ms | 3 ms | 0.25 |
| pairwise(140) | 74 ms | 66 ms | 3.03 |

Chunking costs a few percent of wall time at these sizes and makes the peak
independent of the number of points: the one-program route is already at 3 GB
for 20k points at a 19.6k-wide input map, which is the scaling that reached
36-61 GB (and OOM) at h11 = 16, 17 in the donor's scan.  Agreement 5e-5
(float32) / exact under x64.

### the transition count

`h11_3`, batch 128: the toric patch-transition loss takes 0.5 ms with the fan
count (8) and 1.7 ms with the projective count (112), and the two differ by
exactly the ratio of the counts (13.85 measured against 14.0), because every
extra slot is a transition of a patch to itself that contributes nothing to
the sum but still divides it.  So the fan count is a 3.2x saving here and the
correct normalisation; at h11 = 10 the projective count is 2.9e8 and the
model cannot be built at all.


Numbers to read with care: these are one geometry on one GPU, the input maps
are synthetic stand-ins for section features (same width and same two
derivatives, not the same arithmetic), and the timings are means of five
calls after a warm-up, so they carry the usual few-percent jitter.  The
agreement figures, not the timings, are what the tests enforce.

## Addendum from independent verification (2026-10-20)

The suite was reported as "98 pass in float32, 97 + 1 skip under x64". That is
true on the CPU and under x64 on either device, but **the default GPU float32
configuration was never run, and 44 tests fail there**:

| configuration | result |
|---|---|
| CPU, float32 | 98 passed |
| CPU, x64 | 97 passed, 1 skipped |
| GPU, x64 | 97 passed, 1 skipped |
| GPU, float32, default | **44 failed** |
| GPU, float32, `JAX_DEFAULT_MATMUL_PRECISION=highest` | 98 passed |

**The cause is TF32, not this port.** On NVIDIA GPUs JAX lowers float32 matmuls
to TF32 by default, which carries about 10 mantissa bits, so the assembled
metric is Hermitian only to roughly 1e-3 relative instead of 1e-6. Setting
`JAX_DEFAULT_MATMUL_PRECISION=highest` (or
`jax.config.update('jax_default_matmul_precision', 'highest')`) recovers
everything. This is upstream behaviour and applies equally to the unmodified
package; it is a property of the default precision, not of the changes here.

It matters anyway, for two reasons: GPU float32 is the configuration users will
actually train in, and a Hermiticity residual of 1e-3 is large enough to be
worth a deliberate decision rather than an accident. Recommended: set the
precision explicitly in the training entry points, or document it loudly.

One test was genuinely too strict and has been fixed:
`test_chunked_pullbacks_precompute_matches_one_program` asserted bit-exact
equality between the chunked and unchunked pullback precompute. Chunking
changes the reduction order, so on a GPU float32 differs by about 5e-7 on
entries of order 1. It now compares against a precision-appropriate relative
tolerance. Equality held on the CPU, which is why it was written as `== 0`.

## Against the upstream baseline, same card (RTX 5090)

Measured with the harness from branch `bench-baseline`, run from inside this
worktree so it resolves this tree (`bench.py` puts its own root on
`sys.path[0]`, so running it from the benchmark worktree measures *that* copy --
which is why the earlier attempt reported 0/3).

| case | upstream | patched | change |
|---|---|---|---|
| step, quintic bs64 | 0.91 ms | 1.06 ms | 0.86x |
| step, quintic bs1024 | 1.10 ms | 0.87 ms | 1.27x |
| step, quintic bs10000 | 3.97 ms | 3.77 ms | 1.05x |
| step, bicubic bs1024 | 0.78 ms | 0.91 ms | 0.86x |
| step, tetraquadric bs1024 | 0.98 ms | 1.12 ms | 0.88x |
| **`train_model`, quintic** | 5.36 s | 4.35 s | **1.23x** |
| **`train_model`, tetraquadric** | 6.03 s | 5.01 s | **1.20x** |
| det g, quintic | 1.38 ms | 1.71 ms | 0.81x |
| det g, tetraquadric | 2.44 ms | 1.39 ms | 1.76x |
| **step, toric h11 = 1** | **blocked** | 1.10 ms | unblocked |
| **step, toric h11 = 5** | **blocked** | 1.03 ms | unblocked |
| **step, toric h11 = 10** | **blocked** | 1.22 ms | unblocked |

Read honestly:

- **The headline is the last three rows.** Upstream cannot construct a JAX toric
  model at all; these now train, at 200, 203 and 263 MB peak device memory.
- **End to end is 1.20-1.23x faster**, which is the host-side gather change.
  That is the number that matters, because an epoch is host-bound: on this same
  harness a 4x faster GPU bought only 1.04-1.14x.
- **The single-step cases wobble by +-15% in both directions and should not be
  read as either a win or a regression.** The harness's own notes put the
  batch-64 case at +-25% within a run and absolute step times moving by a third
  between runs of the same script; bs1024 is small enough to be host-latency
  bound too. The one solid step result is bs10000 (1.05x, tight spread), and the
  one solid det g result is the tetraquadric (1.76x, the wide-input case the
  low-rank route was built for).
- det g on the quintic at 0.81x is within that same noise band, and the
  route chooser deliberately keeps the narrow CICY path on the full Hessian.

To reproduce: copy `benchmarks/` from `bench-baseline` into this worktree and
run `CUDA_VISIBLE_DEVICES=0 python benchmarks/bench.py --groups
train_jax_toric,train_jax,train_model_jax,detg_jax`.
