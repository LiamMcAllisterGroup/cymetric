# How the CY-metric training step got 2x faster, and how the fused kernel lost: a profiling story

For someone who knows cProfile. On the GPU the analogue of "which function is hot" is
"which kernels run, how long, and how much of the wall clock is the device actually busy";
the tools change but the loop is the same: measure, find the hot spot, ask whether the
work is necessary, change one thing, measure again. This is that loop, run on one stage of
the Yukawa pipeline on 2026-09-05 (RTX 5090, jax 0.10.2). Every number below was measured in
a private repository that is no longer maintained, so the regeneration scripts it refers
to (`git checkout state-S6`, `benchmarks/profiling/run_all.sh`) are not in this fork. The
method is the point, and it transfers; `benchmarks/bench.py` here is the maintained way to
measure this package.

## 1. The code we started from

Every training step of the CY metric evaluates, for each point x in R^16, the 3x3 complex
metric g = g_FS + J (d dbar phi) J^H, where phi is a small MLP (16 -> 128 -> 128 -> 128 -> 1,
gelu) averaged over a 4-element symmetry orbit, and J is the fixed 3x8 pullback. The
second derivative was computed the way cymetric does it (`train._g_pred`):

```python
def _g_pred(phi, x, J_i, fs_i):
    T  = ...                                         # 6 real tangent vectors from J (6, 16)
    HT = jax.vmap(lambda t: jax.jvp(jax.grad(phi), (x,), (t,))[1])(T)   # 6 Hessian-vector products
    S  = T @ HT.T                                    # 6x6 block T Hess(phi) T^T
    return fs_i + 0.25 * ((S[:3,:3] + S[3:,3:]) + 1j * (S[3:,:3] - S[:3,3:]))
```

vmapped over the batch, followed by `|1 - det g / (|Omega|^2 kappa)|`, `jax.grad` with
respect to the weights, Adam. 25 epochs at batches 256 and 512 took 42 s of a ~150 s
moduli point.

## 2. First measurement: is it big kernels or many kernels?

Before any profiler: time one step as a function of batch size. A step that costs the
same for 2x the points is paying per launch, not per point.

```python
def timeit(fn, *a, reps=20):
    jax.block_until_ready(fn(*a)); jax.block_until_ready(fn(*a))   # compile, warm
    t0 = time.perf_counter()
    for _ in range(reps): out = fn(*a)
    jax.block_until_ready(out)                                    # dispatch is asynchronous
    return (time.perf_counter() - t0) / reps
```

| batch | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| ms / step | 0.66 | 0.74 | 0.98 | 1.32 | 2.25 |
| us / point | 5.1 | 2.9 | 1.9 | 1.3 | 1.1 |

Doubling the batch twice from 128 costs 50%. At the batches we use (256, 512) the step is
launch-bound; the arithmetic would fit in a fraction of the time. That is the whole
diagnosis at this level: the hot spot is not any one computation, it is the *count* of
computations.

## 3. Second measurement: how many kernels, and how busy is the device?

Two tools, both cheap.

**What XLA emitted.** The compiled HLO lists every fused kernel; `cost_analysis` counts
FLOPs and bytes.

```python
comp = jax.jit(step).lower(*args).compile()
n_kernels = comp.as_text().count("kind=k")          # kLoop / kInput / kOutput fusions
flops = comp.cost_analysis()["flops"]
```

394 fusions and 3.4 GFLOP per step. 3.4 GFLOP is nothing for this GPU (~30 us at peak);
394 launches at 3-10 us each is the whole 600 us.

**The timeline.** Nsight Systems records every launch with its duration:

```bash
nsys profile -o run -t cuda --stats=false python benchmarks/profiling/step_only.py hvp
nsys stats --report cuda_gpu_trace --format csv run.nsys-rep > trace.csv
python benchmarks/profiling/nsys_summary.py trace.csv 50 0.603
```

Per step: 58 kernels launched (XLA's 394 fusions collapse into fewer launches once
autotuning picks GEMM implementations), 210 us of kernel time inside a 603 us step, so the
device is **35% busy**; mean kernel 3.6 us. The top kernels are all `gemm_fusion_dot_*`,
the 128x128 layer matmuls fused with their elementwise neighbours, 3-7 us each: no single
one is worth optimising. Pitfalls that cost an hour: the first ~30 steps are XLA autotuning
(kernels named `RedzoneAllocator`, `xla_fp_comparison`; exclude them), and nsys's own
tracing inflates the gaps between kernels, so take per-kernel durations from nsys and the
step's wall time from section 2.

So the necessary work is ~200 us of small GEMMs and the waste is ~400 us of gaps. Nothing
about the GEMMs themselves is wrong. The lever is the number of launches, i.e. the
structure of the autodiff graph that produces them.

## 4. Why the autodiff graph is so fragmented

`jvp(grad(phi))` is autodiff of autodiff. The reverse pass through a 4-layer MLP is ~10
ops; pushing a tangent through it doubles that; vmapping over 6 tangents and 4 orbit
images multiplies the *tensor sizes*, not the op count, but XLA cannot fuse across the
matmul boundaries, so each layer's forward, backward, tangent-forward and tangent-backward
matmul is its own kernel, times 3 layers, times the outer `grad` with respect to the
weights, which differentiates through all of it again. That is where 394 fusions come
from. There is no redundancy in FLOPs, only in launches.

## 5. First fix: change the formulation, not the kernels

Two ways to get the 6x6 block S with fewer, fatter operations:

**Forward-over-forward (Taylor).** Carry the value, the 6 first derivatives and the 21
second derivatives (pairs) as 28 channels through each layer; the nonlinearity's chain rule
is explicit (`fused.taylor_layer`):

```python
A  = einsum("rci,io->rco", Z, Wt) ; A[:,0] += b       # one matmul for all 28 channels
s, s1, s2 = gelu3(A[:,0])
dy  = s1 * dA ;  d2y = s1 * d2A + s2 * dA[k] * dA[l]  # 6 and 21 channels
```

3 matmuls per layer, no backward pass. HLO: 67 fusions, but 6.2 GFLOP (2x), because
second derivatives along n directions cost O(n^2) this way.

**Forward-over-reverse, written by hand** (`fused.fwdrev`). The same 14 channels autodiff
uses (value + 6 tangents up, gradient + 6 tangents down) but as explicit matmuls:

```python
a  = z @ Wt + b ; da = einsum("rki,io->rko", dz, Wt) ; z, dz = s(a), s1(a) * da      # up
ga = s1 * g ; dga = s2 * da * g + s1 * dg ; g = ga @ Wt.T ; dg = einsum(dga, Wt)     # down
```

HLO: 145 fusions, 3.1 GFLOP, 334 MB moved (the least of the three).

Measured (section 2's harness, then nsys):

| formulation | fusions | GFLOP | kernels/step | device busy | step @256 | step @1024 |
|---|---|---|---|---|---|---|
| autodiff HVP | 394 | 3.4 | 58 | 35 % | 0.60 ms | 1.20 ms |
| Taylor (XLA) | 67 | 6.2 | 20 | 80 % | 0.43 ms | 1.87 ms |
| forward-over-reverse (XLA) | 145 | 3.1 | 60 | 88 % | 0.31 ms | 0.68 ms |

The Taylor route confirms the diagnosis (few launches -> 80% busy) and then loses at large
batch because it does twice the arithmetic: once the device is saturated, FLOPs decide.
The hand-written forward-over-reverse route has the same FLOPs as autodiff and 88% busy,
and wins at every batch. Adopted as `train_metric(mode="fwdrev")`: 1.9x at the production
batches. Validation sigma after 25 epochs is unchanged (0.0164 vs 0.0175).

Why did the hand-written version launch better than autodiff's if it has similar kernel
counts (60 vs 58)? Look at the kernel durations, not the counts: its kernels are 4.6 us on
average against 3.6 us, i.e. each launch does more work (the 6 tangents ride as a 6-column
block through one matmul instead of six vmapped ones), and its kernels sit back to back.
The count is similar; the gaps are gone.

## 6. Second attempt: fuse the kernels ourselves (Pallas/Triton)

With the device 88% busy the only remaining lever is fewer, bigger kernels, which means
leaving XLA. Two kernels were written (`fused.py`, `fused_fwdrev.py`):

**Taylor recursion as one kernel per block of rows.** All 28 channels live in registers;
weights loaded once per block. Measured with nsys (`Reg/Trd` column, no counters needed):

| rows/block | time (1024 rows) | registers/thread |
|---|---|---|
| 16 | 60 us | 255 |
| 32 | 142 us | 255 |

255 is the hardware maximum: the register file is full and anything more spills to local
memory; 2x the rows costing 2.4x the time is the spill signature (Nsight Compute's
`local_op_ld/st` bytes would show it directly, but the driver's `RmProfilingAdminOnly=1`
blocks `ncu` for non-root here; the fix is in section 8). At its best the kernel *ties*
XLA's forward (0.076 vs 0.073 ms) and its backward is still XLA's: no gain.

**Forward-over-reverse with a hand-written adjoint.** The right formulation this time
(14 channels), a forward kernel that saves the per-layer state, and the exact reverse of
the recursion as kernels (needs gelu''' as well). Verified in float64 against `jax.vjp`
to 1e-16 before any kernel was written. Then:

- the forward kernel ties XLA again (0.051 vs 0.059 ms at 8 rows per block; 16 rows
  collapses to 40 registers per thread and 2.3 ms: the compiler gave up and spilled
  everything);
- the adjoint kernel produced wrong gradients for the lower layers while every intermediate
  written out of it was right. Twelve bisection scripts later
  (`benchmarks/fused_fwdrev/pallas_bug_repro*.py`) the cause is a Triton miscompile: a
  loaded tile kept in a Python variable and reused in two expressions is miscomputed when
  the kernel has an extra output. Re-indexing the ref at each use fixes it
  (`docs/pallas_triton_bug_report.md`);
- with the fix the adjoint is exact (8e-5) and **4x slower than XLA's** (0.52 vs 0.13 ms):
  each 8-row block writes three 128x128 weight partials that are summed afterwards, and
  its transposed dots have K = 8, half a tensor-core tile. Larger blocks spill.

Full step: 0.71 ms against 0.32 ms for the same math left to XLA. Recorded as
`train_metric(mode="fwdrev_pallas")` and not used.

The lesson worth keeping: on this problem XLA's fusion is already at the throughput
floor once the *formulation* stops fragmenting the graph; hand fusion could only win by
also redesigning the data layout (streaming channels to fit bigger blocks, accumulating
weight gradients across blocks in the kernel), which is a different project.

## 7. What the same method found elsewhere in the pipeline

The per-stage wall times (`timing.json` per run) pointed at three more things, none of
them GPU kernels:

- **Compilation** was a third of every run and never cached, because jitted functions
  closed over the model weights, the data arrays and the Python-float moduli, baking them
  into the program: every moduli point was a new program. Passing them as arguments
  (`eqx.partition`) let the persistent cache hit; a warm scan point went 98 s -> 46 s.
- **Retraining from scratch at every moduli point**, when everything is smooth in psi:
  warm-starting from the neighbour cut training 150 s -> 25 s at equal residuals.
- **The mass integrals**, not compute-bound at all: their integrand cancels 30-200x and
  they were being taken on 5k points. Their error, not their speed, was the problem.

## 8. Tools, in the order to reach for them

1. `timeit` with `block_until_ready` and a batch-size sweep (section 2).
2. `jax.jit(f).lower(...).compile().as_text()` and `.cost_analysis()` (section 3);
   `XLA_FLAGS=--xla_dump_to=DIR --xla_dump_hlo_as_text --xla_gpu_dump_llvmir` for the PTX
   of every XLA fusion (`ld.local` = spills).
3. `nsys profile -t cuda` + `nsys stats --report cuda_gpu_trace` (section 3): per-launch
   duration, grid, block, registers per thread. Exclude autotuning; trust durations, not
   gaps. Nsight Systems is in the `gpuprof` conda env
   (`~/miniforge3/envs/gpuprof/nsight-compute-*/host/target-linux-x64/nsys`).
4. `ncu --set full` (same env, `.../ncu`) for occupancy, achieved vs peak throughput,
   spill bytes, tensor-core utilisation per kernel. Blocked here until, as root:
   `echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf`
   and a reboot. `benchmarks/profiling/run_all.sh` then fills in the level-3 logs.
5. `pl.pallas_call(..., debug=True)` prints the Triton IR (op counts per kernel);
   `interpret=True` runs the kernel body in plain JAX for correctness debugging.

`jax.profiler.trace` would give the nsys timeline from inside JAX, but it segfaults with
`lax.scan` in this build.

All logs: `benchmarks/profiling/logs/2026-09-05/`; scripts: `benchmarks/profiling/`,
`benchmarks/fused_metric/`, `benchmarks/fused_fwdrev/`.
