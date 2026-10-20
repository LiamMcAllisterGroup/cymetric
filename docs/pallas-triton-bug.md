# Pallas (Triton backend) bug report: a loaded tile reused from a Python variable is miscomputed

Status 2026-10-20: **still reproduces on jax 0.12.0**, two minor versions after it was
found on 0.10.2. Reproduced on two GPUs, isolated to a self-contained script, workaround
found. **Not yet filed upstream** -- it should be; the reproducer is
`benchmarks/pallas_bug_repro.py` here and needs nothing but jax.

Confirmation run, jax 0.12.0, RTX 5060 Ti (`dW` is the quantity that goes wrong):

    NT=6 share=True  rowsum=True  preload=True : dW 5.6e-01
    NT=6 share=True  rowsum=True  preload=False: dW 5.6e-01
    NT=6 share=True  rowsum=False preload=True : dW 5.6e-01
    NT=6 share=False rowsum=True  preload=True : dW 2.4e-07
    NT=6 share=False rowsum=False preload=False: dW 2.4e-07

i.e. sharing the loaded tile is wrong by O(1) and not sharing it is correct to float32
rounding, independently of the other two switches.

## Environment

- jax 0.10.2, jaxlib 0.10.2 (CUDA 12 wheels), Python 3.14, Linux 7.0.0
- NVIDIA driver 595.84; RTX 5090 (device 0) and RTX 5060 Ti (device 1): both reproduce
- Pallas GPU with `compiler_params=jax.experimental.pallas.triton.CompilerParams(num_warps=8)`
  (the default Mosaic-GPU backend has no `dot` lowering in this version)

## Reproducer

`benchmarks/fused_fwdrev/pallas_bug_repro.py` (self-contained, no project imports; the ladder of variants it was reduced through, `pallas_bug_repro2.py` to `repro9.py`, is in git at tag `state-S26`). Kernel
over blocks of R = 8 rows, tiles of width 128, NT = 6 channels:

```python
s1, s2 = s1_ref[...], s2_ref[...]
ca = ca0_ref[...] + s1 * cz_ref[...]
c_dz = [cdz_ref[:, k, :] for k in range(NT)]                 # load each tile once ...
for k in range(NT):
    ca = ca + s2 * da_ref[:, k, :] * c_dz[k]                  # ... use it here
cda = [cda0_ref[:, k, :] + s1 * c_dz[k] for k in range(NT)]  # ... and here
dW = plt.dot(zp_ref[...], ca, trans_a=True)
for k in range(NT):
    dW = dW + plt.dot(dzp_ref[:, k, :], cda[k], trans_a=True)
dw_ref[...] = dW                      # (None, 128, 128) per-block output
db_ref[...] = jnp.sum(ca, axis=0)     # (None, 128) per-block output
ca_ref[...] = ca                      # (R, 128) output, for checking
for k in range(NT): cda_ref[:, k, :] = cda[k]
```

Observed (relative max error against the same arithmetic in plain jnp):

| variant | ca | cda | dW |
|---|---|---|---|
| tile held in `c_dz` and reused (above) | 9e-8 | **0.56** | **0.56** |
| same, `cdz_ref[:, k, :]` re-indexed at each use | 9e-8 | 0 | 2e-7 |

`ca`, which is built from the same shared tiles, is correct; `cda` is wrong, and so is
anything computed from it. The failure does not depend on: the `db` output being written or
zero-filled, whether the extra output is `(None, 128)`, `(1, 128)` or `(R, 128)`
(`pallas_bug_repro9.py`), the operands of the dots being pre-loaded, NT (1, 2 and 6 fail),
`num_warps`, or the block size R (8, 16, 32 all fail in the full kernel).

The failure does depend on the kernel having the additional output(s): the same body with
only `dW` and `ca` as outputs, or without the dots, is correct (`pallas_bug_repro3.py`,
`pallas_bug_repro8.py`). So the minimal trigger observed is: a loaded tile reused from a
Python variable in two elementwise expressions, one of whose results feeds a `trans_a` dot,
in a kernel with a third output.

## How it was found

The hand-written adjoint of a small MLP's forward-over-reverse Hessian
(`src/extensionbundle/fused_fwdrev.py`) gave wrong weight gradients for the lower layers
while every intermediate written out of the kernel was right. Bisection from the failing
kernel body down (`pallas_bug_repro2.py` ... `repro9.py`) removed one ingredient at a time;
building up from a minimal kernel (`repro1`, `repro3`-`repro6`) never reproduced it,
because each of those variants happened to lack one of the three ingredients.

## Workaround

Re-index the ref at every use instead of keeping the loaded tile in a variable. Applied in
`_up_adj_layer_kernel`; the layered adjoint then agrees with the reference to 8e-5 (float32).
The monolithic three-layer kernel cannot use this fix because its reused tiles are dot
results rather than loads, and stays wrong.
