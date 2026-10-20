# The Pallas/Triton "bug" was not a bug

**Retracted 2026-10-20.** A private repository carried a report of a Pallas
(Triton backend) miscompilation: a tile loaded from a ref into a Python
variable and used twice was said to be computed wrongly, by O(1), on two GPUs
and across jax 0.10.2 and 0.12.0. It was never filed upstream. It should not
be: the reproducer was wrong, and Pallas is correct.

## What the reproducer actually did

The failing branch built its list of outputs like this:

```python
c_dz = [cdz_ref[:, k, :] for k in range(nt)]
for k in range(nt):
    ca = ca + s2 * da_ref[:, k, :] * c_dz[k]
cda = [cda0_ref[:, k, :] + s1 * e for e in c_dz]      # <-- k is stale
```

In that last line `k` is not the comprehension variable. It is the loop
variable left over from the `for` loop above, frozen at `nt - 1`, so every one
of the `nt` entries of `cda` was built from slice `nt - 1` of `cda0_ref`. The
branch it was being compared against wrote its own `for k in range(nt)` and
was correct:

```python
cda = [cda0_ref[:, k, :] + s1 * cdz_ref[:, k, :] for k in range(nt)]
```

So the two branches never computed the same function, and the O(1) discrepancy
was the reproducer's own Python scoping error.

## The check

`benchmarks/pallas_shared_tile_check.py` is the original script with that one
line corrected to `enumerate(c_dz)`. Every variant now agrees with the
reference to float32 rounding, with the elementwise output exact:

    NT=6 share=True  rowsum=True  preload=True : dW 2.4e-07  ca 8.7e-08  cda 0.0e+00
    NT=6 share=True  rowsum=True  preload=False: dW 2.4e-07  ca 8.7e-08  cda 0.0e+00
    NT=6 share=True  rowsum=False preload=True : dW 2.4e-07  ca 8.7e-08  cda 0.0e+00
    NT=6 share=False rowsum=True  preload=True : dW 2.4e-07  ca 8.7e-08  cda 0.0e+00
    NT=6 share=False rowsum=False preload=False: dW 2.4e-07  ca 8.7e-08  cda 0.0e+00

jax 0.12.0, RTX 5060 Ti. The script is kept as a regression check, not as a bug
report: **loading a tile once and reusing it from a Python variable is sound**,
which is the pattern a fused kernel wants.

## Why this matters beyond the retraction

Work in the donor repository was routed around this supposed defect -- see the
row about the fused forward-over-reverse kernel in that project's optimisation
log, which records a workaround. Any design choice made to avoid reusing a
loaded tile rests on nothing and is worth revisiting.

The general lesson is the duller one: when two code paths disagree, rule out
the harness before blaming the compiler. Minimising the reproducer is what
exposed this, after two GPUs and two jax versions had been taken as
corroboration.
