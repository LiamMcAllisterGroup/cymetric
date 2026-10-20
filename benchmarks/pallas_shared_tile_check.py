# Minimal reproducer search, step 7: shared loaded tile (each cdz_k loaded once, used twice) x row-sum.
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as plt
R, W, n = 8, 128, 64; nb = n // R

def build(nt, share, rowsum, preload):
    def kern(ca0_ref, cz_ref, zp_ref, s1_ref, s2_ref, da_ref, cda0_ref, cdz_ref, dzp_ref, dw_ref, db_ref, ca_ref, cda_ref):
        s1, s2 = s1_ref[...], s2_ref[...]
        ca = ca0_ref[...] + s1 * cz_ref[...]
        if share:
            c_dz = [cdz_ref[:, k, :] for k in range(nt)]
            for k in range(nt): ca = ca + s2 * da_ref[:, k, :] * c_dz[k]
            cda = [cda0_ref[:, kk, :] + s1 * e for kk, e in enumerate(c_dz)]
        else:
            for k in range(nt): ca = ca + s2 * da_ref[:, k, :] * cdz_ref[:, k, :]
            cda = [cda0_ref[:, k, :] + s1 * cdz_ref[:, k, :] for k in range(nt)]
        if preload:
            zp_ = zp_ref[...]; dzp_ = [dzp_ref[:, k, :] for k in range(nt)]
            dW = plt.dot(zp_, ca, trans_a=True)
            for k in range(nt): dW = dW + plt.dot(dzp_[k], cda[k], trans_a=True)
        else:
            dW = plt.dot(zp_ref[...], ca, trans_a=True)
            for k in range(nt): dW = dW + plt.dot(dzp_ref[:, k, :], cda[k], trans_a=True)
        dw_ref[...] = dW
        db_ref[...] = jnp.sum(ca, axis=0) if rowsum else jnp.zeros((W,), jnp.float32)
        ca_ref[...] = ca
        for k in range(nt): cda_ref[:, k, :] = cda[k]
    return kern

def run(nt, share, rowsum, preload, seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 9)
    ca0, cz, zp, s1, s2 = (jax.random.normal(k, (n, W)) for k in ks[:5])
    da, cda0, cdz, dzp = (jax.random.normal(k, (n, nt, W)) for k in ks[5:9])
    rows = lambda *tail: pl.BlockSpec((R,) + tail, lambda i: (i,) + (0,) * len(tail))
    f32 = lambda *s: jax.ShapeDtypeStruct(s, jnp.float32)
    dW, db, ca_o, cda_o = pl.pallas_call(build(nt, share, rowsum, preload), grid=(nb,),
        in_specs=[rows(W)] * 5 + [rows(nt, W)] * 4,
        out_specs=[pl.BlockSpec((None, W, W), lambda i: (i, 0, 0)), pl.BlockSpec((None, W), lambda i: (i, 0)), rows(W), rows(nt, W)],
        out_shape=[f32(nb, W, W), f32(nb, W), f32(n, W), f32(n, nt, W)],
        compiler_params=plt.CompilerParams(num_warps=8))(ca0, cz, zp, s1, s2, da, cda0, cdz, dzp)
    ca = ca0 + s1 * cz + sum(s2 * da[:, k] * cdz[:, k] for k in range(nt))
    cda = jnp.stack([cda0[:, k] + s1 * cdz[:, k] for k in range(nt)], 1)
    dW_ref = zp.T @ ca + sum(dzp[:, k].T @ cda[:, k] for k in range(nt))
    rel = lambda x, y: float(jnp.max(jnp.abs(x - y)) / jnp.max(jnp.abs(y)))
    return f"dW {rel(dW.sum(0), dW_ref):.1e}  ca {rel(ca_o, ca):.1e}  cda {rel(cda_o, cda):.1e}"

print(jax.__version__, jax.devices()[0])
for share, rowsum, preload in ((True, True, True), (True, True, False), (True, False, True), (False, True, True), (False, False, False)):
    print(f"NT=6 share={share!s:5} rowsum={rowsum!s:5} preload={preload!s:5}: {run(6, share, rowsum, preload)}")
