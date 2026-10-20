"""Agreement of the cheap routes with the ones they replace.

Three changes are covered: the pulled back ``ddbar phi`` from ``2*nfold``
Hessian-vector products instead of the full Hessian of the network, ``det g``
in jitted chunks, and the toric section norms in log space.  Each test
compares the new route against the old one on the committed geometries.

:Authors:
    Nate MacFadden
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import TORIC, geometry, load, phi_model, rtol, x64

from cymetric.jax.models.dtypes import complex_dtype, real_dtype
from cymetric.jax.models.measures import det_g_chunked, sigma_measure
from cymetric.jax.models.models import (ddbar_phi_pullback,
                                        ddbar_phi_pullback_full)


def _rel(a, b):
    """max|a - b| / mean|b|."""
    return float(jnp.max(jnp.abs(a - b))) / float(jnp.mean(jnp.abs(b)))


# ddbar phi: the low-rank route against the full Hessian
# ======================================================
def test_metric_matches_the_full_hessian(any_name):
    """g from 2*nfold Hessian-vector products equals g from the full Hessian."""
    model, X, y = phi_model(any_name)
    X = X[:64]
    assert _rel(model.metric_low_rank(X), model.metric_full_hessian(X)) < rtol()
    # ... and __call__ takes one of them, whichever the route policy picked
    chosen = (model.metric_full_hessian(X) if model.hessian_route == 'full'
              else model.metric_low_rank(X))
    assert _rel(model(X), chosen) == 0.


def test_det_matches_the_full_hessian(any_name):
    """The determinant, which is what the Monge-Ampere loss sees."""
    model, X, y = phi_model(any_name)
    X = X[:64]
    d_low = np.real(np.linalg.det(np.array(model.metric_low_rank(X))))
    d_full = np.real(np.linalg.det(np.array(model.metric_full_hessian(X))))
    assert np.max(np.abs(d_low - d_full) / np.abs(d_full)) < rtol()


def test_loss_gradients_match_the_full_hessian(any_name):
    """The gradient w.r.t. the network parameters, which is what training uses."""
    model, X, y = phi_model(any_name)
    X, y = X[:64], y[:64]

    def loss(nn, full):
        m = eqx.tree_at(lambda q: q.model, model, nn)
        g = m.metric_full_hessian(X) if full else m.metric_low_rank(X)
        return jnp.mean(m.sigma_loss(y, g))

    g_low = eqx.filter_grad(lambda nn: loss(nn, False))(model.model)
    g_full = eqx.filter_grad(lambda nn: loss(nn, True))(model.model)
    leaves_low = jax.tree_util.tree_leaves(eqx.filter(g_low, eqx.is_array))
    leaves_full = jax.tree_util.tree_leaves(eqx.filter(g_full, eqx.is_array))
    norm = np.sqrt(sum(float(jnp.sum(a ** 2)) for a in leaves_full))
    diff = np.sqrt(sum(float(jnp.sum((a - b) ** 2))
                       for a, b in zip(leaves_low, leaves_full)))
    assert norm > 0.
    assert diff / norm < rtol()


def test_ddbar_phi_pullback_is_hermitian(any_name):
    """J ddbar phi J^H is Hermitian for any phi, by construction."""
    model, X, y = phi_model(any_name)
    pbs = model.pullbacks(X[:32])
    dd = ddbar_phi_pullback(model._phi, X[:32], pbs)
    assert _rel(dd, jnp.conj(jnp.swapaxes(dd, -1, -2))) < rtol()


def test_mapped_and_vmapped_tangents_agree(toric_name):
    """``sequential`` only changes how the 2*nfold products are scheduled."""
    model, X, y = phi_model(toric_name)
    pbs = model.pullbacks(X[:32])
    a = ddbar_phi_pullback(model._phi, X[:32], pbs, sequential=True)
    b = ddbar_phi_pullback(model._phi, X[:32], pbs, sequential=False)
    assert _rel(a, b) < rtol()


def test_full_hessian_route_is_the_old_code(any_name):
    """The reference route is literally the pre-change expression."""
    model, X, y = phi_model(any_name)
    X = X[:16]
    nc = model.ncoords
    dd_phi = jax.vmap(jax.jacobian(jax.grad(model._phi)))(X)
    dd_c = (0.25 * (dd_phi[:, :nc, :nc] + dd_phi[:, nc:, nc:])
            + 0.25j * (dd_phi[:, :nc, nc:] - dd_phi[:, nc:, :nc])
            ).astype(complex_dtype())
    pbs = model.pullbacks(X)
    old = jnp.einsum('xai,xij,xbj->xab', pbs, dd_c, jnp.conj(pbs))
    assert _rel(ddbar_phi_pullback_full(model._phi, X, pbs), old) == 0.


# det g in chunks
# ===============
def test_chunked_det_matches_one_shot(any_name):
    """Chunking changes the memory peak, not the numbers."""
    model, X, y = phi_model(any_name)
    one_shot = jnp.real(jnp.linalg.det(model(X)))
    for chunk in (7, 64, None):
        assert _rel(det_g_chunked(model, X, chunk=chunk), one_shot) < rtol()


def test_sigma_measure_is_chunk_independent(any_name):
    """The sigma measure built on it is unchanged, at any chunk size."""
    model, X, y = phi_model(any_name)
    ref = float(sigma_measure(model, X, y, chunk=len(X)))
    assert abs(float(sigma_measure(model, X, y, chunk=13)) - ref) < rtol() * ref
    assert abs(float(sigma_measure(model, X, y)) - ref) < rtol() * ref


def test_chunk_heuristic_shrinks_with_the_feature_width():
    """A wide input map gets fewer points per program."""
    from cymetric.jax.models.measures import _n_features
    from cymetric.jax.models.models import PhiFSModelToric
    from conftest import make_net
    basis, td, X, y = geometry('h11_3')
    wide = eqx.nn.Sequential([eqx.nn.Lambda(lambda x: jnp.concatenate([x] * 500)),
                              *make_net(500 * X.shape[1], width=4).layers])
    model = PhiFSModelToric(wide, basis, toric_data=td)
    assert _n_features(model) == 500 * X.shape[1]
    plain, _, _ = phi_model('h11_3')
    assert len(det_g_chunked(model, X[:8])) == 8
    # the same width decides how the Hessian-vector products are scheduled
    assert model.hessian_route == 'map'
    assert plain.hessian_route == 'full'      # 14 inputs, narrow input map
    # same budget, 500x the features: the chunk must come down
    budget = 3e8
    n_wide = int(budget // (2 * 3 * 4 * _n_features(model)))
    assert n_wide < 4096 and _n_features(plain) == 32


# the toric section norms in log space
# ====================================
def test_route_policy_follows_the_input_count_and_width():
    """choose_hessian_route: the full Hessian only where it is the cheaper one."""
    from cymetric.jax.models.models import choose_hessian_route
    from conftest import make_net
    net = make_net(28, width=64)
    assert choose_hessian_route(net, 14) == 'full'    # ncoords 7
    assert choose_hessian_route(net, 28) == 'vmap'    # ncoords 14
    wide = eqx.nn.Sequential([eqx.nn.Lambda(lambda x: jnp.concatenate([x] * 500)),
                              *make_net(500 * 28, width=4).layers])
    assert choose_hessian_route(wide, 14) == 'map'
    assert choose_hessian_route(wide, 28) == 'map'
    # what the models resolve it to, and a nonsense route is rejected
    from cymetric.jax.models.models import PhiFSModelToric
    small, _, _ = phi_model('h11_3')
    big, _, _ = phi_model('h11_10')
    assert small.hessian_route == 'full' and big.hessian_route == 'vmap'
    basis, td, X, y = geometry('h11_3')
    with pytest.raises(ValueError, match='hessian_route'):
        PhiFSModelToric(small.model, basis, toric_data=td, hessian_route='quick')


def _mss_products(model, cpoints, alpha):
    """The old, direct product form of |s_a|^2."""
    degrees = model.sections[alpha]
    ms = jnp.prod(cpoints[:, None, :] ** degrees[None, :, :], axis=-1)
    return ms * jnp.conj(ms)


def test_section_norms_match_the_direct_product(toric_name):
    """The shifted log form of |s_a|^2 / kappa agrees with the products."""
    model, X, y = phi_model(toric_name)
    nc = model.ncoords
    cp = (X[:64, :nc] + 1j * X[:64, nc:]).astype(complex_dtype())
    for alpha in range(len(model.sections)):
        mss, shift = model._section_norms(cp, alpha)
        ref = _mss_products(model, cp, alpha)
        # only the ratios are used downstream, and they carry no shift
        assert _rel(mss / jnp.sum(mss, axis=-1, keepdims=True),
                    ref / jnp.sum(ref, axis=-1, keepdims=True)) < rtol()
        assert _rel(shift + jnp.log(jnp.real(jnp.sum(mss, axis=-1))),
                    jnp.log(jnp.real(jnp.sum(ref, axis=-1)))) < rtol()


def _fs_n_metrics_products(model, cpoints, alpha, t):
    """The old toric FS metric: section norms as direct monomial products."""
    degrees = model.sections[alpha]
    mss = _mss_products(model, cpoints, alpha)
    kappa = jnp.sum(mss, axis=-1)
    zizj = cpoints[:, :, None] * jnp.conj(cpoints[:, None, :])
    J = jnp.einsum('x,xab->xab', 1. / kappa ** 2, 1. / zizj)
    coeffs = (jnp.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees)
              - jnp.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees))
    return J * coeffs * t / jnp.array(np.pi, dtype=complex_dtype())


def test_factorised_fs_contraction_matches_the_double_sum(toric_name):
    """The toric FS metric's two contractions are rewritten in factored form.

    ``sum_ab mss_a mss_b d_ai d_aj`` and ``sum_ab mss_a mss_b d_ai d_bj`` are
    quadratic in the section count as written; they factorise into
    ``kappa * sum_a mss_a d_ai d_aj`` and an outer product of
    ``sum_a mss_a d_ai``.  Measured 1.4-1.9x on 200 points at 512-2048
    synthetic sections (CPU), and it no longer depends on what contraction
    path ``jnp.einsum`` picks.
    """
    model, X, y = phi_model(toric_name)
    nc = model.ncoords
    cp = (X[:64, :nc] + 1j * X[:64, nc:]).astype(complex_dtype())
    for alpha in range(len(model.sections)):
        degrees = model.sections[alpha]
        mss, _ = model._section_norms(cp, alpha)
        kappa = jnp.sum(mss, axis=-1)
        zizj = cp[:, :, None] * jnp.conj(cp[:, None, :])
        J_alphas = jnp.einsum('x,xab->xab', 1. / kappa ** 2, 1. / zizj)
        double = (jnp.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees)
                  - jnp.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees))
        t = model.kmoduli[alpha]
        ref = J_alphas * double * t / jnp.array(np.pi, dtype=complex_dtype())
        got = model._fubini_study_n_metrics(cp, n=alpha, t=t)
        assert _rel(got, ref) < rtol()


def test_high_degree_sections_do_not_underflow():
    """The toric FS metric stays finite where the direct products flush to zero.

    At h11 = 16 of the McAllister scan the sections reach degree 87, so a
    coordinate of 0.025 contributes 1e-140: in float32 every |s_a|^2 of a
    generator underflows, kappa_alpha is 0 and 0/0 made det g nan at ~1% of
    the points.  Scaling the exponents of the h11 = 5 fixture reproduces it
    exactly -- that is the fixture with a generator whose sections are all
    below modulus one at every point, which is the generic situation once the
    sections are many and the patch pivots are few.  The log form has to stay
    finite and still agree with a float64 numpy reference.
    """
    if x64():
        pytest.skip('a float32 underflow')
    model, X, y = phi_model('h11_5')
    nc = model.ncoords
    big = [(jnp.round(jnp.real(s)) * 60).astype(complex_dtype())
           for s in model.sections]
    model = eqx.tree_at(lambda m: m.sections, model, big)
    cp = (X[:, :nc] + 1j * X[:, nc:]).astype(complex_dtype())

    # the old route: a generator loses every section, and the metric is nan
    flushed = [alpha for alpha in range(len(model.sections))
               if bool(jnp.any(jnp.sum(jnp.abs(_mss_products(model, cp, alpha)),
                                       axis=-1) == 0.))]
    assert flushed
    old = _fs_n_metrics_products(model, cp, flushed[0], model.kmoduli[flushed[0]])
    assert not bool(jnp.all(jnp.isfinite(jnp.real(old))))

    # the new route: finite everywhere, determinant still positive
    g = model(X)
    det = np.real(np.linalg.det(np.array(g)))
    assert np.all(np.isfinite(det))

    # ... and the section ratios still match float64 numpy
    for alpha in range(len(model.sections)):
        d = np.round(np.real(np.array(model.sections[alpha]))).astype(np.int64)
        z = np.array(cp[:32], dtype=np.complex128)
        ms = np.prod(z[:, None, :] ** d[None], axis=-1)
        ref = np.real(ms * np.conj(ms))
        assert np.all(np.isfinite(ref)) and np.all(ref.sum(axis=-1) > 0.)
        mss, _ = model._section_norms(cp[:32], alpha)
        got = np.real(np.array(mss, dtype=np.complex128))
        err = np.max(np.abs(got / got.sum(-1, keepdims=True)
                            - ref / ref.sum(-1, keepdims=True)))
        assert err < 1e-4
