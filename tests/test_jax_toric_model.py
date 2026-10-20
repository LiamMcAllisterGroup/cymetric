"""The JAX toric models: construction, the fan's patch data, the FS metric.

Before the fix in this branch no toric JAX model could be constructed at all:
``FSModel.__init__`` inlined the projective-space helpers instead of calling
``_generate_helpers``, so ``ToricModel._generate_helpers`` -- the only place
that assigns ``patch_degrees``, ``transition_coefficients`` and
``transition_degrees`` -- was dead code, and the projection matrix was built
from a toric ``AMBIENT`` (a list of section counts) and raised first.

:Authors:
    Nate MacFadden
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (UNIMODULAR, fan_patch_transitions, geometry, phi_model,
                      projective_patch_transitions, rtol, x64, zero_net)

from cymetric.jax.models.models import (MatrixFSModelToric, PhiFSModelToric,
                                        ToricModel)


def test_toric_phi_model_constructs(toric_name):
    """The model builds and every helper field is a real array of the right shape."""
    model, X, y = phi_model(toric_name)
    basis, td, _, _ = geometry(toric_name)
    npatches, ncoords = np.array(td['patch_masks']).shape
    nfold = model.nfold
    assert isinstance(model, PhiFSModelToric)
    assert model.ncoords == ncoords and nfold == 3
    assert model.patch_degrees.shape == (npatches, ncoords, ncoords)
    assert model.transition_coefficients.shape == (
        ncoords, npatches, npatches, nfold, nfold)
    assert model.transition_degrees.shape == (
        ncoords, npatches, npatches, nfold, nfold, ncoords)
    assert model.fixed_patches.shape == (ncoords, npatches, model.nTransitions)
    assert model.fixed_patches_n is None
    # the toric routines do not use the projective helpers, and a toric BASIS
    # cannot build them
    assert model.proj_matrix is None and model._proj_indices is None


def test_patch_transitions_is_the_fan_count(toric_name):
    """nTransitions comes from the fan, not from AMBIENT.

    h11 >= 3 is the regime the Cornell group works in; there the projective
    count is simply unrelated to the number of patches a coordinate can be
    transitioned to (too small for a BASIS whose AMBIENT is a partition of
    ncoords, far too large for one whose AMBIENT holds section counts).
    """
    model, _, _ = phi_model(toric_name)
    basis, td, _, _ = geometry(toric_name)
    assert model.nTransitions == fan_patch_transitions(td)
    # the patch table must have room for every patch a coordinate is free in
    pm = np.array(td['patch_masks'], dtype=bool)
    for i in range(model.ncoords):
        assert int(np.sum(~pm[:, i])) <= model.nTransitions
    # with cymetric's own AMBIENT convention (one entry per generator, the
    # number of its sections plus one) the projective count is unrelated:
    # 192 against 8 on the tetraquadric, 2.9e8 against 16 at h11 = 10, where
    # the patch table it asks for would be 780 GB
    assert model.nTransitions != projective_patch_transitions(basis)


def test_toric_metric_is_a_finite_hermitian_metric(toric_name):
    """g = g_FS + ddbar phi is Hermitian with a positive determinant."""
    model, X, y = phi_model(toric_name)
    g = model(X[:64])
    assert g.shape == (64, 3, 3)
    assert bool(jnp.all(jnp.isfinite(g.view(jnp.float32 if not x64() else jnp.float64))))
    herm = jnp.max(jnp.abs(g - jnp.conj(jnp.swapaxes(g, -1, -2))))
    assert float(herm) < 1e-5 * float(jnp.mean(jnp.abs(g)))
    det = np.real(np.linalg.det(np.array(g)))
    assert np.all(det > 0.)


@pytest.mark.parametrize('name', UNIMODULAR)
def test_fs_metric_transitions_between_toric_patches(name):
    """The Fubini-Study metric itself has zero toric transition loss.

    This is the end-to-end check of the ported patch data (patch_degrees,
    transition_coefficients, transition_degrees and nTransitions): g_FS is a
    genuine tensor, so T^dagger g_i T - g_j must vanish on every one of the
    nTransitions transitions of every point.  Only fans whose charts are all
    smooth are tested -- `cymetric.pointgen.nphelper.compute_all_w_of_x`
    cannot write the transition functions as monomials otherwise (it prints
    "WARNING W(X): NO INTEGER COEFFICIENTS" and rounds), which is an upstream
    numpy-side limitation, not a JAX one.
    """
    basis, td, X, y = geometry(name)
    nfold = int(np.real(basis['NFOLD']))
    model = MatrixFSModelToric(zero_net(X.shape[1], nfold ** 2), basis,
                               toric_data=td)
    loss = model.compute_transition_loss(X[:32])
    scale = float(jnp.mean(jnp.abs(model(X[:32]))))
    # the x64 floor is the sampled points themselves: they sit on the
    # hypersurface to a relative 1e-9, and the transition relates the
    # pullbacks, so the residual tracks that rather than the arithmetic
    assert float(jnp.max(loss)) < (1e-6 if x64() else 1e-4) * scale


@pytest.mark.parametrize('name', ('h11_5', 'h11_10'))
def test_fs_transition_residual_on_the_harder_fans(name):
    """A loose bound on the fans where upstream's transition data is not exact.

    Recorded rather than fixed: the data comes from
    `cymetric.pointgen.nphelper.compute_all_w_of_x` (numpy, shared with the
    TF and torch backends), which least-squares-fits the exponents of each
    patch transition and rounds.  On h11_5 (chart order 2) it warns; on
    h11_10 (a smooth fan, every generator base-point free) it does not warn
    and the Fubini-Study transition loss is still 2.4e-4 at 6% of the points
    under x64 (0.028 in float32, where the transition monomials of that fan
    lose digits), against 5e-8 on the fans of UNIMODULAR.  This test only
    pins the magnitude so that a JAX-side regression would still show up.
    """
    basis, td, X, y = geometry(name)
    nfold = int(np.real(basis['NFOLD']))
    model = MatrixFSModelToric(zero_net(X.shape[1], nfold ** 2), basis,
                               toric_data=td)
    loss = model.compute_transition_loss(X[:32])
    scale = float(jnp.mean(jnp.abs(model(X[:32]))))
    bound = 1e-1 if name == 'h11_10' else 1e3
    assert float(jnp.max(loss)) < bound * scale


def test_toric_phi_transition_losses_run(toric_name):
    """Both toric transition losses are jittable and finite.

    The patch version (``num_random_scalings=None``) used
    ``jnp.where(condition)`` twice, whose output shape depends on the data, so
    it raised a ConcretizationTypeError under jit.
    """
    model, X, y = phi_model(toric_name)
    scalings = model.compute_transition_loss(X[:32])
    patches = model.compute_transition_loss(X[:32], num_random_scalings=None)
    assert scalings.shape == (32,) and patches.shape == (32,)
    assert bool(jnp.all(jnp.isfinite(scalings)))
    assert bool(jnp.all(jnp.isfinite(patches)))


def test_training_batches_are_the_same_after_the_device_shuffle(toric_name):
    """The shuffle moved to the device; the batches it hands out are the same.

    `train_model` used to gather each batch out of the device arrays with a
    numpy index array, which costs 1.6 ms on top of a 0.9 ms step at batch 64
    (RTX 5090).  It now permutes once and slices, which is the same grouping
    of the same permutation -- this pins that, and that the permutation is
    still drawn from numpy's global RNG so a seeded run is reproducible.
    """
    from cymetric.jax.models.helper import _shuffled
    model, X, y = phi_model(toric_name)
    pb = model.pullbacks(X)
    sw = y[:, -2]
    np.random.seed(7)
    perm = np.random.permutation(len(X))
    np.random.seed(7)
    sx, sy, spb, ssw = _shuffled(X, y, pb, sw, len(X))
    for start in range(0, len(X), 23):
        idx = perm[start:start + 23]
        assert np.array_equal(np.array(sx[start:start + 23]), np.array(X[idx]))
        assert np.array_equal(np.array(sy[start:start + 23]), np.array(y[idx]))
        assert np.array_equal(np.array(spb[start:start + 23]), np.array(pb[idx]))
        assert np.array_equal(np.array(ssw[start:start + 23]), np.array(sw[idx]))
    assert _shuffled(X, y, pb, None, len(X))[3] is None


def test_chunked_pullbacks_precompute_matches_one_program(toric_name):
    """train_model precomputes the pullbacks in chunks; same tensor.

    Not bit-exact on a GPU: chunking changes the reduction order, so float32
    differs by about 5e-7 on entries of order 1. Equality held on the CPU,
    which is why this was written as `== 0`.
    """
    model, X, y = phi_model(toric_name)
    one = model.pullbacks(X)
    chunked = jnp.concatenate([model.pullbacks(X[i:i + 37])
                               for i in range(0, len(X), 37)])
    tol = 1e-12 if x64() else 1e-5
    assert float(jnp.max(jnp.abs(one - chunked))) < tol * float(jnp.max(jnp.abs(one)))


def test_toric_phi_training_step_runs(toric_name):
    """One epoch of the standard training loop on a toric model."""
    import optax
    from cymetric.jax.models.helper import train_model
    model, X, y = phi_model(toric_name, width=8)
    data = {'X_train': np.array(X[:64]), 'y_train': np.array(y[:64])}
    out, hist = train_model(model, data, optimizer=optax.adam(1e-3), epochs=1,
                            batch_sizes=(32, 64), verbose=0)
    assert np.isfinite(hist['General loss phase'][0])
    assert np.isfinite(hist['Volume loss phase'][0])
    # the weights moved and the metric is still finite
    g = out(X[:16])
    assert bool(jnp.all(jnp.isfinite(jnp.real(g)))) and bool(jnp.all(jnp.isfinite(jnp.imag(g))))
