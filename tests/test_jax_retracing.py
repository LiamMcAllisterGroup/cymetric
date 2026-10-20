"""A jitted program taking a model must not recompile per model instance.

A scan over moduli points or geometries builds one model per point and hands
it to the same jitted programs.  Everything on an ``eqx.Module`` that is not
an array is static, so a Python-float ``kappa`` -- or a per-instance loss
closure -- made every one of those programs recompile.  Both are measured
here by counting XLA compilations.

:Authors:
    Nate MacFadden
"""
import io
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import geometry, make_net, phi_model

from cymetric.jax.models.models import PhiFSModelToric


@eqx.filter_jit
def _det_g(model, X):
    return jnp.real(jnp.linalg.det(model(X)))


def _compiles(fn):
    """Number of XLA compilations fn triggers."""
    logs = io.StringIO()
    handler = logging.StreamHandler(logs)
    logger = logging.getLogger('jax')
    logger.addHandler(handler)
    try:
        with jax.log_compiles():
            fn()
    finally:
        logger.removeHandler(handler)
    return logs.getvalue().lower().count('compiling')


def test_no_retrace_across_model_instances():
    """A rebuilt model, a new kappa and new weights all reuse the program."""
    model, X, y = phi_model('h11_3', seed=0)
    basis, td, _, _ = geometry('h11_3')
    X = X[:32]
    if _compiles(lambda: _det_g(model, X).block_until_ready()) == 0:
        pytest.skip('compilation logging not available in this JAX build')
    assert _compiles(lambda: _det_g(model, X).block_until_ready()) == 0

    # the same geometry and network, built a second time
    again, _, _ = phi_model('h11_3', seed=0)
    assert _compiles(lambda: _det_g(again, X).block_until_ready()) == 0

    # a different moduli point of the same geometry: only kappa changes
    other = dict(basis)
    other['KAPPA'] = 1.37           # as a BASIS pickle carries it: a float
    moved = PhiFSModelToric(model.model, other, toric_data=td)
    assert float(moved.kappa) != float(model.kappa)
    assert _compiles(lambda: _det_g(moved, X).block_until_ready()) == 0

    # new weights, as a training step produces
    trained = eqx.tree_at(lambda q: q.model, model,
                          make_net(X.shape[1], width=32, seed=7))
    assert _compiles(lambda: _det_g(trained, X).block_until_ready()) == 0


def test_kappa_is_an_array_and_still_reaches_the_loss():
    """kappa is dynamic, and the MA loss is the same number it always was."""
    model, X, y = phi_model('h11_3')
    assert isinstance(model.kappa, jax.Array)
    g = model(X[:32])
    from cymetric.jax.models.losses import sigma_loss
    closure = sigma_loss(float(model.kappa), float(model.nfold))
    assert np.allclose(np.array(model.sigma_loss(y[:32], g)),
                       np.array(closure(y[:32], g)), rtol=1e-6)


def test_flat_sigma_loss_matches_the_closure():
    """The flat (vector prediction) branch too."""
    from cymetric.jax.models.losses import sigma_loss
    model, X, y = phi_model('h11_3')
    flat = jax.random.normal(jax.random.PRNGKey(0), (32, model.nfold ** 2))
    closure = sigma_loss(float(model.kappa), float(model.nfold), flat=True)
    assert np.allclose(np.array(model.sigma_loss(y[:32], flat, flat=True)),
                       np.array(closure(y[:32], flat)), rtol=1e-6)
