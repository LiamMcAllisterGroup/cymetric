"""The projective (CICY) path still behaves after the _generate_helpers move.

``FSModel.__init__`` now delegates its helper fields to ``_generate_helpers``
so that the toric override is reachable.  These tests pin the projective
helpers it used to build inline, on one and on four projective factors.

:Authors:
    Nate MacFadden
"""
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import (CICY, geometry, projective_patch_transitions, rtol, x64,
                      zero_net)

from cymetric.jax.models.models import AddFSModel, PhiFSModel


@pytest.mark.parametrize('name', CICY)
def test_projective_helpers_are_built(name):
    """proj_matrix, the patch table and the transition count come from AMBIENT."""
    basis, _, X, y = geometry(name)
    nfold = int(np.real(basis['NFOLD']))
    model = PhiFSModel(zero_net(X.shape[1], 1), basis)
    degrees = [int(np.real(a)) + 1 for a in np.array(basis['AMBIENT'])]
    assert model.nProjective == len(degrees)
    assert list(np.array(model.degrees)) == degrees
    assert set(model.proj_matrix) == {str(i) for i in range(len(degrees))}
    for i, d in enumerate(degrees):
        assert model.proj_matrix[str(i)].shape == (d, model.ncoords)
    assert model.nTransitions == projective_patch_transitions(basis)
    assert model.fixed_patches.shape == (model.ncoords, model.nTransitions,
                                         len(degrees))
    assert model.fixed_patches_n is None     # nhyper == 1


@pytest.mark.parametrize('name', CICY)
def test_fs_metric_transitions_between_projective_patches(name):
    """The Fubini-Study metric has zero transition loss on a CICY too."""
    basis, _, X, y = geometry(name)
    nfold = int(np.real(basis['NFOLD']))
    model = AddFSModel(zero_net(X.shape[1], nfold ** 2), basis)
    loss = model.compute_transition_loss(X[:32])
    scale = float(jnp.mean(jnp.abs(model(X[:32]))))
    assert float(jnp.max(loss)) < (1e-6 if x64() else 1e-4) * scale


@pytest.mark.parametrize('name', CICY)
def test_phi_model_is_kaehler(name):
    """ddbar phi is closed, so the Kahler loss vanishes identically."""
    from conftest import phi_model
    model, X, y = phi_model(name)
    assert not model.learn_kaehler
    loss = model.compute_kaehler_loss(X[:16])
    g = model(X[:16])
    assert float(jnp.max(loss)) < (1e-6 if x64() else 1e-2) * float(jnp.mean(jnp.abs(g)))
