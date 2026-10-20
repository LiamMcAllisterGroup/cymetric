"""Shared fixtures for the JAX backend tests.

The geometries live in ``tests/data`` as small pickles, each holding the
``BASIS`` that cymetric's own ``prepare_basis_pickle`` writes, the points and
labels of a few hundred samples, and (for the toric ones) the ``toric_data``
dict that ``cymetric.sage.sagelib.prepare_toric_cy_data`` produces.  They are
committed so that the tests need neither SageMath, Mathematica nor cytools.

:Authors:
    Nate MacFadden
"""
import os
import pickle

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(__file__), 'data')

# the toric fixtures, with the number of Kahler cone generators
TORIC = ('tetraquadric', 'h11_3', 'h11_5', 'h11_10')
# fans on which upstream's patch-transition data is exact.  It is built by
# `cymetric.pointgen.nphelper.compute_all_w_of_x`, which least-squares-fits
# and rounds the monomial exponents of each transition: on an orbifold chart
# it warns ("WARNING W(X): NO INTEGER COEFFICIENTS") and rounds anyway
# (h11_5), and even on the smooth h11_10 fan a few patch pairs come out
# wrong without a warning (see PORTING.md).  A numpy-side limitation shared
# by all three backends.
UNIMODULAR = ('tetraquadric', 'h11_3')
# the complete-intersection fixtures (one and four projective factors)
CICY = ('quintic', 'tetraquadric_cicy')


def x64():
    """True when the test session runs with jax_enable_x64."""
    return bool(jax.config.jax_enable_x64)


def rtol():
    """Relative tolerance for 'the two routes agree to rounding'."""
    return 1e-12 if x64() else 1e-5


def load(name):
    """The raw fixture dict."""
    with open(os.path.join(DATA, name + '.pickle'), 'rb') as f:
        return pickle.load(f)


def make_net(n_in, width=32, depth=2, seed=0, n_out=1):
    """A small MLP, (n_in,) -> (n_out,), as the models expect."""
    keys = jax.random.split(jax.random.PRNGKey(seed), depth + 1)
    layers = [eqx.nn.Linear(n_in, width, key=keys[0]), eqx.nn.Lambda(jax.nn.gelu)]
    for i in range(depth - 1):
        layers += [eqx.nn.Linear(width, width, key=keys[i + 1]),
                   eqx.nn.Lambda(jax.nn.gelu)]
    layers += [eqx.nn.Linear(width, n_out, key=keys[-1])]
    return eqx.nn.Sequential(layers)


def zero_net(n_in, n_out, seed=0):
    """A net that returns exactly zero, so that g_out is the FS metric."""
    net = eqx.nn.Sequential([eqx.nn.Linear(n_in, n_out, key=jax.random.PRNGKey(seed))])
    return eqx.tree_at(lambda m: [m.layers[0].weight, m.layers[0].bias], net,
                       [jnp.zeros((n_out, n_in)), jnp.zeros(n_out)])


def geometry(name):
    """(BASIS as JAX arrays, toric_data or None, X, y) of a fixture."""
    from cymetric.jax.models.helper import prepare_basis
    from cymetric.jax.models.dtypes import real_dtype
    d = load(name)
    basis = prepare_basis(d['BASIS'])
    X = jnp.array(d['X'], dtype=real_dtype())
    y = jnp.array(d['y'], dtype=real_dtype())
    return basis, d.get('toric_data'), X, y


def phi_model(name, width=32, seed=0):
    """A phi model (toric or CICY, matching the fixture) with a small MLP."""
    from cymetric.jax.models.models import PhiFSModel, PhiFSModelToric
    basis, td, X, y = geometry(name)
    net = make_net(X.shape[1], width=width, seed=seed)
    if td is None:
        return PhiFSModel(net, basis), X, y
    return PhiFSModelToric(net, basis, toric_data=td), X, y


def projective_patch_transitions(basis, nhyper=1):
    """The projective-space transition count FSModel._patch_transitions gives."""
    from cymetric.pointgen.nphelper import generate_monomials
    degrees = [int(np.real(a)) + 1 for a in np.array(basis['AMBIENT'])]
    best = 0
    for t in generate_monomials(len(degrees), nhyper):
        best = max(best, int(np.prod([d - t[j] for j, d in enumerate(degrees)])))
    return best


def fan_patch_transitions(toric_data):
    """The toric transition count: patches in which a coordinate is not a pivot."""
    pm = np.array(toric_data['patch_masks'], dtype=bool)
    return int(np.max(np.sum(~pm, axis=-2)))


@pytest.fixture(scope='module', params=TORIC)
def toric_name(request):
    return request.param


@pytest.fixture(scope='module', params=TORIC + CICY)
def any_name(request):
    return request.param
