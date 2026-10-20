"""The scanned training epoch reproduces the host-loop epoch, step for step.

``helper.train_model`` used to dispatch one jitted ``train_step`` per batch from
a Python ``for`` loop; it now runs each phase as a single ``lax.scan`` over the
batch axis.  That is a pure restructuring, so for a fixed seed it must give the
same batches in the same order, the same optimizer updates, and the same
history.

The reference in ``tests/data/epoch_scan_reference.npz`` was captured by running
this file as a script against the pre-scan ``train_model`` (branch
``perf-jax-models``, commit 6a01f82):

    PYTHONPATH=<worktree> JAX_PLATFORMS=cpu python tests/test_jax_epoch_scan.py

Regenerating it after a deliberate change of training semantics is the only
reason to run that; re-running it against the new code would make the test
tautological.

:Authors:
    Nate MacFadden
"""
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from conftest import DATA, phi_model, rtol, x64
from test_jax_retracing import _compiles

from cymetric.jax.models.callbacks import SigmaCallback
from cymetric.jax.models.helper import train_model
from cymetric.jax.models.metrics import (SigmaLoss, TotalLoss, TransitionLoss,
                                         VolkLoss)

REFERENCE = os.path.join(DATA, 'epoch_scan_reference.npz')

# (fixture, epochs, batch_sizes, sample weights).  The batch sizes are chosen
# so that phase 1 has a ragged last batch (200 points, batch 64 -> 3 full
# batches plus one of 8) and phase 2 does not, because the ragged tail is the
# one step that cannot sit inside the scan.
CONFIGS = (
    ('quintic', 3, (64, 100), False),
    ('h11_3', 2, (64, 200), True),
)

# the per-component history keys train_model merges into its return value
COMPONENTS = ('sigma_loss', 'kaehler_loss', 'transition_loss', 'ricci_loss',
              'volk_loss')


def _run(name, epochs, batch_sizes, sw):
    """One short two-phase training run, deterministic for a fixed seed.

    Returns a flat dict of float arrays: the two phase histories, the merged
    per-component histories, the callback's per-epoch value, the custom
    metrics' running means, and the final network parameters.
    """
    model, X, y = phi_model(name, width=8, seed=1)
    data = {'X_train': X, 'y_train': y}
    metrics = [SigmaLoss(), TransitionLoss(), VolkLoss(), TotalLoss()]
    callbacks = [SigmaCallback((X, y))]
    optimizer = optax.adam(1e-3)

    np.random.seed(7)
    trained, history = train_model(
        model, data, optimizer=optimizer, epochs=epochs,
        batch_sizes=list(batch_sizes), verbose=0,
        custom_metrics=metrics, callbacks=callbacks, sw=sw)

    out = {}
    for key in ('General loss phase', 'Volume loss phase', 'sigma_val'):
        out[key] = np.asarray(history[key], dtype=np.float64)
    for key in COMPONENTS:
        out[key] = np.asarray(history[key], dtype=np.float64)
    out['epochs'] = np.asarray(history['epochs'], dtype=np.float64)
    out['metrics'] = np.asarray([m.result() for m in metrics], dtype=np.float64)
    leaves = jax.tree_util.tree_leaves(eqx.filter(trained.model, eqx.is_array))
    out['params'] = np.concatenate(
        [np.asarray(leaf, dtype=np.float64).ravel() for leaf in leaves])
    return out


def _capture():
    """Writes the reference file from whatever train_model is importable."""
    saved = {}
    for name, epochs, batch_sizes, sw in CONFIGS:
        for key, value in _run(name, epochs, batch_sizes, sw).items():
            saved[name + '|' + key] = value
    np.savez(REFERENCE, **saved)
    return saved


@pytest.mark.skipif(not os.path.exists(REFERENCE),
                    reason='reference not captured')
@pytest.mark.parametrize('config', CONFIGS, ids=[c[0] for c in CONFIGS])
def test_scanned_epoch_reproduces_the_host_loop(config):
    """Every history key and the final parameters match the pre-scan run."""
    if x64():
        pytest.skip('reference was captured in float32')
    name, epochs, batch_sizes, sw = config
    ref = np.load(REFERENCE)
    got = _run(name, epochs, batch_sizes, sw)
    keys = [k[len(name) + 1:] for k in ref.files if k.startswith(name + '|')]
    assert sorted(keys) == sorted(got.keys())
    for key in keys:
        want = ref[name + '|' + key]
        have = got[key]
        assert have.shape == want.shape, key
        scale = max(float(np.max(np.abs(want))), 1e-30)
        err = float(np.max(np.abs(have - want))) / scale
        # float32 arithmetic, and a scan body is fused differently from a
        # sequence of separate programs, so this is a rounding tolerance
        # rather than bit equality.  Measured on the CPU at the time of the
        # change: most keys bit-exact, worst 1.7e-7 relative, i.e. float32
        # eps.  The tolerance is the suite's usual float32 band, which is
        # still orders of magnitude below what a dropped batch or a changed
        # update rule would do.
        assert err < rtol(), (key, err)


@pytest.mark.skipif(not os.path.exists(REFERENCE),
                    reason='reference not captured')
def test_history_keys_are_unchanged():
    """The returned dict still has exactly the keys it used to have."""
    name, epochs, batch_sizes, sw = CONFIGS[0]
    model, X, y = phi_model(name, width=8, seed=1)
    np.random.seed(7)
    _, history = train_model(model, {'X_train': X, 'y_train': y},
                             epochs=1, batch_sizes=list(batch_sizes),
                             verbose=0)
    assert set(history.keys()) == {
        'General loss phase', 'Volume loss phase', 'epochs'} | set(COMPONENTS)


def test_phase_batches_cover_the_shuffle_exactly():
    """The scan's full batches plus the ragged tail are the whole permutation.

    A reshape-based scan silently drops the tail if the batch count is computed
    with ceil instead of floor, which the loss history would hide (it is an
    average) but the trajectory would not.
    """
    from cymetric.jax.models.helper import _phase_batching
    for n_train, batch in ((200, 64), (200, 100), (200, 200), (64, 64),
                           (63, 64), (1, 64), (10000, 64)):
        n_full, tail = _phase_batching(n_train, batch)
        assert n_full * batch + tail == n_train
        assert 0 <= tail < batch
        n_steps = (n_train + min(batch, n_train) - 1) // min(batch, n_train)
        assert n_full + (1 if tail else 0) == n_steps


def _host_loop(name, epochs, batch_sizes, sw):
    """The pre-scan epoch, written out: one jitted step dispatched per batch.

    Deliberately a transcription of the loop ``train_model`` used to run, not a
    call into it, so that this test keeps working as a reference after the
    library's own loop has changed.  It uses the library's shuffle and flag
    helpers, because those are the parts whose semantics must be shared.
    """
    from cymetric.jax.models.dtypes import real_dtype
    from cymetric.jax.models.helper import (_make_train_step, _shuffled,
                                            _with_flags)

    model, X, y = phi_model(name, width=8, seed=1)
    X = jnp.asarray(X, dtype=real_dtype())
    y = jnp.asarray(y, dtype=real_dtype())
    pullbacks = model.pullbacks(X)
    weights = y[:, -2] if sw else None
    n_train = len(X)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model.model, eqx.is_array))
    step = _make_train_step(optimizer)
    flags = (model.learn_kaehler, model.learn_transition, model.learn_ricci,
             model.learn_ricci_val)

    history = {'General loss phase': [], 'Volume loss phase': []}
    np.random.seed(7)
    for _ in range(epochs):
        phases = (('General loss phase', batch_sizes[0],
                   flags + (False,)),
                  ('Volume loss phase', min(batch_sizes[1], n_train),
                   (False, False, False, False, True)))
        for key, batch_size, phase_flags in phases:
            model = _with_flags(model, *phase_flags)
            sx, sy, spb, ssw = _shuffled(X, y, pullbacks, weights, n_train)
            total, n_batches = 0., 0
            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                bsw = None if ssw is None else ssw[start:end]
                model, opt_state, loss, _parts = step(
                    model, opt_state, sx[start:end], sy[start:end], bsw,
                    spb[start:end])
                total = total + loss
                n_batches += 1
            history[key].append(float(total / n_batches))

    leaves = jax.tree_util.tree_leaves(eqx.filter(model.model, eqx.is_array))
    params = np.concatenate(
        [np.asarray(leaf, dtype=np.float64).ravel() for leaf in leaves])
    return history, params


@pytest.mark.parametrize('config', CONFIGS, ids=[c[0] for c in CONFIGS])
def test_the_scan_is_the_host_loop(config):
    """The scanned phase equals a per-batch dispatch loop, at any precision.

    The stored reference above is a float32 CPU capture of the code as it was;
    this is the same claim without a stored artefact, so it also runs under
    ``jax_enable_x64`` and on a device whose float32 rounds differently.
    """
    name, epochs, batch_sizes, sw = config
    want, want_params = _host_loop(name, epochs, batch_sizes, sw)
    got = _run(name, epochs, batch_sizes, sw)
    for key in want:
        have = np.asarray(got[key])
        ref = np.asarray(want[key])
        scale = max(float(np.max(np.abs(ref))), 1e-30)
        assert float(np.max(np.abs(have - ref))) / scale < rtol(), key
    scale = max(float(np.max(np.abs(want_params))), 1e-30)
    assert float(np.max(np.abs(got['params'] - want_params))) / scale < rtol()


def test_the_programs_are_built_once_not_once_per_epoch():
    """A four-epoch run compiles exactly what a one-epoch run compiles.

    The phase programs are built outside the epoch loop, so the number of XLA
    compilations must not grow with ``epochs``.  Each run below uses a
    different ``n_train``, so none of them can be answered out of a previous
    run's cache; the first is a warm-up for the shape-independent programs.
    """
    model, X, y = phi_model('quintic', width=8, seed=1)

    def run(n_train, epochs):
        data = {'X_train': X[:n_train], 'y_train': y[:n_train]}
        np.random.seed(3)
        return _compiles(lambda: train_model(
            model, data, epochs=epochs, batch_sizes=[64, 100], verbose=0))

    if run(192, 1) == 0:
        pytest.skip('compilation logging not available in this JAX build')
    one = run(184, 1)
    four = run(176, 4)
    assert one > 0
    assert four == one


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if jax.config.jax_enable_x64:
        raise SystemExit('capture the reference in float32')
    captured = _capture()
    print('wrote', REFERENCE)
    for k in sorted(captured):
        v = captured[k]
        print('  %-40s %s' % (k, np.array2string(
            v.ravel()[:4], precision=8)), v.shape)
