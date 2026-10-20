"""Benchmark case definitions for the cymetric harness (see bench.py).

Every case function returns a dict with at least::

    {'timing': <output of bench.time_calls>, 'headline': '<one line>'}

and may add 'derived' (throughputs, per-step times) and 'info' (anything
needed to interpret the numbers: shapes, geometry sizes, tuned chain
parameters, ...).  Cases raise :class:`KnownBlocked` when the library cannot
run them because of a defect that is being tracked; the harness records those
with the exact error and keeps the case, so the day the defect is fixed the
case starts producing numbers.

Geometries
----------
``quintic``       : the Fermat quintic in P^4.                 h11 = 1, 5 coords
``bicubic``       : a (3,3) hypersurface in P^2 x P^2.         h11 = 2, 6 coords
``tetraquadric``  : a (2,2,2,2) hypersurface in (P^1)^4.       h11 = 4, 8 coords
``toric_h11_N``   : Kreuzer-Skarke hypersurfaces, from benchmarks/data/.

The three CICY geometries are the h11 axis that the JAX metric model can
actually be run on today (see the toric cases below), and they grow the two
things that drive cost in the model: the number of ambient coordinates and the
number of projective factors.
"""

import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _p in (REPO_ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import time_calls  # noqa: E402

CACHE = os.environ.get('CYMETRIC_BENCH_CACHE', os.path.join(HERE, 'cache'))
DATA = os.path.join(HERE, 'data')


class KnownBlocked(Exception):
    """Raised when a case cannot run because of a tracked upstream defect."""


# ---------------------------------------------------------------------------
# geometries
# ---------------------------------------------------------------------------

#: CICY geometries.  ``conf`` is the configuration matrix (rows = projective
#: factors, the single column = the hypersurface multidegree; the ambient P^n
#: has n = sum(row) - 1).  ``monomials='fermat'`` keeps only the diagonal
#: monomials z_i^d (the Fermat quintic, cymetric's standard easy case);
#: ``'all'`` is the dense monomial basis of that multidegree with fixed
#: pseudo-random complex coefficients, i.e. a generic complex structure.
#:
#: The monomial count is part of the cost of both point generation and the
#: pullbacks, and it is *not* constant along the h11 axis below, so
#: ``quintic_dense`` is included to separate the two effects.
CICY_GEOMETRIES = {
    'quintic':       dict(conf=[[5]], monomials='fermat'),
    'quintic_dense': dict(conf=[[5]], monomials='all'),
    'bicubic':       dict(conf=[[3], [3]], monomials='all'),
    'tetraquadric':  dict(conf=[[2], [2], [2], [2]], monomials='all'),
}


def cicy_pointgen(geometry, verbose=3):
    """A :class:`cymetric.pointgen.pointgen.PointGenerator` for a CICY."""
    from cymetric.pointgen.pointgen import PointGenerator
    from cymetric.pointgen.nphelper import conf_to_monomials
    spec = CICY_GEOMETRIES[geometry]
    conf = np.array(spec['conf'], dtype=np.int64)
    ambient = np.sum(conf, axis=-1) - 1
    monomials = np.array(conf_to_monomials(conf)[0], dtype=np.int64)
    if spec['monomials'] == 'fermat':
        degree = int(np.max(monomials.sum(axis=-1)))
        monomials = monomials[np.max(monomials, axis=-1) == degree]
        coefficients = np.ones(len(monomials))
    else:
        rng = np.random.default_rng(0)
        coefficients = (rng.standard_normal(len(monomials))
                        + 1j * rng.standard_normal(len(monomials)))
    kmoduli = np.ones(len(ambient))
    return PointGenerator(monomials, coefficients, kmoduli, ambient,
                          verbose=verbose)


def load_toric_data(h11):
    """Toric geometry input (cymetric's ``toric_data`` dict) from data/."""
    path = os.path.join(DATA, 'toric_h11_%d.json' % h11)
    if not os.path.exists(path):
        raise FileNotFoundError(
            'missing %s; see benchmarks/make_toric_data.py' % path)
    with open(path) as fh:
        blob = json.load(fh)
    td = blob['toric_data']
    td['coeff_aK'] = [complex(re, im) for re, im in td['coeff_aK']]
    blob['kmoduli'] = np.array(blob['kmoduli'], dtype=float)
    return blob


def toric_pointgen(h11, verbose=3, **kwargs):
    from cymetric.pointgen.pointgen_mc import ToricPointGeneratorMC
    blob = load_toric_data(h11)
    pg = ToricPointGeneratorMC(blob['toric_data'], blob['kmoduli'],
                               verbose=verbose, seed=0, **kwargs)
    return pg, blob


def toric_basis(pg, blob):
    """BASIS dict for cymetric's toric models.

    cymetric's toric point generators do not produce an ``AMBIENT`` entry that
    satisfies the base model's ``sum(AMBIENT + 1) == ncoords`` requirement
    (``ToricPointGenerator`` overwrites ``self.ambient`` with ``2 * selected_t``,
    a sampling helper), so the harness supplies a partition of ``ncoords`` into
    ``len(sections)`` blocks, which is what the toric models' own code path
    needs; the toric metric never reads the values, only the count.  This is
    harness-side data preparation, not a change to the library.
    """
    td = blob['toric_data']
    n_gen = len(td['exps_sections'])
    base, rem = divmod(pg.ncoords, n_gen)
    ambient = np.array([base - 1 + (1 if i < rem else 0) for i in range(n_gen)])
    basis = dict(pg.BASIS)
    basis.update(dict(AMBIENT=ambient, KMODULI=blob['kmoduli'],
                      NFOLD=td['dim_cy'], NHYPER=1,
                      INTNUMS=np.array(blob['int_nums']), KAPPA=1.0))
    return basis


# ---------------------------------------------------------------------------
# datasets (setup, never timed)
# ---------------------------------------------------------------------------

def cicy_dataset(geometry, n_points):
    """Generate (and cache) a training set + BASIS for a CICY geometry."""
    from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle
    dirname = os.path.join(CACHE, '%s_%d' % (geometry, n_points))
    dataset = os.path.join(dirname, 'dataset.npz')
    basis = os.path.join(dirname, 'basis.pickle')
    if not (os.path.exists(dataset) and os.path.exists(basis)):
        os.makedirs(dirname, exist_ok=True)
        np.random.seed(0)
        pg = cicy_pointgen(geometry)
        kappa = prepare_dataset(pg, n_points, dirname)
        prepare_basis_pickle(pg, dirname, kappa)
    data = np.load(dataset, allow_pickle=True)
    with open(basis, 'rb') as fh:
        BASIS = pickle.load(fh)
    return data, BASIS


def make_mlp(n_in, n_out, width=64, depth=3, seed=42):
    """The standard cymetric metric network (notebooks/0.GettingStarted)."""
    import jax
    import jax.nn as jnn
    import equinox as eqx
    keys = jax.random.split(jax.random.PRNGKey(seed), depth + 1)
    layers = []
    prev = n_in
    for i in range(depth):
        layers += [eqx.nn.Linear(prev, width, key=keys[i]),
                   eqx.nn.Lambda(jnn.gelu)]
        prev = width
    layers += [eqx.nn.Linear(prev, n_out, use_bias=False, key=keys[depth])]
    return eqx.nn.Sequential(layers)


def jax_model(geometry, n_points, model_name='PhiFSModel'):
    """Build a JAX model + on-device training data for a CICY geometry."""
    import jax.numpy as jnp
    from cymetric.jax.models import models as jax_models
    from cymetric.jax.models.helper import prepare_basis
    data, BASIS_np = cicy_dataset(geometry, n_points)
    BASIS = prepare_basis(BASIS_np)
    cls = getattr(jax_models, model_name)
    n_in = data['X_train'].shape[1]
    n_out = 1 if model_name.startswith('Phi') else int(BASIS_np['NFOLD']) ** 2
    model = cls(make_mlp(n_in, n_out), BASIS)
    X = jnp.array(data['X_train'], dtype=jnp.float32)
    Y = jnp.array(data['y_train'], dtype=jnp.float32)
    pb = model.pullbacks(X)
    return model, X, Y, pb


# ---------------------------------------------------------------------------
# cases: point generation
# ---------------------------------------------------------------------------

def pointgen_cicy(geometry, n_points, nproc, method, repeats):
    """Throughput of cymetric's CICY (line-intersection) point generator.

    ``method='points'`` times :meth:`generate_points` (sampling only).
    ``method='weights'`` times :meth:`generate_point_weights`, which is what
    ``prepare_dataset`` calls: sampling plus integration weights plus Omega,
    i.e. the real cost of building a training set.
    """
    pg = cicy_pointgen(geometry)
    if method == 'points':
        def call():
            return pg.generate_points(n_points, nproc=nproc)
    elif method == 'weights':
        def call():
            return pg.generate_point_weights(n_points, omega=True)
    else:
        raise ValueError('method must be points or weights')

    timing = time_calls(call, repeats)
    rate = n_points / timing['median_s']
    return {
        'timing': timing,
        'derived': {'points_per_s': rate,
                    'us_per_point': 1e6 * timing['median_s'] / n_points},
        'info': {'ncoords': int(pg.ncoords), 'nfold': int(pg.nfold),
                 'n_projective': int(len(pg.ambient)),
                 'n_monomials': int(pg.nmonomials),
                 'note': 'the first call also pays joblib worker start-up'},
        'headline': '%.0f points/s' % rate,
    }


def pointgen_toric(h11, n_points, repeats):
    """Throughput of cymetric's toric Monte-Carlo point generator.

    Every call re-runs the sampler's auto-tuning (pilot chain for the step
    size, autocorrelation estimate for the thinning), because the tuned values
    are not cached on the generator unless they are passed to the constructor.
    That fixed per-call cost is included here, exactly as a user pays it; the
    n_p=500 case at the same h11 exposes how large it is.
    """
    pg, blob = toric_pointgen(h11)

    def call():
        return pg.generate_points(n_points)

    timing = time_calls(call, repeats)
    rate = n_points / timing['median_s']
    td = blob['toric_data']
    return {
        'timing': timing,
        'derived': {'points_per_s': rate,
                    'ms_per_point': 1e3 * timing['median_s'] / n_points},
        'info': {'h11': h11, 'ncoords': int(pg.ncoords),
                 'n_monomials': int(len(td['exp_aK'])),
                 'n_patches': int(len(td['patch_masks'])),
                 'n_kaehler_cone_generators': int(len(td['exps_sections'])),
                 'n_sections_total': int(sum(len(s) for s in td['exps_sections'])),
                 'includes_auto_tuning': True},
        'headline': '%.0f points/s' % rate,
    }


# ---------------------------------------------------------------------------
# cases: JAX metric training
# ---------------------------------------------------------------------------

def train_jax(geometry, batch_size, n_points, repeats, model_name='PhiFSModel'):
    """One optimizer step of the JAX metric model: compile vs steady state.

    Times the jitted training step built by
    ``cymetric.jax.models.helper._make_train_step`` (the function
    ``train_model`` runs in its inner loop), on batches that are already on the
    device, cycling over distinct batches so the measurement is not a single
    cached slice.  ``first_call_s`` is the XLA compilation of that step.

    The epoch estimate times a contiguous block of steps with a single
    synchronisation at the end (so host dispatch overlaps compute, as in a real
    epoch) and scales it to ``n_train / batch_size`` steps.
    """
    import optax
    import equinox as eqx
    from cymetric.jax.models.helper import _make_train_step

    model, X, Y, pb = jax_model(geometry, n_points, model_name)
    n_train = X.shape[0]
    if batch_size > n_train:
        raise ValueError('batch_size %d > n_train %d' % (batch_size, n_train))
    n_batches = max(1, min(8, n_train // batch_size))
    batches = [(X[i * batch_size:(i + 1) * batch_size],
                Y[i * batch_size:(i + 1) * batch_size],
                pb[i * batch_size:(i + 1) * batch_size])
               for i in range(n_batches)]

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model.model, eqx.is_array))
    step = _make_train_step(optimizer)

    state = {'model': model, 'opt': opt_state, 'i': 0}

    def call():
        bx, by, bpb = batches[state['i'] % n_batches]
        state['i'] += 1
        m, o, loss, _ = step(state['model'], state['opt'], bx, by, None, bpb)
        state['model'], state['opt'] = m, o
        return loss

    timing = time_calls(call, repeats, block=lambda r: r.block_until_ready())

    # contiguous block, single sync at the end -> realistic epoch rate
    block_steps = min(32, max(4, n_train // batch_size))
    t0 = time.perf_counter()
    loss = None
    for _ in range(block_steps):
        bx, by, bpb = batches[state['i'] % n_batches]
        state['i'] += 1
        state['model'], state['opt'], loss, _ = step(
            state['model'], state['opt'], bx, by, None, bpb)
    loss.block_until_ready()
    block_s = time.perf_counter() - t0

    steps_per_epoch = max(1, n_train // batch_size)
    per_step = block_s / block_steps
    return {
        'timing': timing,
        'derived': {
            'ms_per_step_median': 1e3 * timing['median_s'],
            'ms_per_step_pipelined': 1e3 * per_step,
            'steps_per_epoch': steps_per_epoch,
            's_per_epoch_est': per_step * steps_per_epoch,
            'points_per_s': batch_size / per_step,
            'compile_s': timing['first_call_s'],
            'block_steps': block_steps,
        },
        'info': {'geometry': geometry, 'model': model_name,
                 'n_train': int(n_train), 'batch_size': batch_size,
                 'ncoords': int(model.ncoords), 'nfold': int(model.nfold),
                 'n_projective': int(model.nProjective),
                 'final_loss': float(loss)},
        'headline': '%.2f ms/step, %.2f s/epoch (%d steps)'
                    % (1e3 * timing['median_s'], per_step * steps_per_epoch,
                       steps_per_epoch),
    }


def train_model_jax(geometry, n_points, epochs, batch_sizes, repeats,
                    model_name='PhiFSModel'):
    """End-to-end ``helper.train_model``: the seconds per epoch users see.

    The per-step cases above time the jitted step in isolation; this one runs
    the library's own two-phase training loop, so it also contains the phase-2
    batch, the host-side shuffling and numpy fancy-indexing of every batch, the
    history bookkeeping, and the XLA compilation of both phases.  Divide by
    ``epochs`` for a per-epoch figure that still carries a share of the
    one-off compilation.

    ``train_model`` builds its jitted step inside the call, but a second
    invocation in the same process still hits XLA's compilation cache, so
    ``first_call_s - median_s`` is a lower bound on the compile share rather
    than the whole of it.  The gap between this case's per-epoch time and
    ``train_jax.<geometry>.bs64``'s ``s_per_epoch_est`` is the overhead the
    library's loop adds around the step itself.
    """
    from cymetric.jax.models.helper import train_model

    model, X, Y, _pb = jax_model(geometry, n_points, model_name)
    data = {'X_train': X, 'y_train': Y}

    def call():
        m, hist = train_model(model, data, epochs=epochs,
                              batch_sizes=list(batch_sizes), verbose=0)
        return hist['General loss phase'][-1]

    timing = time_calls(call, repeats, block=lambda r: float(r))
    per_epoch = timing['median_s'] / epochs if 'median_s' in timing \
        else timing['first_call_s'] / epochs
    return {
        'timing': timing,
        'derived': {'s_per_epoch_incl_compile_share': per_epoch,
                    'epochs': epochs,
                    'steps_per_epoch_phase1': int(len(X) // batch_sizes[0]),
                    'n_train': int(len(X))},
        'info': {'geometry': geometry, 'model': model_name,
                 'batch_sizes': list(batch_sizes),
                 'what_is_timed': 'cymetric.jax.models.helper.train_model, '
                                  'both phases, incl. compilation'},
        'headline': '%.2f s/epoch end-to-end (%d epochs/call)'
                    % (per_epoch, epochs),
    }


def detg_jax(geometry, n_points, chunk, repeats, model_name='PhiFSModel'):
    """Throughput of ``det g`` evaluation over a point set.

    This is the quantity every downstream measure (sigma, Ricci, volumes) is
    built on, and the known memory hot spot: ``model(x)`` builds a per-point
    Hessian of the network before contracting with the pullbacks.  The point
    set is processed in fixed-size chunks so that only one shape is compiled.
    """
    import jax.numpy as jnp
    import equinox as eqx

    model, X, _Y, pb = jax_model(geometry, n_points, model_name)
    # prepare_dataset holds back a validation split, so fewer than n_points
    # are available; evaluate whole chunks only and report what was evaluated.
    avail = len(X)
    chunk = min(chunk, avail)
    n_chunks = max(1, avail // chunk)
    xs = [X[i * chunk:(i + 1) * chunk] for i in range(n_chunks)]
    pbs = [pb[i * chunk:(i + 1) * chunk] for i in range(n_chunks)]

    @eqx.filter_jit
    def detg(m, x, p):
        return jnp.real(jnp.linalg.det(m(x, pb=p)))

    def call():
        out = [detg(model, x, p) for x, p in zip(xs, pbs)]
        return out[-1]

    timing = time_calls(call, repeats, block=lambda r: r.block_until_ready())
    n_eval = n_chunks * chunk
    rate = n_eval / timing['median_s']
    return {
        'timing': timing,
        'derived': {'points_per_s': rate,
                    'us_per_point': 1e6 * timing['median_s'] / n_eval,
                    'n_points_evaluated': n_eval,
                    'chunk': chunk,
                    'compile_s': timing['first_call_s']},
        'info': {'geometry': geometry, 'model': model_name,
                 'ncoords': int(model.ncoords),
                 'n_projective': int(model.nProjective),
                 'points_available': int(avail), 'n_chunks': n_chunks},
        'headline': '%.3g points/s' % rate,
    }


# ---------------------------------------------------------------------------
# cases: toric metric model (currently blocked upstream)
# ---------------------------------------------------------------------------

def train_jax_toric(h11, batch_size, n_points, repeats,
                    model_name='PhiFSModelToric'):
    """One optimizer step of the JAX *toric* metric model.

    Known blocked on upstream be772af: ``ToricModel._generate_helpers`` is
    defined in ``cymetric/jax/models/models.py`` but never called by the JAX
    backend (``torch`` and ``tensorflow`` both call theirs), so
    ``patch_degrees``, ``transition_coefficients`` and ``transition_degrees``
    are never assigned and the model cannot be constructed.  The case is kept
    so that it starts producing numbers as soon as that is fixed.
    """
    import jax.numpy as jnp
    import optax
    import equinox as eqx
    from cymetric.jax.models import models as jax_models
    from cymetric.jax.models.helper import prepare_basis, _make_train_step

    pg, blob = toric_pointgen(h11)
    BASIS = prepare_basis(toric_basis(pg, blob))
    cls = getattr(jax_models, model_name)
    n_out = 1 if model_name.startswith('Phi') else int(blob['toric_data']['dim_cy']) ** 2
    nn = make_mlp(2 * pg.ncoords, n_out)
    try:
        model = cls(nn, BASIS, toric_data=blob['toric_data'])
    except Exception as exc:
        raise KnownBlocked(
            'JAX toric model %s could not be constructed at h11=%d: %s: %s'
            % (model_name, h11, type(exc).__name__, exc)) from exc

    points = pg.generate_points(max(batch_size, n_points))
    X = jnp.array(np.concatenate([points.real, points.imag], axis=-1),
                  dtype=jnp.float32)
    Y = jnp.ones((len(X), 2), dtype=jnp.float32)
    pb = model.pullbacks(X)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(model.model, eqx.is_array))
    step = _make_train_step(optimizer)
    state = {'model': model, 'opt': opt_state}

    def call():
        m, o, loss, _ = step(state['model'], state['opt'], X[:batch_size],
                             Y[:batch_size], None, pb[:batch_size])
        state['model'], state['opt'] = m, o
        return loss

    timing = time_calls(call, repeats, block=lambda r: r.block_until_ready())
    return {
        'timing': timing,
        'derived': {'ms_per_step_median': 1e3 * timing['median_s'],
                    'compile_s': timing['first_call_s']},
        'info': {'h11': h11, 'ncoords': int(pg.ncoords),
                 'batch_size': batch_size},
        'headline': '%.2f ms/step' % (1e3 * timing['median_s']),
    }


def metric_torch_toric(h11, batch_size, n_points, repeats, device='auto'):
    """Forward evaluation of the *torch* toric metric model, for reference.

    The torch backend does call ``_generate_helpers``, so it is the only
    backend on this commit in which a toric metric can be evaluated at all.
    It is the cross-backend reference the JAX toric cases will be compared
    against once they unblock.  Two further upstream defects show up here and
    are recorded rather than worked around:

    * at h11 > 1 the construction fails for torch too, because the base class
      builds ``fixed_patches`` through the toric override while
      ``nTransitions`` still holds the CICY value;
    * at h11 = 1 the forward pass works but ``compute_transition_loss`` (and
      hence ``compute_loss``, and hence training) raises a shape error, so
      only the forward pass can be timed.  The attempt is made once and its
      error text is stored in ``info['compute_loss_error']``.
    """
    import torch
    from cymetric.torch.models import models as torch_models
    from cymetric.torch.models.helper import prepare_basis

    wanted = ['cuda', 'cpu'] if (device == 'auto' and torch.cuda.is_available()) \
        else ([device] if device != 'auto' else ['cpu'])
    pg, blob = toric_pointgen(h11)
    BASIS = prepare_basis(toric_basis(pg, blob))
    cls = getattr(torch_models, 'PhiFSModelToric')
    points = pg.generate_points(max(batch_size, n_points))
    points = np.concatenate([points.real, points.imag], axis=-1)

    device_errors = {}
    model = xb = None
    for dev in wanted:
        net = torch.nn.Sequential(
            torch.nn.Linear(2 * pg.ncoords, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 64), torch.nn.GELU(),
            torch.nn.Linear(64, 1, bias=False)).to(dev)
        try:
            cand = cls(net, prepare_basis(toric_basis(pg, blob)),
                       toric_data=blob['toric_data'], device=dev)
            x = torch.tensor(points, dtype=torch.float32,
                             device=dev)[:batch_size]
            cand(x)          # the toric model keeps some buffers on the host
            model, xb, device = cand, x, dev
            break
        except Exception as exc:
            device_errors[dev] = '%s: %s' % (type(exc).__name__, exc)
    if model is None:
        raise KnownBlocked(
            'torch toric model unusable at h11=%d on %s: %s'
            % (h11, '/'.join(wanted), '; '.join(
                '%s -> %s' % (k, v) for k, v in device_errors.items())))

    loss_error = None
    try:
        model.compute_loss(xb, torch.ones((len(xb), 2), dtype=torch.float32,
                                          device=device))
    except Exception as exc:
        loss_error = '%s: %s' % (type(exc).__name__, exc)

    # no torch.no_grad(): PhiFSModel's forward differentiates the network
    # twice internally, so it needs grad mode even for pure evaluation.
    def call():
        return torch.real(torch.linalg.det(model(xb)))

    def block(r):
        if device == 'cuda':
            torch.cuda.synchronize()
        return float(r[0])

    timing = time_calls(call, repeats, block=block)
    rate = batch_size / timing['median_s']
    return {
        'timing': timing,
        'derived': {'ms_per_call_median': 1e3 * timing['median_s'],
                    'points_per_s': rate},
        'info': {'h11': h11, 'ncoords': int(pg.ncoords), 'device': device,
                 'batch_size': batch_size, 'backend': 'torch',
                 'compute_loss_error': loss_error,
                 'device_errors': device_errors,
                 'what_is_timed': 'det g on one batch, forward only'},
        'headline': '%.2f ms/batch det g (torch, %s)'
                    % (1e3 * timing['median_s'], device),
    }


# ---------------------------------------------------------------------------
# quick mode
# ---------------------------------------------------------------------------

def shrink_params(params):
    """Smaller problem sizes for --quick (smoke runs, not baselines)."""
    out = dict(params)
    for key, factor, floor in (('n_points', 10, 200), ('batch_size', 1, 1)):
        if key in out and isinstance(out[key], int):
            out[key] = max(floor, out[key] // factor)
    if 'batch_size' in out and 'n_points' in out:
        out['batch_size'] = min(out['batch_size'], out['n_points'])
    return out


# ---------------------------------------------------------------------------
# the case list
# ---------------------------------------------------------------------------

def _case(name, group, fn, what, params, repeats):
    return {'name': name, 'group': group, 'fn': fn, 'what': what,
            'params': params, 'repeats': repeats}


CASES = [
    # ---- point generation -------------------------------------------------
    _case('pointgen.quintic.points', 'pointgen', 'pointgen_cicy',
          'CICY line-intersection sampler, points only (Fermat quintic, h11=1)',
          dict(geometry='quintic', n_points=20000, nproc=4, method='points'), 5),
    _case('pointgen.quintic.weights', 'pointgen', 'pointgen_cicy',
          'CICY sampler incl. weights and Omega, i.e. prepare_dataset cost',
          dict(geometry='quintic', n_points=20000, nproc=4, method='weights'), 5),
    _case('pointgen.quintic_dense.points', 'pointgen', 'pointgen_cicy',
          'same P^4, dense quintic (126 monomials): the monomial-count axis',
          dict(geometry='quintic_dense', n_points=20000, nproc=4,
               method='points'), 3),
    _case('pointgen.quintic_dense.weights', 'pointgen', 'pointgen_cicy',
          'dense quintic, weights and Omega',
          dict(geometry='quintic_dense', n_points=20000, nproc=4,
               method='weights'), 3),
    _case('pointgen.bicubic.points', 'pointgen', 'pointgen_cicy',
          'CICY sampler, points only, P^2xP^2 (h11=2)',
          dict(geometry='bicubic', n_points=20000, nproc=4, method='points'), 3),
    _case('pointgen.bicubic.weights', 'pointgen', 'pointgen_cicy',
          'CICY sampler incl. weights and Omega, P^2xP^2 (h11=2)',
          dict(geometry='bicubic', n_points=20000, nproc=4, method='weights'), 3),
    _case('pointgen.tetraquadric.points', 'pointgen', 'pointgen_cicy',
          'CICY sampler, points only, (P^1)^4 (h11=4)',
          dict(geometry='tetraquadric', n_points=20000, nproc=4,
               method='points'), 3),
    _case('pointgen.tetraquadric.weights', 'pointgen', 'pointgen_cicy',
          'CICY sampler incl. weights and Omega, (P^1)^4 (h11=4)',
          dict(geometry='tetraquadric', n_points=20000, nproc=4,
               method='weights'), 3),
    _case('pointgen.toric.h11_1', 'pointgen', 'pointgen_toric',
          'toric Metropolis sampler on a KS hypersurface, h11=1',
          dict(h11=1, n_points=2000), 3),
    _case('pointgen.toric.h11_5', 'pointgen', 'pointgen_toric',
          'toric Metropolis sampler on a KS hypersurface, h11=5',
          dict(h11=5, n_points=2000), 3),
    _case('pointgen.toric.h11_5.n500', 'pointgen', 'pointgen_toric',
          'same geometry at n_p=500: exposes the per-call auto-tuning cost',
          dict(h11=5, n_points=500), 3),
    _case('pointgen.toric.h11_10', 'pointgen', 'pointgen_toric',
          'toric Metropolis sampler on a KS hypersurface, h11=10',
          dict(h11=10, n_points=2000), 3),

    # ---- JAX metric training ---------------------------------------------
    _case('train_jax.quintic.bs64', 'train_jax', 'train_jax',
          'PhiFSModel training step, quintic, small batch (phase-1 default)',
          dict(geometry='quintic', batch_size=64, n_points=20000), 30),
    _case('train_jax.quintic.bs1024', 'train_jax', 'train_jax',
          'PhiFSModel training step, quintic, medium batch',
          dict(geometry='quintic', batch_size=1024, n_points=20000), 20),
    _case('train_jax.quintic.bs10000', 'train_jax', 'train_jax',
          'PhiFSModel training step, quintic, phase-2 batch size',
          dict(geometry='quintic', batch_size=10000, n_points=20000), 10),
    _case('train_jax.bicubic.bs1024', 'train_jax', 'train_jax',
          'PhiFSModel training step, bicubic in P^2xP^2 (h11=2)',
          dict(geometry='bicubic', batch_size=1024, n_points=20000), 20),
    _case('train_jax.tetraquadric.bs1024', 'train_jax', 'train_jax',
          'PhiFSModel training step, tetraquadric (h11=4): the h11 axis',
          dict(geometry='tetraquadric', batch_size=1024, n_points=20000), 20),

    _case('train_model_jax.quintic', 'train_jax', 'train_model_jax',
          'end-to-end helper.train_model (both phases, incl. compile), quintic',
          dict(geometry='quintic', n_points=20000, epochs=3,
               batch_sizes=(64, 10000)), 2),
    _case('train_model_jax.tetraquadric', 'train_jax', 'train_model_jax',
          'end-to-end helper.train_model, tetraquadric (h11=4)',
          dict(geometry='tetraquadric', n_points=20000, epochs=3,
               batch_sizes=(64, 10000)), 2),

    # ---- det g ------------------------------------------------------------
    _case('detg_jax.quintic.n20000', 'detg_jax', 'detg_jax',
          'det g over a point set, quintic, chunks of 5000',
          dict(geometry='quintic', n_points=20000, chunk=5000), 7),
    _case('detg_jax.tetraquadric.n20000', 'detg_jax', 'detg_jax',
          'det g over a point set, tetraquadric (h11=4)',
          dict(geometry='tetraquadric', n_points=20000, chunk=5000), 7),

    # ---- toric metric models (blocked upstream) --------------------------
    _case('train_jax.toric.h11_1', 'train_jax_toric', 'train_jax_toric',
          'JAX toric training step, h11=1 (known blocked: see case docstring)',
          dict(h11=1, batch_size=512, n_points=512), 10),
    _case('train_jax.toric.h11_5', 'train_jax_toric', 'train_jax_toric',
          'JAX toric training step, h11=5 (known blocked)',
          dict(h11=5, batch_size=512, n_points=512), 10),
    _case('train_jax.toric.h11_10', 'train_jax_toric', 'train_jax_toric',
          'JAX toric training step, h11=10 (known blocked)',
          dict(h11=10, batch_size=512, n_points=512), 10),
    _case('metric_torch.toric.h11_1', 'metric_torch_toric',
          'metric_torch_toric',
          'torch toric det g, h11=1: the only runnable toric metric reference',
          dict(h11=1, batch_size=512, n_points=512), 10),
    _case('metric_torch.toric.h11_5', 'metric_torch_toric',
          'metric_torch_toric',
          'torch toric det g, h11=5 (blocked: second defect, see docstring)',
          dict(h11=5, batch_size=512, n_points=512), 10),
]
