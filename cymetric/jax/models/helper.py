"""
Helper functions for JAX Calabi-Yau metric training.

Faithful translation of tensorflow/models/helper.py.
Uses optax for optimization (analogous to tf.keras.optimizers).
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from cymetric.jax.models.dtypes import complex_dtype, real_dtype

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def prepare_basis(basis, dtype=None):
    r"""Casts each numpy array in BASIS to a JAX array of the given dtype.

    Equivalent to tensorflow/models/helper.py::prepare_basis.

    Args:
        basis (dict or str): Dictionary of geometric data or path to a
            pickle file containing such a dictionary.
        dtype (jax dtype, optional): Target dtype. Defaults to complex128
            when ``jax_enable_x64`` is set and complex64 otherwise.  Resolved
            at call time, not import time, so enabling x64 after importing
            this module still takes effect.

    Returns:
        dict: same keys, but numpy arrays replaced with JAX arrays.
    """
    if dtype is None:
        dtype = complex_dtype()
    if isinstance(basis, str):
        import pickle
        with open(basis, 'rb') as f:
            basis = pickle.load(f)

    new_basis = {}
    for key in basis:
        if isinstance(basis[key], np.ndarray):
            new_basis[key] = jnp.array(basis[key], dtype=dtype)
        else:
            new_basis[key] = basis[key]
    return new_basis


# ---------------------------------------------------------------------------
# Per-step loss and gradient computation
# ---------------------------------------------------------------------------

def _train_step(model, opt_state, x, y, sample_weight, pb, optimizer):
    r"""One optimizer step, as a plain (traceable) function.

    The body of the training step, with no jit of its own, so that the same
    arithmetic can serve both as a standalone jitted call (see
    :func:`_make_train_step`) and as the body of a ``lax.scan`` over a whole
    phase (see :func:`_make_phase_runner`).

    Differentiates ONLY through the NN backbone (``model.model``), matching
    TF's ``trainable_variables = model.model.trainable_variables``.

    Args:
        model: a JAX FreeModel (or subclass) instance.
        opt_state: the optax state for ``eqx.filter(model.model, eqx.is_array)``.
        x, y (jnp.ndarray): one batch of points and labels.
        sample_weight (jnp.ndarray or None): optional per-point weights.
        pb (jnp.ndarray or None): precomputed pullbacks for ``x``.
        optimizer (optax.GradientTransformation): the optimizer.

    Returns:
        tuple: ``(new_model, new_opt_state, mean_loss, loss_dict)``.
    """
    def loss_fn(nn_model):
        # Reconstruct the full model with the candidate NN
        full_model = eqx.tree_at(lambda m: m.model, model, nn_model)
        total_loss, loss_dict = full_model.compute_loss(x, y, sample_weight, pb=pb)
        # Sum (not mean) over the batch for the gradient target: TF's
        # tape.gradient(total_loss, ...) on a non-scalar [bSize] target
        # implicitly uses output_gradients=ones_like(total_loss), i.e. it
        # differentiates sum(total_loss). Matching that here keeps the
        # raw gradient magnitude (and hence gclipping's effect) the same
        # scale as TF; using mean() shrinks gradients by batch_size and
        # makes gclipping's global-norm clip rarely engage.
        return jnp.sum(total_loss), (jnp.mean(total_loss), loss_dict)

    (_, (mean_loss, loss_dict)), grads = eqx.filter_value_and_grad(
        loss_fn, has_aux=True)(model.model)

    # Replace NaNs with small gradient (matches TF: tf.where(is_nan, 1e-8, g))
    grads = jax.tree_util.tree_map(
        lambda g: jnp.where(jnp.isnan(g), jnp.full_like(g, 1e-8), g),
        grads)

    # Global norm clipping  (matches TF: tf.clip_by_global_norm)
    leaves, treedef = jax.tree_util.tree_flatten(grads)
    global_norm = jnp.sqrt(
        sum(jnp.sum(g ** 2) for g in leaves) + 1e-12)
    clip_factor = jnp.minimum(1.0, model.gclipping / global_norm)
    grads = jax.tree_util.tree_unflatten(
        treedef, [g * clip_factor for g in leaves])

    updates, new_opt_state = optimizer.update(grads, opt_state,
                                              eqx.filter(model.model, eqx.is_array))
    new_nn = eqx.apply_updates(model.model, updates)
    new_model = eqx.tree_at(lambda m: m.model, model, new_nn)
    return new_model, new_opt_state, mean_loss, loss_dict


def _make_train_step(optimizer):
    r"""Returns a JIT-compiled single training-step function.

    Args:
        optimizer (optax.GradientTransformation): An optax optimizer.

    Returns:
        callable: ``train_step(model, opt_state, x, y, sw, pb)``
    """

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, sample_weight=None, pb=None):
        return _train_step(model, opt_state, x, y, sample_weight, pb, optimizer)

    return train_step


# ---------------------------------------------------------------------------
# One whole phase of an epoch, on the device
# ---------------------------------------------------------------------------

def _phase_batching(n_train, batch_size):
    r"""Splits a phase into equal-shaped batches plus a ragged last one.

    The host loop took ``range(0, n_train, batch_size)`` and clipped the last
    slice, so a phase is ``n_train // batch_size`` batches of exactly
    ``batch_size`` points and, when ``n_train`` is not a multiple of it, one
    shorter batch.  The short batch cannot join the scan -- its shape differs,
    and padding it would change the summed gradient it contributes and the mean
    loss it reports -- so it runs as one extra step inside the same program.

    Args:
        n_train (int): number of training points.
        batch_size (int): the phase's batch size.

    Returns:
        tuple: ``(n_full, tail)``, with ``n_full * batch_size + tail ==
        n_train`` and ``0 <= tail < batch_size``.
    """
    n_full = n_train // batch_size
    return n_full, n_train - n_full * batch_size


def _make_phase_runner(optimizer, batch_size, collect_batches=False):
    r"""Returns a jitted function that runs one whole phase of an epoch.

    The phase is a ``lax.scan`` over the batch axis, carrying the NN
    parameters, the optimizer state and the running loss sum, so the Python
    interpreter sees one dispatch per phase instead of one per batch.  Only the
    network parameters are carried: the rest of the model (BASIS, the patch
    tables, kappa) is a free variable of the enclosing jitted function and so a
    loop-invariant operand of the scan rather than something copied on every
    iteration.

    The returned function is built once per phase, outside the epoch loop, and
    called once per epoch; the ``learn_*`` flags are static on an
    ``eqx.Module``, so the two phases are two programs and each compiles once
    for a whole run.

    Args:
        optimizer (optax.GradientTransformation): An optax optimizer.
        batch_size (int): the phase's batch size (static: it is part of the
            compiled shapes).
        collect_batches (bool): If True, also stack the per-batch loss
            components, mean losses and sample weights, which is what custom
            metrics need.  They cost device memory proportional to the number
            of steps, so they are only collected when there are metrics to
            feed.  Defaults to False.

    Returns:
        callable: ``run_phase(model, opt_state, sx, sy, spb, ssw)`` returning
        ``(model, opt_state, loss_sum, last_component_means, scanned, tailed)``
        on already-shuffled arrays.  ``scanned`` and ``tailed`` are None unless
        ``collect_batches``; ``loss_sum`` is the sum over batches of the
        per-batch mean loss, accumulated in the scan carry so that the
        summation order is the host loop's.
    """

    @eqx.filter_jit
    def run_phase(model, opt_state, sx, sy, spb, ssw):
        n_train = sx.shape[0]
        n_full, tail = _phase_batching(n_train, batch_size)

        nn_params, nn_static = eqx.partition(model.model, eqx.is_array)

        def rebuild(params):
            return eqx.tree_at(lambda m: m.model, model,
                               eqx.combine(params, nn_static))

        def one_step(carry, batch):
            params, state, loss_sum = carry
            bx, by, bpb, bsw = batch
            new_model, new_state, loss, parts = _train_step(
                rebuild(params), state, bx, by, bsw, bpb, optimizer)
            new_params, _ = eqx.partition(new_model.model, eqx.is_array)
            out = {'means': {k: jnp.mean(v) for k, v in parts.items()}}
            if collect_batches:
                out['loss'] = loss
                out['parts'] = parts
                if bsw is not None:
                    out['sample_weight'] = bsw
            return (new_params, new_state, loss_sum + loss), out

        carry = (nn_params, opt_state, jnp.zeros((), dtype=real_dtype()))

        scanned = None
        if n_full > 0:
            def chop(a):
                return a[:n_full * batch_size].reshape(
                    (n_full, batch_size) + a.shape[1:])
            carry, scanned = jax.lax.scan(
                one_step, carry,
                (chop(sx), chop(sy), chop(spb),
                 None if ssw is None else chop(ssw)))

        tailed = None
        if tail > 0:
            start = n_full * batch_size
            carry, tailed = one_step(
                carry, (sx[start:], sy[start:], spb[start:],
                        None if ssw is None else ssw[start:]))

        params, new_opt_state, loss_sum = carry
        # The history records the component losses of the LAST batch of the
        # phase, which is the ragged one when there is one.
        if tail > 0:
            last = tailed['means']
        else:
            last = jax.tree_util.tree_map(lambda a: a[-1], scanned['means'])
        return rebuild(params), new_opt_state, loss_sum, last, scanned, tailed

    return run_phase


def _replay_metrics(custom_metrics, scanned, tailed, n_full):
    r"""Feeds the per-batch losses of a finished phase to the custom metrics.

    ``metrics._RunningMean.update_state`` is host-side Python with a
    ``float()`` in it, so it cannot run inside the scan.  The values it needs
    are per-batch and known once the phase has run, so the phase stacks them
    and they are replayed here in batch order: the running means come out
    bit-identical to the host loop's, and no device dispatch happens per step.

    Args:
        custom_metrics (list): metric objects (see metrics.py). Empty is a
            no-op, and then the phase does not collect anything either.
        scanned (dict or None): the stacked outputs of the scanned batches.
        tailed (dict or None): the outputs of the ragged last batch.
        n_full (int): number of scanned batches.
    """
    if not custom_metrics:
        return

    def host(tree):
        return jax.tree_util.tree_map(np.asarray, tree)

    steps = []
    if scanned is not None:
        s = host(scanned)
        for i in range(n_full):
            steps.append(({k: v[i] for k, v in s['parts'].items()},
                          s['loss'][i],
                          s['sample_weight'][i] if 'sample_weight' in s
                          else None))
    if tailed is not None:
        t = host(tailed)
        steps.append((dict(t['parts']), t['loss'], t.get('sample_weight')))

    for parts, loss_val, bsw in steps:
        loss_dict = dict(parts)
        loss_dict['loss'] = jnp.array([loss_val])
        for m in custom_metrics:
            m.update_state(loss_dict, bsw)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _shuffled(X, y, pullbacks, sample_weights, n_train):
    r"""One device-side shuffle of a phase's training data.

    Draws the permutation with ``np.random.permutation`` exactly as the
    per-batch gather did, applies it once, and lets the batch loop take
    contiguous slices.

    Args:
        X, y, pullbacks (jnp.ndarray): Training arrays, first axis the points.
        sample_weights (jnp.ndarray or None): Optional per-point weights.
        n_train (int): Number of points.

    Returns:
        tuple: the four arrays in the shuffled order (the last is None if
        ``sample_weights`` is None).
    """
    perm = jnp.asarray(np.random.permutation(n_train))
    return (X[perm], y[perm], pullbacks[perm],
            None if sample_weights is None else sample_weights[perm])


def _with_flags(model, kaehler, transition, ricci, ricci_val, volk):
    r"""Returns MODEL with the five ``learn_*`` flags set to these values.

    They are static fields, so this is what selects which loss terms the
    phase's program contains.

    Args:
        model: a JAX FreeModel (or subclass) instance.
        kaehler, transition, ricci, ricci_val, volk (bool): the flags.

    Returns:
        the model with the flags replaced.
    """
    model = eqx.tree_at(lambda m: m.learn_kaehler, model, kaehler)
    model = eqx.tree_at(lambda m: m.learn_transition, model, transition)
    model = eqx.tree_at(lambda m: m.learn_ricci, model, ricci)
    model = eqx.tree_at(lambda m: m.learn_ricci_val, model, ricci_val)
    model = eqx.tree_at(lambda m: m.learn_volk, model, volk)
    return model


def train_model(fsmodel, data, optimizer=None, epochs=50,
                batch_sizes=(64, 10000), verbose=1,
                custom_metrics=None, callbacks=None, sw=False,
                pullback_chunk=4096):
    r"""Two-phase training loop for fixing the Kähler class.

    Equivalent to tensorflow/models/helper.py::train_model.

    Phase 1: small batch, volk loss disabled.
    Phase 2: large batch, only MA + volk loss.

    Each phase of each epoch is one jitted program that scans over the phase's
    batches (:func:`_make_phase_runner`), so the Python interpreter sees a
    handful of dispatches per epoch rather than one per batch.  Callbacks and
    the custom metrics are unchanged: callbacks already ran at epoch
    granularity, and the metrics' per-batch running means are replayed from the
    phase's stacked outputs (:func:`_replay_metrics`).

    Args:
        fsmodel: Any JAX FreeModel (or subclass) instance.
        data (dict): {'X_train': ndarray, 'y_train': ndarray}.
        optimizer (optax.GradientTransformation, optional): Defaults to Adam.
        epochs (int): Number of training epochs. Defaults to 50.
        batch_sizes (list of int): [small, large]. Defaults to [64, 10000].
        verbose (int): If > 0, prints epoch info. Defaults to 1.
        custom_metrics (list, optional): Metric objects (see metrics.py).
        callbacks (list, optional): Callback objects (see callbacks.py).
        sw (bool): If True, use integration weights as sample weights.
        pullback_chunk (int): Points per program when the training set's
            pullbacks are precomputed. Defaults to 4096.

    Returns:
        (model, training_history): updated model and loss history dict.
    """
    if callbacks is None:
        callbacks = []
    if custom_metrics is None:
        custom_metrics = []

    X_train = jnp.array(data['X_train'], dtype=real_dtype())
    y_train = jnp.array(data['y_train'], dtype=real_dtype())

    # The pullbacks depend only on the points, not on the network weights, so
    # compute them once here instead of in every forward pass.  In chunks:
    # one program over the whole training set holds a
    # (n_p, ncoords, n_terms, ncoords) intermediate of dQ/dz, which at a few
    # hundred monomials and 100k points is tens of GB.
    train_pullbacks = jnp.concatenate(
        [fsmodel.pullbacks(X_train[i:i + pullback_chunk])
         for i in range(0, len(X_train), pullback_chunk)])

    sample_weights = y_train[:, -2] if sw else None

    if optimizer is None:
        optimizer = optax.adam(1e-3)

    # Initialise optimizer state on the NN parameters only
    opt_state = optimizer.init(eqx.filter(fsmodel.model, eqx.is_array))

    training_history = {}

    # Notify callbacks
    for cb in callbacks:
        if hasattr(cb, 'on_train_begin'):
            cb.on_train_begin(logs=training_history, model=fsmodel)

    # Snapshot learning flags
    learn_kaehler = fsmodel.learn_kaehler
    learn_transition = fsmodel.learn_transition
    learn_ricci = fsmodel.learn_ricci
    learn_ricci_val = fsmodel.learn_ricci_val

    training_history['General loss phase'] = []
    training_history['Volume loss phase'] = []
    training_history['epochs'] = list(range(epochs))

    hist1, hist2 = {}, {}
    n_train = len(X_train)

    # Build both phases' programs once, outside the epoch loop.  Rebuilding
    # them per epoch made XLA recompile and retain an executable per epoch,
    # which leaked memory on long runs.
    batch_size1 = batch_sizes[0]
    batch_size2 = min(batch_sizes[1], n_train)
    n_full1, tail1 = _phase_batching(n_train, batch_size1)
    n_full2, tail2 = _phase_batching(n_train, batch_size2)
    num_batches1 = n_full1 + (1 if tail1 else 0)
    num_batches2 = n_full2 + (1 if tail2 else 0)
    want_batches = bool(custom_metrics)
    run_phase1 = _make_phase_runner(optimizer, batch_size1, want_batches)
    run_phase2 = _make_phase_runner(optimizer, batch_size2, want_batches)

    epoch_it = range(epochs)
    if HAS_TQDM and verbose > 0:
        epoch_it = tqdm.tqdm(epoch_it)

    for epoch in epoch_it:
        if verbose > 0:
            print('\nEpoch {:2d}/{:d}'.format(epoch + 1, epochs))

        # ---- Phase 1: small batch, volk disabled ----
        fsmodel = _with_flags(fsmodel, learn_kaehler, learn_transition,
                              learn_ricci, learn_ricci_val, False)

        # Shuffle once and then take contiguous slices.  A numpy fancy-index
        # gather per batch (the former X_train[indices[start:end]]) copied the
        # indices to the device and launched a gather for each of X, y and the
        # pullbacks: 1.6 ms on top of a 0.9 ms step at batch 64 on an RTX
        # 5090, i.e. most of an epoch.  The batches themselves are unchanged
        # -- same permutation, same grouping -- so the trajectory is the same.
        sx, sy, spb, ssw = _shuffled(
            X_train, y_train, train_pullbacks, sample_weights, n_train)
        fsmodel, opt_state, loss_sum1, last1, scanned, tailed = run_phase1(
            fsmodel, opt_state, sx, sy, spb, ssw)
        _replay_metrics(custom_metrics, scanned, tailed, n_full1)

        avg_loss1 = loss_sum1 / max(num_batches1, 1)
        training_history['General loss phase'].append(float(avg_loss1))
        for k, v in last1.items():
            if k not in hist1:
                hist1[k] = []
            hist1[k].append(float(v))

        # ---- Phase 2: large batch, only MA + volk ----
        fsmodel = _with_flags(fsmodel, False, False, False, False, True)

        sx, sy, spb, ssw = _shuffled(
            X_train, y_train, train_pullbacks, sample_weights, n_train)
        fsmodel, opt_state, loss_sum2, last2, scanned, tailed = run_phase2(
            fsmodel, opt_state, sx, sy, spb, ssw)
        _replay_metrics(custom_metrics, scanned, tailed, n_full2)

        avg_loss2 = loss_sum2 / max(num_batches2, 1)
        training_history['Volume loss phase'].append(float(avg_loss2))
        for k, v in last2.items():
            if k not in hist2:
                hist2[k] = []
            hist2[k].append(float(v))

        if verbose > 0:
            print(' - General loss phase: {:.6f}'.format(
                training_history['General loss phase'][-1]))
            print(' - Volume loss phase:  {:.6f}'.format(
                training_history['Volume loss phase'][-1]))

        # Run epoch-end callbacks.
        # Use a per-epoch dict so callback scalars are accumulated into lists
        # in training_history rather than overwriting each epoch.
        epoch_logs = {}
        for cb in callbacks:
            if hasattr(cb, 'on_epoch_end'):
                cb.on_epoch_end(epoch, logs=epoch_logs, model=fsmodel)
        for k, v in epoch_logs.items():
            if k not in training_history:
                training_history[k] = []
            training_history[k].append(v)

    # Restore flags
    fsmodel = _with_flags(fsmodel, learn_kaehler, learn_transition,
                          learn_ricci, learn_ricci_val, False)

    # Merge histories (matches TF helper logic)
    for k in set(list(hist1.keys()) + list(hist2.keys())):
        if k in hist2 and (k not in hist1 or max(hist2[k], default=0) != 0):
            training_history[k] = hist2[k]
        else:
            training_history[k] = hist1.get(k, [])

    # Run train-end callbacks
    for cb in callbacks:
        if hasattr(cb, 'on_train_end'):
            cb.on_train_end(logs=training_history, model=fsmodel)

    return fsmodel, training_history
