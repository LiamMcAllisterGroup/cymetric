"""
Various error measures for JAX neural networks representing (Ricci-flat) Kähler metrics.

Faithful translation of tensorflow/models/measures.py.
Uses jax.grad / jax.vmap / jax.jacobian instead of tf.GradientTape.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cymetric.jax.models.dtypes import complex_dtype, real_dtype


@eqx.filter_jit
def _det_g_one_chunk(model, points):
    r"""``Re det g`` of one chunk of points, in a single jitted program."""
    return jnp.real(jnp.linalg.det(model(points)))


def _n_features(model):
    r"""Width of the first layer's output of the NN backbone, or None.

    The input map of a toric phi net is what makes the metric expensive: the
    nets of the McAllister scan take all pairwise section products, 87k real
    numbers per point at h11 = 16.  Knowing that width lets
    :func:`det_g_chunked` pick a batch that fits.
    """
    from cymetric.jax.models.models import input_map_width
    return input_map_width(getattr(model, 'model', None), 2 * model.ncoords)


def det_g_chunked(model, points, chunk=None, budget=3e8):
    r"""``Re det g`` on ``points``, one jitted chunk at a time.

    ``model(points)`` holds, per point, the intermediates of ``2*nfold``
    Hessian-vector products through the network's input map.  Asking for a
    whole validation set at once is what put the toric models of the
    McAllister scan at 36-61 GB (and an OOM at h11 = 16, 17); chunking makes
    the peak independent of the number of points.

    Args:
        model: Any FSModel subclass (JAX/equinox version).
        points (jnp.ndarray, [n_p, 2*ncoord], float): NN input.
        chunk (int, optional): Points per program.  None sizes it from the
            width of the network's input map, between 128 and 4096.
        budget (float): Target number of real intermediates per chunk used by
            that heuristic.  Defaults to 3e8 (~1 GB in float32).

    Returns:
        jnp.ndarray, [n_p], real: det g at every point.
    """
    points = jnp.asarray(points)
    if chunk is None:
        n_feat = _n_features(model) or 2 * model.ncoords
        chunk = int(max(128, min(4096,
                                 budget // (2 * model.nfold * 4 * max(n_feat, 1)))))
    return jnp.concatenate([_det_g_one_chunk(model, points[i:i + chunk])
                            for i in range(0, points.shape[0], chunk)])


def sigma_measure(model, points, y_true, chunk=None):
    r"""Monge-Ampere (sigma) measure.

    .. math::

        \sigma = \frac{1}{\mathrm{Vol_{CY}}\,n_p}
            \sum_i \left|1 -
            \frac{\det(g)\,\mathrm{Vol_{CY}}}
                 {|\Omega|^2\,\mathrm{Vol_K}}\right|

    Equivalent to tensorflow/models/measures.py::sigma_measure.

    Args:
        model: Any FSModel subclass (JAX/equinox version).
        points (jnp.ndarray, [n_p, 2*ncoord], float32): NN input.
        y_true (jnp.ndarray, [n_p, 2], float32): (weights, |Omega|^2).
        chunk (int, optional): Points per jitted ``det g`` program; see
            :func:`det_g_chunked`.  None lets that function choose.

    Returns:
        float: sigma measure.
    """
    weights = y_true[:, -2]
    omega = y_true[:, -1]
    det = det_g_chunked(model, points, chunk=chunk)
    det_over_omega = det / omega
    volume_cy = jnp.mean(weights)
    vol_k = jnp.mean(det_over_omega * weights)
    ratio = volume_cy / vol_k
    sigma_integrand = jnp.abs(jnp.ones_like(det_over_omega)
                              - det_over_omega * ratio) * weights
    sigma = jnp.mean(sigma_integrand) / volume_cy
    return sigma


def ricci_measure(model, points, y_true, pullbacks=None, verbose=0):
    r"""Ricci measure for a Kähler metric.

    .. math::

        \|R\| \equiv
            \frac{\mathrm{Vol}_K^{1/\mathrm{nfold}}}{\mathrm{Vol_{CY}}}
            \int_X d\mathrm{Vol}_K\, |R|

    Equivalent to tensorflow/models/measures.py::ricci_measure.
    Uses jax.grad and jax.vmap instead of tf.GradientTape.

    Args:
        model: Any FSModel subclass (JAX/equinox version).
        points (jnp.ndarray, [n_p, 2*ncoord], float32): NN input.
        y_true (jnp.ndarray, [n_p, 2], float32): (weights, |Omega|^2).
        pullbacks (jnp.ndarray, [n_p, nfold, ncoord], complex64, optional):
            Pre-computed pullback tensors.
        verbose (int): If > 0 prints intermediate info.

    Returns:
        float: Ricci measure.
    """
    nfold = float(model.nfold)
    ncoords = model.ncoords
    weights = y_true[:, -2]
    omega = y_true[:, -1]

    if pullbacks is None:
        pullbacks = model.pullbacks(points)

    def log_det_single(x_single):
        pred = model(x_single[None], training=False)[0]
        det = jnp.real(jnp.linalg.det(pred))
        return jnp.log(det)

    def di_dg_single(x_single):
        return jax.grad(log_det_single)(x_single)

    didj_dg = jax.vmap(jax.jacobian(di_dg_single))(points).astype(complex_dtype())

    ricci_ij = (didj_dg[:, :ncoords, :ncoords]
                + 1j * didj_dg[:, :ncoords, ncoords:]
                - 1j * didj_dg[:, ncoords:, :ncoords]
                + didj_dg[:, ncoords:, ncoords:])
    ricci_ij = ricci_ij * 0.25

    prediction = model(points)
    det = jnp.real(jnp.linalg.det(prediction))
    pred_inv = jnp.linalg.inv(prediction)

    ricci_scalar = jnp.einsum(
        'xba,xai,xij,xbj->x',
        pred_inv, pullbacks, ricci_ij, jnp.conj(pullbacks))
    ricci_scalar = jnp.abs(jnp.real(ricci_scalar))

    if verbose > 0:
        print(' - Avg ricci scalar is', float(jnp.mean(ricci_scalar)))
        if verbose > 1:
            print(' - Max ricci scalar is', float(jnp.max(ricci_scalar)))
            print(' - Min ricci scalar is', float(jnp.min(ricci_scalar)))

    det_over_omega = det / omega
    volume_cy = jnp.mean(weights)
    vol_k = jnp.mean(det_over_omega * weights)
    ricci_meas = ((vol_k ** (1. / nfold)) / volume_cy
                  * jnp.mean(det_over_omega * ricci_scalar * weights))
    return ricci_meas


def ricci_scalar_fn(model, points, pullbacks=None, verbose=0, rdet=True):
    r"""Computes the Ricci scalar for a Kähler metric.

    .. math::

        R = g^{ij} \partial_i \bar\partial_j \log \det g

    Equivalent to tensorflow/models/measures.py::ricci_scalar_fn.

    Args:
        model: Any FSModel subclass (JAX/equinox version).
        points (jnp.ndarray, [n_p, 2*ncoord], float32): NN input.
        pullbacks (jnp.ndarray, [n_p, nfold, ncoord], complex64, optional):
            Pre-computed pullback tensors.
        verbose (int): If > 0 prints intermediate info.
        rdet (bool): If True also returns det. Defaults to True.

    Returns:
        ricci_scalar (jnp.ndarray, [n_p], float32).
        det (jnp.ndarray, [n_p], float32)  — only if rdet=True.
    """
    ncoords = model.ncoords

    if pullbacks is None:
        pullbacks = model.pullbacks(points)

    def log_det_single(x_single):
        pred = model(x_single[None], training=False)[0]
        det = jnp.real(jnp.linalg.det(pred))
        return jnp.log(det)

    def di_dg_single(x_single):
        return jax.grad(log_det_single)(x_single)

    didj_dg = jax.vmap(jax.jacobian(di_dg_single))(points).astype(complex_dtype())

    ricci_ij = (didj_dg[:, :ncoords, :ncoords]
                + 1j * didj_dg[:, :ncoords, ncoords:]
                - 1j * didj_dg[:, ncoords:, :ncoords]
                + didj_dg[:, ncoords:, ncoords:])
    ricci_ij = ricci_ij * 0.25

    prediction = model(points)
    det = jnp.real(jnp.linalg.det(prediction))
    pred_inv = jnp.linalg.inv(prediction)

    ricci_scalar = jnp.einsum(
        'xba,xai,xij,xbj->x',
        pred_inv, pullbacks, ricci_ij, jnp.conj(pullbacks))
    ricci_scalar = jnp.real(ricci_scalar)

    if verbose > 0:
        print(' - Avg ricci scalar is', float(jnp.mean(ricci_scalar)))
        if verbose > 1:
            print(' - Max ricci scalar is', float(jnp.max(ricci_scalar)))
            print(' - Min ricci scalar is', float(jnp.min(ricci_scalar)))

    if rdet:
        return ricci_scalar, det
    else:
        return ricci_scalar


def sigma_measure_loss(model, points, omegas):
    r"""Mean sigma loss over points.

    Equivalent to tensorflow/models/measures.py::sigma_measure_loss.

    Args:
        model: FreeModel (or subclass) with a ``sigma_loss`` attribute.
        points (jnp.ndarray, [n_p, 2*ncoord], float32): NN input.
        omegas (jnp.ndarray, [n_p, x], float32): Array whose last column is
            |Omega|^2. Same convention as TF: passed directly as ``y_true``.

    Returns:
        float: mean sigma loss.
    """
    # Pass omegas directly — sigma_loss reads y_true[:, -1] as omega_squared.
    # This is identical to the TF counterpart which does:
    #   tf.math.reduce_mean(model.sigma_loss(omegas, model(points)))
    return jnp.mean(model.sigma_loss(omegas, model(points)))


def kaehler_measure_loss(model, points):
    r"""Mean Kähler loss measure.

    Equivalent to tensorflow/models/measures.py::kaehler_measure_loss.

    Args:
        model: Any FSModel subclass (JAX/equinox version).
        points (jnp.ndarray, [n_p, 2*ncoord], float32): NN input.

    Returns:
        float: mean Kähler loss.
    """
    return jnp.mean(model.compute_kaehler_loss(points))


def transition_measure_loss(model, points):
    r"""Mean transition loss measure.

    Equivalent to tensorflow/models/measures.py::transition_measure_loss.

    Args:
        model: Any FSModel subclass (JAX/equinox version).
        points (jnp.ndarray, [n_p, 2*ncoord], float32): NN input.

    Returns:
        float: mean transition loss.
    """
    return jnp.mean(model.compute_transition_loss(points.astype(real_dtype())))
