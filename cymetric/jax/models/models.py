"""
JAX / Equinox models for learning Calabi-Yau metrics.

Faithful translation of tensorflow/models/models.py.

Key design:
* Each model is an ``equinox.Module`` (analogous to tf.keras.Model).
* ``@eqx.filter_jit`` replaces ``@tf.function`` for XLA compilation.
* ``jax.grad`` / ``jax.jacobian`` + ``jax.vmap`` replace tf.GradientTape.
* ``optax`` replaces tf.keras optimizers.
* Training flags (learn_kaehler, learn_volk, …) are plain Python bools
  (static under jit; changing them requires a new JIT trace, same as
  changing tf.bool Variables before tracing in TF).
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from functools import partial

from .losses import sigma_integrand
from .fubinistudy import FSModel
from cymetric.pointgen.nphelper import (
    get_all_patch_degrees, compute_all_w_of_x, get_levicivita_tensor,
)
from cymetric.jax.models.dtypes import complex_dtype, real_dtype


# ---------------------------------------------------------------------------
# Helper: Hermitian matrix from flat real vector
# (used in multiple models; matches to_hermitian in TF FreeModel)
# ---------------------------------------------------------------------------

def _to_hermitian(x, nfold):
    r"""Return a Hermitian (nfold×nfold) matrix from a flat real vector.

    Faithful translation of FreeModel.to_hermitian in TF.

    Args:
        x (jnp.ndarray, [bSize, nfold**2], float32): Flat prediction.
        nfold (int): CY dimension.

    Returns:
        jnp.ndarray, [bSize, nfold, nfold], complex64.
    """
    t1 = jnp.reshape(x + 0j, (-1, nfold, nfold)).astype(complex_dtype())
    up = jnp.triu(t1)           # tf.linalg.band_part(t1, 0, -1)
    low = jnp.tril(1j * t1)    # tf.linalg.band_part(1j*t1, -1, 0)
    # diagonal matrix
    diag_vals = jnp.diagonal(t1, axis1=-2, axis2=-1)       # (bSize, nfold)
    diag_mat = jnp.einsum('...i,ij->...ij', diag_vals,
                          jnp.eye(nfold, dtype=complex_dtype()))
    out = up + jnp.swapaxes(up, -2, -1) - diag_mat
    return out + low + jnp.conj(jnp.swapaxes(low, -2, -1))


# ---------------------------------------------------------------------------
# ddbar phi, pulled back: the low-rank route and the full-Hessian reference
# ---------------------------------------------------------------------------

def _real_tangents(J):
    r"""The 2*nfold real directions of one point from its pullback.

    Rows ``[Re J_a, Im J_a]`` and ``[Im J_a, -Re J_a]``, so that
    ``S = T Hess(phi) T^T`` holds the real and imaginary parts of
    ``J (del delbar phi) J^H`` (see :func:`ddbar_phi_pullback`).

    Args:
        J (jnp.ndarray, [nfold, ncoords], complex): Pullback at one point.

    Returns:
        jnp.ndarray, [2*nfold, 2*ncoords], real.
    """
    return jnp.concatenate(
        [jnp.concatenate([jnp.real(J), jnp.imag(J)], axis=1),
         jnp.concatenate([jnp.imag(J), -jnp.real(J)], axis=1)], axis=0)


# Above this many values out of the network's input map, the 2*nfold
# Hessian-vector products of ddbar phi are mapped rather than vmapped: each
# one carries a copy of those intermediates.  Below it the six products are
# small and six kernel launches cost more than the memory saves.
TANGENT_MAP_THRESHOLD = 4096


def input_map_width(nn_model, n_in):
    r"""Number of values the backbone's first layer makes of one point.

    For a plain MLP that is the hidden width; for a net whose first layer is
    an input map (the toric nets of the McAllister scan start with all
    pairwise section products) it is the width of that map, which is what
    decides whether ``ddbar phi`` should map or vmap its tangents and how
    many points fit in a ``det g`` program.

    Args:
        nn_model (eqx.Module): The network backbone.
        n_in (int): Number of real inputs per point (``2 * ncoords``).

    Returns:
        int or None: None when the backbone is not a layer sequence.
    """
    layers = getattr(nn_model, 'layers', None)
    if not layers:
        return None
    try:
        x0 = jnp.zeros(n_in, dtype=real_dtype())
        return int(np.prod(jax.eval_shape(layers[0], x0).shape))
    except Exception:
        return None


def ddbar_phi_pullback(phi_fn, points, pullbacks, sequential=True):
    r"""Pulled back ``del delbar phi`` without ever forming the full Hessian.

    ``(del delbar phi)_pb = J (del delbar phi) J^H`` only ever contracts the
    Hessian of phi with the ``nfold`` rows of the pullback on either side, so
    the ``2*ncoords`` squared second derivatives of
    :func:`ddbar_phi_pullback_full` are ``2*nfold`` directional ones (six for
    a threefold), each a forward-over-reverse Hessian-vector product.  With

    .. math::

        T = \begin{pmatrix} \Re J & \Im J \\ \Im J & -\Re J\end{pmatrix},
        \qquad S = T\,\mathrm{Hess}(\phi)\,T^T

    one has
    ``J (del delbar phi) J^H = [(S_RR + S_II) + i (S_IR - S_RI)] / 4``.
    Exact, not an approximation: it agrees with the full Hessian route to
    float32 rounding (see ``tests/test_jax_metric_routes.py``).

    Args:
        phi_fn (callable): scalar phi of a single point, (2*ncoords,) -> ().
        points (jnp.ndarray, [bSize, 2*ncoords], real): Points.
        pullbacks (jnp.ndarray, [bSize, nfold, ncoords], complex): Pullbacks.
        sequential (bool): If True (default) the ``2*nfold`` Hessian-vector
            products run in a ``lax.map`` rather than a ``vmap``.  Each one
            carries the intermediates of phi's input map, which for toric
            models are all pairwise section products (87k values per point at
            h11 = 16), so mapping keeps one copy instead of ``2*nfold``.

    Returns:
        jnp.ndarray, [bSize, nfold, nfold], complex.
    """
    nfold = pullbacks.shape[1]
    cdtype = complex_dtype()

    def single(x, J):
        T = _real_tangents(J).astype(x.dtype)
        grad_phi = jax.grad(phi_fn)
        hvp = lambda t: jax.jvp(grad_phi, (x,), (t,))[1]
        HT = jax.lax.map(hvp, T) if sequential else jax.vmap(hvp)(T)
        S = T @ HT.T
        dd = ((S[:nfold, :nfold] + S[nfold:, nfold:])
              + 1j * (S[nfold:, :nfold] - S[:nfold, nfold:]))
        return (0.25 * dd).astype(cdtype)

    return jax.vmap(single)(points, pullbacks)


def ddbar_phi_pullback_full(phi_fn, points, pullbacks):
    r"""Pulled back ``del delbar phi`` through the full Hessian of phi.

    The reference route (what the JAX models did everywhere before): a
    ``(2*ncoords, 2*ncoords)`` Hessian per point, contracted with the
    pullbacks afterwards.  Kept for the agreement tests of
    :func:`ddbar_phi_pullback`.

    Args:
        phi_fn (callable): scalar phi of a single point, (2*ncoords,) -> ().
        points (jnp.ndarray, [bSize, 2*ncoords], real): Points.
        pullbacks (jnp.ndarray, [bSize, nfold, ncoords], complex): Pullbacks.

    Returns:
        jnp.ndarray, [bSize, nfold, nfold], complex.
    """
    ncoords = pullbacks.shape[-1]
    dd_phi = jax.vmap(jax.jacobian(jax.grad(phi_fn)))(points)
    nc = ncoords
    dx_dx_phi = 0.25 * dd_phi[:, :nc, :nc]
    dx_dy_phi = 0.25 * dd_phi[:, :nc, nc:]
    dy_dx_phi = 0.25 * dd_phi[:, nc:, :nc]
    dy_dy_phi = 0.25 * dd_phi[:, nc:, nc:]
    dd_phi_c = (dx_dx_phi + dy_dy_phi
                + 1j * (dx_dy_phi - dy_dx_phi)).astype(complex_dtype())
    return jnp.einsum('xai,xij,xbj->xab', pullbacks, dd_phi_c,
                      jnp.conj(pullbacks))


# ---------------------------------------------------------------------------
# FreeModel
# ---------------------------------------------------------------------------

class FreeModel(FSModel):
    r"""FreeModel: learns g_out = g_NN (a Hermitian tensor).

    Equivalent to tensorflow/models/models.py::FreeModel.

    The NN backbone (``self.model``) is any ``equinox.Module`` that maps
    ``(bSize, 2*ncoords)`` → ``(bSize, nfold**2)`` real-valued outputs.
    Training is done externally via :func:`cymetric.jax.models.helper.train_model`.

    NOTE:
        * ``learn_ricci`` is False by default, matching TF.
        * Loss contributions are weighted by ``self.alpha``.
        * ``self.sigma_loss`` is the MA loss (a method, taking kappa from
          the model's dynamic ``kappa`` field).
    """
    model: eqx.Module
    NLOSS: int
    alpha: list          # [float] × NLOSS
    learn_kaehler: bool
    learn_transition: bool
    learn_ricci: bool
    learn_ricci_val: bool
    learn_volk: bool
    # kappa is an array, not a Python float: it changes from one moduli point
    # of a geometry to the next, and a float would make every jitted program
    # that takes the model recompile when it does (measured: see PORTING.md).
    kappa: jnp.ndarray
    gclipping: float
    custom_metrics: list   # [metric objects]
    map_tangents: bool     # see TANGENT_MAP_THRESHOLD

    def __init__(self, nn_model, BASIS, alpha=None, map_tangents=None, **kwargs):
        r"""
        Args:
            nn_model (eqx.Module): NN mapping (bSize, 2*ncoords) → (bSize, nfold²).
            BASIS (dict): Geometry dictionary from PointGenerator.
            alpha (list of float, optional): Loss weights. Defaults to all 1.
            map_tangents (bool, optional): Whether the 2*nfold
                Hessian-vector products of ``ddbar phi`` are scheduled one at
                a time (see :func:`ddbar_phi_pullback`).  None decides from
                the width of the network's input map, which is what the
                memory peak is proportional to. Defaults to None.
        """
        super(FreeModel, self).__init__(BASIS=BASIS, **kwargs)
        self.model = nn_model
        if map_tangents is None:
            width = input_map_width(nn_model, 2 * self.ncoords)
            map_tangents = bool(width is not None
                                and width > TANGENT_MAP_THRESHOLD)
        self.map_tangents = bool(map_tangents)
        self.NLOSS = 5

        if alpha is not None:
            self.alpha = [float(a) for a in alpha]
        else:
            self.alpha = [1.0] * self.NLOSS

        self.learn_kaehler = True
        self.learn_transition = True
        self.learn_ricci = False
        self.learn_ricci_val = False
        self.learn_volk = True

        self.kappa = jnp.asarray(jnp.real(BASIS['KAPPA']), dtype=real_dtype())
        self.gclipping = 5.0
        self.custom_metrics = []

    # ------------------------------------------------------------------
    # The MA (sigma) loss
    # ------------------------------------------------------------------

    def sigma_loss(self, y_true, y_pred, flat=False):
        r"""``|1 - det(g) / (|Omega|^2 kappa)|`` of this model's kappa.

        Was a closure built in ``__init__`` (``sigma_loss(kappa, nfold)``)
        and stored on the module.  Two models of the same geometry then had
        different static parts -- a fresh function object each time -- so
        every jitted program that takes the model recompiled per instance.
        """
        return sigma_integrand(y_true, y_pred, self.kappa,
                               float(self.nfold), flat=flat)

    # ------------------------------------------------------------------
    # Forward pass (equivalent to FreeModel.call in TF)
    # ------------------------------------------------------------------

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out = g_NN (Hermitian).

        Equivalent to FreeModel.call in TF.

        Args:
            input_tensor (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            training (bool): Passed to NN backbone (for dropout/BN).
            j_elim: Not used in FreeModel; kept for API compatibility.

        Returns:
            jnp.ndarray, [bSize, nfold, nfold], complex64.
        """
        return self.to_hermitian(jax.vmap(self.model)(input_tensor))

    @eqx.filter_jit
    def to_hermitian(self, x):
        r"""Hermitian matrix from flat vector.

        Equivalent to FreeModel.to_hermitian in TF.
        """
        return _to_hermitian(x, self.nfold)

    # ------------------------------------------------------------------
    # Volk loss  (equivalent to FreeModel.compute_volk_loss in TF)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def compute_volk_loss(self, input_tensor, wo, pred=None):
        r"""Volk loss (integral over batch).

        Equivalent to FreeModel.compute_volk_loss in TF.

        Args:
            input_tensor (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            wo (jnp.ndarray, [bSize, ≥2], float32): y_train; wo[:,0]=weights,
                wo[:,1]=|Omega|^2.
            pred (jnp.ndarray, [bSize, nfold, nfold], complex64, optional):
                Pre-computed metric prediction.

        Returns:
            jnp.ndarray, [bSize], float32: repeated scalar loss.
        """
        if pred is None:
            pred = self(input_tensor)

        aux_weights = (wo[:, 0] / wo[:, 1]).astype(complex_dtype())
        nk = len(self.BASIS['KMODULI'])
        aux_weights = jnp.repeat(aux_weights[None, :], nk, axis=0)
        # (nk, bSize)

        ks = jnp.eye(nk, dtype=complex_dtype())

        # Build actual_slopes via a Python loop (matches TF while_loop logic)
        actual_slopes = None
        for ki in range(nk):
            f_a = self.fubini_study_pb(input_tensor, ts=ks[ki])
            s = self._calculate_slope(pred, f_a)[None, :]     # (1, bSize)
            if actual_slopes is None:
                actual_slopes = s
            else:
                actual_slopes = jnp.concatenate([actual_slopes, s], axis=0)
        # actual_slopes: (nk, bSize)

        actual_slopes = jnp.mean(aux_weights * actual_slopes, axis=-1)
        # actual_slopes: (nk,)

        loss = jnp.mean(jnp.abs(actual_slopes - self.slopes) ** self.n[4])
        return jnp.repeat(loss[None], len(wo), axis=0)

    # ------------------------------------------------------------------
    # Combined loss (used internally by train_step / compute_loss)
    # ------------------------------------------------------------------

    def compute_loss(self, x, y, sample_weight=None, pb=None):
        r"""Computes the full combined loss for a batch.

        Equivalent to the body of FreeModel.train_step in TF.

        Args:
            x (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            y (jnp.ndarray, [bSize, ≥2], float32): Labels.
            sample_weight (jnp.ndarray, [bSize], float32, optional).
            pb (jnp.ndarray, [bSize, nfold, ncoords], complex, optional):
                Precomputed pullbacks for x. They do not depend on the network
                weights, so training passes them in once instead of rebuilding
                them every step. If None they are computed from the points.
                Defaults to None.

        Returns:
            jnp.ndarray, [bSize], float32: per-sample loss.
        """
        y_pred = self(x, pb=pb)

        cijk_loss = (self.compute_kaehler_loss(x)
                     if self.learn_kaehler
                     else jnp.zeros(x.shape[0], dtype=real_dtype()))

        t_loss = (self.compute_transition_loss(x)
                  if self.learn_transition
                  else jnp.zeros_like(cijk_loss))

        r_loss = (self.compute_ricci_loss(x)
                  if self.learn_ricci
                  else jnp.zeros_like(cijk_loss))

        volk_loss = (self.compute_volk_loss(x, y, y_pred)
                     if self.learn_volk
                     else jnp.zeros_like(cijk_loss))

        omega = y[:, -1:]
        sigma_loss_cont = self.sigma_loss(omega, y_pred) ** self.n[0]

        total_loss = (self.alpha[0] * sigma_loss_cont
                      + self.alpha[1] * cijk_loss
                      + self.alpha[2] * t_loss
                      + self.alpha[3] * r_loss
                      + self.alpha[4] * volk_loss)

        if sample_weight is not None:
            total_loss = total_loss * sample_weight

        return total_loss, {
            'sigma_loss': sigma_loss_cont,
            'kaehler_loss': cijk_loss,
            'transition_loss': t_loss,
            'ricci_loss': r_loss,
            'volk_loss': volk_loss,
        }

    def save(self, filepath, **kwargs):
        r"""Saves the NN backbone to ``filepath`` using equinox serialisation.

        Equivalent to FreeModel.save in TF (saves only the NN).

        Args:
            filepath (str): destination path.
        """
        eqx.tree_serialise_leaves(filepath, self.model)


# ---------------------------------------------------------------------------
# MultFSModel
# ---------------------------------------------------------------------------

class MultFSModel(FreeModel):
    r"""g_out = g_FS * (1 + g_NN)  (element-wise).

    Equivalent to tensorflow/models/models.py::MultFSModel.
    """

    def __init__(self, *args, **kwargs):
        super(MultFSModel, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out_ij = g_FS_ij * (1 + g_NN_ij).

        Equivalent to MultFSModel.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, pb=pb, j_elim=j_elim)
        return fs_cont + jnp.multiply(fs_cont, nn_cont)


# ---------------------------------------------------------------------------
# MatrixFSModel
# ---------------------------------------------------------------------------

class MatrixFSModel(FreeModel):
    r"""g_out = g_FS + g_FS @ g_NN  (matrix multiplication).

    Equivalent to tensorflow/models/models.py::MatrixFSModel.
    """

    def __init__(self, *args, **kwargs):
        super(MatrixFSModel, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out_ik = g_FS_ij (delta_jk + g_NN_jk).

        Equivalent to MatrixFSModel.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, pb=pb, j_elim=j_elim)
        return fs_cont + jnp.matmul(fs_cont, nn_cont)


# ---------------------------------------------------------------------------
# AddFSModel
# ---------------------------------------------------------------------------

class AddFSModel(FreeModel):
    r"""g_out = g_FS + g_NN  (element-wise addition).

    Equivalent to tensorflow/models/models.py::AddFSModel.
    """

    def __init__(self, *args, **kwargs):
        super(AddFSModel, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out_ij = g_FS_ij + g_NN_ij.

        Equivalent to AddFSModel.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, pb=pb, j_elim=j_elim)
        return fs_cont + nn_cont


# ---------------------------------------------------------------------------
# PhiFSModel
# ---------------------------------------------------------------------------

class PhiFSModel(FreeModel):
    r"""g_out = g_FS + del dbar phi_NN.

    Equivalent to tensorflow/models/models.py::PhiFSModel.

    The NN outputs a scalar phi; the metric correction is computed via
    second-order JAX autodiff (replacing tf.GradientTape nested calls).
    By construction Kähler, so learn_kaehler is disabled.
    """

    def __init__(self, *args, **kwargs):
        super(PhiFSModel, self).__init__(*args, **kwargs)
        # Phi model is automatically Kähler
        self.learn_kaehler = False

    def _phi(self, x_single):
        """Scalar NN output for a single point.

        x_single has shape (n_in,); equinox Linear layers expect 1-D input,
        so we call self.model directly without adding a batch dimension.
        """
        return self.model(x_single)[0]   # scalar: output shape (1,) -> ()

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out_ij = g_FS_ij + partial_i bar_partial_j phi_NN.

        Equivalent to PhiFSModel.call in TF, but the pulled back
        ``del delbar phi`` comes from :func:`ddbar_phi_pullback` (2*nfold
        Hessian-vector products) rather than from the full Hessian of phi.
        """
        pbs = self.pullbacks(input_tensor, j_elim=j_elim) if pb is None else pb
        dd_phi_pb = ddbar_phi_pullback(self._phi, input_tensor, pbs,
                                       sequential=self.map_tangents)
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return fs_cont + dd_phi_pb

    def metric_full_hessian(self, input_tensor, training=True, j_elim=None,
                            pb=None):
        r"""``__call__`` through the full Hessian of phi (reference route).

        Only the agreement tests of :func:`ddbar_phi_pullback` need this; it
        costs ``2*ncoords`` Hessian-vector products per point instead of
        ``2*nfold`` and holds the whole Hessian in memory.
        """
        pbs = self.pullbacks(input_tensor, j_elim=j_elim) if pb is None else pb
        dd_phi_pb = ddbar_phi_pullback_full(self._phi, input_tensor, pbs)
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return fs_cont + dd_phi_pb

    def compute_transition_loss(self, points):
        r"""Transition loss for Phi model: phi(lambda^q z) == phi(z).

        Equivalent to PhiFSModel.compute_transition_loss in TF.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        # Each row of ~inv_one_mask has exactly nProjective True entries (patch
        # coords).  jnp.where(condition) used as nonzero requires a static size
        # inside jit; use jax.vmap + jnp.nonzero(size=nP) per row instead.
        nP = self.nProjective
        patch_indices = jax.vmap(
            lambda row: jnp.nonzero(~row, size=nP, fill_value=0)[0].astype(jnp.int32)
        )(inv_one_mask)   # (bSize, nProjective)
        current_patch_mask = self._indices_to_mask(patch_indices)
        fixed = self._find_max_dQ_coords(points)
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(complex_dtype())

        if self.nhyper == 1:
            other_patches = self.fixed_patches[fixed[:, 0]]
        else:
            # General nhyper case: vectorized lookup into the precomputed
            # patch table (see FSModel._generate_patches_table), fully
            # jit-compatible (no Python loop over the batch, no data
            # dependent shapes).
            combo_idx = jnp.zeros(fixed.shape[0], dtype=jnp.int32)
            for j in range(self.nhyper):
                combo_idx = combo_idx * self.ncoords + fixed[:, j]
            patches_slice = self.fixed_patches[combo_idx]
            npatches = self.fixed_patches_n[combo_idx]
            valid = (jnp.arange(self.nTransitions)[None, :]
                     < npatches[:, None])
            other_patches = jnp.where(
                valid[:, :, None], patches_slice, patch_indices[:, None, :])

        other_patches = other_patches.reshape(-1, self.nProjective)
        other_patch_mask = self._indices_to_mask(other_patches)
        exp_points = jnp.repeat(cpoints, self.nTransitions, axis=0)
        patch_points = self._get_patch_coordinates(
            exp_points, other_patch_mask.astype(bool))
        real_patch_points = jnp.concatenate(
            [jnp.real(patch_points), jnp.imag(patch_points)], axis=-1)
        gj = jax.vmap(self.model)(real_patch_points)
        gi = jnp.repeat(jax.vmap(self.model)(points), self.nTransitions, axis=0)
        all_t_loss = jnp.abs(gi - gj)
        all_t_loss = all_t_loss.reshape(-1, self.nTransitions)
        all_t_loss = jnp.sum(all_t_loss ** self.n[2], axis=-1)
        return all_t_loss / (self.nTransitions * self.nfold ** 2)

    def get_kahler_potential(self, points):
        r"""K_FS + phi_NN.

        Equivalent to PhiFSModel.get_kahler_potential in TF.
        """
        if self.nProjective > 1:
            s0 = 0
            e0 = int(self.degrees[0])
            cpoints = (points[:, s0:e0]
                       + 1j * points[:, self.ncoords + s0: self.ncoords + e0]).astype(complex_dtype())
            k_fs = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][0])
            for i in range(1, self.nProjective):
                s = int(jnp.sum(self.degrees[:i]))
                e = s + int(self.degrees[i])
                cpoints = (points[:, s:e]
                           + 1j * points[:, self.ncoords + s: self.ncoords + e]).astype(complex_dtype())
                k_fs = k_fs + self._fubini_study_n_potentials(
                    cpoints, t=self.BASIS['KMODULI'][i])
        else:
            cpoints = (points[:, :self.ncoords]
                       + 1j * points[:, self.ncoords:2 * self.ncoords]).astype(complex_dtype())
            k_fs = self._fubini_study_n_potentials(
                cpoints, t=self.BASIS['KMODULI'][0])

        k_fs = k_fs + jax.vmap(self.model)(points).reshape(-1)
        return k_fs


# ---------------------------------------------------------------------------
# ToricModel
# ---------------------------------------------------------------------------

class ToricModel(FreeModel):
    r"""Toric CY model (generalisation of FS metric for toric spaces).

    Equivalent to tensorflow/models/models.py::ToricModel.
    """
    # Extra toric attributes (equinox: non-array attrs are static by default)
    kmoduli: jnp.ndarray
    sections: list
    patch_masks_arr: jnp.ndarray   # bool array (nPatches, ncoords)
    glsm_charges: jnp.ndarray
    nPatches: int
    patch_degrees: jnp.ndarray
    transition_coefficients: jnp.ndarray
    transition_degrees: jnp.ndarray

    def __init__(self, *args, toric_data, **kwargs):
        r"""
        Args:
            *args: Positional args forwarded to FreeModel (nn_model, BASIS).
            toric_data (dict): Toric geometry data from sage_lib.
            **kwargs: Keyword args forwarded to FreeModel.
        """
        self.nfold = toric_data['dim_cy']
        self.sections = [
            jnp.array(m, dtype=complex_dtype())
            for m in toric_data['exps_sections']
        ]
        self.patch_masks_arr = jnp.array(
            np.array(toric_data['patch_masks'], dtype=bool))
        self.glsm_charges = jnp.array(
            np.array(toric_data['glsm_charges']))
        self.nPatches = len(self.patch_masks_arr)
        self.nProjective = len(toric_data['glsm_charges'])
        super(ToricModel, self).__init__(*args, **kwargs)
        self.kmoduli = self.BASIS['KMODULI']
        self.lc = jnp.array(
            get_levicivita_tensor(self.nfold), dtype=complex_dtype())
        self.slopes = self._target_slopes()

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""Returns the toric FS metric J = t^alpha J_alpha.

        Equivalent to ToricModel.call in TF.
        """
        return self.fubini_study_pb(input_tensor, pb=pb, j_elim=j_elim)

    def fubini_study_pb(self, points, pb=None, j_elim=None, ts=None):
        r"""Pullbacked toric FS metric.

        Equivalent to ToricModel.fubini_study_pb in TF.
        """
        if ts is None:
            ts = self.BASIS['KMODULI']
        pullbacks = self.pullbacks(points, j_elim=j_elim) if pb is None else pb
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(complex_dtype())
        Js = self._fubini_study_n_metrics(cpoints, n=0, t=ts[0])
        for i in range(1, len(self.kmoduli)):
            Js = Js + self._fubini_study_n_metrics(cpoints, n=i, t=ts[i])
        gFSpb = jnp.einsum('xai,xij,xbj->xab', pullbacks, Js, jnp.conj(pullbacks))
        return gFSpb

    def _section_norms(self, points, alpha):
        r"""|s_a|^2 of the sections of generator alpha, up to one shift per point.

        Returns ``mss`` with ``max_a mss[x, a] == 1`` and the ``log`` of the
        factor taken out, i.e. ``|s_a(z)|^2 = exp(shift) * mss[:, a]``.  The
        direct product ``prod(z_i ** d_ai)`` underflows float32 at large h11
        -- a degree-87 monomial of a coordinate near 0.025 is 1e-140 -- and
        when every section underflows ``kappa_alpha`` is 0, so the toric FS
        metric and ``det g`` came out nan at about 1% of the points of the
        h11 = 16 fan of the McAllister scan.  Working with the logs and
        pulling out the largest section keeps the ratios (which is all the
        metric and the potential use) exact to float32 rounding and finite at
        any degree.

        Args:
            points (jnp.ndarray, [bSize, ncoords], complex): Coordinates.
            alpha (int): Index of the Kahler cone generator.

        Returns:
            tuple: (mss, shift), shapes [bSize, n_sections] complex and
            [bSize] real.
        """
        degrees = jnp.real(self.sections[alpha]).astype(real_dtype())
        tiny = jnp.finfo(real_dtype()).tiny
        # |z|^2, floored: log of an exactly vanishing coordinate would be -inf
        # (and its gradient nan) while a zero exponent must contribute 0.
        z2 = jnp.real(points * jnp.conj(points)).astype(real_dtype())
        log_abs2 = jnp.log(jnp.maximum(z2, tiny))                    # (bSize, nc)
        log_mss = log_abs2 @ degrees.T                                # (bSize, n_sections)
        shift = jnp.max(log_mss, axis=-1)
        mss = jnp.exp(log_mss - shift[:, None]).astype(complex_dtype())
        return mss, shift

    @eqx.filter_jit
    def _fubini_study_n_metrics(self, points, n=None, t=None):
        r"""Toric FS metric contribution g_alpha = d_i dbar_j ln rho_alpha.

        Equivalent to ToricModel._fubini_study_n_metrics in TF, with the
        section norms from :meth:`_section_norms` (every term below is
        homogeneous of degree zero in them, so the shift drops out).
        """
        if t is None:
            t = jnp.asarray(1. + 0j, dtype=complex_dtype())
        alpha = 0 if n is None else n
        degrees = self.sections[alpha]
        mss, _ = self._section_norms(points, alpha)                   # (bSize, n_sections)
        kappa_alphas = jnp.sum(mss, axis=-1)                          # (bSize,)
        zizj = points[:, :, None] * jnp.conj(points[:, None, :])     # (bSize, nc, nc)
        J_alphas = 1. / zizj
        J_alphas = jnp.einsum('x,xab->xab',
                              1. / kappa_alphas ** 2, J_alphas)
        coeffs = (jnp.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees)
                  - jnp.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees))
        return J_alphas * coeffs * t / jnp.array(np.pi, dtype=complex_dtype())

    def _patch_transitions(self):
        r"""Number of patch transitions with the same fixed variables (toric).

        The toric count: the largest number of patches in which one
        coordinate is not a patch coordinate (i.e. may be transitioned to).
        The projective-space count of :meth:`FSModel._patch_transitions` is
        unrelated to the fan and is wrong here in both directions -- too
        small as soon as a coordinate is free in more patches than that
        (every scanned Kreuzer-Skarke polytope with h11 >= 3), and far too
        large for a BASIS whose ``AMBIENT`` entries are section counts, which
        then pads every transition tensor with unused slots.
        """
        return int(np.max(
            np.sum(~np.array(self.patch_masks_arr, dtype=bool), axis=-2)))

    def _generate_helpers(self):
        r"""Toric-specific helpers (patch degrees, transition data).

        Equivalent to ToricModel._generate_helpers in TF/torch.  Nothing used
        to call this in the JAX backend, so ``patch_degrees``,
        ``transition_coefficients`` and ``transition_degrees`` stayed
        unassigned and no toric model could be constructed at all;
        ``proj_matrix`` was built from ``AMBIENT`` instead, which for toric
        data is a list of section counts and raises before that.
        """
        self.nTransitions = self._patch_transitions()
        self.fixed_patches = self._generate_all_patches()
        self.fixed_patches_n = None          # nhyper == 1 for a toric hypersurface
        pm_np = np.array(self.patch_masks_arr, dtype=bool)
        gc_np = np.array(np.real(self.glsm_charges)).astype(int)
        patch_degrees = get_all_patch_degrees(gc_np, pm_np)
        w_of_x, del_w_of_x, del_w_of_z = compute_all_w_of_x(
            patch_degrees, pm_np, dim_cy=self.nfold)
        self.patch_degrees = jnp.array(patch_degrees, dtype=complex_dtype())
        self.transition_coefficients = jnp.array(w_of_x, dtype=complex_dtype())
        self.transition_degrees = jnp.array(del_w_of_z, dtype=complex_dtype())
        # The toric routines index patches by the fan, not by a product of
        # projective factors; these two are unused and the ambient data a
        # toric BASIS carries would not build them.
        self.proj_matrix = None
        self._proj_indices = None

    def _generate_all_patches(self):
        """Toric fixed_patches (ncoords, nPatches, nTransitions)."""
        pm_np = np.array(self.patch_masks_arr)
        fixed_patches = np.repeat(
            np.arange(self.nPatches), self.nTransitions)
        fixed_patches = np.tile(fixed_patches, self.ncoords)
        fixed_patches = fixed_patches.reshape(
            self.ncoords, self.nPatches, self.nTransitions)
        for i in range(self.ncoords):
            all_patches = ~pm_np[:, i]
            all_indices = np.where(all_patches)[0]
            fixed_patches[i, all_indices, :len(all_indices)] = (
                all_indices * np.ones((len(all_indices), len(all_indices)),
                                      dtype=int))
        return jnp.array(fixed_patches, dtype=jnp.int32)

    @eqx.filter_jit
    def _get_patch_coordinates(self, points, patch_index):
        r"""Transforms to patch specified by patch_index.

        Equivalent to ToricModel._get_patch_coordinates in TF.
        """
        degrees = self.patch_degrees[patch_index[:, 0]]
        scaled = points[:, None, :] ** degrees
        return jnp.prod(scaled, axis=-1)

    @eqx.filter_jit
    def _mask_to_patch_index(self, mask):
        """Returns the patch index in self.patch_masks_arr of each mask row.

        Every row matches exactly one patch, so argmax picks it out; the
        former ``jnp.where(match)`` has a data-dependent shape and raises a
        ConcretizationTypeError under jit.
        """
        match = jnp.all(mask[:, None, :] == self.patch_masks_arr[None], axis=-1)
        return jnp.argmax(match, axis=-1).astype(jnp.int32)

    @eqx.filter_jit
    def compute_transition_loss(self, points):
        r"""Toric transition loss.

        Equivalent to ToricModel.compute_transition_loss in TF.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        current_patch_mask = ~inv_one_mask
        current_patch_index = self._mask_to_patch_index(
            current_patch_mask).reshape(-1, 1)
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(complex_dtype())
        fixed = self._find_max_dQ_coords(points)
        other_patches = self.fixed_patches[
            jnp.concatenate([fixed, current_patch_index], axis=-1)[:, 0],
            jnp.concatenate([fixed, current_patch_index], axis=-1)[:, 1]]
        other_patch_mask = self.patch_masks_arr[other_patches]
        other_patch_mask = other_patch_mask.reshape(-1, self.ncoords)
        exp_points = jnp.repeat(cpoints, self.nTransitions, axis=0)
        patch_points = self._get_patch_coordinates(
            exp_points, other_patches.reshape(-1, 1))
        fixed_exp = jnp.reshape(
            jnp.tile(fixed, (1, self.nTransitions)), (-1, self.nhyper))
        real_points = jnp.concatenate(
            [jnp.real(patch_points), jnp.imag(patch_points)], axis=-1)
        gj = self(real_points, training=True, j_elim=fixed_exp)
        gi = jnp.repeat(self(points), self.nTransitions, axis=0)
        current_patch_mask_exp = jnp.repeat(
            current_patch_mask, self.nTransitions, axis=0)
        Tij = self.get_transition_matrix(
            patch_points, other_patch_mask, current_patch_mask_exp, fixed_exp)
        all_t_loss = jnp.abs(self.transition_loss_matrices(gj, gi, Tij))
        all_t_loss = jnp.sum(all_t_loss, axis=(1, 2))
        all_t_loss = all_t_loss.reshape(-1, self.nTransitions)
        all_t_loss = jnp.sum(all_t_loss, axis=-1)
        return all_t_loss / (self.nTransitions * self.nfold ** 2)

    @eqx.filter_jit
    def get_transition_matrix(self, points, i_mask, j_mask, fixed):
        r"""Toric transition matrix (uses pre-computed polynomial coefficients).

        Equivalent to ToricModel.get_transition_matrix in TF.
        """
        # Which rows stay in their own patch.  Selecting the two groups with
        # jnp.where(...) gives data-dependent shapes (not jittable), so the
        # polynomial is evaluated for every row and masked afterwards.  The
        # coordinates of the same-patch rows are replaced by ones first, so
        # that negative exponents on a vanishing coordinate cannot put an inf
        # (or a nan gradient) into the discarded branch.
        same = jnp.all(i_mask == j_mask, axis=-1)
        i_patch_indices = self._mask_to_patch_index(i_mask).reshape(-1, 1)
        j_patch_indices = self._mask_to_patch_index(j_mask).reshape(-1, 1)

        tij_indices = jnp.concatenate(
            [fixed, i_patch_indices, j_patch_indices], axis=-1)
        tij_degrees = self.transition_degrees[
            tij_indices[:, 0], tij_indices[:, 1], tij_indices[:, 2]]
        tij_coeff = self.transition_coefficients[
            tij_indices[:, 0], tij_indices[:, 1], tij_indices[:, 2]]
        safe_points = jnp.where(same[:, None], jnp.ones_like(points), points)
        tij_all = jnp.prod(
            safe_points[:, None, None, :] ** tij_degrees, axis=-1)
        tij_all = jnp.transpose(tij_coeff * tij_all, (0, 2, 1))

        tij_eye = jnp.eye(self.nfold, dtype=complex_dtype())[None]
        return jnp.where(same[:, None, None], tij_eye, tij_all)

    @eqx.filter_jit
    def _fubini_study_n_potentials(self, points, n=None, t=None):
        r"""Toric FS Kähler potential for Kahler modulus n.

        Equivalent to PhiFSModelToric._fubini_study_n_potentials in TF.
        """
        if t is None:
            t = jnp.asarray(1. + 0j, dtype=complex_dtype())
        alpha = 0 if n is None else n
        mss, shift = self._section_norms(points, alpha)
        kappa_alphas = jnp.sum(mss, axis=-1)
        # ln rho_alpha = shift + ln(sum_a mss_a); see _section_norms
        log_kappa = shift + jnp.real(jnp.log(kappa_alphas)).astype(real_dtype())
        return (jnp.real(t / np.pi).astype(real_dtype())
                * log_kappa.astype(real_dtype()))

    def get_kahler_potential(self, points):
        r"""Toric FS Kähler potential K = sum_alpha t_alpha ln rho_alpha.

        Equivalent to PhiFSModelToric.get_kahler_potential in TF.
        """
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(complex_dtype())
        k_fs = self._fubini_study_n_potentials(cpoints, t=self.kmoduli[0])
        for i in range(1, len(self.kmoduli)):
            k_fs = k_fs + self._fubini_study_n_potentials(cpoints, i, t=self.kmoduli[i])
        k_fs = k_fs + jax.vmap(self.model)(points).reshape(-1)
        return k_fs


# ---------------------------------------------------------------------------
# PhiFSModelToric
# ---------------------------------------------------------------------------

class PhiFSModelToric(ToricModel):
    r"""g_out = g_FS' + del dbar phi_NN  (toric version).

    Equivalent to tensorflow/models/models.py::PhiFSModelToric.
    """

    def __init__(self, *args, **kwargs):
        super(PhiFSModelToric, self).__init__(*args, **kwargs)
        self.learn_kaehler = False

    def _phi(self, x_single):
        """Scalar NN output for a single point (1-D input, no batch axis)."""
        return self.model(x_single)[0]

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out_ij = g_FS'_ij + partial_i bar_partial_j phi_NN.

        Equivalent to PhiFSModelToric.call in TF, with the pulled back
        ``del delbar phi`` from :func:`ddbar_phi_pullback` (2*nfold
        Hessian-vector products).  The full Hessian of phi is what made this
        model need tens of GB at large h11: the toric nets of the McAllister
        scan take all pairwise section products as inputs, so one Hessian is
        quadratic in a feature count that reaches 87k.
        """
        pbs = self.pullbacks(input_tensor, j_elim=j_elim) if pb is None else pb
        dd_phi_pb = ddbar_phi_pullback(self._phi, input_tensor, pbs,
                                       sequential=self.map_tangents)
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return fs_cont + dd_phi_pb

    def metric_full_hessian(self, input_tensor, training=True, j_elim=None,
                            pb=None):
        r"""``__call__`` through the full Hessian of phi (reference route)."""
        pbs = self.pullbacks(input_tensor, j_elim=j_elim) if pb is None else pb
        dd_phi_pb = ddbar_phi_pullback_full(self._phi, input_tensor, pbs)
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return fs_cont + dd_phi_pb

    def compute_transition_loss(self, points, num_random_scalings=10):
        r"""Toric Phi transition loss: phi(lambda^q z) == phi(z).

        Equivalent to PhiFSModelToric.compute_transition_loss in TF.
        """
        if num_random_scalings is None:
            return super(PhiFSModelToric, self).compute_transition_loss(points)

        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(complex_dtype())
        num_pns = self.glsm_charges.shape[0]

        # Random scalings (same logic as TF version)
        key = jax.random.PRNGKey(0)
        key, k1, k2 = jax.random.split(key, 3)
        scale_factor_rand = jax.random.uniform(
            k1, shape=(num_random_scalings, num_pns),
            minval=0.1, maxval=0.9, dtype=real_dtype()).astype(complex_dtype())
        scale_factor_rand = jnp.repeat(
            scale_factor_rand[:, :, None], self.ncoords, axis=-1)

        lambdas_rand_2 = jax.random.uniform(
            k2, shape=(num_random_scalings, num_pns, 2),
            minval=-1., maxval=1., dtype=real_dtype())
        lambdas_rand = (lambdas_rand_2[:, :, 0]
                        + 1j * lambdas_rand_2[:, :, 1]).astype(complex_dtype())
        lambdas_rand = jnp.repeat(lambdas_rand[:, :, None], self.ncoords, axis=-1)
        lambdas_rand = (scale_factor_rand * lambdas_rand
                        / (lambdas_rand * jnp.conj(lambdas_rand)) ** 0.5)
        lambdas_rand = lambdas_rand ** self.glsm_charges
        lambdas_rand = jnp.prod(lambdas_rand, axis=1)   # (num_random_scalings, ncoords)

        scaled_points = jnp.einsum('xi,ai->xai', cpoints, lambdas_rand)
        scaled_points = scaled_points.reshape(-1, self.ncoords)

        real_patch_points = jnp.concatenate(
            [jnp.real(scaled_points), jnp.imag(scaled_points)], axis=-1)
        gj = jax.vmap(self.model)(real_patch_points)
        gi = jnp.repeat(jax.vmap(self.model)(points), num_random_scalings, axis=0)
        all_t_loss = jnp.abs(gi - gj)
        all_t_loss = all_t_loss.reshape(-1, num_random_scalings)
        all_t_loss = jnp.sum(all_t_loss ** self.n[2], axis=-1)
        return all_t_loss / num_random_scalings


# ---------------------------------------------------------------------------
# MatrixFSModelToric
# ---------------------------------------------------------------------------

class MatrixFSModelToric(ToricModel):
    r"""g_out = g_FS' + g_FS' @ g_NN  (toric, matrix multiplication).

    Equivalent to tensorflow/models/models.py::MatrixFSModelToric.
    """

    def __init__(self, *args, **kwargs):
        super(MatrixFSModelToric, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None, pb=None):
        r"""g_out_ik = g_FS'_ij (delta_jk + g_NN_jk).

        Equivalent to MatrixFSModelToric.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, pb=pb, j_elim=j_elim)
        return fs_cont + jnp.matmul(fs_cont, nn_cont)
