import numpy as np
from numba import njit
from typing import Tuple, List
from numpy.linalg import LinAlgError
from scipy.interpolate import interp1d

try:
    from frites.core.gcmi_nd import cmi_nd_ggg
    from frites.core.gcmi_nd import nd_reshape, nd_shape_checking
    from scipy.special import psi
except:
    print("Frites module is not activated")


def compute_te(
    v_sample: np.ndarray,
    nmove: int = 5,
    nmin_delay: int = 1,
    nmax_delay: int = 40,
    nstep_delay: int = 1,
    nrel_points: Tuple | List = None,
    method="naive",
):

    if nrel_points is None:
        nrel_points = [0]
    if np.any(np.array(nrel_points) > 0):
        raise ValueError("nrel_points must be negative")

    assert method in ("spo", "naive", "mit")

    if method == "spo":
        rollout_points = rollout_points_spo
    elif method == "naive":
        rollout_points = rollout_points_naive
    elif method == "mit":

        raise NotImplemented(
            "momentary information transfer method have not yet implemted"
        )

    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]

    ndelays = -np.arange(nmin_delay, nmax_delay + 1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)

    pair = ((0, 1), (1, 0))
    te_pair = np.zeros((2, len(ndelays)))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks]
        y = data[kd]

        te, nte = 0, 0
        for n0 in npoint_pos:
            yt_curr, yt_prev, xt_prev = rollout_points(x, y, n0, ndelays, nrel_points)

            if yt_curr.shape[-1] < 5 * len(nrel_points):
                continue
            try:
                te_sub = cmi_nd_ggg(
                    yt_curr, xt_prev, yt_prev, mvaxis=-2, demeaned=False
                )
            except:
                continue

            te += te_sub
            nte += 1

        te_pair[ntp] = te / nte

    nlag = -ndelays - np.average(nrel_points)

    return te_pair, nlag


def compute_te_2d(
    v_sample: np.ndarray,
    nmove: int = 5,
    nmin_delay: int = 1,
    nmax_delay: int = 80,
    nstep_delay: int = 1,
    num_time_stack: int = 50,
):
    """
    Compute TE without considering time components

    Inputs:
    v_sample (nsamples, sources, ntimes)

    Outputs:
    te_pair_2d (npairs, ndelays, ndelays): TE (src, dst)
    """
    assert not np.any(np.isnan(v_sample))
    data = np.transpose(v_sample, (1, 0, 2))

    ndelays = -np.arange(nmin_delay, nmax_delay + 1, nstep_delay)
    num_delays = len(ndelays)

    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)

    pair = ((0, 1), (1, 0))
    te_pair_2d = np.zeros((2, num_delays, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks]
        y = data[kd]

        nstack = 0
        nte = np.zeros(num_delays)
        yt_curr, yt_prev, xt_prev = [], [], []
        for n0 in npoint_pos:

            yt_curr.append(y[:, n0])
            yt_prev.append(y[:, n0 + ndelays])
            xt_prev.append(x[:, n0 + ndelays])
            nstack += 1

            if nstack == num_time_stack or n0 == npoint_pos[-1]:

                yt_curr = np.concatenate(yt_curr)
                yt_prev = np.vstack(yt_prev)
                xt_prev = np.vstack(xt_prev)

                yt_curr = np.tile(yt_curr[None, None, :], (num_delays, 1, 1))
                yt_prev = np.transpose(yt_prev)[:, np.newaxis, :]
                xt_prev = np.tile(
                    np.transpose(xt_prev)[:, None, None, :], (1, num_delays, 1, 1)
                )

                for i in range(num_delays):
                    te = cmi_nd_ggg(
                        yt_curr, xt_prev[i], yt_prev, mvaxis=1, demeaned=False
                    )
                    te_pair_2d[ntp, i, :] += te
                    nte[i] += 1

                nstack = 0
                yt_curr, yt_prev, xt_prev = [], [], []

        nte[nte == 0] = 1
        te_pair_2d[ntp] /= nte[:, None]

    nlag = -ndelays

    return te_pair_2d, nlag


def compute_te_2d_reverse(
    v_sample: np.ndarray,
    nmove: int = 5,
    nmin_delay: int = 1,
    nmax_delay: int = 80,
    nstep_delay: int = 1,
    num_time_stack: int = 50,
):
    """
    Compute TE without considering time components

    Inputs:
    v_sample (nsamples, sources, ntimes)

    Outputs:
    te_pair_2d (npairs, ndelays, ndelays): TE (src, dst)
    """
    assert not np.any(np.isnan(v_sample))
    data = np.transpose(v_sample, (1, 0, 2))

    ndelays = np.arange(nmin_delay, nmax_delay + 1, nstep_delay)
    ndelays_pre = np.arange(nmin_delay - nstep_delay, nmax_delay, nstep_delay)
    num_delays = len(ndelays)

    npoint_pos = np.arange(0, v_sample.shape[-1] - nmax_delay, nmove)

    pair = ((0, 1), (1, 0))
    te_pair_2d = np.zeros((2, num_delays, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks]
        y = data[kd]

        nstack = 0
        nte = np.zeros(num_delays)
        yt_next, yt_curr, xt_curr = [], [], []
        for n0 in npoint_pos:

            yt_next.append(y[:, n0 + ndelays])
            yt_curr.append(y[:, n0 + ndelays_pre])
            xt_curr.append(x[:, n0])
            nstack += 1

            if nstack == num_time_stack or n0 == npoint_pos[-1]:

                yt_next = np.vstack(yt_next)
                yt_curr = np.vstack(yt_curr)
                xt_curr = np.concatenate(xt_curr)

                yt_next = np.tile(
                    np.transpose(yt_next)[:, None, None, :], (1, num_delays, 1, 1)
                )
                yt_curr = np.transpose(yt_curr)[:, np.newaxis, :]
                xt_curr = np.tile(xt_curr[None, None, :], (num_delays, 1, 1))

                for i in range(num_delays):
                    te = cmi_nd_ggg(
                        yt_next[i, : i + 1],
                        xt_curr[: i + 1],
                        yt_curr[: i + 1],
                        mvaxis=1,
                        demeaned=False,
                    )
                    te_pair_2d[ntp, i, : i + 1] += te
                    nte[i] += 1

                nstack = 0
                yt_next, yt_curr, xt_curr = [], [], []

        nte[nte == 0] = 1
        te_pair_2d[ntp] /= nte[:, None]

    nlag = ndelays

    return te_pair_2d, nlag


def compute_te_embedding(
    v_sample: np.ndarray,
    nmove: int = 5,
    nmin_delay: int = 1,
    nmax_delay: int = 80,
    nstep_delay: int = 1,
    num_emb_dim=2,
    num_time_stack: int = 50,
):
    """
    Compute TE without considering time components

    Arguments
    num_emb_dim: int, number of embedding dimensions

    """
    assert not np.any(np.isnan(v_sample))
    data = np.transpose(v_sample, (1, 0, 2))

    ndelays = -np.arange(nmin_delay, nmax_delay + 1, nstep_delay)
    num_delays = len(ndelays)

    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)

    pair = ((0, 1), (1, 0))
    te_pair_2d = np.zeros((2, num_delays, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks]
        y = data[kd]

        nstack = 0
        nte = np.zeros(num_delays)
        yt_curr, yt_prev, xt_prev = [], [], []
        for n0 in npoint_pos:

            yt_curr.append(y[:, n0])

            xt_prev.append(x[:, n0 + ndelays])
            yt_prev_tmp = np.zeros((x.shape[0], num_emb_dim, num_delays))
            for n in range(len(ndelays)):
                xp = np.arange(n0 + ndelays[n], n0 + 1)
                yp = y[:, n0 + ndelays[n] : n0 + 1]
                xq = np.linspace(xp[0], xp[-1], num_emb_dim + 1)[:-1]
                yq = interp1d(xp, yp, kind="linear", axis=-1)(xq)
                yt_prev_tmp[:, :, n] = yq
                if np.any(np.isnan(yq)):
                    raise ValueError("Nan detected")

            yt_prev.append(yt_prev_tmp)

            nstack += 1

            if nstack == num_time_stack or n0 == npoint_pos[-1]:

                yt_curr = np.concatenate(yt_curr)
                yt_prev = np.vstack(yt_prev)
                xt_prev = np.vstack(xt_prev)

                yt_curr = np.tile(yt_curr[None, None, :], (num_delays, 1, 1))

                yt_prev = np.transpose(yt_prev)
                xt_prev = np.tile(
                    np.transpose(xt_prev)[:, None, None, :], (1, num_delays, 1, 1)
                )

                for i in range(num_delays):
                    te = _cmi_nd_ggg(
                        yt_curr, xt_prev[i], yt_prev, mvaxis=1, demeaned=False
                    )
                    te_pair_2d[ntp, i, :] += te
                    nte[i] += 1

                nstack = 0
                yt_curr, yt_prev, xt_prev = [], [], []

        nte[nte == 0] = 1
        te_pair_2d[ntp] /= nte[:, None]

    nlag = -ndelays

    return te_pair_2d, nlag


def safe_cholesky(A, eps=1e-13, nstack=0):
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        if nstack > 0 and nstack % 5 == 0:
            eps *= 10
            if nstack == 20:
                raise ValueError(
                    "Cholesky decomposition failed after multiple attempts."
                )

        e = np.linalg.eigvalsh(A)

        if nstack == 0:
            _A = A.copy()
        else:
            _A = A
        for i in range(_A.shape[0]):
            if np.any(e[i] < 0):
                _A[i] += eps * np.eye(_A.shape[1])
        return safe_cholesky(_A, eps=eps, nstack=nstack + 1)


def _cmi_nd_ggg(
    x,
    y,
    z,
    mvaxis=None,
    traxis=-1,
    biascorrect=True,
    demeaned=False,
    shape_checking=True,
):
    """wrapper for cmi_nd_ggg to avoid numerical error determining pos-def matrix in cholesky decomposition"""

    if shape_checking:
        x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        z = nd_reshape(z, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
        nd_shape_checking(x, z, mvaxis, traxis)

    ntrl = x.shape[-1]
    nvarx, nvary, nvarz = x.shape[-2], y.shape[-2], z.shape[-2]
    nvarxy = nvarx + nvary
    nvaryz = nvary + nvarz
    nvarxy = nvarx + nvary
    nvarxz = nvarx + nvarz
    nvarxyz = nvarx + nvaryz

    xyz = np.concatenate((x, y, z), axis=-2)
    if not demeaned:
        xyz -= xyz.mean(axis=-1, keepdims=True)
    cxyz = np.einsum("...ij, ...kj->...ik", xyz, xyz)
    cxyz /= float(ntrl - 1.0)

    cz = cxyz[..., nvarxy:, nvarxy:]
    cyz = cxyz[..., nvarx:, nvarx:]
    sh = list(cxyz.shape)
    sh[-1], sh[-2] = nvarxz, nvarxz
    cxz = np.zeros(tuple(sh), dtype=float)
    cxz[..., :nvarx, :nvarx] = cxyz[..., :nvarx, :nvarx]
    cxz[..., :nvarx, nvarx:] = cxyz[..., :nvarx, nvarxy:]
    cxz[..., nvarx:, :nvarx] = cxyz[..., nvarxy:, :nvarx]
    cxz[..., nvarx:, nvarx:] = cxyz[..., nvarxy:, nvarxy:]

    chcz = safe_cholesky(cz)
    chcxz = safe_cholesky(cxz)
    chcyz = safe_cholesky(cyz)
    chcxyz = safe_cholesky(cxyz)

    hz = np.log(np.einsum("...ii->...i", chcz)).sum(-1)
    hxz = np.log(np.einsum("...ii->...i", chcxz)).sum(-1)
    hyz = np.log(np.einsum("...ii->...i", chcyz)).sum(-1)
    hxyz = np.log(np.einsum("...ii->...i", chcxyz)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxyz + 1)
        psiterms = psi((ntrl - vec).astype(float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hz = hz - nvarz * dterm - psiterms[:nvarz].sum()
        hxz = hxz - nvarxz * dterm - psiterms[:nvarxz].sum()
        hyz = hyz - nvaryz * dterm - psiterms[:nvaryz].sum()
        hxyz = hxyz - nvarxyz * dterm - psiterms[:nvarxyz].sum()

    i = (hxz + hyz - hxyz - hz) / ln2
    return i


def compute_te_full(
    v_sample: np.ndarray,
    nmove: int = 5,
    nmin_delay: int = 1,
    nmax_delay: int = 40,
    nstep_delay: int = 1,
    verbose=False,
):

    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]

    ndelays = -np.arange(nmin_delay, nmax_delay + 1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
    num_delays = len(ndelays)

    pair = ((0, 1), (1, 0))
    te_pair_full = np.zeros((2, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks]
        y = data[kd]

        nte = np.zeros(num_delays)
        for n0 in npoint_pos:
            for i, nd in enumerate(ndelays):
                yt_curr = y[n0, ...][np.newaxis, ...]

                xt_prev = x[n0 + nd, ...][np.newaxis, ...]
                yt_prev = np.hstack([y[[n0 + d], ...] for d in ndelays])
                try:
                    te = cmi_nd_ggg(
                        yt_curr, xt_prev, yt_prev, mvaxis=-2, demeaned=False
                    )
                except LinAlgError as e:
                    pass

                if np.any(abs(te) > 1e5) or np.any(np.isnan(te)):
                    continue

                te_pair_full[ntp] += te
                nte += 1

        nte[nte == 0] = 1
        te_pair_full[ntp] /= nte

    nlag = -ndelays

    return te_pair_full, nlag


def compute_te_full2(
    v_sample: np.ndarray,
    nmove: int = 5,
    nmin_delay: int = 1,
    nmax_delay: int = 80,
    nstep_delay: int = 1,
):
    """
    Compute transfer entropy (TE) considering full history
    TE_{X->Y}(k) = I(Y(t), X(t-k) | {Y_{t-k_max},...,Y_{t-1}}\\Y_{t-k}})

    Parameters
    -----------
    v_sample : np.ndarray
        A 3D array of shape (n, m, t) where 'n' is the number of samples,
        'm' is the number of nodes (2) and 't' is the number of time steps
    nmove : int
        The number of steps the window move
    nmin_delay : int (default 1)
        The minimal time step for lag
    nmax_delay : int (default 80)
        The maximal time step for lag
    nstep_delay : int (default 1)

    Returns
    -------
    te_pair_2d : np.ndarray
        A 2D array of shape (m, k) where 'k' is equal to (nmax_delay-nmin_delay)//2
    """

    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]

    ndelays = -np.arange(nmin_delay, nmax_delay + 1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
    num_delays = len(ndelays)

    pair = ((0, 1), (1, 0))
    te_pair_full = np.zeros((2, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks]
        y = data[kd]

        nte = np.zeros(num_delays)
        for n0 in npoint_pos:

            for i, nd in enumerate(ndelays):
                yt_curr = y[n0, ...][np.newaxis, ...]
                xt_prev = x[n0 + nd, ...][np.newaxis, ...]
                yt_prev = np.hstack([y[[n0 + d], ...] for d in ndelays if d != nd])

                te = _cmi_nd_ggg(yt_curr, xt_prev, yt_prev)
                if abs(te) > 1e5 or np.isnan(te):
                    continue

                te_pair_full[ntp, i] += te
                nte[i] += 1

        nte[nte == 0] = 1
        te_pair_full[ntp] /= nte

    nlag = -ndelays

    return te_pair_full, nlag


def clean_null_points(x, y, z):

    is_in = ~np.isnan(x[-1, 0, :])
    return x[:, :, is_in], y[:, :, is_in], z[:, :, is_in]


def rollout_points_naive(x, y, n0, ndelays, nrel_points):
    yt_curr = np.tile(y[n0, ...], (len(ndelays), 1, 1))
    yt_prev = roll_hstack(y, n0, ndelays, nrel_points)
    xt_prev = roll_hstack(x, n0, ndelays, nrel_points)

    return clean_null_points(yt_curr, yt_prev, xt_prev)


def rollout_points_spo(x, y, n0, ndelays, nrel_points):
    yt_curr = np.tile(y[n0, ...], (len(ndelays), 1, 1))

    yt_prev = roll_hstack(y, n0, np.array([-1]), nrel_points)
    yt_prev = np.tile(yt_prev, (len(ndelays), 1, 1))
    xt_prev = roll_hstack(x, n0, ndelays, nrel_points)

    return clean_null_points(yt_curr, yt_prev, xt_prev)


def roll_hstack(data, ncurr, ndelays: int | np.ndarray, nrel_points):
    return np.hstack([data[ncurr + ndelays + nd, ...] for nd in nrel_points])


def _sampling(
    f_sampling,
    v_set: np.ndarray,
    nchunks: int = 1000,
    chunk_size: int = 100,
    max_delay: int = 0,
    nadd: int = 0,
    reverse=False,
):
    """
    Sampling by chunking the signal.
    The first 'max_delay' number of signals will be overlapped
    """

    assert max_delay < nadd

    nlen = chunk_size + max_delay
    v_sample = np.zeros((nchunks, 2, nlen))

    n = 0
    refresh = True
    while n < nchunks:

        if refresh:
            v_sel = f_sampling(v_set)

            nmax = v_sel.shape[1]
            n0 = np.random.randint(nlen)
            n1 = n0 + nlen
            refresh = False

        if n1 <= nmax:
            v_sample[n] = v_sel[:, n0:n1]
            n0 = n1 - max_delay
            n1 = n0 + nlen

        else:
            n -= 1
            refresh = True

        n += 1

    return v_sample


def sample_true(
    v_set: np.ndarray,
    nchunks: int = 1000,
    chunk_size: int = 100,
    nmax_delay: int = 0,
    nadd: int = 0,
    reverse: bool = False,
):

    def f_sampling(v_set):
        idx = np.random.randint(0, v_set.shape[0])
        if not reverse:
            v_sel = v_set[idx, :, nadd - nmax_delay :]
        else:
            v_sel = v_set[idx, :, : -(nadd - nmax_delay)]
        is_in = ~np.isnan(v_sel[0, :])
        v_sel = v_sel[:, is_in]
        return v_sel

    return _sampling(
        f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd, reverse=reverse
    )


def sample_surrogate(
    v_set: np.ndarray,
    nchunks: int = 1000,
    chunk_size: int = 100,
    nmax_delay: int = 0,
    nadd: int = 0,
    warp_range=(0.8, 1.2),
    reverse=False,
):

    dr = 0.05
    ratio_set = np.arange(warp_range[0], warp_range[1] + dr // 2, dr)

    def f_sampling(v_set):
        nt, ns = np.random.choice(v_set.shape[0], 2, replace=True)
        if nt == ns:
            if not reverse:
                vsub = v_set[nt, :, nadd - nmax_delay :]
            else:
                vsub = v_set[nt, :, : -(nadd - nmax_delay)]
            vsub = vsub[:, ~np.isnan(vsub[0])]
            return vsub

        idt = ~np.isnan(v_set[nt, 0, :])
        ids = ~np.isnan(v_set[ns, 0, :])

        if np.sum(idt) > np.sum(ids) * warp_range[1] - dr:
            nt, ns = ns, nt
            idt, ids = ids, idt

        v_sel = v_set[nt][:, idt]
        n0 = nadd - nmax_delay
        if not reverse:
            vf = v_sel[0, n0:]
            vs = v_sel[1, n0:]
        else:
            vf, vs = v_sel[0, :-n0], v_sel[1, :-n0]

        vs_surr = warp_surrogate_set(vs, v_set[ns, 1, ids], ratio_set)
        assert not np.any(np.isnan(vf))
        assert not np.any(np.isnan(vs_surr))

        return np.array([vf, vs_surr])

    return _sampling(
        f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd, reverse=reverse
    )


def sample_surrogate_iaaft(
    v_set: np.ndarray,
    nchunks: int = 1000,
    chunk_size: int = 100,
    nmax_delay: int = 0,
    nadd: int = 0,
    reverse=False,
):

    def f_sampling(v_set):

        idx = np.random.randint(0, v_set.shape[0])
        if not reverse:
            v_sel = v_set[idx, :, nadd - nmax_delay :]
        else:
            v_sel = v_set[idx, :, : -(nadd - nmax_delay)]
        is_in = ~np.isnan(v_sel[0, :])
        v_sel = v_sel[:, is_in]

        vs_surr = bivariate_surrogates(v_sel[0], v_sel[1])
        return np.array(vs_surr)

    return _sampling(
        f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd, reverse=reverse
    )


def bivariate_surrogates(x1, x2, tol_pc=5.0, maxiter=1e3):
    """
    Returns bivariate IAAFT surrogates of given two time series.

    Parameters
    ----------
    x1, x2 : numpy.ndarray, with shape (N,)
        Input time series for which bivariate IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.

    Returns
    -------
    xs1, xs2 : numpy.ndarray, with shape (ns, N)
        Arrays containing the bivariate IAAFT surrogates of `x1` and `x2` such that
        each row of `xs1` and `xs2` are individual surrogate time series.

    """
    nx = x1.shape[0]
    ii = np.arange(nx)

    x1_amp = np.abs(np.fft.fft(x1))
    x2_amp = np.abs(np.fft.fft(x2))

    x1_srt = np.sort(x1)
    x2_srt = np.sort(x2)

    r_orig1 = np.argsort(x1)
    r_orig2 = np.argsort(x2)

    count = 0
    r_prev1 = np.random.permutation(ii)
    r_prev2 = np.random.permutation(ii)
    r_curr1 = r_orig1
    r_curr2 = r_orig2
    z_n1 = x1[r_prev1]
    z_n2 = x2[r_prev2]
    percent_unequal = 100.0

    phi1 = np.angle(np.fft.fft(x1))
    phi2 = np.angle(np.fft.fft(x2))

    while (percent_unequal > tol_pc) and (count < maxiter):
        r_prev1 = r_curr1
        r_prev2 = r_curr2

        y_prev1 = z_n1

        fft_prev1 = np.fft.fft(y_prev1)

        phi_prev1 = np.angle(fft_prev1)

        mean_phase_diff = phi2 - phi1

        e_i_phi1 = np.exp(phi_prev1 * 1j)
        e_i_phi2 = np.exp((phi_prev1 + mean_phase_diff) * 1j)

        z_n1 = np.fft.ifft(x1_amp * e_i_phi1)
        z_n2 = np.fft.ifft(x2_amp * e_i_phi2)

        r_curr1 = np.argsort(z_n1, kind="quicksort")
        r_curr2 = np.argsort(z_n2, kind="quicksort")

        z_n1[r_curr1] = x1_srt.copy()
        z_n2[r_curr2] = x2_srt.copy()

        percent_unequal = (r_curr2 != r_prev2).sum() * 100.0 / nx

        count += 1

    if count >= (maxiter - 1):
        print("Maximum number of iterations reached!")

    xs1 = np.real(z_n1)
    xs2 = np.real(z_n2)

    return xs1, xs2


@njit
def warp_surrogate_set(v_template, v_surr, ratio_set):

    num = len(ratio_set)
    cmax_set = np.zeros(num)
    vw_set = np.zeros((num, len(v_template)))

    for n in range(num):

        if len(v_surr) * ratio_set[n] < len(v_template):
            continue

        cmax, vw_align = warp_surrogate(v_template, v_surr, ratio_set[n])

        cmax_set[n] = cmax
        vw_set[n] = vw_align

    nid = np.argmax(cmax_set)
    return vw_set[nid]


@njit
def warp_surrogate(v_template, v_surr, ratio):
    N = len(v_surr)
    nmax = int(N * ratio)

    vw = np.interp(np.arange(0, nmax + 1e-10), np.linspace(0, nmax, N), v_surr)

    vw = (vw - vw.mean()) / vw.std()
    c = np.correlate(vw, v_template) / len(v_template)

    nc = np.argmax(c)
    cmax = c[nc]
    vw_align = vw[nc : nc + len(v_template)]

    return cmax, vw_align


def check_overlap(x1, x2, p_ranges):
    x1_min = np.percentile(x1, p_ranges[0], axis=0)
    x1_max = np.percentile(x1, p_ranges[1], axis=0)
    x2_min = np.percentile(x2, p_ranges[0], axis=0)
    x2_max = np.percentile(x2, p_ranges[1], axis=0)

    is_overlap = np.zeros(len(x1_min), dtype=bool)
    is_overlap = is_overlap | ((x2_min < x1_max) & (x2_max > x1_min))
    is_overlap = is_overlap | ((x1_min < x2_max) & (x1_max > x2_min))
    return is_overlap


def bool2ind(bool_array):
    idx = []

    is_find_start = True
    N = len(bool_array)
    for n in range(len(bool_array)):
        if is_find_start and bool_array[n]:
            idx.append([n, N - 1])
            is_find_start = False
        elif not bool_array[n] and not is_find_start:
            idx[-1][1] = n - 1
            is_find_start = True

    return idx
