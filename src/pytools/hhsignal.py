import numpy as np
from scipy.signal import correlate, butter, sosfiltfilt


def get_fft(x, fs, nbin=None, nbin_t=None, frange=None, real=True):
    if nbin is None and nbin_t is None:
        N = len(x)
    elif nbin_t is not None:
        N = int(nbin_t * fs)
    elif nbin is not None:
        N = nbin

    yf = np.fft.fft(x, axis=0, n=N)
    yf = 2 / N * yf[: N // 2]
    if real:
        yf = np.abs(yf)
    freq = np.linspace(0, 1 / 2 * fs, N // 2)

    if frange is not None:
        if frange[0] is None:
            frange[0] = freq[0]
        if frange[1] is None:
            frange[1] = freq[-1]
        idf = (freq >= frange[0]) & (freq <= frange[1])
        yf = yf[idf]
        freq = freq[idf]

    return yf, freq


def get_stfft(x, t, fs, mbin_t=0.1, wbin_t=1, frange=None, buf_size=100, real=True):

    wbin = int(wbin_t * fs)
    mbin = int(mbin_t * fs)
    window = np.hanning(wbin)

    ind = np.arange(wbin // 2, len(t) - wbin // 2, mbin, dtype=int)
    psd = np.zeros([wbin // 2, len(ind)], dtype=float if real else complex)

    n_id = 0
    while n_id < len(ind):
        n_buf = min([buf_size, len(ind) - n_id])
        y = np.zeros([wbin, n_buf])

        for i in range(n_buf):
            n = i + n_id
            n0 = max([0, ind[n] - wbin // 2])
            n1 = min([ind[n] + wbin // 2, len(t)])
            y[n0 - (ind[n] - wbin // 2) : wbin - (ind[n] + wbin // 2) + n1, i] = x[
                n0:n1
            ]
        y = y * window[:, np.newaxis]
        yf, fpsd = get_fft(y, fs, real=real)
        psd[:, n_id : n_id + n_buf] = yf

        n_id += n_buf

    if frange is not None:
        idf = (fpsd >= frange[0]) & (fpsd <= frange[1])
        psd = psd[idf, :]
        fpsd = fpsd[idf]

    tpsd = t[ind]

    return psd, fpsd, tpsd


def get_frequency_peak(vlfp, fs=2000):
    from scipy.signal import find_peaks

    yf, freq = get_fft(vlfp, fs)
    idf = (freq >= 2) & (freq < 200)
    yf = yf[idf]
    freq = freq[idf]

    inds = find_peaks(yf)[0]
    n = np.argmax(yf[inds])
    return freq[inds[n]]


def get_correlation(x, y, srate, max_lag=None, norm=True):

    if norm:
        xn = x - np.average(x)
        yn = y - np.average(y)
        std = [np.std(xn), np.std(yn)]
    else:
        xn, yn = x, y
        std = [1, 1]

    if max_lag is None:
        max_lag = len(x) / srate
    max_pad = int(max_lag * srate)
    tlag = np.arange(-max_lag, max_lag + 1 / srate / 10, 1 / srate)

    if (std[0] == 0) or (std[1] == 0):
        return np.zeros(2 * max_pad + 1), tlag

    pad = np.zeros(max_pad)
    xn = np.concatenate((pad, xn, pad))
    cc = correlate(xn, yn, mode="valid", method="fft") / std[0] / std[1]

    num_use = len(yn)
    cc = cc / num_use

    return cc, tlag


def detect_peak(c, prominence=0.01, mode=0):

    from scipy.signal import find_peaks

    if (mode < 0) or (mode > 3):
        raise ValueError("Invalid mode: %d" % (mode))

    ind_peaks, _ = find_peaks(c, prominence=prominence)
    amp_peaks = c[ind_peaks]
    n0 = ind_peaks[np.argmax(amp_peaks)]

    dn = np.abs(ind_peaks - n0)
    ind_sort = np.argsort(dn)

    if mode == 0:
        return ind_peaks[ind_sort]

    tmp_peaks = ind_peaks[ind_sort[:5]]
    dn2 = np.abs(tmp_peaks - n0)

    if dn2[1] == dn2[2]:
        if dn2[3] == dn2[4]:
            ind_peaks = [tmp_peaks[0], n0 + dn2[1], n0 + dn2[3]]
        else:
            ind_peaks = [tmp_peaks[0], n0 + dn2[1], tmp_peaks[3]]
    else:
        ind_peaks = tmp_peaks[:3]

    if mode == 1:
        return ind_peaks[:2]
    else:
        c1, c2 = c[ind_peaks[1]], c[ind_peaks[2]]
        if c1 < c2:
            ind_peaks_l = [ind_peaks[0], ind_peaks[2]]
        else:
            ind_peaks_l = ind_peaks[:2]

        if mode == 2:
            return ind_peaks_l
        else:
            return ind_peaks[:2], ind_peaks_l


def get_sosfilter(frange: list, srate, fo=5, filter="butter"):
    if filter == "butter":
        sos = butter(
            fo,
            np.array(frange) / srate * 2,
            btype="bandpass",
            output="sos",
            analog=False,
        )
    elif filter == "cheby1":
        from scipy.signal import cheby1

        sos = cheby1(
            fo,
            rp=2,
            Wn=np.array(frange) / srate * 2,
            btype="bandpass",
            output="sos",
            analog=False,
        )
    return sos


def draw_sosfilter_response(sos):
    pass


def filt(sig, sos):
    return sosfiltfilt(sos, sig)


def bandpass_filter(sig, frange, srate, fo=5, filter="butter"):
    sos = get_sosfilter(frange, srate, fo=fo, filter=filter)
    return filt(sig, sos)


def smooth(x, window_size, porder):
    from scipy.signal import savgol_filter

    return savgol_filter(x, window_size, porder)


def downsample(x, srate, srate_new):
    n = int(srate / srate_new)
    return x[::n]


def get_eq_dynamics(x, t, teq):
    idt_eq = t >= teq
    return x[idt_eq], t[idt_eq]


def get_mua(detail, dt=0.01, st=0.001):
    """
    Compute MUA activity for Fast (F) and Slow (S) regions with sigma=st
    out: mua of [F; S]
    """
    from scipy.ndimage import gaussian_filter1d

    tmax = detail["ts"][-1]
    nmax = int((tmax + dt) * 1e3 / dt)

    spk_array = np.zeros((2, nmax))
    for n, n_spk in enumerate(detail["step_spk"]):
        ntp = n // 1000
        spk_array[ntp, n_spk] += 1

    s = int(st * 1e3 / dt)
    spk_array[0] = gaussian_filter1d(spk_array[0], s)
    spk_array[1] = gaussian_filter1d(spk_array[1], s)

    t = np.arange(nmax) * 1e-3 * dt
    return _downsample(detail["ts"], t, spk_array)


def _downsample(tq, t, y):
    yq = [np.interp(tq, t, y[0]), np.interp(tq, t, y[1])]
    return np.array(yq)
