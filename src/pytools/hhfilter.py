import numpy as np
from scipy.signal import fftconvolve


def get_wv_filter(
    fs, fc, bandwidth=None, n_cycles=7, zero_mean=True, extract_complex=False
):
    """
    Get wavelet based filter
    https://github.com/pactools/pactools/blob/master/pactools/bandpass_filter.py#L9

    """
    if bandwidth is None and n_cycles is not None:
        half_order = int(n_cycles * fs / fc / 2)
    elif bandwidth is not None and n_cycles is None:
        half_order = int(1.65 * fs / bandwidth) // 2
    else:
        raise ValueError("Specify exactly one of bandwidth or n_cycles.")

    order = half_order * 2 + 1
    t = np.linspace(-half_order, half_order, order)
    phase = 2 * np.pi * fc / fs * t
    window = np.blackman(order)

    fir_real = np.cos(phase) * window
    assert np.all(np.abs(fir_real - fir_real[::-1]) < 1e-15)

    if zero_mean:
        fir_real -= fir_real.mean()
    gain = np.sum(fir_real * np.cos(phase))
    fir_real /= gain

    if extract_complex:
        fir_imag = np.sin(phase) * window
        fir_imag /= gain
        return fir_real, fir_imag
    else:
        return fir_real


def bandpass_wv(
    data,
    fir_real=None,
    fir_imag=None,
    fc=None,
    fs=None,
    bandwidth=None,
    n_cycles=7,
    zero_mean=True,
    extract_complex=False,
):
    """
    Apply FIR filter to 1D or 2D signal along the last axis.

    Args:
        data: 1D array of shape (T,)
        fir: 1D FIR filter
    Returns:
        filtered: ndarray of same shape
    """
    data = np.asarray(data)
    if fir_real is None:
        assert fir_imag is None, "If fir_real is None, fir_imag must also be None."
        fir_set = get_wv_filter(
            fs,
            fc,
            bandwidth=bandwidth,
            n_cycles=n_cycles,
            zero_mean=zero_mean,
            extract_complex=extract_complex,
        )
        if not extract_complex:
            fir_set = [fir_set]
    else:
        assert fc is None
        fir_set = (fir_real, fir_imag) if fir_imag is not None else [fir_real]

    data_filt_real = fftconvolve(data, fir_set[0], mode="same")
    if len(fir_set) > 1:
        data_filt_imag = fftconvolve(data, fir_set[1], mode="same")
    else:
        data_filt_imag = None

    return data_filt_real, data_filt_imag
