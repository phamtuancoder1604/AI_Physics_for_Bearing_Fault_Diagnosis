import numpy as np
import scipy.io as sio
import scipy.signal as signal

from scipy.stats import kurtosis


def kurtogram_find_frequency_band(
    residual_image,
    img_size,
    max_freq_of_image,
    bandwidth,
    min_lowcut
):
    if len(residual_image.shape) == 3:
        gray_residual = np.mean(
            residual_image,
            axis=2
        )
    else:
        gray_residual = residual_image

    kurtosis_profile = kurtosis(
        gray_residual,
        axis=1,
        fisher=True
    )

    kurtosis_profile = np.nan_to_num(
        kurtosis_profile
    )

    peak_y_pixel = int(
        np.argmax(kurtosis_profile)
    )

    freq_per_pixel = (
        max_freq_of_image / img_size
    )

    peak_frequency = (
        peak_y_pixel * freq_per_pixel
    )

    lowcut = max(
        min_lowcut,
        peak_frequency - bandwidth
    )

    highcut = min(
        max_freq_of_image,
        peak_frequency + bandwidth
    )

    print(
        f"\n[BRIDGE] AI Kurtogram captured the pulse center at: {peak_frequency:.1f} Hz"
    )

    print(
        f"[BRIDGE] Selected Bandpass range: {lowcut:.1f} - {highcut:.1f} Hz\n"
    )

    return lowcut, highcut


def load_signal(mat_path, fs):
    data = sio.loadmat(mat_path)

    key = next(
        (
            k for k in data.keys()
            if "DE_time" in k
        ),
        None
    )

    if key is None:
        raise ValueError(
            f"Could not find the DE_time variable in {mat_path}"
        )

    sig = data[key].flatten()[:fs]

    sig = sig - np.mean(sig)

    return sig


def butter_bandpass_filter(
    data,
    lowcut,
    highcut,
    fs,
    order=4
):
    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    b, a = signal.butter(
        order,
        [low, high],
        btype='band'
    )

    return signal.filtfilt(b, a, data)


def perform_envelope_analysis(
    filtered_signal,
    fs
):
    analytic_signal = signal.hilbert(
        filtered_signal
    )

    amplitude_envelope = np.abs(
        analytic_signal
    )

    amplitude_envelope = (
        amplitude_envelope -
        np.mean(amplitude_envelope)
    )

    n = len(amplitude_envelope)

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / fs
    )

    fft_values = (
        np.abs(
            np.fft.rfft(amplitude_envelope)
        ) * 2 / n
    )

    return (
        frequencies,
        fft_values,
        amplitude_envelope
    )