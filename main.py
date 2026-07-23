import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import (
    UltraTightAutoencoder,
    load_and_preprocess_image,
    run_autoencoder
)

from signal_processing import (
    kurtogram_find_frequency_band,
    load_signal,
    butter_bandpass_filter,
    perform_envelope_analysis
)

from diagnosis import auto_diagnose


with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_PATH = cfg["paths"]["model_path"]
TEST_IMAGE_PATH = cfg["paths"]["test_image_path"]
TEST_FILE_MAT = cfg["paths"]["test_file_mat"]

FS = cfg["signal"]["fs"]
Fr = cfg["signal"]["fr"]

BPFO = (
    cfg["signal"]["bpfo_multiplier"] * Fr
)

BPFI = (
    cfg["signal"]["bpfi_multiplier"] * Fr
)

LATENT_DIM = cfg["model"]["latent_dim"]

IMAGE_SIZE = cfg["image"]["image_size"]
MAX_FREQ_OF_IMAGE = cfg["image"]["max_freq_of_image"]

BANDWIDTH = cfg["filter"]["bandwidth"]
MIN_LOWCUT = cfg["filter"]["min_lowcut"]

THRESHOLD = cfg["diagnosis"]["threshold"]
TOLERANCE = cfg["diagnosis"]["tolerance"]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def run_end_to_end_pipeline():
    print("STARTING END-TO-END PIPELINE")

    model = UltraTightAutoencoder(
        latent_dim=LATENT_DIM
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    tensor_input = load_and_preprocess_image(
        TEST_IMAGE_PATH,
        IMAGE_SIZE,
        DEVICE
    )

    tensor_output, tensor_residual = run_autoencoder(
        model,
        tensor_input
    )

    img_in_show = (
        tensor_input
        .squeeze()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    img_out_show = (
        tensor_output
        .squeeze()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    img_res_show = (
        tensor_residual
        .squeeze()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    LOWCUT, HIGHCUT = (
        kurtogram_find_frequency_band(
            img_res_show,
            IMAGE_SIZE,
            MAX_FREQ_OF_IMAGE,
            BANDWIDTH,
            MIN_LOWCUT
        )
    )

    raw_signal = load_signal(
        TEST_FILE_MAT,
        FS
    )

    filtered_signal = butter_bandpass_filter(
        raw_signal,
        LOWCUT,
        HIGHCUT,
        FS
    )

    freqs, fft_amps, envelope = (
        perform_envelope_analysis(
            filtered_signal,
            FS
        )
    )

    status, report_str = auto_diagnose(
        freqs,
        fft_amps,
        BPFO,
        BPFI,
        THRESHOLD,
        TOLERANCE
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10)
    )

    color_map = {
        "NORMAL": "green",
        "RED ALERT": "red",
        "YELLOW ALERT": "orange"
    }

    title_color = color_map.get(
        status,
        "black"
    )

    fig.suptitle(
        f"HYBRID DIAGNOSIS DASHBOARD\n{report_str}",
        fontsize=14,
        fontweight='bold',
        color=title_color,
        y=1.02
    )

    axes[0, 0].imshow(img_in_show)
    axes[0, 0].set_title("1. Original Input")

    axes[0, 1].imshow(img_out_show)
    axes[0, 1].set_title("2. AI Reconstruction (Normal)")

    axes[0, 2].imshow(img_res_show)
    axes[0, 2].set_title("3. Residual (AI Highlight)")

    for ax in axes[0]:
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Frequency Bins")

    time_axis = np.linspace(
        0,
        1,
        FS
    )

    zoom_len = int(0.05 * FS)

    axes[1, 0].plot(
        time_axis[:zoom_len],
        raw_signal[:zoom_len],
        color='lightgray',
        label="Raw"
    )

    axes[1, 0].plot(
        time_axis[:zoom_len],
        filtered_signal[:zoom_len],
        color='blue',
        label=f"Filtered ({LOWCUT:.0f}-{HIGHCUT:.0f}Hz)"
    )

    axes[1, 0].set_title(
        "4. Impact Wave Extraction (Bandpass)"
    )

    axes[1, 0].legend(
        loc="upper right",
        fontsize=8
    )

    axes[1, 1].plot(
        time_axis[:zoom_len],
        filtered_signal[:zoom_len],
        color='blue',
        alpha=0.4
    )

    axes[1, 1].plot(
        time_axis[:zoom_len],
        envelope[:zoom_len],
        color='red',
        linewidth=1.5
    )

    axes[1, 1].set_title(
        "5. Envelope Signal (Hilbert)"
    )

    mask = freqs <= 500

    axes[1, 2].plot(
        freqs[mask],
        fft_amps[mask],
        color='purple'
    )

    axes[1, 2].set_title(
        "6. FFT Spectrum - Fault Confirmation"
    )

    axes[1, 2].axvline(
        x=Fr,
        color='green',
        linestyle='--',
        alpha=0.5,
        label=f'Fr = {Fr:.1f}Hz'
    )

    axes[1, 2].axvline(
        x=BPFO,
        color='orange',
        linestyle='--',
        alpha=0.7,
        label=f'BPFO = {BPFO:.1f}Hz'
    )

    axes[1, 2].axvline(
        x=BPFI,
        color='red',
        linestyle='--',
        alpha=0.7,
        label=f'BPFI = {BPFI:.1f}Hz'
    )

    axes[1, 2].legend(
        loc="upper right",
        fontsize=8
    )

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_end_to_end_pipeline()