import os
import cv2
import numpy as np
import scipy.io as sio

from scipy.signal import stft

DATA_DIR = "data"

NORMAL_FILE = os.path.join(DATA_DIR, "97.mat")

FAULT_DIRS = [
    "error_0.007_inch",
    "error_0.014_inch",
    "error_0.021_inch",
    "error_0.028_inch"
]

OUTPUT_DIR = "bearing_dataset"

WINDOW_SIZE = 2048
STRIDE = 512

IMG_SIZE = 224
FS = 48000

TRAIN_GOOD = os.path.join(
    OUTPUT_DIR,
    "train/good"
)

TEST_GOOD = os.path.join(
    OUTPUT_DIR,
    "test/good"
)

TEST_BAD = os.path.join(
    OUTPUT_DIR,
    "test/bad"
)

os.makedirs(TRAIN_GOOD, exist_ok=True)
os.makedirs(TEST_GOOD, exist_ok=True)
os.makedirs(TEST_BAD, exist_ok=True)

def load_cwru_signal(mat_path):

    data = sio.loadmat(mat_path)

    key = None

    for k in data.keys():

        if "DE_time" in k:
            key = k
            break

    if key is None:
        raise ValueError(
            f"Cannot find DE_time in {mat_path}"
        )

    signal = data[key].flatten()

    return signal


def create_windows(
    signal,
    window_size=2048,
    stride=512
):

    windows = []

    for i in range(
        0,
        len(signal) - window_size,
        stride
    ):

        w = signal[i:i + window_size]

        windows.append(w)

    return windows

def signal_to_spectrogram(signal):

    f, t, Zxx = stft(
        signal,
        fs=FS,
        nperseg=128
    )
    spec = np.abs(Zxx)
    # log scale
    spec = np.log1p(spec)
    # normalize
    spec = (
        spec - spec.min()
    ) / (
        spec.max() - spec.min() + 1e-8
    )

    spec = cv2.resize(
        spec,
        (IMG_SIZE, IMG_SIZE)
    )
    spec = (spec * 255).astype(np.uint8)
    spec = cv2.cvtColor(
        spec,
        cv2.COLOR_GRAY2BGR
    )

    return spec

print("Processing normal signal...")

normal_signal = load_cwru_signal(
    NORMAL_FILE
)

normal_windows = create_windows(
    normal_signal,
    WINDOW_SIZE,
    STRIDE
)

print(
    "Normal windows:",
    len(normal_windows)
)



for i, w in enumerate(normal_windows[:-50]):

    spec = signal_to_spectrogram(w)

    save_path = os.path.join(
        TRAIN_GOOD,
        f"normal_{i}.png"
    )

    cv2.imwrite(save_path, spec)



for i, w in enumerate(normal_windows[-50:]):

    spec = signal_to_spectrogram(w)

    save_path = os.path.join(
        TEST_GOOD,
        f"good_{i}.png"
    )

    cv2.imwrite(save_path, spec)



print("Processing fault signals...")

fault_index = 0

for folder in FAULT_DIRS:

    folder_path = os.path.join(
        DATA_DIR,
        folder
    )

    if not os.path.exists(folder_path):
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".mat")
    ]

    for file in files:

        file_path = os.path.join(
            folder_path,
            file
        )

        print("Loading:", file_path)

        signal = load_cwru_signal(
            file_path
        )

        windows = create_windows(
            signal,
            WINDOW_SIZE,
            STRIDE
        )

        for w in windows[:30]:

            spec = signal_to_spectrogram(w)

            # Lấy tên file gốc (ví dụ: 'IR007_0') bỏ đuôi '.mat'
            base_name = os.path.splitext(file)[0]
            
            save_path = os.path.join(
                TEST_BAD,
                f"fault_{base_name}_{fault_index}.png"
            )

            cv2.imwrite(save_path, spec)

            fault_index += 1


print("DATASET CREATION COMPLETE")
