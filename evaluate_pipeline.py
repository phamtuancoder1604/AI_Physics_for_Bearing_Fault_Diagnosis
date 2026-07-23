# import os
# import glob
# import time
# import re
# import yaml
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix,roc_auc_score, average_precision_score
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
# from model import UltraTightAutoencoder, load_and_preprocess_image, run_autoencoder
# from signal_processing import kurtogram_find_frequency_band, load_signal, butter_bandpass_filter, perform_envelope_analysis
# from diagnosis import auto_diagnose
# # import random
# # torch.manual_seed(0)
# # np.random.seed(0)
# # random.seed(0)
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.size'] = 12

# def add_awgn_noise(signal_1d, snr_db):
#     if snr_db is None:
#         return signal_1d
#     snr_linear = 10 ** (snr_db / 10.0)
#     signal_power = np.mean(signal_1d ** 2)
#     noise_power = signal_power / snr_linear
#     noise = np.sqrt(noise_power) * np.random.randn(len(signal_1d))
#     return signal_1d + noise

# def parse_predicted_label(report_str):
#     if "Outer Ring Defect" in report_str:
#         return "BPFO"
#     elif "Inner Ring Defect" in report_str:
#         return "BPFI"
#     else:
#         return "NORMAL"

# def get_test_pairs(mat_base_dir):
#     pairs = []
#     normal_img_dir = os.path.join("bearing_dataset", "test", "good")
#     normal_mat_path = os.path.join(mat_base_dir, "97.mat")
    
#     if os.path.exists(normal_img_dir):
#         normal_images = glob.glob(os.path.join(normal_img_dir, "*.png"))
#         for img_path in normal_images:
#             if os.path.exists(normal_mat_path):
#                 pairs.append({"img_path": img_path, "mat_path": normal_mat_path, "true_label": "NORMAL"})
                
#     fault_classes = ["BPFI", "BPFO"]
#     for cls in fault_classes:
#         img_pattern = os.path.join("bearing_dataset", cls, "*.png")
#         for img_path in glob.glob(img_pattern):
#             basename = os.path.basename(img_path)
#             match = re.search(r'fault_(\d+)', basename)
#             if not match:
#                 match = re.search(r'(\d+)', basename)
#             if match:
#                 file_id = match.group(1)
#                 mat_path = os.path.join(mat_base_dir, cls, f"{file_id}.mat")
#                 if os.path.exists(mat_path):
#                     pairs.append({"img_path": img_path, "mat_path": mat_path, "true_label": cls})
#     return pairs

# def visualize_residual_comparison(model, test_pairs, img_size, device, out_dir):
#     normal_img, fault_img = None, None
#     for item in test_pairs:
#         if item["true_label"] == "NORMAL" and normal_img is None:
#             normal_img = load_and_preprocess_image(item["img_path"], img_size, device)
#         elif item["true_label"] != "NORMAL" and fault_img is None:
#             fault_img = load_and_preprocess_image(item["img_path"], img_size, device)
#         if normal_img is not None and fault_img is not None:
#             break
            
#     if normal_img is None or fault_img is None:
#         return

#     model.eval()
#     with torch.no_grad():
#         n_out = model(normal_img).cpu()
#         f_out = model(fault_img).cpu()
        
#     n_in = normal_img.cpu().squeeze().permute(1,2,0).numpy()
#     f_in = fault_img.cpu().squeeze().permute(1,2,0).numpy()
    
#     n_res = np.mean(np.abs(n_in - n_out.squeeze().permute(1,2,0).numpy()), axis=2)
#     f_res = np.mean(np.abs(f_in - f_out.squeeze().permute(1,2,0).numpy()), axis=2)
    
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#     vmax = max(np.max(n_res), np.max(f_res))
    
#     axes[0].imshow(n_res, cmap='jet', vmin=0, vmax=vmax)
#     axes[0].set_title('NORMAL Residual')
#     axes[0].axis('off')
    
#     im2 = axes[1].imshow(f_res, cmap='jet', vmin=0, vmax=vmax)
#     axes[1].set_title('FAULT Residual (BPFI/BPFO)')
#     axes[1].axis('off')
    
#     plt.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
#     plt.savefig(os.path.join(out_dir, 'residual_comparison.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def visualize_model_architecture(out_dir):
#     model = UltraTightAutoencoder()
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.text(0.01, 0.5, str(model), fontsize=8, family='monospace', va='center')
#     ax.axis('off')
#     plt.savefig(os.path.join(out_dir, 'model_architecture.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def plot_confusion_matrix_custom(y_true, y_pred, out_dir):
#     classes = ["NORMAL", "BPFI", "BPFO"]
#     cm = confusion_matrix(y_true, y_pred, labels=classes)
#     plt.figure(figsize=(7, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.title('Hybrid System Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
#     plt.close()

# def plot_robustness_chart(robust_df, out_dir):
#     plt.figure(figsize=(8, 5))
#     acc_percent = robust_df['Accuracy'] * 100
#     plt.plot(robust_df['SNR_Level_dB'].astype(str), acc_percent, marker='s', markersize=8, color='#d62728', linewidth=2.5, markerfacecolor='white', markeredgewidth=2)
#     plt.grid(True, linestyle=':', alpha=0.6)
#     plt.gca().set_axisbelow(True)
#     for i, acc in enumerate(acc_percent):
#         plt.text(i, acc - 1.5, f"{acc:.2f}%", ha='center', fontsize=11)
#     plt.ylim(max(0, acc_percent.min() - 5), 105)
#     plt.title('Diagnostic Accuracy Under Varying Noise Conditions', pad=15)
#     plt.xlabel('Signal-to-Noise Ratio (SNR)')
#     plt.ylabel('Accuracy (%)')
#     plt.savefig(os.path.join(out_dir, 'robustness_chart.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def plot_envelope_spectrum_custom(freqs, fft_amps, bpfo, bpfi, out_dir):
#     mask = freqs <= 500
#     plt.figure(figsize=(10, 4))
#     plt.plot(freqs[mask], fft_amps[mask], color='#1f77b4', linewidth=1.2)
    
#     for i in range(1, 4):
#         plt.axvline(x=bpfo * i, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='BPFO' if i==1 else "")
#         plt.axvline(x=bpfi * i, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='BPFI' if i==1 else "")
        
#     plt.xlim(0, 500)
#     plt.title('Kinematic Verification via Hilbert Envelope Spectrum', pad=10)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Amplitude')
    
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
#     plt.savefig(os.path.join(out_dir, 'envelope_spectrum.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     with open("configs/config.yaml", "r") as f:
#         cfg = yaml.safe_load(f)

#     FS = cfg["signal"]["fs"]
#     Fr = cfg["signal"]["fr"]
#     BPFO_FREQ = cfg["signal"]["bpfo_multiplier"] * Fr
#     BPFI_FREQ = cfg["signal"]["bpfi_multiplier"] * Fr
#     IMAGE_SIZE = cfg["image"]["image_size"]
#     MAX_FREQ = cfg["image"]["max_freq_of_image"]
#     BANDWIDTH = cfg["filter"]["bandwidth"]
#     MIN_LOWCUT = cfg["filter"]["min_lowcut"]
#     THRESHOLD = cfg["diagnosis"]["threshold"]
#     TOLERANCE = cfg["diagnosis"]["tolerance"]
#     LATENT_DIM = cfg["model"]["latent_dim"]
    
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     OUTPUT_DIR = "evaluation_outputs"
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     model = UltraTightAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
#     model.load_state_dict(torch.load(cfg["paths"]["model_path"], map_location=DEVICE, weights_only=True))
#     model.eval()

#     MAT_BASE_DIR = "data"
#     test_pairs = get_test_pairs(MAT_BASE_DIR)
    
#     if not test_pairs:
#         return

#     y_true = []
#     y_pred = []
#     anomaly_scores = []
#     inference_times = []
#     kinematic_errors = []
    
#     sample_fault_freqs = None
#     sample_fault_amps = None

#     for item in test_pairs:
#         start_time = time.perf_counter()

#         tensor_input = load_and_preprocess_image(item["img_path"], IMAGE_SIZE, DEVICE)
#         _, tensor_residual = run_autoencoder(model, tensor_input)
#         img_res = tensor_residual.squeeze().cpu().numpy().transpose(1, 2, 0)
#         anomaly_scores.append(tensor_residual.abs().mean().item())
#         LOWCUT, HIGHCUT = kurtogram_find_frequency_band(img_res, IMAGE_SIZE, MAX_FREQ, BANDWIDTH, MIN_LOWCUT)

#         raw_signal = load_signal(item["mat_path"], FS)
#         filtered_signal = butter_bandpass_filter(raw_signal, LOWCUT, HIGHCUT, FS)
#         freqs, fft_amps, _ = perform_envelope_analysis(filtered_signal, FS)
        
#         status, report_str = auto_diagnose(freqs, fft_amps, BPFO_FREQ, BPFI_FREQ, THRESHOLD, TOLERANCE)
        
#         end_time = time.perf_counter()
#         pred_label = parse_predicted_label(report_str)
        
#         y_true.append(item["true_label"])
#         y_pred.append(pred_label)
#         inference_times.append((end_time - start_time) * 1000)

#         if item["true_label"] != "NORMAL":
#             if sample_fault_freqs is None:
#                 sample_fault_freqs = freqs
#                 sample_fault_amps = fft_amps
                
#             mask = (freqs > 40.0) & (freqs <= 500.0)
#             valid_freqs = freqs[mask]
#             valid_amps = fft_amps[mask]
            
#             if len(valid_amps) > 0:
#                 f_max = valid_freqs[valid_amps.argmax()]
#                 theo_freq = BPFO_FREQ if item["true_label"] == "BPFO" else BPFI_FREQ
#                 err = abs(f_max - theo_freq) / theo_freq * 100
#                 kinematic_errors.append(err)

#     # acc = accuracy_score(y_true, y_pred)
#     # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
#     # metrics_df = pd.DataFrame({
#     #     "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Avg Inference Time (ms)", "Avg Kinematic Error (%)"],
#     #     "Value": [acc, precision, recall, f1, np.mean(inference_times), np.mean(kinematic_errors) if kinematic_errors else 0.0]
#     # })
#     y_true_binary = [0 if label == "NORMAL" else 1 for label in y_true]
    
#     acc = accuracy_score(y_true, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
#     try:
#         auc_roc = roc_auc_score(y_true_binary, anomaly_scores)
#         auc_pr = average_precision_score(y_true_binary, anomaly_scores)
#     except ValueError:
#         auc_roc = 0.0
#         auc_pr = 0.0
    
#     metrics_df = pd.DataFrame({
#         "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC (Binary)", "AUC-PR (Binary)", "Avg Inference Time (ms)", "Avg Kinematic Error (%)"],
#         "Value": [acc, precision, recall, f1, auc_roc, auc_pr, np.mean(inference_times), np.mean(kinematic_errors) if kinematic_errors else 0.0]
#     })
#     metrics_df.to_csv(os.path.join(OUTPUT_DIR, "classification_kinematic_metrics.csv"), index=False)

#     snr_levels = [None, 10, 4, 0, -4]
#     robustness_results = []

#     for snr in snr_levels:
#         snr_preds = []
#         for item in test_pairs:
#             raw_signal = load_signal(item["mat_path"], FS)
#             noisy_signal = add_awgn_noise(raw_signal, snr)
            
#             tensor_input = load_and_preprocess_image(item["img_path"], IMAGE_SIZE, DEVICE)
#             _, tensor_res = run_autoencoder(model, tensor_input)
#             img_r = tensor_res.squeeze().cpu().numpy().transpose(1, 2, 0)
#             L_cut, H_cut = kurtogram_find_frequency_band(img_r, IMAGE_SIZE, MAX_FREQ, BANDWIDTH, MIN_LOWCUT)
            
#             f_sig = butter_bandpass_filter(noisy_signal, L_cut, H_cut, FS)
#             f_q, f_a, _ = perform_envelope_analysis(f_sig, FS)
#             _, r_str = auto_diagnose(f_q, f_a, BPFO_FREQ, BPFI_FREQ, THRESHOLD, TOLERANCE)
            
#             snr_preds.append(parse_predicted_label(r_str))
            
#         acc_snr = accuracy_score(y_true, snr_preds)
#         robustness_results.append({"SNR_Level_dB": "Clean" if snr is None else snr, "Accuracy": acc_snr})

#     robust_df = pd.DataFrame(robustness_results)
#     robust_df.to_csv(os.path.join(OUTPUT_DIR, "snr_robustness.csv"), index=False)

#     visualize_residual_comparison(model, test_pairs, IMAGE_SIZE, DEVICE, OUTPUT_DIR)
#     visualize_model_architecture(OUTPUT_DIR)
#     plot_confusion_matrix_custom(y_true, y_pred, OUTPUT_DIR)
#     plot_robustness_chart(robust_df, OUTPUT_DIR)
#     if sample_fault_freqs is not None:
#         plot_envelope_spectrum_custom(sample_fault_freqs, sample_fault_amps, BPFO_FREQ, BPFI_FREQ, OUTPUT_DIR)

# if __name__ == "__main__":
#     main()
import os
import glob
import time
import re
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
from model import UltraTightAutoencoder, load_and_preprocess_image, run_autoencoder
from signal_processing import kurtogram_find_frequency_band, load_signal, butter_bandpass_filter, perform_envelope_analysis
from diagnosis import auto_diagnose

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

def add_awgn_noise(signal_1d, snr_db):
    if snr_db is None:
        return signal_1d
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = np.mean(signal_1d ** 2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal_1d))
    return signal_1d + noise

def parse_predicted_label(report_str):
    if "Outer Ring Defect" in report_str:
        return "BPFO"
    elif "Inner Ring Defect" in report_str:
        return "BPFI"
    else:
        return "NORMAL"

def get_test_pairs(mat_base_dir):
    pairs = []
    normal_img_dir = os.path.join("bearing_dataset", "test", "good")
    normal_mat_path = os.path.join(mat_base_dir, "97.mat")
    
    if os.path.exists(normal_img_dir):
        normal_images = glob.glob(os.path.join(normal_img_dir, "*.png"))
        for img_path in normal_images:
            if os.path.exists(normal_mat_path):
                pairs.append({"img_path": img_path, "mat_path": normal_mat_path, "true_label": "NORMAL"})
                
    fault_classes = ["BPFI", "BPFO"]
    for cls in fault_classes:
        img_pattern = os.path.join("bearing_dataset", cls, "*.png")
        for img_path in glob.glob(img_pattern):
            basename = os.path.basename(img_path)
            match = re.search(r'fault_(\d+)', basename)
            if not match:
                match = re.search(r'(\d+)', basename)
            if match:
                file_id = match.group(1)
                mat_path = os.path.join(mat_base_dir, cls, f"{file_id}.mat")
                if os.path.exists(mat_path):
                    pairs.append({"img_path": img_path, "mat_path": mat_path, "true_label": cls})
    return pairs

def visualize_residual_comparison(model, test_pairs, img_size, device, out_dir):
    normal_img, fault_img = None, None
    for item in test_pairs:
        if item["true_label"] == "NORMAL" and normal_img is None:
            normal_img = load_and_preprocess_image(item["img_path"], img_size, device)
        elif item["true_label"] != "NORMAL" and fault_img is None:
            fault_img = load_and_preprocess_image(item["img_path"], img_size, device)
        if normal_img is not None and fault_img is not None:
            break
            
    if normal_img is None or fault_img is None:
        return

    model.eval()
    with torch.no_grad():
        n_out = model(normal_img).cpu()
        f_out = model(fault_img).cpu()
        
    n_in = normal_img.cpu().squeeze().permute(1,2,0).numpy()
    f_in = fault_img.cpu().squeeze().permute(1,2,0).numpy()
    
    n_res = np.mean(np.abs(n_in - n_out.squeeze().permute(1,2,0).numpy()), axis=2)
    f_res = np.mean(np.abs(f_in - f_out.squeeze().permute(1,2,0).numpy()), axis=2)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmax = max(np.max(n_res), np.max(f_res))
    
    axes[0].imshow(n_res, cmap='jet', vmin=0, vmax=vmax)
    axes[0].set_title('NORMAL Residual')
    axes[0].axis('off')
    
    im2 = axes[1].imshow(f_res, cmap='jet', vmin=0, vmax=vmax)
    axes[1].set_title('FAULT Residual (BPFI/BPFO)')
    axes[1].axis('off')
    
    plt.colorbar(im2, ax=axes, fraction=0.02, pad=0.04)
    plt.savefig(os.path.join(out_dir, 'residual_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_model_architecture(out_dir):
    model = UltraTightAutoencoder()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.01, 0.5, str(model), fontsize=8, family='monospace', va='center')
    ax.axis('off')
    plt.savefig(os.path.join(out_dir, 'model_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, out_dir):
    classes = ["NORMAL", "BPFI", "BPFO"]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Hybrid System Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

def plot_robustness_chart(robust_df, out_dir):
    plt.figure(figsize=(8, 5))
    acc_percent = robust_df['Accuracy'] * 100
    plt.plot(robust_df['SNR_Level_dB'].astype(str), acc_percent, marker='s', markersize=8, color='#d62728', linewidth=2.5, markerfacecolor='white', markeredgewidth=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().set_axisbelow(True)
    for i, acc in enumerate(acc_percent):
        plt.text(i, acc - 1.5, f"{acc:.2f}%", ha='center', fontsize=11)
    plt.ylim(max(0, acc_percent.min() - 5), 105)
    plt.title('Mean Diagnostic Accuracy Under 100 Noise Iterations', pad=15)
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(out_dir, 'robustness_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_envelope_spectrum_custom(freqs, fft_amps, bpfo, bpfi, out_dir):
    mask = freqs <= 500
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[mask], fft_amps[mask], color='#1f77b4', linewidth=1.2)
    
    for i in range(1, 4):
        plt.axvline(x=bpfo * i, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='BPFO' if i==1 else "")
        plt.axvline(x=bpfi * i, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='BPFI' if i==1 else "")
        
    plt.xlim(0, 500)
    plt.title('Kinematic Verification via Hilbert Envelope Spectrum', pad=10)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.savefig(os.path.join(out_dir, 'envelope_spectrum.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    FS = cfg["signal"]["fs"]
    Fr = cfg["signal"]["fr"]
    BPFO_FREQ = cfg["signal"]["bpfo_multiplier"] * Fr
    BPFI_FREQ = cfg["signal"]["bpfi_multiplier"] * Fr
    IMAGE_SIZE = cfg["image"]["image_size"]
    MAX_FREQ = cfg["image"]["max_freq_of_image"]
    BANDWIDTH = cfg["filter"]["bandwidth"]
    MIN_LOWCUT = cfg["filter"]["min_lowcut"]
    THRESHOLD = cfg["diagnosis"]["threshold"]
    TOLERANCE = cfg["diagnosis"]["tolerance"]
    LATENT_DIM = cfg["model"]["latent_dim"]
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = "evaluation"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = UltraTightAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(cfg["paths"]["model_path"], map_location=DEVICE, weights_only=True))
    model.eval()

    MAT_BASE_DIR = "data"
    test_pairs = get_test_pairs(MAT_BASE_DIR)
    
    if not test_pairs:
        return

    y_true = []
    y_pred = []
    anomaly_scores = []
    inference_times = []
    kinematic_errors = []
    
    sample_fault_freqs = None
    sample_fault_amps = None

    for item in test_pairs:
        start_time = time.perf_counter()

        tensor_input = load_and_preprocess_image(item["img_path"], IMAGE_SIZE, DEVICE)
        _, tensor_residual = run_autoencoder(model, tensor_input)
        img_res = tensor_residual.squeeze().cpu().numpy().transpose(1, 2, 0)
        anomaly_scores.append(tensor_residual.abs().mean().item())
        LOWCUT, HIGHCUT = kurtogram_find_frequency_band(img_res, IMAGE_SIZE, MAX_FREQ, BANDWIDTH, MIN_LOWCUT)

        raw_signal = load_signal(item["mat_path"], FS)
        filtered_signal = butter_bandpass_filter(raw_signal, LOWCUT, HIGHCUT, FS)
        freqs, fft_amps, _ = perform_envelope_analysis(filtered_signal, FS)
        
        status, report_str = auto_diagnose(freqs, fft_amps, BPFO_FREQ, BPFI_FREQ, THRESHOLD, TOLERANCE)
        
        end_time = time.perf_counter()
        pred_label = parse_predicted_label(report_str)
        
        y_true.append(item["true_label"])
        y_pred.append(pred_label)
        inference_times.append((end_time - start_time) * 1000)

        if item["true_label"] != "NORMAL":
            if sample_fault_freqs is None:
                sample_fault_freqs = freqs
                sample_fault_amps = fft_amps
                
            mask = (freqs > 40.0) & (freqs <= 500.0)
            valid_freqs = freqs[mask]
            valid_amps = fft_amps[mask]
            
            if len(valid_amps) > 0:
                f_max = valid_freqs[valid_amps.argmax()]
                theo_freq = BPFO_FREQ if item["true_label"] == "BPFO" else BPFI_FREQ
                err = abs(f_max - theo_freq) / theo_freq * 100
                kinematic_errors.append(err)

    y_true_binary = [0 if label == "NORMAL" else 1 for label in y_true]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true_binary, anomaly_scores)
        auc_pr = average_precision_score(y_true_binary, anomaly_scores)
    except ValueError:
        auc_roc = 0.0
        auc_pr = 0.0
    
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC (Binary)", "AUC-PR (Binary)", "Avg Inference Time (ms)", "Avg Kinematic Error (%)"],
        "Value": [acc, precision, recall, f1, auc_roc, auc_pr, np.mean(inference_times), np.mean(kinematic_errors) if kinematic_errors else 0.0]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "classification_kinematic_metrics.csv"), index=False)

    snr_levels = [None, 10, 4, 0, -4]
    robustness_results = []
    so_lan_thu_nghiem = 100

    for snr in snr_levels:
        if snr is None:
            snr_preds = []
            for item in test_pairs:
                raw_signal = load_signal(item["mat_path"], FS)
                tensor_input = load_and_preprocess_image(item["img_path"], IMAGE_SIZE, DEVICE)
                _, tensor_res = run_autoencoder(model, tensor_input)
                img_r = tensor_res.squeeze().cpu().numpy().transpose(1, 2, 0)
                L_cut, H_cut = kurtogram_find_frequency_band(img_r, IMAGE_SIZE, MAX_FREQ, BANDWIDTH, MIN_LOWCUT)
                f_sig = butter_bandpass_filter(raw_signal, L_cut, H_cut, FS)
                f_q, f_a, _ = perform_envelope_analysis(f_sig, FS)
                _, r_str = auto_diagnose(f_q, f_a, BPFO_FREQ, BPFI_FREQ, THRESHOLD, TOLERANCE)
                snr_preds.append(parse_predicted_label(r_str))
            acc_tb = accuracy_score(y_true, snr_preds)
        else:
            danh_sach_acc = []
            for _ in range(so_lan_thu_nghiem):
                snr_preds = []
                for item in test_pairs:
                    raw_signal = load_signal(item["mat_path"], FS)
                    noisy_signal = add_awgn_noise(raw_signal, snr)
                    
                    tensor_input = load_and_preprocess_image(item["img_path"], IMAGE_SIZE, DEVICE)
                    _, tensor_res = run_autoencoder(model, tensor_input)
                    img_r = tensor_res.squeeze().cpu().numpy().transpose(1, 2, 0)
                    L_cut, H_cut = kurtogram_find_frequency_band(img_r, IMAGE_SIZE, MAX_FREQ, BANDWIDTH, MIN_LOWCUT)
                    
                    f_sig = butter_bandpass_filter(noisy_signal, L_cut, H_cut, FS)
                    f_q, f_a, _ = perform_envelope_analysis(f_sig, FS)
                    _, r_str = auto_diagnose(f_q, f_a, BPFO_FREQ, BPFI_FREQ, THRESHOLD, TOLERANCE)
                    snr_preds.append(parse_predicted_label(r_str))
                danh_sach_acc.append(accuracy_score(y_true, snr_preds))
            acc_tb = np.mean(danh_sach_acc)
            
        robustness_results.append({"SNR_Level_dB": "Clean" if snr is None else snr, "Accuracy": acc_tb})

    robust_df = pd.DataFrame(robustness_results)
    robust_df.to_csv(os.path.join(OUTPUT_DIR, "snr_robustness.csv"), index=False)

    visualize_residual_comparison(model, test_pairs, IMAGE_SIZE, DEVICE, OUTPUT_DIR)
    visualize_model_architecture(OUTPUT_DIR)
    plot_confusion_matrix_custom(y_true, y_pred, OUTPUT_DIR)
    plot_robustness_chart(robust_df, OUTPUT_DIR)
    if sample_fault_freqs is not None:
        plot_envelope_spectrum_custom(sample_fault_freqs, sample_fault_amps, BPFO_FREQ, BPFI_FREQ, OUTPUT_DIR)

if __name__ == "__main__":
    main()