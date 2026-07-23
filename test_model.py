import os
import glob
import cv2
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


MODELS_DIR = r"train_outputs" # Thư mục chứa các file .pth
TEST_GOOD_DIR = r"bearing_dataset\\test\\good" # Tập test bình thường
TEST_BAD_DIR = r"bearing_dataset\\test\\bad"   # Tập test lỗi
OUTPUT_CSV = "model_evaluation_results.csv" # Tên file xuất ra

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16


class UltraTightAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.encoder_linear = nn.Linear(128 * 14 * 14, latent_dim)
        
        self.decoder_linear = nn.Linear(latent_dim, 128 * 14 * 14)
        self.relu = nn.ReLU()
        self.unflatten = nn.Unflatten(1, (128, 14, 14))
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)
        x = self.decoder_linear(x)
        x = self.relu(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x

class AnomalyEvalDataset(Dataset):
    def __init__(self, root_good, root_bad):
        self.good_paths = glob.glob(os.path.join(root_good, "*.png"))
        self.bad_paths = glob.glob(os.path.join(root_bad, "*.png"))
        
        # Nhãn: 0 là Bình thường (Good), 1 là Lỗi (Bad)
        self.images = [(p, 0) for p in self.good_paths] + [(p, 1) for p in self.bad_paths]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Tiền xử lý giống hệt lúc train (ToTensorV2)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label

def evaluate_model_auroc(model, dataloader):
    model.eval()
    all_labels = []
    all_anomaly_scores = []
    
    criterion_mse = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # Tính MSE (Input - Output)^2 cho từng pixel
            mse_per_pixel = criterion_mse(outputs, inputs)
            
            # Gom lại thành 1 con số (Reconstruction Error) cho mỗi ảnh trong batch
            # Shape của mse_per_pixel đang là [Batch, Channel, Height, Width]
            # Ta tính trung bình trên các chiều Channel, Height, Width (dim 1, 2, 3)
            anomaly_scores = mse_per_pixel.mean(dim=[1, 2, 3]).cpu().numpy()
            
            all_anomaly_scores.extend(anomaly_scores)
            all_labels.extend(labels.numpy())

    # Tính AUROC từ tập nhãn và tập điểm số
    auroc = roc_auc_score(all_labels, all_anomaly_scores)
    return auroc


def main():
    print("1. Đang chuẩn bị tập dữ liệu Test...")
    test_dataset = AnomalyEvalDataset(TEST_GOOD_DIR, TEST_BAD_DIR)
    
    if len(test_dataset) == 0:
        print("[!] Không tìm thấy ảnh test. Vui lòng kiểm tra lại đường dẫn!")
        return
        
    print(f"Tổng số ảnh test: {len(test_dataset)} (Good: {len(test_dataset.good_paths)} | Bad: {len(test_dataset.bad_paths)})")
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Tìm tất cả file .pth
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pth")))
    if not model_files:
        print(f"[!] Không tìm thấy file .pth nào trong {MODELS_DIR}")
        return
    
    print(f"2. Tìm thấy {len(model_files)} file model. Bắt đầu đánh giá...")
    
    # Khởi tạo model base để nạp trọng số
    model = UltraTightAutoencoder(latent_dim=128).to(DEVICE)
    
    results = []

    # Quét từng file model
    for filepath in tqdm(model_files, desc="Đánh giá Models"):
        filename = os.path.basename(filepath)
        
        try:
            # Nạp trọng số
            model.load_state_dict(torch.load(filepath, map_location=DEVICE))
            
            # Tính AUROC
            auroc_score = evaluate_model_auroc(model, test_loader)
            
            results.append([filename, f"{auroc_score:.4f}"])
            
        except Exception as e:
            print(f"\n[!] Lỗi khi đánh giá model {filename}: {e}")
            results.append([filename, "ERROR"])

    # Ghi kết quả ra CSV
    print(f"\n3. Đang ghi kết quả ra file: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Filename", "AUROC_Score"])
        writer.writerows(results)
        


if __name__ == "__main__":
    main()