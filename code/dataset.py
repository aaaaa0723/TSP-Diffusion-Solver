import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TSPDataset(Dataset):
    def __init__(self, npz_path):
        print(f"📦 載入資料集: {npz_path}...")
        # 讀取剛剛辛苦生出來的二進位檔
        data = np.load(npz_path)
        
        # 轉成 PyTorch 專用的 Tensor，並指定浮點數格式
        self.coords = torch.tensor(data['coords'], dtype=torch.float32)
        self.adjacencies = torch.tensor(data['adjacencies'], dtype=torch.float32)
        
        self.num_samples = len(self.coords)
        print(f"✅ 成功載入 {self.num_samples} 筆資料！")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        coord = self.coords[idx]
        adj = self.adjacencies[idx]
        
        # 【極致優化時間】在這裡「即時」把距離矩陣算回來！
        # 利用 PyTorch 的 cdist 高速計算 20x20 的歐幾里得距離
        dist_matrix = torch.cdist(coord, coord)
        
        return coord, dist_matrix, adj

# ==========================================
# 本地端快速測試 (Dry Run)
# ==========================================
if __name__ == "__main__":
    from experiment_utils import DATASET_PATH
    dataset = TSPDataset(str(DATASET_PATH))
    
    # 建立 DataLoader，batch_size 設 32 跑跑看
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 抽一把來看看形狀對不對
    for batch_coords, batch_dists, batch_adjs in dataloader:
        print("\n🚀 DataLoader 測試成功！")
        print(f"📍 座標矩陣 Batch Shape: {batch_coords.shape}  (預期: 32, 20, 2)")
        print(f"📏 距離矩陣 Batch Shape: {batch_dists.shape}  (預期: 32, 20, 20)")
        print(f"🎯 目標標籤 Batch Shape: {batch_adjs.shape}  (預期: 32, 20, 20)")
        break