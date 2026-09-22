import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from dataset import TSPDataset
from model import TSPPureGNNModel
from evaluation import adjacency_to_path, calculate_path_distance, greedy_decoder
from experiment_utils import DATASET_PATH, get_latest_run

# 最佳超參數（由 Optuna 調優結果更新）
MODEL_HIDDEN_DIM = int(os.getenv("TSP_HIDDEN_DIM", "256"))


def run_test_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏳ 載入資料並執行 70/15/15 切割... (hidden_dim={MODEL_HIDDEN_DIM})")
    
    # 1. 資料切割 (核心防禁機制)
    dataset = TSPDataset(str(DATASET_PATH))
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 設定亂數種子，保證每次切出來的考卷都一樣
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    # 為了快速產出圖表，我們從 Test Set 抽 1000 題來考
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 載入模型（使用最新一次 train.py 產出的 run 資料夾）
    run_dir = get_latest_run()
    model = TSPPureGNNModel(hidden_dim=MODEL_HIDDEN_DIM).to(device)
    checkpoint = torch.load(run_dir / "tsp_gnn_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    gaps = []
    
    sample_limit = min(1000, len(test_dataset))
    print(f"🏃‍♂️ 開始批次處理 {sample_limit} 題測資...")
    for i, (coord, dist_matrix, ground_truth_adj) in enumerate(tqdm(test_loader, total=sample_limit)):
        if i >= sample_limit: break
            
        with torch.no_grad():
            logits = model(coord.to(device), dist_matrix.to(device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        coords_np = coord.squeeze(0).cpu().numpy()
        gt_adj_np = ground_truth_adj.squeeze(0).cpu().numpy()
        
        # 計算 AI 距離
        ai_path = greedy_decoder(probs, coords=coords_np)
        ai_dist = calculate_path_distance(ai_path, coords_np)
        
        # 計算 OR-Tools 最佳距離
        optimal_path = adjacency_to_path(gt_adj_np)
        optimal_dist = calculate_path_distance(optimal_path, coords_np)
        
        # 結算 Gap
        gap = ((ai_dist - optimal_dist) / optimal_dist) * 100
        gaps.append(gap)

    # 3. 繪製 Error Histogram (誤差直方圖)
    print("🖼️ 正在產生誤差直方圖...")
    plt.figure(figsize=(10, 6))
    sns.histplot(gaps, bins=30, kde=True, color='blue')
    
    mean_gap = np.mean(gaps)
    plt.axvline(mean_gap, color='red', linestyle='dashed', linewidth=2, label=f'Mean Gap: {mean_gap:.2f}%')
    
    plt.title(f'Reference Route Gap Distribution (Test Set - {len(gaps)} instances)')
    plt.xlabel('Optimality Gap (%)')
    plt.ylabel('Frequency')
    plt.legend()
    error_hist_path = run_dir / "error_histogram.png"
    plt.savefig(error_hist_path)
    
    print(f"🎉 測試完成！{len(gaps)} 題的平均 Reference Route Gap 為: {mean_gap:.2f}%")
    print(f"📊 誤差直方圖已儲存為 {error_hist_path}")

if __name__ == "__main__":
    run_test_and_plot()