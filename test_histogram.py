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

# (保留原本的 Greedy Decoder 與算距離函數)
def greedy_decoder(prob_matrix, start_node=0):
    num_nodes = prob_matrix.shape[0]
    visited = set([start_node])
    path = [start_node]
    current_node = start_node
    for _ in range(num_nodes - 1):
        step_probs = prob_matrix[current_node].copy()
        step_probs[list(visited)] = -999.0 
        next_node = np.argmax(step_probs)
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node
    path.append(start_node)
    return path

def calculate_path_distance(path, coords_np):
    dist = 0.0
    for i in range(len(path) - 1):
        dist += np.linalg.norm(coords_np[path[i]] - coords_np[path[i+1]])
    return dist

def run_test_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("⏳ 載入資料並執行 70/15/15 切割...")
    
    # 1. 資料切割 (核心防禦機制)
    dataset = TSPDataset("tsp_dataset_lite.npz")
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 設定亂數種子，保證每次切出來的考卷都一樣
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    # 為了快速產出圖表，我們從 Test Set 抽 1000 題來考
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 載入模型
    model = TSPPureGNNModel(hidden_dim=256).to(device)
    model.load_state_dict(torch.load("tsp_gnn_model.pth", map_location=device))
    model.eval()

    gaps = []
    
    print("🏃‍♂️ 開始批次處理 1,000 題全新測資...")
    # 只取前 1000 筆測試
    for i, (coord, dist_matrix, ground_truth_adj) in enumerate(tqdm(test_loader, total=1000)):
        if i >= 1000: break
            
        with torch.no_grad():
            logits = model(coord.to(device), dist_matrix.to(device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        coords_np = coord.squeeze(0).cpu().numpy()
        gt_adj_np = ground_truth_adj.squeeze(0).cpu().numpy()
        
        # 計算 AI 距離
        ai_path = greedy_decoder(probs)
        ai_dist = calculate_path_distance(ai_path, coords_np)
        
        # 計算 OR-Tools 最佳距離
        optimal_path = [0]
        curr = 0
        for _ in range(19):
            next_node = np.argmax(gt_adj_np[curr])
            optimal_path.append(next_node)
            curr = next_node
        optimal_path.append(0)
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
    
    plt.title('Optimality Gap Distribution (Test Set - 1000 instances)')
    plt.xlabel('Optimality Gap (%)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('error_histogram.png')
    
    print(f"🎉 測試完成！1,000 題的真實平均 Gap 為: {mean_gap:.2f}%")
    print("📊 誤差直方圖已儲存為 error_histogram.png")

if __name__ == "__main__":
    run_test_and_plot()