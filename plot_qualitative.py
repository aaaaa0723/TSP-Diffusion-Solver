import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

from dataset import TSPDataset
from model import TSPPureGNNModel
from evaluation import adjacency_to_path, calculate_path_distance, greedy_decoder

def run_qualitative_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 嚴格重現期末考卷 (Test Set)
    dataset = TSPDataset("tsp_dataset_lite.npz")
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. 載入訓練好的 AI 腦袋
    model = TSPPureGNNModel(hidden_dim=128).to(device)
    model.load_state_dict(torch.load("tsp_gnn_model.pth", map_location=device))
    model.eval()

    worst_gap = -1
    worst_data = None

    sample_limit = min(1000, len(test_dataset))
    print(f"🕵️‍♂️ 正在 {sample_limit} 份測資中搜尋最差作答...")
    for i, (coord, dist_matrix, ground_truth_adj) in enumerate(test_loader):
        if i >= sample_limit: break
            
        with torch.no_grad():
            logits = model(coord.to(device), dist_matrix.to(device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
        coords_np = coord.squeeze(0).cpu().numpy()
        gt_adj_np = ground_truth_adj.squeeze(0).cpu().numpy()
        
        ai_path = greedy_decoder(probs)
        ai_dist = calculate_path_distance(ai_path, coords_np)
        
        optimal_path = adjacency_to_path(gt_adj_np)
        optimal_dist = calculate_path_distance(optimal_path, coords_np)
        
        gap = ((ai_dist - optimal_dist) / optimal_dist) * 100
        
        # 紀錄最慘烈的戰況
        if gap > worst_gap:
            worst_gap = gap
            worst_data = (coords_np, ai_path, optimal_path, ai_dist, optimal_dist)

    coords_np, ai_path, optimal_path, ai_dist, optimal_dist = worst_data
    
    # 3. 畫出對比圖
    print(f"📸 抓到戰犯！最大誤差高達 {worst_gap:.2f}%，正在繪製犯罪現場...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 畫出 OR-Tools 完美路線
    ax1.scatter(coords_np[:, 0], coords_np[:, 1], c='red', s=50, zorder=5)
    for i in range(len(optimal_path) - 1):
        p1, p2 = optimal_path[i], optimal_path[i+1]
        ax1.plot([coords_np[p1, 0], coords_np[p2, 0]], 
                 [coords_np[p1, 1], coords_np[p2, 1]], 'k-', zorder=1)
    ax1.set_title(f"OR-Tools Reference\nDistance: {optimal_dist:.4f} km")
    
    # 畫出 AI 迷宮路線
    ax2.scatter(coords_np[:, 0], coords_np[:, 1], c='red', s=50, zorder=5)
    for i in range(len(ai_path) - 1):
        p1, p2 = ai_path[i], ai_path[i+1]
        ax2.plot([coords_np[p1, 0], coords_np[p2, 0]], 
                 [coords_np[p1, 1], coords_np[p2, 1]], 'b-', zorder=1)
    ax2.set_title(f"AI Greedy Prediction\nDistance: {ai_dist:.4f} km (Gap: {worst_gap:.2f}%)")
    
    plt.savefig('worst_case_comparison.png')
    print("🎉 質化實驗完成！請查看 worst_case_comparison.png")

if __name__ == "__main__":
    run_qualitative_analysis()