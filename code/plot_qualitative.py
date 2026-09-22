import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

from dataset import TSPDataset
from model import TSPPureGNNModel
from evaluation import adjacency_to_path, calculate_path_distance, divide_and_conquer_decoder
from experiment_utils import DATASET_PATH, get_latest_run

# 最佳超參數（由 Optuna 調優結果更新）
MODEL_HIDDEN_DIM = 256


def _build_case_record(coords_np, ai_path, optimal_path, ai_dist, optimal_dist):
    gap = ((ai_dist - optimal_dist) / optimal_dist) * 100
    return {
        "coords": coords_np,
        "ai_path": ai_path,
        "optimal_path": optimal_path,
        "ai_dist": ai_dist,
        "optimal_dist": optimal_dist,
        "gap": gap,
    }


def _plot_case(ax, coords_np, path, title, color='b'):
    ax.scatter(coords_np[:, 0], coords_np[:, 1], c='red', s=30, zorder=5)
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i + 1]
        ax.plot(
            [coords_np[p1, 0], coords_np[p2, 0]],
            [coords_np[p1, 1], coords_np[p2, 1]],
            color,
            zorder=1,
        )
    ax.set_title(title)
    ax.set_aspect('equal')


def run_qualitative_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TSPDataset(str(DATASET_PATH))
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    run_dir = get_latest_run()
    model = TSPPureGNNModel(hidden_dim=MODEL_HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(run_dir / "tsp_gnn_model.pth", map_location=device))
    model.eval()

    records = []
    sample_limit = min(1000, len(test_dataset))
    print(f"🕵️‍♂️ 正在 {sample_limit} 份測資中比對最佳、平均與最差案例...")

    for i, (coord, dist_matrix, ground_truth_adj) in enumerate(test_loader):
        if i >= sample_limit:
            break

        with torch.no_grad():
            logits = model(coord.to(device), dist_matrix.to(device))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        coords_np = coord.squeeze(0).cpu().numpy()
        gt_adj_np = ground_truth_adj.squeeze(0).cpu().numpy()

        ai_path = divide_and_conquer_decoder(probs, coords=coords_np)
        ai_dist = calculate_path_distance(ai_path, coords_np)

        optimal_path = adjacency_to_path(gt_adj_np)
        optimal_dist = calculate_path_distance(optimal_path, coords_np)

        records.append(_build_case_record(coords_np, ai_path, optimal_path, ai_dist, optimal_dist))

    gaps = np.array([record["gap"] for record in records])
    mean_gap = float(np.mean(gaps))

    best_record = min(records, key=lambda record: record["gap"])
    avg_record = min(records, key=lambda record: abs(record["gap"] - mean_gap))
    worst_record = max(records, key=lambda record: record["gap"])

    print(f"📊 平均 gap: {mean_gap:.2f}% | 最佳 gap: {best_record['gap']:.2f}% | 最差 gap: {worst_record['gap']:.2f}%")

    # 1) Best case: AI route vs OR-Tools reference
    best_coords = best_record["coords"]
    best_ai_path = best_record["ai_path"]
    best_opt_path = best_record["optimal_path"]
    best_gap = best_record["gap"]

    fig_best, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_case(ax1, best_coords, best_opt_path, f"OR-Tools Reference\nDistance: {best_record['optimal_dist']:.4f} km", color='black')
    _plot_case(ax2, best_coords, best_ai_path, f"AI Divide-and-Conquer\nDistance: {best_record['ai_dist']:.4f} km (Gap: {best_gap:.2f}%)", color='green')
    plt.tight_layout()
    best_path = run_dir / "best_case_comparison.png"
    plt.savefig(best_path)
    print(f"🎉 Best case 圖已儲存：{best_path}")

    # 2) Average case: AI route vs OR-Tools reference
    avg_coords = avg_record["coords"]
    avg_ai_path = avg_record["ai_path"]
    avg_opt_path = avg_record["optimal_path"]
    avg_gap = avg_record["gap"]

    fig_avg, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_case(ax1, avg_coords, avg_opt_path, f"OR-Tools Reference\nDistance: {avg_record['optimal_dist']:.4f} km", color='black')
    _plot_case(ax2, avg_coords, avg_ai_path, f"AI Divide-and-Conquer\nDistance: {avg_record['ai_dist']:.4f} km (Gap: {avg_gap:.2f}%)", color='orange')
    plt.tight_layout()
    avg_path = run_dir / "average_case_comparison.png"
    plt.savefig(avg_path)
    print(f"🎉 Average case 圖已儲存：{avg_path}")

    # 3) Worst case: AI route vs OR-Tools reference
    worst_coords = worst_record["coords"]
    worst_ai_path = worst_record["ai_path"]
    worst_opt_path = worst_record["optimal_path"]
    worst_gap = worst_record["gap"]

    fig_worst, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _plot_case(ax1, worst_coords, worst_opt_path, f"OR-Tools Reference\nDistance: {worst_record['optimal_dist']:.4f} km", color='black')
    _plot_case(ax2, worst_coords, worst_ai_path, f"AI Divide-and-Conquer\nDistance: {worst_record['ai_dist']:.4f} km (Gap: {worst_gap:.2f}%)", color='blue')
    plt.tight_layout()
    worst_path = run_dir / "worst_case_comparison.png"
    plt.savefig(worst_path)
    print(f"🎉 Worst case 圖已儲存：{worst_path}")


if __name__ == "__main__":
    run_qualitative_analysis()
