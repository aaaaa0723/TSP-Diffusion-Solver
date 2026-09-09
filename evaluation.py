import numpy as np
import torch


def greedy_decoder(prob_matrix, start_node=0):
    num_nodes = prob_matrix.shape[0]
    visited = {start_node}
    path = [start_node]
    current_node = start_node

    for _ in range(num_nodes - 1):
        step_probs = prob_matrix[current_node].copy()
        step_probs[list(visited)] = -np.inf
        next_node = int(np.argmax(step_probs))
        path.append(next_node)
        visited.add(next_node)
        current_node = next_node

    path.append(start_node)
    return path


def adjacency_to_path(adj_matrix, start_node=0):
    num_nodes = adj_matrix.shape[0]
    visited = {start_node}
    path = [start_node]
    current_node = start_node

    for _ in range(num_nodes - 1):
        candidates = np.flatnonzero(adj_matrix[current_node] > 0)
        candidates = [node for node in candidates if node not in visited]
        if not candidates:
            raise ValueError("Reference adjacency does not describe a valid tour")
        current_node = int(candidates[0])
        path.append(current_node)
        visited.add(current_node)

    path.append(start_node)
    return path


def calculate_path_distance(path, coords_np):
    return sum(
        np.linalg.norm(coords_np[path[index]] - coords_np[path[index + 1]])
        for index in range(len(path) - 1)
    )


def evaluate_route_gap(model, data_loader, device, max_samples=None):
    model.eval()
    gaps = []
    processed = 0

    with torch.no_grad():
        for coords, dist_matrix, ground_truth_adj in data_loader:
            logits = model(coords.to(device), dist_matrix.to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            coords_np = coords.numpy()
            ground_truth_np = ground_truth_adj.numpy()

            for index in range(len(coords_np)):
                ai_path = greedy_decoder(probabilities[index])
                reference_path = adjacency_to_path(ground_truth_np[index])
                ai_distance = calculate_path_distance(ai_path, coords_np[index])
                reference_distance = calculate_path_distance(
                    reference_path, coords_np[index]
                )
                gaps.append(
                    (ai_distance - reference_distance) / reference_distance * 100
                )
                processed += 1
                if max_samples is not None and processed >= max_samples:
                    return summarize_route_gaps(gaps)

    return summarize_route_gaps(gaps)


def summarize_route_gaps(gaps):
    mean_gap = float(np.mean(gaps))
    percentile_95_gap = float(np.percentile(gaps, 95))
    objective = mean_gap + 0.5 * percentile_95_gap
    return mean_gap, percentile_95_gap, objective
