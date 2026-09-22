import numpy as np
import torch


def _choose_next_from_group(current_node, candidate_nodes, prob_matrix):
    """在候選集合內選出最有希望的下一個城市，以邊機率為主，全域熱度僅作極小 tie-breaker。"""
    best_node = None
    best_score = -np.inf

    for node in candidate_nodes:
        forward_score = prob_matrix[current_node, node]
        outgoing_heat = prob_matrix[node].sum() / max(prob_matrix.shape[0] - 1, 1)
        incoming_heat = prob_matrix[:, node].sum() / max(prob_matrix.shape[0] - 1, 1)
        score = forward_score + 0.05 * outgoing_heat + 0.02 * incoming_heat
        if score > best_score:
            best_score = score
            best_node = node

    if best_node is None:
        return candidate_nodes[0]
    return best_node


def divide_and_conquer_decoder(prob_matrix, start_node=0, coords=None):
    """把候選節點分成左右兩塊分別找局部最佳解，再依邊機率選出整體較優的分支。

    有座標時依「相對於前進方向的夾角」做真正的空間切分；沒有座標時退回依邊機率排序切半。
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    num_nodes = prob_matrix.shape[0]

    if num_nodes <= 1:
        return [start_node, start_node]

    coords_np = np.asarray(coords, dtype=np.float64) if coords is not None else None

    visited = {start_node}
    path = [start_node]
    current_node = start_node

    while len(path) < num_nodes:
        remaining = [node for node in range(num_nodes) if node not in visited]
        if not remaining:
            break

        if len(remaining) <= 8:
            next_node = _choose_next_from_group(current_node, remaining, prob_matrix)
        elif coords_np is not None:
            origin = coords_np[current_node]
            reference_vector = (
                origin - coords_np[path[-2]] if len(path) >= 2 else np.array([1.0, 0.0])
            )
            reference_angle = np.arctan2(reference_vector[1], reference_vector[0])
            angles = np.array(
                [
                    np.arctan2(
                        coords_np[node, 1] - origin[1], coords_np[node, 0] - origin[0]
                    )
                    - reference_angle
                    for node in remaining
                ]
            )
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            left_group = [node for node, angle in zip(remaining, angles) if angle >= 0]
            right_group = [node for node, angle in zip(remaining, angles) if angle < 0]

            if not left_group or not right_group:
                next_node = _choose_next_from_group(current_node, remaining, prob_matrix)
            else:
                left_choice = _choose_next_from_group(current_node, left_group, prob_matrix)
                right_choice = _choose_next_from_group(current_node, right_group, prob_matrix)
                left_score = prob_matrix[current_node, left_choice]
                right_score = prob_matrix[current_node, right_choice]
                next_node = left_choice if left_score >= right_score else right_choice
        else:
            remaining_sorted = sorted(
                remaining, key=lambda node: prob_matrix[current_node, node], reverse=True
            )
            split_index = max(1, len(remaining_sorted) // 2)
            left_group = remaining_sorted[:split_index]
            right_group = remaining_sorted[split_index:]

            left_choice = _choose_next_from_group(current_node, left_group, prob_matrix)
            right_choice = _choose_next_from_group(current_node, right_group, prob_matrix)
            left_score = prob_matrix[current_node, left_choice]
            right_score = prob_matrix[current_node, right_choice]
            next_node = left_choice if left_score >= right_score else right_choice

        path.append(next_node)
        visited.add(next_node)
        current_node = next_node

    path.append(start_node)
    return path


def greedy_decoder(prob_matrix, start_node=0, coords=None):
    return divide_and_conquer_decoder(prob_matrix, start_node=start_node, coords=coords)


def adjacency_to_path(adj_matrix, start_node=0):
    num_nodes = adj_matrix.shape[0]
    if not np.all(adj_matrix.sum(axis=1) == 1) or not np.all(adj_matrix.sum(axis=0) == 1):
        raise ValueError("Reference adjacency must have one incoming and outgoing edge per node")
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
    if len(set(path[:-1])) != num_nodes or path[-1] != start_node:
        raise ValueError("Reference adjacency does not describe one Hamiltonian cycle")
    return path


def is_valid_tour(path, num_nodes):
    return len(path) == num_nodes + 1 and path[0] == path[-1] and len(set(path[:-1])) == num_nodes


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
                ai_path = greedy_decoder(probabilities[index], coords=coords_np[index])
                reference_path = adjacency_to_path(ground_truth_np[index])
                if not is_valid_tour(ai_path, len(coords_np[index])):
                    continue
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
    objective = mean_gap + 0.1 * percentile_95_gap
    return mean_gap, percentile_95_gap, objective
