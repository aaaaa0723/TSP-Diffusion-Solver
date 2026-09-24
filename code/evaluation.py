import math
import warnings
import numpy as np
import torch
from sklearn.cluster import KMeans

# ===== 目前正式 decoder 的可調整參數 =====
COST_EPSILON = 1e-9
MAX_SUBGRAPH_SIZE = 12  # Held-Karp 的葉細胞最大節點數
BEAM_WIDTH = 2          # 區塊合併時的波束寬度 (同時保留的最佳路線數)
CANDIDATE_LIMIT = 2     # 每次合併時，最多考慮幾個距離最近的相鄰區塊


def _recursive_kmeans_bisection(nodes, coords_np, max_size):
    """
    遞迴 K-Means 切割法：
    將節點集一分為二，直到每個子集的節點數 <= max_size。
    這樣能確保地理位置相近的節點被分在同一組，不會被死板的幾何網格切斷。
    """
    if len(nodes) <= max_size:
        return [nodes]
    
    sub_coords = coords_np[nodes]
    
    # 忽略 KMeans 可能在極少樣本時產生的警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(sub_coords)
    
    labels = kmeans.labels_
    nodes_0 = [nodes[i] for i, label in enumerate(labels) if label == 0]
    nodes_1 = [nodes[i] for i, label in enumerate(labels) if label == 1]
    
    # 防呆：如果 KMeans 無法有效切分 (例如點都在同一個座標)，則強行平分
    if not nodes_0 or not nodes_1:
        mid = len(nodes) // 2
        nodes_0, nodes_1 = nodes[:mid], nodes[mid:]
        
    return _recursive_kmeans_bisection(nodes_0, coords_np, max_size) + \
           _recursive_kmeans_bisection(nodes_1, coords_np, max_size)


def divide_and_conquer_kmeans_beam_decoder(
    prob_matrix, start_node=0, coords=None, max_subgraph_size=MAX_SUBGRAPH_SIZE,
    apply_two_opt=False, beam_width=BEAM_WIDTH, candidate_limit=CANDIDATE_LIMIT
):
    """
    K-Means 聚類 + 區塊級波束搜索 (Beam Search) 求解器。
    先用 KMeans 把全圖切成合理的聚落，用 Held-Karp 完美求解聚落，
    最後用 Beam Search 探索最佳的區塊合併順序。
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    num_nodes = prob_matrix.shape[0]
    
    if prob_matrix.ndim != 2 or prob_matrix.shape[1] != num_nodes:
        raise ValueError("prob_matrix must be a square matrix")
    if not 0 <= start_node < max(num_nodes, 1):
        raise ValueError("start_node is outside prob_matrix")
    if num_nodes <= 1:
        return [start_node, start_node]
    if coords is None:
        raise ValueError("Decoder requires coords")

    coords_np = np.asarray(coords, dtype=np.float64)
    if coords_np.shape[0] != num_nodes:
        raise ValueError("coords and prob_matrix must contain the same number of nodes")
    
    cost_matrix = compute_cost_matrix(prob_matrix)
    
    # --- 1. Divide: 使用 K-Means 切割 ---
    all_nodes = list(range(num_nodes))
    leaf_node_lists = _recursive_kmeans_bisection(all_nodes, coords_np, max_subgraph_size)
    
    # --- 2. Conquer: 局部 Held-Karp 完美求解 ---
    leaves = []
    centroids = []
    for nodes in leaf_node_lists:
        cycle = held_karp_tsp(nodes, cost_matrix)
        leaves.append(cycle)
        centroids.append(coords_np[nodes].mean(axis=0))  # 計算區塊的幾何中心
        
    centroids = np.array(centroids)
    # 預先計算區塊中心之間的距離矩陣，用於尋找相鄰區塊
    cluster_dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)

    # --- 3. Merge: 區塊級波束搜索 (Beam Search) ---
    # 找出包含起點城市的區塊
    start_cluster = next(i for i, leaf in enumerate(leaf_node_lists) if start_node in leaf)
    
    # Beam state 結構: (合併累積成本, 當前區塊 index, 已經走訪的區塊集合(frozenset), 當前串接的大迴圈)
    beam = [(0.0, start_cluster, frozenset([start_cluster]), leaves[start_cluster])]
    
    # 總共需要執行 (葉子數量 - 1) 次合併
    for _ in range(len(leaves) - 1):
        new_beam = []
        
        for score, curr_idx, visited, cycle in beam:
            # 找出尚未走訪的區塊
            unvisited = [i for i in range(len(leaves)) if i not in visited]
            # 依據幾何中心距離，由近到遠排序
            unvisited.sort(key=lambda x: cluster_dist[curr_idx, x])
            
            # 只取距離最近的前 candidate_limit 個區塊進行嘗試，避免窮舉過於耗時
            candidates = unvisited[:candidate_limit]
            
            for nxt_idx in candidates:
                # 計算合併成本
                delta, next_cycle = _best_merge(cycle, leaves[nxt_idx], cost_matrix)
                new_score = score + delta
                new_visited = visited | frozenset([nxt_idx])
                new_beam.append((new_score, nxt_idx, new_visited, next_cycle))
                
        # 將所有探索到的可能路徑依成本排序，只保留前 beam_width 條
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]

    # 取出最終成本最低的那條大迴圈
    best_cycle = beam[0][3]

    # 旋轉陣列，確保從指定的 start_node 出發
    start_index = best_cycle.index(start_node)
    final_cycle = best_cycle[start_index:] + best_cycle[:start_index] + [start_node]

    # --- 4. Refine: 2-Opt 修補 ---
    if apply_two_opt:
        distance_matrix = np.linalg.norm(coords_np[:, None, :] - coords_np[None, :, :], axis=-1)
        final_cycle = two_opt(final_cycle, distance_matrix)
        
    return final_cycle


def divide_and_conquer_decoder(prob_matrix, start_node=0, coords=None):
    """正式使用的 decoder：已經升級為 K-Means + Beam Search 引擎。"""
    return divide_and_conquer_kmeans_beam_decoder(
        prob_matrix, start_node, coords, apply_two_opt=True
    )


def compute_cost_matrix(prob_matrix, eps=COST_EPSILON):
    """把邊機率轉成 Decoder 用的成本：W_ij = -log(P_ij + eps)，機率越高成本越低。"""
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    return -np.log(prob_matrix + eps)


def held_karp_tsp(nodes, cost_matrix):
    """Conquer 階段：bitmask DP（Held-Karp）在子圖內窮舉求出成本最低的封閉迴圈。"""
    num_nodes = len(nodes)
    if num_nodes <= 2:
        return list(nodes)

    def edge_cost(a, b):
        node_a, node_b = nodes[a], nodes[b]
        return (cost_matrix[node_a, node_b] + cost_matrix[node_b, node_a]) / 2.0

    size = 1 << num_nodes
    dp = [[math.inf] * num_nodes for _ in range(size)]
    parent = [[-1] * num_nodes for _ in range(size)]
    dp[1][0] = 0.0  # 固定 nodes[0] 為起點，避免重複列舉旋轉對稱的排列

    for mask in range(size):
        if not (mask & 1):
            continue
        for last in range(num_nodes):
            if not (mask & (1 << last)) or dp[mask][last] == math.inf:
                continue
            base_cost = dp[mask][last]
            for nxt in range(num_nodes):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                candidate = base_cost + edge_cost(last, nxt)
                if candidate < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = candidate
                    parent[new_mask][nxt] = last

    full_mask = size - 1
    best_last, best_cost = -1, math.inf
    for last in range(1, num_nodes):
        if dp[full_mask][last] == math.inf:
            continue
        total = dp[full_mask][last] + edge_cost(last, 0)
        if total < best_cost:
            best_cost = total
            best_last = last

    if best_last == -1:
        return list(nodes)

    order = []
    mask = full_mask
    current = best_last
    while current != -1:
        order.append(nodes[current])
        prev = parent[mask][current]
        mask &= ~(1 << current)
        current = prev
    order.reverse()
    return order


def _cycle_edges(cycle):
    length = len(cycle)
    return [(cycle[i], cycle[(i + 1) % length]) for i in range(length)]


def _break_cycle_at_edge(cycle, edge):
    """把封閉迴圈在指定邊處剪開，回傳一條覆蓋所有節點、從邊的一端走到另一端的開放路徑。"""
    u, v = edge
    length = len(cycle)
    i = cycle.index(u)
    if cycle[(i + 1) % length] == v:
        start_index = (i + 1) % length  # 路徑從 v 開始、u 結束
    else:
        start_index = i  # 邊方向相反，路徑從 u 開始、v 結束
    return cycle[start_index:] + cycle[:start_index]


def _symmetric_cost(cost_matrix, a, b):
    return (cost_matrix[a, b] + cost_matrix[b, a]) / 2.0


def _best_merge(cycle_a, cycle_b, cost_matrix):
    """窮舉 A、B 兩子迴圈的所有邊配對與兩種接法，找出邊交換成本 Delta 最小的合併結果。"""
    best_delta = math.inf
    best_cycle = None

    for u1, v1 in _cycle_edges(cycle_a):
        old_a = _symmetric_cost(cost_matrix, u1, v1)
        path_a = _break_cycle_at_edge(cycle_a, (u1, v1))

        for u2, v2 in _cycle_edges(cycle_b):
            old_b = _symmetric_cost(cost_matrix, u2, v2)
            path_b = _break_cycle_at_edge(cycle_b, (u2, v2))

            # 接法一：path_a 尾接 path_b 頭，path_b 尾接回 path_a 頭
            delta = (
                _symmetric_cost(cost_matrix, path_a[-1], path_b[0])
                + _symmetric_cost(cost_matrix, path_b[-1], path_a[0])
                - old_a - old_b
            )
            if delta < best_delta:
                best_delta = delta
                best_cycle = path_a + path_b

            # 接法二：path_b 反向後再接
            reversed_b = path_b[::-1]
            delta = (
                _symmetric_cost(cost_matrix, path_a[-1], reversed_b[0])
                + _symmetric_cost(cost_matrix, reversed_b[-1], path_a[0])
                - old_a - old_b
            )
            if delta < best_delta:
                best_delta = delta
                best_cycle = path_a + reversed_b

    return best_delta, best_cycle


def two_opt(path, distance_matrix, max_passes=5):
    """標準 2-opt 區域優化：反覆交換路徑段，消除交叉或成本偏高的邊。"""
    path = list(path)
    n = len(path) - 1  # 最後一個節點是回到起點，不參與交換
    if n < 4:
        return path

    path_arr = np.array(path)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            a, b = path_arr[i - 1], path_arr[i]
            c_arr = path_arr[i + 1:n]
            d_arr = path_arr[i + 2:n + 1]
            old_cost = distance_matrix[a, b] + distance_matrix[c_arr, d_arr]
            new_cost = distance_matrix[a, c_arr] + distance_matrix[b, d_arr]
            gains = old_cost - new_cost
            if gains.size == 0:
                continue
            best_idx = int(np.argmax(gains))
            if gains[best_idx] > 1e-9:
                j = i + 1 + best_idx
                path_arr[i:j + 1] = path_arr[i:j + 1][::-1]
                improved = True

    return path_arr.tolist()


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
                ai_path = divide_and_conquer_decoder(probabilities[index], coords=coords_np[index])
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