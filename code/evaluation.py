import itertools
import math

import networkx as nx
import numpy as np
import torch

# ===== Decoder 參數（可依需求調整） =====
MIN_GROUP_SIZE = 6           # 遞迴切分時，片段小於此值就窮舉最佳排列，不再往下切（(n-1)! 需可負擔）
HEAT_OUTGOING_WEIGHT = 0.05  # tie-breaker：候選節點對外邊機率的整體熱度權重
HEAT_INCOMING_WEIGHT = 0.02  # tie-breaker：候選節點對內邊機率的整體熱度權重
DISTANCE_PENALTY_WEIGHT = 0.15  # 有座標時，距離（正規化後）在評分中的權重，僅作 tie-breaker，避免蓋掉學到的邊機率

# ===== D&C v2 參數：機率驅動圖分割 + Held-Karp 精確求解 + 邊交換合併 =====
COST_EPSILON = 1e-9      # W = -log(P + eps)，避免 log(0)
MAX_SUBGRAPH_SIZE = 12   # Held-Karp 純 Python 下的可負擔規模上限（狀態數 2^S）
LOUVAIN_RESOLUTION = 1.0  # Louvain 解析度，越大切出的社群越小、越多


class _DecodeContext:
    """把整趟解碼共用的查表資料（機率、熱度、距離）預先算好，避免在遞迴／排列窮舉中重複計算。"""

    __slots__ = ("prob", "outgoing_heat", "incoming_heat", "dist", "distance_scale")

    def __init__(self, prob_matrix, coords_np):
        num_nodes = prob_matrix.shape[0]
        self.prob = prob_matrix.tolist()
        self.outgoing_heat = (prob_matrix.sum(axis=1) / max(num_nodes - 1, 1)).tolist()
        self.incoming_heat = (prob_matrix.sum(axis=0) / max(num_nodes - 1, 1)).tolist()

        if coords_np is not None:
            diff = coords_np[:, None, :] - coords_np[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)
            nearest = dist_matrix.copy()
            np.fill_diagonal(nearest, np.inf)
            scale = float(nearest.min(axis=1).mean())
            self.distance_scale = scale if scale > 1e-9 else 1.0
            self.dist = dist_matrix.tolist()
        else:
            self.dist = None
            self.distance_scale = None


def _edge_score(ctx, node_a, node_b):
    """評估一條邊的優先程度，以邊機率為主，全域熱度僅作極小 tie-breaker。"""
    return (
        ctx.prob[node_a][node_b]
        + HEAT_OUTGOING_WEIGHT * ctx.outgoing_heat[node_b]
        + HEAT_INCOMING_WEIGHT * ctx.incoming_heat[node_b]
    )


def _connection_score(ctx, node_a, node_b):
    """接邊的綜合評分：邊機率為主，若有座標則扣掉正規化後的距離作 tie-breaker，避免接出長邊。"""
    score = _edge_score(ctx, node_a, node_b)
    if ctx.dist is not None:
        score -= DISTANCE_PENALTY_WEIGHT * (ctx.dist[node_a][node_b] / ctx.distance_scale)
    return score


def _path_score(ctx, path):
    """整條開放路徑的總評分，供小片段窮舉排列時挑選全域最佳解使用。"""
    total = 0.0
    for node_a, node_b in zip(path[:-1], path[1:]):
        total += _connection_score(ctx, node_a, node_b)
    return total


def _build_initial_order(ctx, num_nodes, start_node):
    """以邊機率（+距離 tie-breaker）為準，貪婪走訪出一個初始節點順序，作為分治切分的依據。

    這一步是讓最終路徑品質跟著模型學習狀況走的關鍵：模型學得越好，這個順序就越貼近真實路線，
    後續的分治重排（見 `_solve_small_group` / `_merge_paths`）只是在這個順序上做局部與銜接優化。
    """
    visited = [False] * num_nodes
    visited[start_node] = True
    order = [start_node]
    current = start_node

    for _ in range(num_nodes - 1):
        best_next = None
        best_score = -np.inf
        for node in range(num_nodes):
            if visited[node]:
                continue
            score = _connection_score(ctx, current, node)
            if score > best_score:
                best_score = score
                best_next = node
        order.append(best_next)
        visited[best_next] = True
        current = best_next

    return order


def _solve_small_group(ctx, node_list):
    """片段夠小時，窮舉所有排列找出真正評分最高的開放路徑，修正初始順序在局部造成的鑽牛角尖。"""
    if len(node_list) <= 2:
        return list(node_list)

    anchor = node_list[0]
    best_path = None
    best_score = -np.inf

    for permutation in itertools.permutations(node_list[1:]):
        candidate = [anchor, *permutation]
        score = _path_score(ctx, candidate)
        if score > best_score:
            best_score = score
            best_path = candidate

    return best_path


def _merge_paths(ctx, left_path, right_path):
    """合併兩條子路徑：嘗試四種頭尾接法，取銜接邊「機率＋距離」綜合分數最高的組合。"""
    if not left_path:
        return right_path
    if not right_path:
        return left_path

    candidates = (
        (left_path, right_path),
        (left_path, right_path[::-1]),
        (left_path[::-1], right_path),
        (left_path[::-1], right_path[::-1]),
    )

    best_merged = None
    best_score = -np.inf
    for head, tail in candidates:
        score = _connection_score(ctx, head[-1], tail[0])
        if score > best_score:
            best_score = score
            best_merged = head + tail

    return best_merged


def divide_and_conquer_decoder(prob_matrix, start_node=0, coords=None):
    """分治解碼：先依邊機率貪婪走訪出一個初始順序，再遞迴對半切分、小片段窮舉最佳排列、
    最後挑最佳頭尾接法合併回整條路徑。

    與單純的空間（座標）切分不同，這裡的「分」是依機率排序後的順序位置切半，因此解碼品質會
    直接跟著模型學到的邊機率走；座標只在評分時作為距離 tie-breaker，避免銜接出不合理的長邊。
    所有機率／熱度／距離查表都會在解碼開始前一次算好（見 `_DecodeContext`），避免遞迴中重複計算拖慢速度。
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    num_nodes = prob_matrix.shape[0]

    if num_nodes <= 1:
        return [start_node, start_node]

    coords_np = np.asarray(coords, dtype=np.float64) if coords is not None else None
    ctx = _DecodeContext(prob_matrix, coords_np)

    initial_order = _build_initial_order(ctx, num_nodes, start_node)

    def solve(segment):
        if len(segment) <= MIN_GROUP_SIZE:
            return _solve_small_group(ctx, segment)

        split_index = len(segment) // 2
        left_path = solve(segment[:split_index])
        right_path = solve(segment[split_index:])
        return _merge_paths(ctx, left_path, right_path)

    full_path = solve(initial_order)

    if start_node in full_path:
        start_index = full_path.index(start_node)
        full_path = full_path[start_index:] + full_path[:start_index]

    full_path.append(full_path[0])
    return full_path


def compute_cost_matrix(prob_matrix, eps=COST_EPSILON):
    """把邊機率轉成 Decoder 用的成本：W_ij = -log(P_ij + eps)，機率越高成本越低。"""
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    return -np.log(prob_matrix + eps)


def compute_affinity_matrix(prob_matrix):
    """把機率矩陣對稱化，作為圖分割用的無向圖權重：A_ij = (P_ij + P_ji) / 2。"""
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    return (prob_matrix + prob_matrix.T) / 2.0


def _split_oversized_cluster(nodes, affinity_matrix, max_size, seed=42):
    """遞迴地用 Kernighan-Lin 對半切過大的社群，直到每塊都不超過 max_size。"""
    if len(nodes) <= max_size:
        return [list(nodes)]

    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    for i_index, i in enumerate(nodes):
        for j in nodes[i_index + 1:]:
            weight = float(affinity_matrix[i, j])
            if weight > 0:
                subgraph.add_edge(i, j, weight=weight)

    if subgraph.number_of_edges() == 0:
        # 完全沒有連結資訊時，退回單純均分，確保一定能切開
        midpoint = len(nodes) // 2
        left, right = list(nodes[:midpoint]), list(nodes[midpoint:])
    else:
        part_a, part_b = nx.algorithms.community.kernighan_lin_bisection(
            subgraph, weight="weight", seed=seed
        )
        left, right = list(part_a), list(part_b)

    return (
        _split_oversized_cluster(left, affinity_matrix, max_size, seed)
        + _split_oversized_cluster(right, affinity_matrix, max_size, seed)
    )


def partition_by_probability(prob_matrix, max_subgraph_size=MAX_SUBGRAPH_SIZE, seed=42):
    """Divide 階段：直接用 GNN 機率矩陣（而非座標）做圖分割，機率高的節點被分進同一子圖。"""
    num_nodes = prob_matrix.shape[0]
    affinity_matrix = compute_affinity_matrix(prob_matrix)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = float(affinity_matrix[i, j])
            if weight > 0:
                graph.add_edge(i, j, weight=weight)

    communities = nx.algorithms.community.louvain_communities(
        graph, weight="weight", resolution=LOUVAIN_RESOLUTION, seed=seed
    )

    clusters = []
    for community in communities:
        clusters.extend(
            _split_oversized_cluster(sorted(community), affinity_matrix, max_subgraph_size, seed)
        )
    return clusters


def partition_balanced_bisection(prob_matrix, max_subgraph_size=MAX_SUBGRAPH_SIZE, seed=42):
    """Divide 階段的替代做法：略過 Louvain，直接對整張機率圖遞迴做 Kernighan-Lin 對半切。

    Louvain 切出來的社群大小可能極不平均（有的 2 個節點、有的 40 個），
    導致後續合併的巨觀結構很差。這裡改成每次都對半切，保證每個子圖大小接近，
    通常能讓 Merge 階段（尤其是巨觀 TSP 排序）表現更穩定。
    """
    num_nodes = prob_matrix.shape[0]
    affinity_matrix = compute_affinity_matrix(prob_matrix)
    return _split_oversized_cluster(list(range(num_nodes)), affinity_matrix, max_subgraph_size, seed)


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


def merge_subtours(subtours, cost_matrix):
    """Merge 階段：Bottom-up 貪心合併，每次挑 Delta 最小的一對子迴圈合併，直到剩下一條 Global Tour。"""
    tours = list(subtours)
    while len(tours) > 1:
        best_i, best_j, best_delta, best_cycle = -1, -1, math.inf, None
        for i in range(len(tours)):
            for j in range(i + 1, len(tours)):
                delta, cycle = _best_merge(tours[i], tours[j], cost_matrix)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j, best_cycle = i, j, cycle

        remaining = [t for idx, t in enumerate(tours) if idx not in (best_i, best_j)]
        remaining.append(best_cycle)
        tours = remaining

    return tours[0]


def _greedy_nn_order(cost_matrix):
    """巨觀 TSP 規模太大、Held-Karp 負擔不起時的備援：單純最近鄰貪婪走訪。"""
    num_nodes = cost_matrix.shape[0]
    visited = [False] * num_nodes
    visited[0] = True
    order = [0]
    current = 0
    for _ in range(num_nodes - 1):
        best_next, best_cost = -1, math.inf
        for node in range(num_nodes):
            if visited[node]:
                continue
            if cost_matrix[current, node] < best_cost:
                best_cost = cost_matrix[current, node]
                best_next = node
        order.append(best_next)
        visited[best_next] = True
        current = best_next
    return order


def _macro_tour_order(macro_cost_matrix):
    """對「子圖之間的合併成本」再解一次巨觀 TSP，決定子圖該用什麼順序串起來。"""
    num_clusters = macro_cost_matrix.shape[0]
    if num_clusters <= 2:
        return list(range(num_clusters))
    if num_clusters <= MAX_SUBGRAPH_SIZE:
        return held_karp_tsp(list(range(num_clusters)), macro_cost_matrix)
    order = _greedy_nn_order(macro_cost_matrix)
    closed = two_opt(order + [order[0]], macro_cost_matrix)
    return closed[:-1]


def merge_subtours_hierarchical(subtours, cost_matrix):
    """Merge 階段的替代做法：先解一次「子圖層級」的巨觀 TSP 決定串接順序，
    再依這個順序逐一把子迴圈接起來，而不是每次都貪心挑當下最省的一對。

    貪心合併（見 `merge_subtours`）只看局部最省，容易把整體路線接得歪七扭八；
    這裡先確保「子圖之間怎麼排」是全域考量過的，通常巨觀結構會平滑很多。
    """
    num_clusters = len(subtours)
    if num_clusters == 1:
        return subtours[0]

    macro_cost = np.zeros((num_clusters, num_clusters))
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            delta, _ = _best_merge(subtours[i], subtours[j], cost_matrix)
            macro_cost[i, j] = macro_cost[j, i] = delta

    order = _macro_tour_order(macro_cost)

    current = subtours[order[0]]
    for idx in range(1, num_clusters):
        _, current = _best_merge(current, subtours[order[idx]], cost_matrix)

    return current


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


def divide_and_conquer_decoder_v2(
    prob_matrix, start_node=0, coords=None, max_subgraph_size=MAX_SUBGRAPH_SIZE,
    seed=42, apply_two_opt=False, partition_method="louvain", merge_method="greedy",
):
    """依機率驅動的分治解碼器（實驗版，供小規模測試）：

    1) Cost：W_ij = -log(P_ij + eps)，機率轉成成本。
    2) Divide：直接用機率矩陣（對稱化後）做圖分割，而非座標。
       - partition_method="louvain"：社群偵測，社群大小可能不平均。
       - partition_method="bisection"：遞迴 Kernighan-Lin 對半切，子圖大小較平均。
    3) Conquer：每個子圖用 Held-Karp bitmask DP 精確求解出最優子迴圈。
    4) Merge：把子迴圈接成一條 Global Tour。
       - merge_method="greedy"：Bottom-up，每次挑 Delta 最小的一對合併。
       - merge_method="hierarchical"：先解一次子圖層級的巨觀 TSP 決定串接順序，
         再依序把子迴圈接起來，巨觀結構通常更平滑。
    5) 2-opt：合併後的路線再做局部優化，清掉子圖合併處殘留的交叉／長邊。

    有座標時 2-opt 用實際距離；沒有座標時退回用機率成本矩陣當作距離依據。
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    num_nodes = prob_matrix.shape[0]

    if num_nodes <= 1:
        return [start_node, start_node]

    cost_matrix = compute_cost_matrix(prob_matrix)

    if num_nodes <= max_subgraph_size:
        cycle = held_karp_tsp(list(range(num_nodes)), cost_matrix)
    else:
        if partition_method == "bisection":
            clusters = partition_balanced_bisection(prob_matrix, max_subgraph_size, seed)
        else:
            clusters = partition_by_probability(prob_matrix, max_subgraph_size, seed)

        subtours = [held_karp_tsp(cluster, cost_matrix) for cluster in clusters]

        if merge_method == "hierarchical":
            cycle = merge_subtours_hierarchical(subtours, cost_matrix)
        else:
            cycle = merge_subtours(subtours, cost_matrix)

    if start_node in cycle:
        start_index = cycle.index(start_node)
        cycle = cycle[start_index:] + cycle[:start_index]

    cycle.append(cycle[0])

    if apply_two_opt:
        if coords is not None:
            coords_np = np.asarray(coords, dtype=np.float64)
            distance_matrix = np.linalg.norm(
                coords_np[:, None, :] - coords_np[None, :, :], axis=-1
            )
        else:
            distance_matrix = (cost_matrix + cost_matrix.T) / 2.0
        cycle = two_opt(cycle, distance_matrix)

    return cycle



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
