import math
import numpy as np

COST_EPSILON = 1e-9
MAX_SUBGRAPH_SIZE = 12
BEAM_WIDTH = 2
CANDIDATE_LIMIT = 2


def _symmetric_road_cost(distance_matrix):
    return (distance_matrix + distance_matrix.T) * 0.5


def _recursive_road_bisection(nodes, road_cost, max_size):
    """Split nodes using only OSRM driving costs, never coordinate distances."""
    if len(nodes) <= max_size:
        return [nodes]
    subset = np.asarray(nodes, dtype=int)
    local = road_cost[np.ix_(subset, subset)]
    left_anchor = 0
    right_anchor = int(np.argmax(local[left_anchor]))
    if right_anchor == left_anchor:
        midpoint = len(nodes) // 2
        return (_recursive_road_bisection(nodes[:midpoint], road_cost, max_size) +
                _recursive_road_bisection(nodes[midpoint:], road_cost, max_size))
    left, right = [], []
    for local_index, node in enumerate(nodes):
        (left if local[local_index, left_anchor] <= local[local_index, right_anchor] else right).append(node)
    if not left or not right:
        midpoint = len(nodes) // 2
        left, right = nodes[:midpoint], nodes[midpoint:]
    return (_recursive_road_bisection(left, road_cost, max_size) +
            _recursive_road_bisection(right, road_cost, max_size))


def _cluster_road_distance(nodes_a, nodes_b, road_cost):
    return float(road_cost[np.ix_(nodes_a, nodes_b)].mean())


def divide_and_conquer_kmeans_beam_decoder(
    prob_matrix, start_node=0, distance_matrix=None, max_subgraph_size=MAX_SUBGRAPH_SIZE,
    apply_two_opt=True, beam_width=BEAM_WIDTH, candidate_limit=CANDIDATE_LIMIT
):
    """Road-cost bisection + beam decoder; every cost-based choice uses OSRM."""
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    num_nodes = prob_matrix.shape[0]
    if prob_matrix.ndim != 2 or prob_matrix.shape[1] != num_nodes:
        raise ValueError("prob_matrix must be a square matrix")
    if not 0 <= start_node < max(num_nodes, 1):
        raise ValueError("start_node is outside prob_matrix")
    if num_nodes <= 1:
        return [start_node, start_node]
    if distance_matrix is None:
        raise ValueError("Decoder requires an OSRM driving distance matrix")
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    if distance_matrix.shape != (num_nodes, num_nodes):
        raise ValueError("distance_matrix and prob_matrix must have matching square shapes")
    road_cost = _symmetric_road_cost(distance_matrix)
    cost_matrix = compute_cost_matrix(prob_matrix)
    all_nodes = list(range(num_nodes))
    leaf_node_lists = _recursive_road_bisection(all_nodes, road_cost, max_subgraph_size)
    leaves = [held_karp_tsp(nodes, cost_matrix) for nodes in leaf_node_lists]
    cluster_dist = np.asarray([[_cluster_road_distance(a, b, road_cost) for b in leaf_node_lists]
                               for a in leaf_node_lists])
    start_cluster = next(i for i, leaf in enumerate(leaf_node_lists) if start_node in leaf)
    beam = [(0.0, start_cluster, frozenset([start_cluster]), leaves[start_cluster])]
    for _ in range(len(leaves) - 1):
        new_beam = []
        for score, curr_idx, visited, cycle in beam:
            unvisited = [i for i in range(len(leaves)) if i not in visited]
            unvisited.sort(key=lambda x: cluster_dist[curr_idx, x])
            for nxt_idx in unvisited[:candidate_limit]:
                delta, next_cycle = _best_merge(cycle, leaves[nxt_idx], cost_matrix)
                new_beam.append((score + delta, nxt_idx, visited | frozenset([nxt_idx]), next_cycle))
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
    best_cycle = beam[0][3]
    start_index = best_cycle.index(start_node)
    final_cycle = best_cycle[start_index:] + best_cycle[:start_index] + [start_node]
    return two_opt(final_cycle, distance_matrix) if apply_two_opt else final_cycle


def divide_and_conquer_decoder(prob_matrix, start_node=0, distance_matrix=None,
                               decoder="road_multistart", starts=16, max_moves=200):
    initial = divide_and_conquer_kmeans_beam_decoder(
        prob_matrix, start_node, distance_matrix, apply_two_opt=True
    )
    if decoder == "legacy":
        return initial
    if decoder != "road_multistart":
        raise ValueError(f"Unknown decoder: {decoder}")
    return road_multistart_decoder(prob_matrix, distance_matrix, initial, start_node, starts, max_moves)


def directed_local_search(path, distance_matrix, max_moves=200):
    """Best-improvement directed 2-opt and relocation of 1--3 consecutive nodes.

    Reversal deltas include every reversed internal arc. Relocations preserve
    internal arc directions. The depot remains fixed and every move lowers cost.
    """
    distance = np.asarray(distance_matrix, dtype=np.float64)
    route = np.asarray(path[:-1], dtype=int)
    n = len(route)
    if n < 3:
        return list(path)
    i, j = np.indices((n, n))
    reversal_mask = (i >= 1) & (j > i)
    for _ in range(max_moves):
        prev, nxt = np.roll(route, 1), np.roll(route, -1)
        difference = distance[nxt, route] - distance[route, nxt]
        prefix = np.concatenate(([0.0], np.cumsum(difference)))
        reversal = (distance[prev[:, None], route[None, :]]
                    + distance[route[:, None], nxt[None, :]]
                    - distance[prev, route][:, None] - distance[route, nxt][None, :]
                    + prefix[j] - prefix[i])
        reversal[~reversal_mask] = np.inf
        a, b = np.unravel_index(np.argmin(reversal), reversal.shape)
        best_delta = reversal[a, b]
        move = ("reverse", a, b)
        for length in (1, 2, 3):
            if length >= n - 1:
                continue
            first = np.arange(1, n - length + 1)
            last = first + length - 1
            u, v = route[first - 1], route[(last + 1) % n]
            head, tail = route[first], route[last]
            delta = ((distance[u, v] - distance[u, head] - distance[tail, v])[:, None]
                     + distance[route[None, :], head[:, None]]
                     + distance[tail[:, None], nxt[None, :]]
                     - distance[route, nxt][None, :])
            positions = np.arange(n)[None, :]
            invalid = (positions >= (first - 1)[:, None]) & (positions <= last[:, None])
            delta[invalid] = np.inf
            row, after = np.unravel_index(np.argmin(delta), delta.shape)
            if delta[row, after] < best_delta:
                best_delta = delta[row, after]
                move = ("relocate", int(first[row]), length, int(route[after]))
        if best_delta >= -1e-9:
            break
        if move[0] == "reverse":
            route[move[1]:move[2] + 1] = route[move[1]:move[2] + 1][::-1]
        else:
            _, first, length, after_node = move
            block = route[first:first + length].copy()
            remaining = np.concatenate((route[:first], route[first + length:]))
            position = int(np.flatnonzero(remaining == after_node)[0]) + 1
            route = np.concatenate((remaining[:position], block, remaining[position:]))
    return route.tolist() + [int(route[0])]


def road_multistart_decoder(prob_matrix, distance_matrix, initial, start_node=0,
                           starts=16, max_moves=200):
    """Neural seed plus road/heatmap-guided greedy starts, selected by road cost.

    No reference adjacency or reference tour is available to this function.
    This is a hybrid decoder, not a claim about raw neural model quality.
    """
    distance = np.asarray(distance_matrix, dtype=np.float64)
    n = len(distance)
    if starts < 1 or max_moves < 0:
        raise ValueError("starts must be positive and max_moves must be nonnegative")
    if distance.shape != (n, n) or not np.isfinite(distance).all() or (distance < 0).any():
        raise ValueError("Road costs must be a finite nonnegative square matrix")
    best = directed_local_search(initial, distance, max_moves)
    best_cost = calculate_path_distance(best, distance)
    probabilities = np.clip(np.asarray(prob_matrix), 1e-6, 1)
    penalty = -np.log(probabilities)
    penalty /= max(float(np.mean(penalty)), 1e-6)
    for trial, first in enumerate(np.linspace(0, n - 1, min(starts, n), dtype=int)):
        # Alternate pure road and learned-edge guided starts for diversity.
        costs = distance * (1 + (0.15 if trial % 2 else 0.0) * penalty)
        route, remaining = [int(first)], np.ones(n, dtype=bool)
        remaining[first] = False
        while remaining.any():
            candidates = np.flatnonzero(remaining)
            node = int(candidates[np.argmin(costs[route[-1], candidates])])
            route.append(node)
            remaining[node] = False
        offset = route.index(start_node)
        route = route[offset:] + route[:offset] + [start_node]
        candidate = directed_local_search(route, distance, max_moves)
        cost = calculate_path_distance(candidate, distance)
        if cost < best_cost:
            best, best_cost = candidate, cost
    return best

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
    """Directed 2-opt evaluated entirely with OSRM costs, including reversed arcs."""
    path = list(path)
    n = len(path) - 1
    for _ in range(max_passes):
        forward = np.asarray([distance_matrix[path[k], path[k + 1]] for k in range(n)])
        reverse = np.asarray([distance_matrix[path[k + 1], path[k]] for k in range(n)])
        forward_prefix = np.concatenate(([0.0], np.cumsum(forward)))
        reverse_prefix = np.concatenate(([0.0], np.cumsum(reverse)))
        current_cost = float(forward_prefix[-1])
        best_cost, best_pair = current_cost, None
        for i in range(1, n - 1):
            a, b = path[i - 1], path[i]
            for j in range(i + 1, n):
                c, d = path[j], path[j + 1]
                candidate_cost = (current_cost - distance_matrix[a, b] - distance_matrix[c, d]
                                  - (forward_prefix[j] - forward_prefix[i])
                                  + distance_matrix[a, c]
                                  + (reverse_prefix[j] - reverse_prefix[i])
                                  + distance_matrix[b, d])
                if candidate_cost + 1e-9 < best_cost:
                    best_cost, best_pair = candidate_cost, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        path = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
    return path


def calculate_path_distance(path, distance_matrix):
    """Return a route length in kilometres from its precomputed cost matrix."""
    return sum(
        float(distance_matrix[path[index], path[index + 1]])
        for index in range(len(path) - 1)
    )
