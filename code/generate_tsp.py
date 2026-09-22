import numpy as np
from multiprocessing import Pool, cpu_count
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

from experiment_utils import DATASET_PATH

NUM_SAMPLES = 1000
NUM_NODES = 494
TAICHUNG_AREA_KM2 = 2215
TAICHUNG_SIDE_KM = np.sqrt(TAICHUNG_AREA_KM2)

def create_data_model(dist_matrix_int):
    data = {}
    data['distance_matrix'] = dist_matrix_int.tolist()
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def solve_single_tsp(seed):
    np.random.seed(seed) 
    
    # 1. 在等面積的台中市近似正方形範圍內隨機撒點，座標單位為公里
    coords = (np.random.rand(NUM_NODES, 2) * TAICHUNG_SIDE_KM).astype(np.float32)
    
    # 2. 算公里直線距離，交給 OR-Tools 找高品質 heuristic reference route
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    dist_matrix_int = (dist_matrix * 1000).astype(int)
    
    data = create_data_model(dist_matrix_int)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30
    
    solution = routing.SolveWithParameters(search_parameters)
    
    # 3. 建立 0/1 相鄰矩陣 (Ground Truth)
    adj_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.int8)
    
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            from_node = manager.IndexToNode(index)
            index = solution.Value(routing.NextVar(index))
            to_node = manager.IndexToNode(index)
            adj_matrix[from_node][to_node] = 1
    else:
        raise RuntimeError(f"OR-Tools failed to find a route for seed {seed}")
            
    # 不再回傳 dist_matrix，只留座標跟解答
    return coords, adj_matrix

def main():
    cores = min(cpu_count(), 16)
    print(f"🚀 啟動 TSP reference route 生成器 (核心數: {cores})...")
    
    start_time = time.time()
    seeds = list(range(NUM_SAMPLES))
    
    with Pool(cores) as pool:
        results = pool.map(solve_single_tsp, seeds)
        
    print(f"✅ 運算完畢！耗時: {time.time() - start_time:.2f} 秒")
    
    # 只收集座標和相鄰矩陣
    all_coords = np.array([r[0] for r in results], dtype=np.float32)
    all_adjs = np.array([r[1] for r in results], dtype=np.int8)
    
    print("💾 準備進行 Byte-level 寫入，存檔中...")
    
    # 存成瘦身版的 numpy 壓縮檔
    np.savez_compressed(
        DATASET_PATH, 
        coords=all_coords, 
        adjacencies=all_adjs
    )
    
    print(f"🎉 存檔完成！獲得 {NUM_SAMPLES} 筆、每筆 {NUM_NODES} 節點的台中市 TSP 資料")
    print(f"📊 檔案維度確認 -> 座標: {all_coords.shape}, 標籤: {all_adjs.shape}")

if __name__ == '__main__':
    main()