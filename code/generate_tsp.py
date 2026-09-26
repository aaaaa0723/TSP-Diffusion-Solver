import numpy as np
from multiprocessing import Pool, cpu_count
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

from experiment_utils import DATASET_PATH

NUM_SAMPLES = 10
NUM_NODES = 494

# 台中市大致的經緯度範圍 (Lat: 緯度, Lon: 經度)
MIN_LAT, MAX_LAT = 24.1, 24.4
MIN_LON, MAX_LON = 120.5, 120.9

def create_data_model(dist_matrix_int):
    data = {}
    data['distance_matrix'] = dist_matrix_int.tolist()
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def solve_single_tsp(seed):
    np.random.seed(seed) 
    
    # 1. 經緯度撒點：在台中市的真實範圍內隨機產生座標 (Lat, Lon)
    lats = np.random.uniform(MIN_LAT, MAX_LAT, NUM_NODES)
    lons = np.random.uniform(MIN_LON, MAX_LON, NUM_NODES)
    coords = np.stack([lats, lons], axis=1).astype(np.float32)

    # 2. Vectorized Haversine 計算真實球面距離 (取代原本的 Euclidean 直線距離)
    # 將經緯度轉換為弧度 (Radians) 以便計算
    coords_rad = np.radians(coords)
    lat = coords_rad[:, 0]
    lon = coords_rad[:, 1]

    dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
    dlon = lon[:, np.newaxis] - lon[np.newaxis, :]
    
    # Haversine 核心公式
    a = (np.sin(dlat / 2.0)**2 +
         np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * np.sin(dlon / 2.0)**2)

    # 【防呆機制】：浮點數運算可能產生 1.0000000002，會讓 arcsin 報錯 (NaN)，強制夾在 0~1 之間
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))

    # 6371 是地球平均半徑 (公里)，所以 dist_matrix 的單位是「公里」
    dist_matrix = 6371.0 * c

    # 轉換成「公尺」並轉為整數，交給 OR-Tools 處理 (它最喜歡整數了)
    dist_matrix_int = (dist_matrix * 1000).astype(int)
    
    # 3. 呼叫 OR-Tools 找高品質 Reference Route (以下邏輯跟原本完全一樣)
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
    
    # 4. 建立 0/1 相鄰矩陣 (Ground Truth)
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
            
    return coords, adj_matrix

def main():
    cores = min(cpu_count(), 16)
    print(f"🌍 啟動【真實經緯度版】TSP reference route 生成器 (核心數: {cores})...")
    
    start_time = time.time()
    seeds = list(range(NUM_SAMPLES))
    
    with Pool(cores) as pool:
        results = pool.map(solve_single_tsp, seeds)
        
    print(f"✅ 運算完畢！耗時: {time.time() - start_time:.2f} 秒")
    
    all_coords = np.array([r[0] for r in results], dtype=np.float32)
    all_adjs = np.array([r[1] for r in results], dtype=np.int8)
    
    print("💾 準備進行 Byte-level 寫入，存檔中...")
    
    np.savez_compressed(
        DATASET_PATH, 
        coords=all_coords, 
        adjacencies=all_adjs
    )
    
    print(f"🎉 存檔完成！獲得 {NUM_SAMPLES} 筆、每筆 {NUM_NODES} 節點的真實經緯度 TSP 資料")

if __name__ == '__main__':
    main()