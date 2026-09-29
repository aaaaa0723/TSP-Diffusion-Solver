# DIFUSCO TSP 專題

目前預設實驗改為 **1,000 筆 × 每筆 300 間台中 7-ELEVEN，OR-Tools 每筆 30 秒**，資料集為 `results/tsp_dataset_7eleven_300nodes_30s.npz`。下方 100 點／5 秒的實測為先前實驗結果，不代表新設定的表現。

在專案根目錄執行 `python code/run_300_experiment.py`，會依序生成資料、檢查 1,000 條參考路線、重新訓練 20 epochs，再評估全部 150 筆 test 樣本。狀態與完整日誌位於 `results/experiment_300nodes_30s/`；成功後更新 `results/runs/latest.txt`。若中途停止，生成階段可沿用 checkpoint；訓練階段重新開始。已存在的完整 300 點／30 秒資料集通過檢查後會直接供新訓練使用。

本專題以 [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) 的 categorical edge diffusion 為基礎：將 TSP tour 表示成城市間的邊矩陣，逐步加入離散雜訊，再用 GNN 預測乾淨的 tour 邊。推論時從隨機邊狀態反向取樣，最後由 divide-and-conquer decoder 輸出 Hamiltonian cycle。方法參考論文 [Graph-based Diffusion Solvers for Combinatorial Optimization](https://arxiv.org/abs/2302.08224)。

本專案用純 PyTorch 實作 dense GNN 和 categorical diffusion，不依賴 PyTorch Geometric、torch-sparse 或 Cython。這是參考 DIFUSCO 方法的精簡實作，不是官方程式碼或官方 checkpoint 的直接移植。

## 專案內容

- `code/difusco_model.py`：GNN 邊去噪模型與分類式擴散程序。
- `code/train.py`：訓練、驗證、checkpoint 選擇和訓練後評估。
- `code/tune.py`：用 Optuna 搜尋訓練超參數。
- `code/evaluate.py`、`code/evaluation.py`：計算 route gap、解碼路線、輸出圖表。
- `code/generate_tsp.py`：產生座標和 OR-Tools 參考路線。
- `results/tsp_dataset_7eleven_300nodes_30s.npz`：目前預設使用的 TSP 資料集；`tsp_dataset_7eleven.npz` 保留先前 100 點／5 秒資料。

## 1. 設定環境

Windows 請安裝 Python 3.11，於專案根目錄開啟 PowerShell：

```powershell
.\setup_windows.ps1
```

啟用虛擬環境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安裝腳本依 `code/requirements.txt` 安裝 PyTorch、NumPy、OR-Tools、scikit-learn、Matplotlib 和 Optuna，並檢查 CUDA。沒有可用的 NVIDIA GPU 時會使用 CPU。


## OSRM 行車距離與台中市範圍

本專案現在只接受 OSRM `driving` 距離矩陣：資料生成、OR-Tools 參考解、模型的邊距離特徵、decoder 的分群與 2-opt、以及 best/average/worst 的公里數均使用同一份 OSRM 矩陣。舊版資料集與 checkpoint 不能混用；請在重新生成資料後重新訓練。

OSRM 是一個 HTTP 路網服務，不能只執行 `generate_tsp.py` 就自動出現。最方便的本機做法是 Docker。先從 [Geofabrik](https://download.geofabrik.de/asia/taiwan.html) 下載 `taiwan-latest.osm.pbf` 到 `data/osrm/`，再在專案根目錄執行：

```powershell
cd .\data\osrm
docker run --rm -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/taiwan-latest.osm.pbf
docker run --rm -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/taiwan-latest.osrm
docker run --rm -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/taiwan-latest.osrm
cd ..\..
docker compose -f .\docker-compose.osrm.yml up -d
```

待 `http://localhost:5000` 就緒後才可生成。`--max-table-size 1000` 已涵蓋預設 300 點；若調大 `TSP_NUM_NODES`，請相應調整 compose 檔。若已有自行部署的 OSRM，可設定 `$env:TSP_OSRM_URL = "http://host:5000"`，不必使用 Docker。

`generate_tsp.py` 首次執行會下載並快取 OpenStreetMap relation 2921154 的臺中市 WGS84 邊界至 `data/taichung_city_boundary.geojson`。拒絕取樣保證每個城市點在該行政邊界內；best/average/worst 圖亦會畫出相同邊界，而不是矩形框。資料來源為 [OpenStreetMap](https://www.openstreetmap.org/relation/2921154)，以 ODbL 授權。

小型重新生成範例：

```powershell
cd .\code
$env:TSP_NUM_SAMPLES = "12"
$env:TSP_NUM_NODES = "100"
$env:TSP_SOLVER_TIME_LIMIT_SECONDS = "5"
..\.venv\Scripts\python.exe generate_tsp.py
Remove-Item Env:TSP_NUM_SAMPLES, Env:TSP_NUM_NODES, Env:TSP_SOLVER_TIME_LIMIT_SECONDS
```

## 2. 執行 smoke test

在專案根目錄執行：

```powershell
cd .\code
..\.venv\Scripts\python.exe train.py --smoke-test
```

Smoke test 使用資料集中的 3 個實例，執行一個訓練 epoch、驗證、反向擴散、路線解碼和圖表輸出。它用來確認整條流程能跑通；單筆測試資料的 mean gap 不能代表模型整體表現。

## 3. Optuna 調參

```powershell
cd .\code
..\.venv\Scripts\python.exe tune.py --trials 5 --samples 12 --epochs 1 --validation-limit 2 --eval-samples 3
```

Optuna 搜尋 learning rate、weight decay、hidden dimension、GNN 層數、訓練擴散步數及推論步數。每個 trial 以 validation set 的下式作為目標，取最小值：

```text
mean gap (%) + 0.1 × p95 gap (%)
```

調參時使用 validation set；test set 保留給最後的成果評估。上方指令適合快速檢查 Optuna 流程。程式預設採用下方正式搜尋設定：20 trials、100 筆樣本、3 epochs；每個 epoch 最多用 32 筆驗證資料選擇 checkpoint，路線指標最多評估 30 筆驗證資料，實際數量不超過 validation set 大小。`--samples` 指資料筆數，不是每筆的門市數。

```powershell
..\.venv\Scripts\python.exe tune.py --trials 20 --samples 100 --epochs 3
```

最佳設定寫入 `results/optuna/best_params.json`。正式訓練時載入該設定：

```powershell
..\.venv\Scripts\python.exe train.py --use-best-params --epochs 20
```

## 4. 完整訓練

若已有 `results/tsp_dataset_7eleven_300nodes_30s.npz`，直接訓練：

```powershell
cd .\code
..\.venv\Scripts\python.exe train.py --use-best-params --epochs 20
```

不載入 Optuna 設定，改用預設值訓練：

```powershell
..\.venv\Scripts\python.exe train.py --epochs 20 --diffusion-steps 32 --inference-steps 8
```

訓練以固定亂數種子切分 70% train、15% validation、15% test；每個 epoch 最多用 32 筆 validation loss 選擇 checkpoint。可用 `--samples 100` 限制訓練使用的資料筆數，或用 `--validation-limit 0` 評估完整 validation set。`--eval-samples 30` 控制訓練後用多少筆 test 資料計算路線指標和圖表；設為 `0` 會評估完整 test set。

## 5. 資料集

資料檔包含：

- `coords`：城市座標，形狀 `[樣本數, 城市數, 2]`。
- `adjacencies`：OR-Tools 參考路線的有向鄰接矩陣，形狀 `[樣本數, 城市數, 城市數]`。

訓練時會把 tour 邊對稱化，使用 categorical diffusion 加入雜訊，並以交叉熵預測乾淨邊。OR-Tools 在時間限制內找到的路線是參考解，未必是精確最佳解；報告中的 gap 表示相對 OR-Tools 參考路線的差距。

如果需要重新產生資料，請先依「OSRM 行車距離與台中市範圍」啟動 OSRM。產生器預設會產生 1,000 筆、每筆 300 間門市，OR-Tools 每筆最多搜尋 30 秒，整批可能需要很長時間。可先用小規模資料測試：

```powershell
$env:TSP_NUM_SAMPLES = "12"
$env:TSP_NUM_NODES = "100"
$env:TSP_SOLVER_TIME_LIMIT_SECONDS = "5"
..\.venv\Scripts\python.exe generate_tsp.py
Remove-Item Env:TSP_NUM_SAMPLES, Env:TSP_NUM_NODES, Env:TSP_SOLVER_TIME_LIMIT_SECONDS
```

## 評估指標與圖表

每次訓練結束後，程式會在 `results/runs/<run_id>/` 儲存 checkpoint、參數、JSON 指標及以下圖表：

- `loss_curve.png`：每個 epoch 的 train loss 與 validation loss。
- `mean_gap_histogram.png`：test route gap 分布和 mean gap。
- `best_case_comparison.png`：相對 OR-Tools gap 最低的案例。
- `average_case_comparison.png`：gap 最接近整體平均的案例。
- `worst_case_comparison.png`：相對 OR-Tools gap 最高的案例。

三張案例圖會疊畫 DIFUSCO 解出的路線和 OR-Tools 參考路線，圖例列出兩條路線長度，標題標出 gap 和路線長度差（公里）。Histogram 也會列出平均 gap 與平均公里差。`gap_metrics.json` 另外保存平均路線長度、平均 OR-Tools 路線長度和 `mean_extra_km_vs_ortools`；正值表示 DIFUSCO 多走的公里數，負值表示少走的公里數。座標為 WGS84 經緯度，但所有公里數與最佳化成本皆為 OSRM 行車距離；案例圖會畫出台中市行政邊界。

`results/runs/latest.txt` 記錄最近一次訓練的 run ID。要重新計算圖表，可執行：

```powershell
..\.venv\Scripts\python.exe evaluate.py --run latest --split test --samples 30
```

`--split validation` 可改評估 validation set；`--samples 0` 會評估所選 split 的全部資料。

## 注意事項

目前模型使用 dense 邊矩陣，記憶體和運算量會隨城市數平方增加。Smoke test 只確認流程可執行；正式模型品質需以較完整的 Optuna 搜尋和 test set 結果判斷。

## 道路搜尋解碼與 gap 實測

預設 decoder 改為 `road_multistart`：先取得模型熱圖的分治解，再加入 16 個道路距離／模型機率引導的不同起點候選，使用有向 2-opt 與 1～3 點連續區段搬移改善路線，最多接受 200 次改善。所有候選只依輸入的道路距離選擇；參考路線僅用於最後評分，沒有提供给 decoder。每次推論固定使用 `42 + 樣本索引` 作為亂數種子。

模型 `20260929_121753` 在 1,000 筆、每筆 100 間 7-ELEVEN 資料上的實測如下，權重未重新訓練：

| 保留資料 | 筆數 | 原版解碼平均 gap | 新混合解碼平均 gap |
| --- | ---: | ---: | ---: |
| Validation | 150 | 25.88% | -1.19% |
| Test | 150 | 27.67% | -1.50% |

先驗證完整 validation set 並固定搜尋設定，再評估完整 test set。Test 的 p95 gap 為 0.41%，每筆平均額外搜尋約 0.34 秒（本次 CPU 環境）。所有測試路線均走訪各點一次並回到起點。負 gap 表示比資料中限時 5 秒的 OR-Tools 參考解更短，不表示超越精確最優解；結果限於這份資料集與目前參考解。

改善主要來自道路局部搜尋：前 30 筆 validation 的混合解碼平均 gap 為 -1.02%，不使用模型機率的道路搜尋為 -1.07%。因此，低 gap 不能單獨證明神經網路有增益，這套方法應描述為「DIFUSCO 模型加道路啟發式搜尋」。調整神經網路參數時也應保留不使用模型的對照。

結果保存在 `results/runs/20260929_121753_roadsearch/`。`decoder_benchmark_validation.json`、`decoder_benchmark_test.json` 包含每筆原版與改版路線、gap 和執行時間。`training_report.json` 是原權重的訓練紀錄，`report.json` 是新解碼的評估結果。

```bash
# 新訓練預設使用改良解碼
python code/train.py
# 重新評估全部測試資料（需要 OSRM 才能取得尚未快取的道路圖形）
python code/evaluate.py --run 20260929_121753_roadsearch --samples 0
# 僅評估指標，不查詢道路圖形
python code/evaluate.py --run 20260929_121753_roadsearch --samples 0 --no-plots
# 使用相同亂數比較原版與改版 decoder
python code/benchmark_decoder.py --run 20260929_121753_roadsearch --split validation --samples 0
```

`train.py --decoder legacy` 可保留原版解碼；`--decoder-starts` 與 `--local-search-moves` 控制新解碼搜尋量。驗證指標另存於 run 的 `validation/`，不覆蓋 test 指標。Optuna 使用同一套新解碼評估 validation，且略過圖形輸出以避免每個 trial 查詢 OSRM。


## 路線圖與道路幾何

評估圖會將路線疊加在台中市界上。若 run 內沒有快取的路線幾何，繪圖時需連線到 OSRM Route API；可先啟動前述本機 OSRM 服務，或設定 `TSP_OSRM_URL` 指向可用的 OSRM `driving` 服務。

```powershell
python code/evaluate.py --run latest
```

若 run 已有 `plot_records.json`，可只重畫圖表：

```powershell
python code/evaluate.py --run latest --plots-only
```

## 7-ELEVEN 資料

`data/taichung_7eleven_osm.json` 保存台中市 7-ELEVEN 門市目錄；OSM 節點經緯度透過 OSRM nearest 對應至可行駛道路。資料集包含座標、門市識別資訊、參考巡迴與距離矩陣。OpenStreetMap 資料依 ODbL 1.0 授權，來源為 OpenStreetMap contributors。

完整資料生成與訓練可由專案根目錄啟動：

```powershell
python code/run_300_experiment.py
```

此流程預設產生 1,000 筆、每筆 300 間門市，OR-Tools 每筆最多搜尋 30 秒，接著訓練 20 epochs 並評估 150 筆 test 樣本。可用 `TSP_NUM_SAMPLES`、`TSP_NUM_NODES`、`TSP_SOLVE_SECONDS` 與 `TSP_NUM_WORKERS` 調整資料生成設定；輸出資料集預設約 848 MB，因此不納入 GitHub。本分支也不包含 300 MB 以上的 OSM PBF 原始檔；請依 OSRM 章節從 Geofabrik 取得。

## 最新實驗

模型使用 `road_multistart` decoder，16 個起始候選、最多 200 次局部搜尋移動，資料切分為 700 train、150 validation、150 test。以下數字來自不同評估範圍，請勿直接視為同一規模的比較：

| Run | Test 樣本 | Mean gap | P95 gap | 路線有效性 |
| --- | ---: | ---: | ---: | --- |
| `20260929_125724_300nodes_30s` | 150 | -1.684% | 0.188% | 全部有效 |
| `20260929_191743`（最新） | 30 | -2.121% | 0.194% | 全部有效 |

Gap 是相對每筆限時 30 秒的 OR-Tools 參考路線計算；負值表示該次評估的路線短於參考 heuristic，不代表已證明達到 TSP 全域最優。完整 checkpoint、JSON 指標與圖表保存在各 run 目錄；`results/runs/latest.txt` 指向最新 run。大型資料集及 OSM PBF 未提交，重新執行完整實驗前需先生成資料並啟動 OSRM。
