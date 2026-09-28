# DIFUSCO TSP 專題

本專題以 [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO) 的 categorical edge diffusion 為基礎：將 TSP tour 表示成城市間的邊矩陣，逐步加入離散雜訊，再用 GNN 預測乾淨的 tour 邊。推論時從隨機邊狀態反向取樣，最後由 divide-and-conquer decoder 輸出 Hamiltonian cycle。方法參考論文 [Graph-based Diffusion Solvers for Combinatorial Optimization](https://arxiv.org/abs/2302.08224)。

本專案用純 PyTorch 實作 dense GNN 和 categorical diffusion，不依賴 PyTorch Geometric、torch-sparse 或 Cython。這是參考 DIFUSCO 方法的精簡實作，不是官方程式碼或官方 checkpoint 的直接移植。

## 專案內容

- `code/difusco_model.py`：GNN 邊去噪模型與分類式擴散程序。
- `code/train.py`：訓練、驗證、checkpoint 選擇和訓練後評估。
- `code/tune.py`：用 Optuna 搜尋訓練超參數。
- `code/evaluate.py`、`code/evaluation.py`：計算 route gap、解碼路線、輸出圖表。
- `code/generate_tsp.py`：產生座標和 OR-Tools 參考路線。
- `results/tsp_dataset_lite.npz`：目前使用的 TSP 資料集。

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

## 2. 執行 smoke test

在專案根目錄執行：

```powershell
cd .\code
..\.venv\Scripts\python.exe train.py --smoke-test
```

Smoke test 使用資料集中的 3 個 494 城市實例，執行一個訓練 epoch、驗證、反向擴散、路線解碼和圖表輸出。它用來確認整條流程能跑通；單筆測試資料的 mean gap 不能代表模型整體表現。

## 3. Optuna 調參

```powershell
cd .\code
..\.venv\Scripts\python.exe tune.py --trials 5 --samples 12 --epochs 1
```

Optuna 搜尋 learning rate、weight decay、hidden dimension、GNN 層數、訓練擴散步數及推論步數。每個 trial 以 validation set 的下式作為目標，取最小值：

```text
mean gap (%) + 0.1 × p95 gap (%)
```

調參時使用 validation set；test set 保留給最後的成果評估。預設參數適合快速檢查 Optuna 流程。正式搜尋可增加資料筆數、epoch 和 trial 數，例如：

```powershell
..\.venv\Scripts\python.exe tune.py --trials 20 --samples 100 --epochs 3
```

最佳設定寫入 `results/optuna/best_params.json`。正式訓練時載入該設定：

```powershell
..\.venv\Scripts\python.exe train.py --use-best-params --epochs 20
```

## 4. 完整訓練

若已有 `results/tsp_dataset_lite.npz`，直接訓練：

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

如果需要重新產生資料，`generate_tsp.py` 預設會產生 1,000 筆、每筆 494 城市，OR-Tools 每筆最多搜尋 30 秒，整批可能需要很長時間。可先用小規模資料測試：

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

三張案例圖會疊畫 DIFUSCO 解出的路線和 OR-Tools 參考路線，圖例列出兩條路線長度，標題標出 gap 和路線長度差（公里）。Histogram 也會列出平均 gap 與平均公里差。`gap_metrics.json` 另外保存平均路線長度、平均 OR-Tools 路線長度和 `mean_extra_km_vs_ortools`；正值表示 DIFUSCO 多走的公里數，負值表示少走的公里數。座標依本專題的 Taichung 區域設定，以公里為單位。

`results/runs/latest.txt` 記錄最近一次訓練的 run ID。要重新計算圖表，可執行：

```powershell
..\.venv\Scripts\python.exe evaluate.py --run latest --split test --samples 30
```

`--split validation` 可改評估 validation set；`--samples 0` 會評估所選 split 的全部資料。

## 注意事項

目前模型使用 dense 邊矩陣，記憶體和運算量會隨城市數平方增加。Smoke test 只確認流程可執行；正式模型品質需以較完整的 Optuna 搜尋和 test set 結果判斷。
