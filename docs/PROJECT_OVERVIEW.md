# TSP-Diffusion-Solver 專案總覽

## 1. 專案宗旨

這個專案的目標，是用圖神經網路（GNN）來學習 494 節點的旅行商問題（TSP）路線預測，並用 OR-Tools 產生參考路線，用來衡量模型預測結果與參考解的差距。

它本質上是一個「supervised graph learning + route evaluation」專案，而不是完整的 diffusion model。從目前程式碼來看，模型是 GNN 邊預測器，最後用 greedy decoder 生成一個巡迴路徑，再與 OR-Tools 產生的 reference route 比較。

---

## 2. 目前狀態與進度

目前專案已經具備以下狀態：

- 已經有資料集檔：`tsp_dataset_lite.npz`
- 已經有模型權重：`tsp_gnn_model.pth`
- 已經有輸出圖表：`loss_curve.png`、`error_histogram.png`、`worst_case_comparison.png`
- 已經有訓練與調參腳本，流程基本完整
- 現有模型屬於一個可運行的 baseline，不代表已經真正解決 494 節點 TSP
- 文件 `TRAINING_FLOW_COMPARISON.md` 也明確指出：目前結果是「有效 baseline」，但不是最終優化解

因此，這個專案目前進度大致可判定為：

- 基礎資料生成與訓練 pipeline 已完成
- 訓練/評估/比較/圖表輸出已經串起來
- 仍處於研究原型階段，而非工程化落地方案

---

## 3. 各檔案分別在幹嘛

### 3.1 `generate_tsp.py`

功能：產生 TSP 訓練資料。

這個檔案做的事情是：

1. 在一個模擬的台中市近似面積範圍內，隨機產生 494 個點
2. 依照 Euclidean distance 計算距離矩陣
3. 用 OR-Tools 進行 TSP heuristic 求解，得到 reference route
4. 將每一個樣本轉成：
   - 座標 `coords`
   - adjacency matrix `adjacencies`
5. 儲存成 `tsp_dataset_lite.npz`

也就是說，這個檔案是資料生成器，負責建立訓練資料集；沒有它，後面的訓練資料就沒有來源。

---

### 3.2 `dataset.py`

功能：定義 `TSPDataset`，把 `.npz` 資料讀成 PyTorch Dataset。

它的工作：

- 讀取 `coords` 和 `adjacencies`
- 轉成 `torch.Tensor`
- 透過 `torch.cdist` 在訓練時即時計算點與點之間的距離矩陣
- 對外提供 `__len__()`、`__getitem__()`，讓 DataLoader 能直接讀資料

這檔案相當於資料處理層，是整個學習 pipeline 的入口。

---

### 3.3 `model.py`

功能：定義模型架構。

核心內容：

- `SimpleGCNLayer`：手刻的 GCN 層，透過 adjacency 進行訊息聚合
- `TSPPureGNNModel`：
  - 編碼節點座標
  - 用兩層 GCN 交換節點資訊
  - 計算每個 edge 的分數（bilinear scorer）
  - 回傳每對節點的 logits

這個模型的直覺是：

- 節點會透過 GCN 交換鄰近節點資訊
- 再用 edge scoring 判斷某個城市對某個城市是否應該連接
- 最終希望預測出 TSP 路線的連接矩陣

注意：它並不是標準 diffusion model，而是 GNN-based edge predictor。

---

### 3.4 `train.py`

功能：正式訓練模型。

它做的工作：

1. 判斷使用 GPU 或 CPU
2. 從 `TSPDataset` 讀資料
3. 依照 70/15/15 切分訓練/驗證/測試資料
4. 建立 DataLoader
5. 建立 optimizer（AdamW）
6. 每個 epoch 進行：
   - 訓練
   - 驗證
   - 計算 mean gap / p95 gap
7. 保存最佳模型 `tsp_gnn_model.pth`
8. 繪製 `loss_curve.png`

這是整個專案最核心的訓練腳本；幾乎所有模型訓練流程都集中在這裡。

---

### 3.5 `tune.py`

功能：超參數調優（Optuna）。

它會：

- 使用 Optuna 搜尋 `hidden_dim`、`learning_rate`、`weight_decay`
- 針對驗證集評估 objective
- 儲存最佳參數
- 最後產生最佳設定供正式訓練使用

這個檔案主要是讓訓練更穩定、效果更佳，也方便比較不同超參數組合。

---

### 3.6 `evaluation.py`

功能：定義評估邏輯。

這裡包含：

- `greedy_decoder`：從 edge probability matrix 生成路線
- `adjacency_to_path`：把 reference adjacency 轉成路徑
- `calculate_path_distance`：計算路線總長
- `evaluate_route_gap`：計算 AI route 與 reference route 的 gap

它是專案的「評估器」，用來判斷模型表現是否有改善。

---

### 3.7 `test_histogram.py`

功能：統計誤差分佈並輸出直方圖。

這個腳本會：

1. 讀取測試集
2. 在測試資料上評估模型
3. 計算每個樣本的 route gap
4. 畫出 `error_histogram.png`

這能幫助了解模型在整體測試集上的分佈狀況，而不只是單一案例。

---

### 3.8 `plot_qualitative.py`

功能：找出最差案例，畫出比較圖。

它會：

- 遍歷測試集
- 找出最大 gap 的案例
- 比較：
  - OR-Tools reference path
  - 模型預測 path
- 輸出 `worst_case_comparison.png`

這個檔案是「質化分析」工具，針對最糟的案例去看模型到底錯在哪裡。

---

### 3.9 `TRAINING_FLOW_COMPARISON.md`

功能：記錄專案設計變更與訓練流程演進。

這份文件說明：

- 以前版本有哪些問題
- 現在版本做了哪些修正
- 目前模型實際是什麼架構
- 這些變更帶來了什麼效果

它比較像研究筆記 / 設計文件，是理解專案歷史與設計脈絡的重要文檔。

---

### 3.10 `requirements.txt`

功能：列出所有 Python 依賴。

套件包括：

- numpy
- torch
- matplotlib
- optuna
- ortools
- seaborn
- tqdm

這份檔案是安裝環境的核心依據。

---

### 3.11 `activate_tsp_env.sh`

功能：幫你快速啟動 Conda 環境。

它會：

- source conda shell integration
- activate `tsp_env`

這個腳本很適合在遠端 SSH 或 server 環境中快速切到專案環境。

---

## 4. 整個專案在幹嘛

整個專案的大流程可以概括為：

1. 生成 TSP 例子
2. 用 OR-Tools 給出 reference route
3. 用 GNN 預測所有 city-to-city 連接機率
4. 利用 greedy decoder 轉成一條巡迴路徑
5. 計算該路徑與 reference route 的 gap
6. 透過訓練、調參、畫圖迭代改進

換句話說，這個專案並不是直接訓練一個「規劃器」，而是：

- 先學一個圖結構上的連接概率模型
- 再從這些機率推導出路線
- 最後用 TSP 評估標準檢視表現

---

## 5. 目前專案的優點與限制

### 優點

- 流程完整：資料生成、模型訓練、評估、比較、圖表輸出都有
- 依賴明確，便於在新環境復現
- 能夠直接產生研究效果與視覺化結果

### 限制

- `model.py` 目前仍是 GNN，不是真正 diffusion-based workflow
- 使用 greedy decoder，沒有嚴格保障 TSP 的一筆合法 Hamiltonian cycle
- OR-Tools 所產生的是 heuristic reference，不是 exact optimum
- 目前結果屬於 baseline，未證明已達到高品質解

---

## 6. 如果要繼續往前做

下一步最合理的方向是：

1. 把 edge probability 轉成更嚴格的 assignment 問題
2. 加入 constrained decoding / TSP projection
3. 改進模型結構與 training objective
4. 讓評估更接近真正的 TSP 優化品質

也就是說，這個專案現在更像「研究基線 + 量化分析框架」，而不是工程成熟的商業級求解器。

---

## 7. 一句話總結

這個專案的核心任務是：利用圖神經網路學習 TSP 邊的連接關係，並用 OR-Tools 產生 reference path，最後檢測模型預測與理想路線之間的差距。它現在已經有完整 pipeline，並且處於「可訓練的研究 baseline」階段。
