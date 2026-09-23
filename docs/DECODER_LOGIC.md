# TSP Decoder 邏輯說明（現況整理）

本文件說明 `code/evaluation.py` 裡目前所有的路徑解碼（decoder）邏輯，
包含「正式流程實際在用的版本」與「還在小規模測試階段、尚未接入正式訓練/評估流程的實驗版本」。

---

## 0. 大背景：Decoder 在整個系統裡的角色

GNN 模型（`TSPPureGNNModel`）針對每一題 TSP，輸出一個 `N x N` 的**邊機率矩陣** `P`，
`P[i][j]` 代表模型認為「從節點 i 走到節點 j」這條邊，屬於最終最佳路線的機率。

但模型輸出的只是機率，**不保證**這些高機率的邊湊起來會是一條合法的
Hamiltonian Cycle（每個節點恰好經過一次、最後回到起點）。Decoder 的工作就是：

> 把一個「N x N 邊機率矩陣」，轉換成一條「合法且盡量短」的實際路線。

目前檔案裡有兩套完全獨立的 decoder 實作：

| decoder | 狀態 | 用在哪裡 |
|---|---|---|
| `divide_and_conquer_decoder`（下稱 **v1**） | ✅ 正式使用中 | `train.py`（訓練時的 validation route gap）、`test_histogram.py`、`plot_qualitative.py` |
| `divide_and_conquer_decoder_v2`（下稱 **v2**） | 🧪 實驗中，尚未接入任何正式流程 | 目前只在手動的小規模測試腳本裡跑過 |

---

## 1. 正式使用中的 Decoder：`divide_and_conquer_decoder`（v1）

這是目前 `train.py` 算 route gap、`test_histogram.py`、`plot_qualitative.py` 實際呼叫的版本。
核心精神是：**用機率本身決定「怎麼切」，而不是用座標空間切**，讓解碼品質確實跟著模型訓練狀況走。

### 1.1 事前查表：`_DecodeContext`

解碼開始前，先把整趟解碼會重複用到的資料算好、存成 Python list（不是 numpy array，
避免每次存取都要付 numpy 的呼叫開銷）：

- `prob`：機率矩陣本身
- `outgoing_heat[i]`：節點 i 對外所有邊機率的平均值（代表這個節點整體有多「熱門」）
- `incoming_heat[i]`：節點 i 被指向的所有邊機率平均值
- `dist` / `distance_scale`（若有給座標）：兩兩節點的歐氏距離，以及「平均最近鄰距離」作為正規化尺度

### 1.2 評分函式

- `_edge_score(ctx, a, b)`：一條邊的分數 = 邊機率本身 + 一點點 `outgoing_heat` / `incoming_heat` 當 tie-breaker（權重極小，只在機率打平時才有影響）
- `_connection_score(ctx, a, b)`：在 `_edge_score` 基礎上，若有座標，再扣掉「正規化後的距離 × `DISTANCE_PENALTY_WEIGHT`（預設 0.15）」。這一項權重刻意設得很小，只當作 tie-breaker，避免蓋掉模型學到的機率訊號（先前版本權重設太大，曾導致 route gap 完全跟模型訓練狀況脫鉤）。

### 1.3 Step 1 — 建立初始順序：`_build_initial_order`

以 `start_node` 為起點，每一步貪婪選「跟目前節點 `_connection_score` 最高」的未拜訪節點，
走訪完所有節點，得到一個初始排列 `initial_order`。

這一步是讓最終路徑品質「跟著模型訓練狀況走」的關鍵：模型學得越好，機率矩陣越準，
這個初始順序就越接近真實最佳路線。

### 1.4 Step 2 — 分治重排：`solve()` 遞迴 + `_solve_small_group`

把 `initial_order` 這個**排列的位置**（不是座標！）對半切：

- 片段長度 ≤ `MIN_GROUP_SIZE`（預設 6）：呼叫 `_solve_small_group`，
  窮舉這個片段內所有排列組合，選出 `_path_score` 總分最高的排法。
  這一步修正了純貪婪鏈在局部產生的次優解（貪婪本身有「鑽牛角尖」的問題）。
- 片段長度 > `MIN_GROUP_SIZE`：對半切成左右兩段，分別遞迴 `solve()`，
  再呼叫 `_merge_paths` 把兩段接回一條。

### 1.5 Step 3 — 合併：`_merge_paths`

左右兩段合併時，嘗試 4 種頭尾接法（左正接右正、左正接右反、左反接右正、左反接右反），
選 `_connection_score` 最高的接法。

### 1.6 目前**沒有** 2-opt 收尾

v1 目前的正式版本合併完就直接輸出，**沒有**額外的 2-opt 局部優化步驟。
（先前版本移除過 2-opt 又重新設計過決策流程，最終決定 v1 保持單純，2-opt 目前只加在 v2 裡。）

---

## 2. 實驗版 Decoder：`divide_and_conquer_decoder_v2`（尚未接入正式流程）

這是依照更嚴謹的「Cost Function → Divide → Conquer → Merge」設計重寫的版本，
目前只在手動小規模測試腳本中驗證過，**還沒有**修改 `train.py` / `test_histogram.py` / `plot_qualitative.py`
去改用它。

### 2.1 Cost Function：`compute_cost_matrix`

$$W_{ij} = -\log(P_{ij} + \epsilon)$$

把「找最大機率路徑」轉換成傳統的「找最小成本路徑」問題。`COST_EPSILON`（預設 `1e-9`）避免 `log(0)`。

另外還有 `compute_affinity_matrix`，把機率矩陣對稱化：
$A_{ij} = (P_{ij} + P_{ji}) / 2$，專門給圖分割（Divide 階段）當作無向圖的邊權重用。

### 2.2 Divide 階段：把大圖切成小子圖

提供兩種切法，用 `partition_method` 參數選：

- **`partition_method="louvain"`（預設）— `partition_by_probability`**：
  直接對「機率矩陣對稱化後」的圖跑 networkx 內建的 **Louvain 社群偵測**
  (`nx.algorithms.community.louvain_communities`)，機率高、常常一起出現的節點會被分進同一群。
  若某個社群仍超過 `MAX_SUBGRAPH_SIZE`（預設 12），用 `_split_oversized_cluster`
  遞迴呼叫 **Kernighan-Lin 對半切**（`kernighan_lin_bisection`）繼續切小。

- **`partition_method="bisection"` — `partition_balanced_bisection`**：
  略過 Louvain，直接對整張圖遞迴做 Kernighan-Lin 對半切。
  Louvain 切出來的社群大小可能極不平均（有的 2 個節點、有的 40 個），
  這個做法保證每個子圖大小都差不多，通常讓後續合併的巨觀結構更穩定。

測試結果：小規模（N ≤ 50）時 `bisection` 通常較好；大規模（N ≥ 80）`louvain` 通常較好，
沒有哪一種全面勝出。

### 2.3 Conquer 階段：`held_karp_tsp`

每個子圖規模夠小（≤ `MAX_SUBGRAPH_SIZE`），直接用 **Held-Karp bitmask 動態規划**
窮舉出「成本最低的封閉迴圈」，保證在該子圖內是最優解。

- 狀態：`dp[mask][last]` = 走過 `mask` 集合、目前停在 `last` 節點的最小成本
- 複雜度：$O(2^S \cdot S^2)$，純 Python 實作，`MAX_SUBGRAPH_SIZE` 設 12 是為了讓
  純 Python 版本可以在合理時間內跑完（$2^{12}=4096$ 個狀態）。
- 已用 N=8 小規模資料跟暴力窮舉全排列比對過，結果完全一致，證明 DP 實作正確。

> 若未來要衝更大的子圖規模（例如 S=20），純 Python 會太慢，需要照原始構想
> 改用 C++/numba 搭配位元運算與手動記憶體管理重寫這段。

### 2.4 Merge 階段：把子迴圈接成一條 Global Tour

提供兩種合併策略，用 `merge_method` 參數選：

- **`merge_method="greedy"`（預設）— `merge_subtours`**：
  Bottom-up 貪心合併。每一輪窮舉所有還沒合併的子迴圈兩兩配對，
  用 `_best_merge` 算出每一對「邊交換」(edge swap) 的最小成本 Delta：

  $$\Delta = W(u_1,u_2) + W(v_1,v_2) - W(u_1,v_1) - W(u_2,v_2)$$

  （`_best_merge` 會窮舉兩個迴圈的所有邊配對、以及正接/反接兩種接法，取 Delta 最小者）
  挑 Delta 全域最小的一對合併，重複直到剩一條完整路線。

- **`merge_method="hierarchical"` — `merge_subtours_hierarchical`**：
  貪心合併只看局部最省，容易把整體路線接得歪七扭八。這個做法先把每個子迴圈當成一個
  「巨觀節點」，兩兩之間的巨觀距離＝ `_best_merge` 算出的最小 Delta，組成一個 `K x K` 的
  巨觀成本矩陣，再對這 K 個子圖解一次「巨觀 TSP」（`_macro_tour_order`：
  子圖數 ≤ `MAX_SUBGRAPH_SIZE` 用 Held-Karp 精確解，否則用最近鄰＋2-opt 當備援），
  決定子圖該用什麼順序串接，最後依照這個順序把子迴圈逐一接起來。

測試結果：`hierarchical` 品質跟 `greedy`相近甚至持平，但**速度快很多**
（N=150、子圖規模上限 10 的測試中，`hierarchical` 只要 0.14 秒，`greedy` 要 0.44 秒），
對「每個 epoch 都要跑一次 validation route gap」這種重複呼叫的場景很有幫助。

### 2.5 2-opt 收尾（可選）：`two_opt`

`apply_two_opt=True` 時，Merge 完的路線會再跑一次標準 2-opt：

- 反覆嘗試把路徑中某一段反轉，只要能讓「新兩條邊成本」小於「舊兩條邊成本」就接受
- 有給座標時用真實座標距離當依據；沒給座標時退回用機率成本矩陣的對稱版本
  `(cost_matrix + cost_matrix.T) / 2` 當距離依據
- 主要用來清掉 Divide/Merge 階段殘留的交叉或不合理長邊

### 2.6 `divide_and_conquer_decoder_v2` 完整參數

```python
divide_and_conquer_decoder_v2(
    prob_matrix,             # N x N 邊機率矩陣
    start_node=0,             # 最終輸出路線的起點
    coords=None,              # 節點座標（可選，給了才能用真實距離做 2-opt）
    max_subgraph_size=12,      # 子圖大小上限（Held-Karp 用，見 MAX_SUBGRAPH_SIZE）
    seed=42,                   # Louvain / Kernighan-Lin 用的亂數種子
    apply_two_opt=False,       # 是否在最後跑 2-opt
    partition_method="louvain",   # "louvain" 或 "bisection"
    merge_method="greedy",        # "greedy" 或 "hierarchical"
)
```

### 2.7 小規模測試結果摘要

用合成資料（座標隨機生成、機率設為 `exp(-3 * 歐氏距離)`，模擬「訓練得不錯的模型」）測試，
`max_subgraph_size=10`、皆有開 `apply_two_opt=True`，跟 v1 比較實際幾何路線長度：

| N | v1（正式版本） | louvain+greedy | louvain+hierarchical | bisection+greedy | bisection+hierarchical |
|---|---|---|---|---|---|
| 30 | 6.057 | 5.138 | 5.138 | **4.798** | **4.798** |
| 50 | 6.241 | 5.948 | 5.948 | 6.080 | **5.677** |
| 80 | 8.771 | **7.866** | 8.008 | 7.912 | 7.912 |
| 150 | 11.506 | **10.042**（0.44s） | 10.318（**0.14s**） | 10.315（0.39s） | 10.379（0.16s） |

結論：v2 的四種組合在測試的所有規模下，幾何路線長度都優於目前正式使用的 v1；
但四種組合彼此之間沒有單一贏家，需要依實際資料規模挑選，或是接下來在真實 494 節點
資料集上跑一次驗證後再決定要不要正式取代 v1。

另外也用 N=8 的極小案例，驗證過 `held_karp_tsp` 求出的解跟暴力窮舉全排列的最佳解完全相同，
確認 DP 實作沒有 bug。

---

## 3. 已知限制 / 待辦事項

1. `divide_and_conquer_decoder_v2` 目前**只做過合成小規模測試**，還沒有：
   - 在真實 494 節點資料集、搭配訓練好的模型機率矩陣上驗證過
   - 接入 `train.py` 的 `evaluate_route_gap`、`test_histogram.py`、`plot_qualitative.py`
2. `held_karp_tsp` 是純 Python 實作，`MAX_SUBGRAPH_SIZE` 目前設 12（狀態數 $2^{12}$）是
   純 Python 效能能負擔的安全值；要衝更大子圖規模需要用 C++/numba 重寫。
3. 四種 `partition_method` × `merge_method` 組合沒有單一全面最優解，可能需要依資料規模
   動態選擇，或針對 494 節點規模再做一次系統性的參數掃描。
