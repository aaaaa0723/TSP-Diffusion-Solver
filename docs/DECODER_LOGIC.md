# TSP Decoder 邏輯

`code/evaluation.py` 現在只保留正式使用中的 decoder：`divide_and_conquer_decoder`。

## 流程

1. 以所有座標的 bounding square 為根，遞迴四分成西南、東南、西北、西北四個小正方形。
2. 當小正方形的點數不超過 `MAX_SUBGRAPH_SIZE`（預設 12），以 Held-Karp 求該格內的精確最優迴圈。
3. 從包含 `start_node` 的葉格開始，優先在尚未走過、且共用邊界的相鄰格子中，選擇 edge-swap 成本最低的一格連接。
4. 沒有未訪問的相鄰格子時，才選最近的葉格作備援，直到走遍所有格子。
5. 輸出完整 Hamiltonian cycle，第一個與最後一個節點皆為 `start_node`。

格內求解與格間連接都使用模型機率轉換的對稱成本：`-log(P + 1e-9)`。

## 自行調整

只要修改 `code/evaluation.py` 即可調整 decoder。最常用的參數是：

- `MAX_SUBGRAPH_SIZE`：葉格最多幾個點；調大會讓局部解更精確，但 Held-Karp 的時間會快速增加。
- `COST_EPSILON`：機率為零時的數值保護。
- `apply_two_opt=True`：呼叫 `divide_and_conquer_quadtree_decoder` 時可額外進行 2-opt 幾何收尾；目前正式預設為關閉。

訓練與視覺化腳本都透過 `divide_and_conquer_decoder(..., coords=...)` 呼叫，因此不需要修改 `train.py`、`test_histogram.py` 或 `plot_qualitative.py`。
