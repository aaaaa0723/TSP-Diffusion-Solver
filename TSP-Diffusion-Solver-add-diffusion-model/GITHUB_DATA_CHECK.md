# GitHub 資料完整性檢查（2026-11-12）

檢查分支：`feat/osm-route-cost-cuda-results`。本次補齊作業不修改 `main`。

## 檢查結果與補齊項目

| 類別 | 檢查結果／處理 |
| --- | --- |
| 成本生成、OSM 特徵、快取與重試程式 | 原分支已齊全 |
| CUDA 訓練、解碼、評估與繪圖程式 | 原分支已齊全 |
| README、教授報告稿 | 原分支已有；本次補入下載入口 |
| 模型權重 | 已存在，SHA-256 與訓練時紀錄一致 |
| 五張圖 | 已存在，逐檔 SHA-256 與紀錄一致 |
| 150 筆路線、指標、設定及道路幾何 | 已存在；路線有效性與雜湊已核對 |
| 完整 1,000 × 300 NPZ | 原先未上傳；本次補至 Release |
| 原始 PBF 與 26 份預處理 OSRM 圖資 | 原先未上傳；本次補至 Release 資料包 |
| 收費／號誌與節點索引、固定站點快照 | 索引原先未上傳；本次補至相同資料包 |
| 下載清單與 SHA-256 | 新增 `data/cost_data_manifest.json` 與 Release 校驗附件 |
| 實際訓練環境版本 | 新增 `code/requirements-training-linux-cu128.txt`，固定 50 個套件 |

原始程式／成果核對基準為已推送的 `5f4bc4b`。新補文件與大型資料為同一個已完成實驗的發佈補充，沒有改變成本權重、模型權重或測試指標。

## 可驗證的依據

1. 使用遠端 Git 分支的實際 tree／blob 核對必要程式及成果，不只檢查本機工作目錄。
2. 模型、五張 PNG、JSON 設定與指標均符合 `reproducibility.json` 記錄的 SHA-256。
3. 完整 NPZ 的 SHA-256 為 `8ed7d46529bd1d07ac72d8e8d6d7e2309c6c55b64247504218fb2063e499e2cb`，與完成訓練時的資料完全一致；原始完整資料驗證紀錄位於 run 目錄的 `dataset_validation.json`。
4. 道路資料包已檢查包含原始 PBF、全部 26 份圖檔、特徵索引、站點、邊界與來源說明，且壓縮內路徑均位於相對目錄。
5. 本次既有程式共 16 項測試已於原訓練環境通過；補齊作業未修改模型或演算法程式。

## 下載位置

[GitHub Release：cost-data-2026-11-11-v1](https://github.com/aaaaa0723/TSP-Diffusion-Solver/releases/tag/cost-data-2026-11-11-v1)

完整步驟見 [DATA_DOWNLOADS.md](DATA_DOWNLOADS.md)。完整 NPZ 與道路重現包合計約 2.14 GB（十進位），每個附件均另有 SHA-256 可驗證。

## 不發布的內容

SSH 金鑰、登入資訊、個人 IDE 設定、暫存／執行紀錄不屬於公開實驗資料。逐筆中途檢查點與衍生 pair cache 亦不需要重複發布：完整資料已提供，其他快取可由相同圖資重新建立。

完整測資與大型道路資料放在 Release，不寫入 Git 歷史；一般 clone 的程式目錄仍保持精簡。
