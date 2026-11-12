# 完整測資與道路重現資料下載

本次完整資料透過 **GitHub Release 附件**提供，不把 GB 級檔案寫進 Git 歷史。

Release：[cost-data-2026-11-11-v1](https://github.com/aaaaa0723/TSP-Diffusion-Solver/releases/tag/cost-data-2026-11-11-v1)。

## 提供的檔案

| 檔案 | 大小（bytes） | 用途 |
| --- | ---: | --- |
| [tsp_dataset_youbike_300nodes_toll_signals.npz](https://github.com/aaaaa0723/TSP-Diffusion-Solver/releases/download/cost-data-2026-11-11-v1/tsp_dataset_youbike_300nodes_toll_signals.npz) | 1,019,230,944 | 已驗證的 1,000 × 300 完整資料集；可直接訓練及評估 |
| [osm-reproduction-taiwan-260925.tar.gz](https://github.com/aaaaa0723/TSP-Diffusion-Solver/releases/download/cost-data-2026-11-11-v1/osm-reproduction-taiwan-260925.tar.gz) | 1,119,604,850 | 原始 PBF、26 份預處理道路檔、收費／號誌索引及固定站點快照 |
| [SHA256SUMS.txt](https://github.com/aaaaa0723/TSP-Diffusion-Solver/releases/download/cost-data-2026-11-11-v1/SHA256SUMS.txt) | 小型文字檔 | 兩個大型附件的 SHA-256 |

機器可讀的下載網址、雜湊與版本資料：[data/cost_data_manifest.json](data/cost_data_manifest.json)。

```text
8ed7d46529bd1d07ac72d8e8d6d7e2309c6c55b64247504218fb2063e499e2cb  tsp_dataset_youbike_300nodes_toll_signals.npz
cda4f029f56dc8e6a0cbc10a35d4796b1e3763a3d9f8dc2ad3556405b4f29b45  osm-reproduction-taiwan-260925.tar.gz
```

## Linux：下載與校驗

以下指令在 `TSP-Diffusion-Solver-add-diffusion-model/` 目錄執行。

```bash
mkdir -p .runtime/downloads results
cd .runtime/downloads
base=https://github.com/aaaaa0723/TSP-Diffusion-Solver/releases/download/cost-data-2026-11-11-v1
curl --fail --location --retry 3 --continue-at - -O "$base/tsp_dataset_youbike_300nodes_toll_signals.npz"
curl --fail --location --retry 3 --continue-at - -O "$base/osm-reproduction-taiwan-260925.tar.gz"
curl --fail --location --retry 3 -O "$base/SHA256SUMS.txt"
sha256sum -c SHA256SUMS.txt
cd ../..
```

只有在兩個檔案都顯示 `OK` 後，再放到程式使用的位置：

```bash
cp .runtime/downloads/tsp_dataset_youbike_300nodes_toll_signals.npz results/
# 解壓到專案根目錄，會產生 data/ 內的道路資料與固定快照
# 若已有不同版本的站點／道路資料，請在全新 clone 或獨立工作目錄解開。
tar -xzf .runtime/downloads/osm-reproduction-taiwan-260925.tar.gz
```

只要重新訓練模型，可僅下載 NPZ；要重新生成測資或繪製未快取的新路線圖，則需要道路資料包與相同 OSRM 服務。

## Windows：校驗

下載相同兩個附件後，以 PowerShell 檢查：

```powershell
Get-FileHash .\tsp_dataset_youbike_300nodes_toll_signals.npz -Algorithm SHA256
Get-FileHash .\osm-reproduction-taiwan-260925.tar.gz -Algorithm SHA256
```

比對上面的 SHA-256，將 NPZ 放入 `results/`；在全新專案目錄執行 `tar -xzf` 解開道路資料包。

## 資料包內含什麼

```text
data/
  osrm/
    taiwan-260925.osm.pbf
    taiwan-260925.osrm.*                 # 26 份預處理圖檔
  osm_cost_index_260925/
    features.json                      # 收費路段／號誌及來源指紋
    node_ids.npy                       # OSM 節點覆蓋索引
  taichung_youbike.json                 # 固定 YouBike 快照
  taichung_city_boundary.geojson
  OSM_DATA_SOURCES.md                   # 資料來源與授權標示
```

OSRM 必須使用 README 固定的 **v26.9.0 映像雜湊**。索引中 `source` 路徑已改為相對路徑以方便搬移；來源大小、時間指紋及特徵內容維持原值，不影響既有 pair cache 的識別方式。

OSM 原始與衍生道路資料來源為 OpenStreetMap contributors，依 ODbL 1.0 提供；原始臺灣 PBF 來自 Geofabrik。YouBike 資料來源與授權記錄於快照目錄及資料集 catalog。詳見資料包的 `data/OSM_DATA_SOURCES.md`。

## 重建實際訓練環境

本次環境為 Linux、Python 3.12.3、PyTorch 2.11.0+cu128、CUDA runtime 12.8、RTX 4090。已補上實際環境的固定套件清單：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-training-linux-cu128.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

此清單針對 Linux/CUDA 訓練環境；Windows 安裝請依 README 的一般方式。重建 PBF 索引才需要另外安裝 `osmium`，直接使用已提供的索引或訓練 NPZ 不需要。

## 完整性範圍

Git 分支提供程式、README、教授報告稿、模型、全部測試指標、150 筆預測路線、道路幾何快取與五張圖；Release 提供完整 NPZ 與原始／預處理道路資料。

SSH 金鑰、登入資訊、IDE 設定、暫存檔、執行中狀態和逐筆生成檢查點不發布。檢查點與 pair cache 可從上述固定資料重新產生，不是使用完整 NPZ 或重現本實驗所缺少的必要輸入。
