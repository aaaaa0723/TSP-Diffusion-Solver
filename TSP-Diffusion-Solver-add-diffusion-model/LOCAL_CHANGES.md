# Diffusion + ORSM：本地差異

相較 GitHub 的 `add-diffusion-model` 分支，本地修改是在既有 TSP 擴散模型上加入道路距離資料支援。Diffusion 模型與訓練流程已存在於 GitHub 分支中。

- `code/generate_tsp.py` 改為產生台中經緯度案例，可使用 OSRM 行車距離（設定 `TSP_DISTANCE_MODE=road`）或 Haversine 距離，並將距離矩陣存入資料集；另加入逐筆 checkpoint 與中斷續跑。
- `code/evaluate.py`、`code/evaluation.py` 改用距離矩陣計算路線長度。舊資料集沒有 `distances` 欄位時，會以座標計算歐氏距離，維持相容。
- 新增 `code/experiment_utils.py` 統一資料集路徑，並在 `code/requirements.txt` 加入 OSRM HTTP 請求所需套件。

GitHub 上的資料集尚未包含 `distances` 欄位；上述 fallback 可讓舊資料集繼續使用。新生成資料集可存放 OSRM 道路距離。產生器預設 `TSP_DISTANCE_MODE=auto`，目前實際使用 Haversine；設定 `road` 才會連線本機 `http://localhost:5000` 的 OSRM 服務。

本分支另包含 `results/runs/20260928_163628/` 的模型 checkpoint、參數、評估報告與圖表，以及更新後的 `results/runs/latest.txt`。859 MB 的 `results/tsp_dataset_lite.npz` 未推送：GitHub 一般 Git push 不接受超過 100 MB 的單一檔案，且目前環境未安裝 Git LFS。空白訓練 log、編輯器設定、虛擬環境、bytecode 與 `.orig` 備份檔也未納入。