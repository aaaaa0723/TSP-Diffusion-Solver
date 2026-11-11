# Diffusion + OSRM：本地差異

相較 GitHub 的 `add-diffusion-model` 分支，本地修改是在既有 TSP 擴散模型上加入道路距離資料支援。Diffusion 模型與訓練流程已存在於 GitHub 分支中。

- `code/generate_tsp.py` 改為產生台中經緯度案例，可使用 OSRM 行車距離（設定 `TSP_DISTANCE_MODE=road`）或 Haversine 距離，並將距離矩陣存入資料集；另加入逐筆 checkpoint 與中斷續跑。
- `code/evaluate.py`、`code/evaluation.py` 改用距離矩陣計算路線長度。舊資料集沒有 `distances` 欄位時，會以座標計算歐氏距離，維持相容。
- 新增 `code/experiment_utils.py` 統一資料集路徑，並在 `code/requirements.txt` 加入 OSRM HTTP 請求所需套件。

GitHub 上的資料集尚未包含 `distances` 欄位；上述 fallback 可讓舊資料集繼續使用。新生成資料集可存放 OSRM 道路距離。產生器預設 `TSP_DISTANCE_MODE=auto`，目前實際使用 Haversine；設定 `road` 才會連線本機 `http://localhost:5000` 的 OSRM 服務。

本分支另包含 `results/runs/20260928_163628/` 的模型 checkpoint、參數、評估報告與圖表，以及更新後的 `results/runs/latest.txt`。859 MB 的 `results/tsp_dataset_lite.npz` 未推送：GitHub 一般 Git push 不接受超過 100 MB 的單一檔案，且目前環境未安裝 Git LFS。空白訓練 log、編輯器設定、虛擬環境、bytecode 與 `.orig` 備份檔也未納入。

## OSM route costs (2026-10-13)

New datasets retain `distances` (km), add `durations` (seconds) and `costs`
(equivalent km). The objective is `0.35 * distance_km + 0.65 * duration_seconds
* median_positive_distance_km / median_positive_duration_seconds`. Medians use
reachable, positive off-diagonal pairs per sample. The 35/65 policy prioritizes
travel time; the conversion coefficient is calibrated from each OSM/OSRM table,
not from live traffic or learned optimal weights. Store coefficients in
`cost_metadata_json`, with `cost_mode=osm_distance_duration_v1`.

OR-Tools labels, training features, evaluation and benchmark decoders use costs.
Physical distances remain separate. Legacy metric keys containing `km` describe
the objective when `objective_unit=equivalent_km`; evaluation records additionally
contain physical route kilometers. Existing datasets keep their distance objective;
regenerate and retrain to use new costs. Cost-specific progress directories prevent
resuming labels from the old objective. Road direction and accessibility are
inherited from the prepared OSRM driving profile. No independent surface, slope,
signal, toll, or live congestion penalty is invented from missing tags.

Validation: four generation/cost tests and Python compilation passed. Local OSRM
at localhost:5000 was unavailable, so no live dataset regeneration was performed.


### Toll and signal extension

`osm_distance_duration_toll_signals_v2` adds +0.25 equivalent km per toll km
and +0.02 equivalent km per traversed signal node. `osm_route_cost.py` matches
OSRM annotations to exact OSM node pairs and signal IDs, batches Overpass queries,
and caches successful JSON responses atomically. Failed queries, missing OSM
nodes, route/table disagreement and malformed feature matrices fail generation.
Toll and signal arrays persist separately in datasets and progress checkpoints.
See README for tag coverage, caching, generation cost and fixed pair-route limits.
Seven generation/cost tests pass; live OSRM regeneration remains unavailable.


### Live dataset generation (2026-10-13)

Started Docker Desktop and reused the existing `osrm-taiwan` service on
localhost:5000, backed by `taiwan-260925.osrm`. Extracted the matching local PBF
into a memory-mapped node index (25,168,107 nodes), 56,899 signal nodes and
29,496 toll edges. Local lookup replaces Overpass for this run; scalar pair
features are cached, reducing storage compared with complete route annotations.
Feature matrices remain numeric arrays in process memory rather than Python
lists to support the full 1,000-by-300 dataset.

`run_cost_generation.py` runs smoke generation, validation, full generation and
final validation without training. A real 1-by-3 smoke dataset passed. The full
1,000-by-300 generation was launched in the background with four workers,
30-second OR-Tools limits and per-sample checkpoints. Logs and status are under
`results/generation_toll_signals/`; the final output is
`results/tsp_dataset_youbike_300nodes_toll_signals.npz`. Eight tests passed,
including local PBF index lookup and cache reuse. Completion is tracked by the
runner status; launching the process does not mean the full dataset is complete.


The initial full-generation startup encountered a transient OSRM dropped
connection during station alignment. Added bounded connection/read retries for
read-only road requests, reran the eight tests successfully and restarted the
background runner. The smoke generation passed again before the full restart.


### Route/Table discrepancy repair

The full run failed on `Route/table mismatch`. Reproduced seed 1 pair (0,121):
Table reported 4,718.7 m / 464 s while a fresh Route reported 4,756.3 m / 464 s.
The cached Route was correct; equal-time path choices differed. Now all physical
and cost factors use the same annotated Route's distance and duration, including
cached features, rather than rejecting valid pairs or mixing route factors.
Changed cost mode to v3 to isolate previous reference labels. Added regression
coverage to the local feature-matrix test; eight tests pass. Verified the actual
failed pair now succeeds with 4.7563 km, 464 seconds and five signal nodes.
Restarted the runner with a larger 30-node smoke generation. Status now includes
completed_samples and retains the underlying error when a child process fails.


### Windows cache publication repair

Generation saved 12/1,000 v3 samples before a concurrent cache-file replacement
failed with WinError 5. Cached route features are optional: reads now tolerate
transient filesystem errors; a publication collision reuses the valid winning
entry, or continues with the fetched route if caching remains unavailable.
Temporary files are cleaned up when possible. Mandatory dataset checkpoint
writes remain strict. Added a regression test for denied cache replacement;
nine tests passed. Restarted the generation runner to resume the 12 saved samples.


### Status-file lock repair

The resumed run stopped when the IDE/Windows held `status.json` during atomic
replacement. Status updates now use unique temporary filenames and bounded
retries. If writing remains blocked, append a monitoring event to
`status_events.jsonl` and continue generation. Monitoring cannot abort the
child process; essential checkpoints still fail explicitly on write errors.
Initial status now retains the 12 existing checkpoints rather than resetting
the completed counter. Added a regression test simulating the status-file lock.
Restarted the runner with the existing checkpoints and route cache retained.


### Unreachable annotated route handling (2026-10-14)

76/1,000 samples were checkpointed before OSRM returned HTTP 400 NoRoute for
an annotated station pair. Classify NoRoute before generic HTTP errors and map
it to the existing unreachable-instance handling, which resamples station sets
(up to 20 attempts) while retaining cached valid route features. Never fill an
unreachable cost with zero or a straight-line approximation. Other HTTP failures
remain errors. Eleven tests passed, including NoRoute response classification
and translation into the resampling exception. Restarted the runner retaining
all 76 checkpoints.

## Completed route-cost experiment (2026-11-11)

Supersedes the earlier in-progress generation notes above. The v3 dataset now
contains 1,000 instances of 300 Taichung YouBike stations, with a 30-second
OR-Tools reference per instance. Generation and full dataset validation completed.
The final NPZ is 1,019,230,944 bytes and is excluded from Git with the PBF,
preprocessed OSRM graph, large feature index, pair cache and private runtime files.

Run `20261111_214014_cost_300nodes_cuda` completed on an RTX 4090: 20 epochs,
700/150/150 splits, full validation each epoch and all 150 held-out test cases.
All reference/predicted test tours are valid and all five PNG artifacts were
verified. Mean gap is -2.9553109223% against the 30-second OR-Tools reference;
146 of 150 test cases have lower cost. These are hybrid model-plus-decoder
results, not proof of global optimality or an equal-time solver comparison.

Added `run_cost_training.py` for CUDA checks, an end-to-end smoke run, full
training/evaluation and mandatory verification of the model and all five plots.
The completed model, metrics, plot records, road geometry and figures are retained
in the new run directory. README and PROFESSOR_REPORT.md explain the objective,
reproduction prerequisites, exact experiment and research limitations.

Pinned the Compose image to the actual graph-compatible OSRM v26.9.0 digest
from ghcr.io/project-osrm/osrm-backend; the older osrm/osrm-backend image cannot
read these graph files. The graph mount is read-only and the published port is
bound to loopback. The runner's fixed snapshot/index/cache namespace must remain
consistent when resuming.
