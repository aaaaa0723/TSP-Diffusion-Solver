# Taichung Shopee single-case evaluation

485 distinct SPX points returned by the official public locator, collected using 53 spatial queries with capped regions subdivided. All points are listed as displayed with point_status=0. This is the locator snapshot, not an independent audit of every operating branch. Source: https://spx.tw/service-point

Every collected point is included; none were subsampled or removed. Maximum road snap: 39.12 m. Automotive OSRM road distances in km.

Checkpoint: results/runs/20260929_191743/difusco_tsp.pth (existing 7-ELEVEN-trained model; no retraining).

| Method | Distance (km) | Time (s) |
|---|---:|---:|
| OR-Tools | 504.683 | 30-second solver limit |
| DIFUSCO + decoder | 477.108 | 55.770 |

Gap: -5.464%; difference: -27.575 km. CPU diffusion: 0.696 s; decoder: 55.075 s. The decoder includes 16 road/heatmap starts and local search. This is not an equal-time comparison or a measurement of the neural network alone. Both tours are valid closed Hamiltonian cycles.

Files: dataset.npz, catalog.json, report.json, plot_records.json, model_tour.csv, ortools_tour.csv, route_comparison.png.

Reproduce: `python code/run_shopee_case.py` (reuses dataset if present, reruns inference and decoder). YouBike generation defaults remain unchanged.
