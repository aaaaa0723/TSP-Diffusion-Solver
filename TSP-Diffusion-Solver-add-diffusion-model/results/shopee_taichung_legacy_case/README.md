# Legacy checkpoint on the same SPX test case

Checkpoint: 20260928_163628. No retraining. Same 485 nodes, order, OSRM distance matrix and saved 30-second OR-Tools reference as ../shopee_taichung_case.

Legacy feature reconstruction was verified from retained code/__pycache__/difusco_model.cpython-311.pyc: center coordinates, torch.cdist, normalize distances by mean; node features normalize centered coordinates by per-axis standard deviation. Disassembly evidence is saved here. The original decoder is loaded from code/evaluation.py.orig (KMeans, Held-Karp, beam merging, Euclidean 2-opt). Neither inference nor decoding receives OSRM costs; OSRM is used only to score and draw the resulting tour.

OSRM road length: 542.944 km; reference: 504.683 km; gap: +7.581%. Inference: 2.056 s; decoding including legacy-module import: 23.234 s. Both tours independently verified to visit all 485 nodes once and close.

Reproduce:

```powershell
python code/run_shopee_case.py --run 20260928_163628 --dataset results/shopee_taichung_case/dataset.npz --output results/shopee_taichung_legacy_case --legacy-euclidean
```

Plot: route_comparison.png. Visit order: model_tour.csv. Full metrics: report.json. scikit-learn dependencies are available in .local-tools/legacy-deps for the current Python environment.
