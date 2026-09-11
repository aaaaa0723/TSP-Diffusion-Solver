# TSP Training Flow Comparison

## Scope

The project models a synthetic Euclidean TSP with 494 randomly generated stops inside a 2,215 km2 approximation of Taichung. The current data is not the geographic locations of the 494 real Shopee stores.

## Before the changes

| Stage | Previous implementation | Consequence |
|---|---|---|
| Data generation | 494 random points, OR-Tools `PATH_CHEAPEST_ARC`, 2-second limit | Produces a feasible heuristic route, not a proven optimum |
| Distance | Kilometer Euclidean distance | Correct scale for route measurement, but raw values were sent into the encoder |
| Graph message passing | Raw/normalized distance used directly as GCN weight | Farther nodes received stronger messages |
| Node model | Two `SimpleGCNLayer` blocks | Dense all-node aggregation |
| Edge model | Concatenate every pair `(x_i, x_j)` then an MLP | Materializes a large `(batch, 494, 494, 512)` tensor |
| Training target | Independent edge BCE | 494 positive edges versus 244,036 matrix entries; model can predict all zeros |
| TSP constraints | None in the model; greedy decoding only | No guarantee of one incoming edge, one outgoing edge, or one cycle |
| Evaluation | Some scripts used `range(19)` | Compared a 494-node AI route with a 20-node reference route |
| Evaluation name | `Optimal` and `Optimality Gap` | Misrepresented a time-limited heuristic route |
| Test display | Hard-coded 1,000 test instances | Dataset split actually contains about 150 test instances |
| Hyperparameter tuning | Tested batch sizes 1, 2, and 4 | Larger batches could exhaust GPU memory |

## After the changes

| Stage | Current implementation | Benefit |
|---|---|---|
| Data generation | 1,000 samples, 494 nodes, coordinates in kilometer-scale Taichung approximation | Matches the requested problem size |
| Reference route | OR-Tools heuristic route with a 2-second per-instance limit; failed solves raise an error | No silent all-zero labels; terminology is honest |
| Distance features | Exponential near-distance affinity followed by row normalization | Nearby stops have stronger graph influence and aggregation is numerically stable |
| Coordinate features | Coordinates are normalized inside the model per sample | Encoder input does not depend on the 0-47 km coordinate scale |
| Node model | Two GCN message-passing layers remain | Preserves the original GNN design |
| Edge model | Bilinear scorer with `einsum` | Avoids the full pairwise hidden feature tensor and reduces memory pressure |
| Training target | BCE with `pos_weight = 493`, diagonal excluded | Positive tour edges are no longer ignored by the loss |
| Batch size | Fixed at 1 for 494-node training and tuning | Keeps memory use predictable |
| Decoder | Greedy decoder is dynamic in the number of nodes | Works for 494 nodes rather than assuming 20 |
| Reference validation | Checks one incoming and one outgoing edge per node and one Hamiltonian cycle | Invalid labels fail early instead of corrupting metrics |
| Metrics wording | `Reference Route Gap` and `OR-Tools Reference` | Does not claim an exact optimum without proof |
| Test display | Uses actual test-set size and actual number of evaluated gaps | Charts report the data they really contain |

## Model details

The current model is still a supervised GNN, not a diffusion model:

1. Normalize each sample's coordinates.
2. Build a near-distance affinity matrix from Euclidean distances.
3. Apply two GCN layers.
4. Score directed edges with a bilinear function:

   `score(i, j) = x_i^T W x_j + b`

5. Train edge logits with masked, positive-weighted BCE.
6. Decode a permutation greedily for evaluation.

The bilinear scorer reduces memory, but greedy decoding still does not mathematically enforce all TSP constraints. For production-quality routes, the next upgrade should project model scores through an assignment/TSP solver and report invalid-tour rate.

## Reproducible commands

```bash
conda activate tsp_env
python generate_tsp.py
TSP_EPOCHS=1 python train.py   # smoke test
python train.py                # full 20-epoch training
python test_histogram.py
python plot_qualitative.py
```

The existing `tsp_gnn_model.pth` was trained with the old architecture and must be regenerated before running the evaluation scripts. The new architecture cannot load the old checkpoint by design.

## Interpretation limits

- The 494 points are synthetic random points, not real store coordinates.
- OR-Tools with a 2-second limit supplies a heuristic reference, not an exact optimum.
- A low BCE value alone is not evidence of a good TSP route.
- Route gap is meaningful only after both routes contain all 494 nodes exactly once.

## Completed tuning run

The short tuning pass used 5 trials, 5 epochs per trial, 100 validation samples, and batch size 1. The best trial returned:

```text
hidden_dim: 128
learning_rate: 6.441072982984653e-05
weight_decay: 0.0005248738212111669
objective: 127.7866
```

Those values were applied to the formal 20-epoch training run. The best validation checkpoint was selected at epoch 13 with objective `182.43`. The final test run evaluated all 150 test instances and reported a mean reference route gap of `114.67%`; the worst qualitative example had a gap of `158.59%`.

These results are a functioning baseline, not evidence that the model has solved the 494-stop TSP well. The next quality improvement is constrained decoding or an assignment/TSP projection after edge scoring.
