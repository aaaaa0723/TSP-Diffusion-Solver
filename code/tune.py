import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torch.utils.data import DataLoader, random_split
from dataset import TSPDataset
from model import TSPPureGNNModel
from evaluation import evaluate_route_gap
from experiment_utils import DATASET_PATH, OPTUNA_DB_PATH


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = TSPDataset(str(DATASET_PATH))
TUNE_EPOCHS = int(os.getenv("TSP_TUNE_EPOCHS", "5"))
TUNE_TRIALS = int(os.getenv("TSP_TUNE_TRIALS", "5"))
TUNE_MAX_SAMPLES = int(os.getenv("TSP_TUNE_MAX_SAMPLES", "100"))

train_size = int(0.7 * len(DATASET))
val_size = int(0.15 * len(DATASET))
test_size = len(DATASET) - train_size - val_size

train_set, val_set, _ = random_split(
    DATASET,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)


def objective(trial):
    hidden_dim = trial.suggest_categorical(
        "hidden_dim", [64, 128, 256]
    )
    batch_size = 1
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 3e-4, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-7, 1e-3, log=True
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    model = TSPPureGNNModel(hidden_dim=hidden_dim).to(DEVICE)
    pos_weight = torch.tensor(DATASET[0][2].shape[-1] - 1, device=DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_objective = float("inf")

    for epoch in range(TUNE_EPOCHS):
        model.train()

        for coords, distances, targets in train_loader:
            coords = coords.to(DEVICE)
            distances = distances.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(coords, distances)
            valid_edges = ~torch.eye(outputs.shape[-1], device=DEVICE, dtype=torch.bool).unsqueeze(0)
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs[valid_edges], targets[valid_edges], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for coords, distances, targets in val_loader:
                coords = coords.to(DEVICE)
                distances = distances.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(coords, distances)
                valid_edges = ~torch.eye(outputs.shape[-1], device=DEVICE, dtype=torch.bool).unsqueeze(0)
                total_val_loss += nn.functional.binary_cross_entropy_with_logits(
                    outputs[valid_edges], targets[valid_edges], pos_weight=pos_weight
                ).item()

        val_loss = total_val_loss / len(val_loader)
        mean_gap, percentile_95_gap, objective = evaluate_route_gap(
            model,
            val_loader,
            DEVICE,
            max_samples=TUNE_MAX_SAMPLES,
        )
        best_objective = min(best_objective, objective)

        trial.set_user_attr(f"val_bce_epoch_{epoch + 1}", val_loss)
        trial.set_user_attr(f"mean_gap_epoch_{epoch + 1}", mean_gap)
        trial.set_user_attr(f"percentile_95_gap_epoch_{epoch + 1}", percentile_95_gap)
        trial.report(objective, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_objective


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="tsp_gnn_mean_p01_p95_tuning",
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3
        ),
    )

    study.optimize(objective, n_trials=TUNE_TRIALS)

    print("\n最佳驗證 Objective (mean gap + 0.1 * p95 gap):", study.best_value)
    print("最佳參數:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")