import gc
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torch.utils.data import DataLoader, random_split
from dataset import TSPDataset
from model import TSPPureGNNModel
from evaluation import evaluate_route_gap


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = TSPDataset("tsp_dataset_lite.npz")

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
    batch_size = trial.suggest_categorical(
        "batch_size", [32, 64, 128]
    )
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 3e-3, log=True
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
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_objective = float("inf")

    for epoch in range(20):
        model.train()

        for coords, distances, targets in train_loader:
            coords = coords.to(DEVICE)
            distances = distances.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(coords, distances)
            loss = criterion(outputs, targets)
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
                total_val_loss += criterion(outputs, targets).item()

        val_loss = total_val_loss / len(val_loader)
        mean_gap, percentile_95_gap, objective = evaluate_route_gap(
            model,
            val_loader,
            DEVICE,
            max_samples=1000,
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
        study_name="tsp_gnn_mean_p95_tuning",
        storage="sqlite:///optuna_mean_p95.db",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3
        ),
    )

    study.optimize(objective, n_trials=20)

    print("\n最佳驗證 Objective (mean gap + 0.5 * p95 gap):", study.best_value)
    print("最佳參數:")

    for key, value in study.best_params.items():
        print(f"{key}: {value}")