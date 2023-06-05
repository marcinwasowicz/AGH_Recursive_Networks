import json
import pickle
from random import randint
import sys
import warnings
import os

import dgl
import numpy as np
import optuna
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.data import DataLoader

sys.path.insert(0, "./")
warnings.filterwarnings("ignore")

from cells import (
    NTreeGRU,
    NTreeLSTM,
    NTreeMGU,
    ChildSumTreeLSTM,
    ChildSumTreeGRU,
    ChildSumTreeMGU,
    SingleForgetGateTreeLSTM,
    SingleForgetGateTreeGRU,
    SingleForgetGateTreeMGU,
)
from networks import TreeNetSimilarityRegressor

NUM_DATA_DIR = os.path.expandvars("$SCRATCH")
CELLS = {
    "lstm": NTreeLSTM,
    "mgu": NTreeMGU,
    "gru": NTreeGRU,
    "child_sum_lstm": ChildSumTreeLSTM,
    "child_sum_gru": ChildSumTreeGRU,
    "child_sum_mgu": ChildSumTreeMGU,
    "single_gate_lstm": SingleForgetGateTreeLSTM,
    "single_gate_gru": SingleForgetGateTreeGRU,
    "single_gate_mgu": SingleForgetGateTreeMGU,
}


def make_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size, False, collate_fn=lambda x: dgl.batch(x))


def make_primitive_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size, False)


def evaluate_regressor(
    regressor,
    dataset_split_a,
    dataset_split_b,
    similarity_split,
    batch_size,
    num_classes,
    device,
):
    total_mse = 0
    total_size = 0
    regressor.eval()
    with th.no_grad():
        for graph_a, graph_b, similarity in zip(
            make_data_loader(dataset_split_a, batch_size),
            make_data_loader(dataset_split_b, batch_size),
            make_primitive_data_loader(similarity_split, batch_size),
        ):
            graph_a = graph_a.to(device)
            graph_b = graph_b.to(device)
            similarity = similarity.to(device)

            response = regressor(graph_a, graph_b, "x")
            pred = th.sum(th.range(1, num_classes).to(device) * response, dim=1).to(
                device
            )
            total_mse += th.sum((similarity - pred) ** 2)
            total_size += graph_a.batch_size
    return total_mse / total_size


def train_regressor(
    train_a,
    train_b,
    train_sim,
    valid_a,
    valid_b,
    valid_sim,
    model_type,
    embeddings,
    lr,
    l2,
    h_size,
    sim_h_size,
    batch_size,
    n_ary,
    num_classes,
    epochs,
    device,
):
    embedding_layer = nn.Embedding.from_pretrained(
        th.load(f"{NUM_DATA_DIR}/sick_constituency_{embeddings}_embeddings.pt")
    )
    cell = CELLS[model_type](embedding_layer.embedding_dim, h_size, n_ary)
    regressor = TreeNetSimilarityRegressor(
        embedding_layer,
        cell,
        h_size,
        sim_h_size,
        num_classes,
    )
    regressor.to(device)
    optimizer = th.optim.Adam(regressor.parameters(), lr=lr, weight_decay=l2)

    best_mse = 16.00  # Theoretically maximal MSE we can get
    batch_size = batch_size

    for epoch in range(epochs):
        regressor.train()
        total_loss = 0
        for graph_a, graph_b, similarity in zip(
            make_data_loader(train_a, batch_size),
            make_data_loader(train_b, batch_size),
            make_primitive_data_loader(train_sim, batch_size),
        ):
            graph_a = graph_a.to(device)
            graph_b = graph_b.to(device)
            similarity = similarity.to(device)

            response = regressor(graph_a, graph_b, "x")
            target_response = th.zeros((len(similarity), num_classes))
            for idx, sim in enumerate(similarity):
                if th.floor(sim).int() == th.ceil(sim).int():
                    target_response[idx, sim.int() - 1] = 1
                    continue
                target_response[idx, th.floor(sim).int() - 1] = th.ceil(sim) - sim
                target_response[idx, th.ceil(sim).int() - 1] = sim - th.floor(sim)

            target_response = target_response.to(device)

            loss = F.kl_div(response.log(), target_response)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mse = evaluate_regressor(
            regressor, valid_a, valid_b, valid_sim, batch_size, num_classes, device
        )
        print(
            "Epoch {:05d} | Train Loss {:.4f} | Val MSE {:.4f}".format(
                epoch, total_loss / batch_size, mse
            )
        )
        if mse < best_mse:
            best_mse = mse
    return best_mse


def objective_factory(
    train_valid_a, train_valid_b, train_valid_sim, model_type, embeddings, device
):
    def obj(trial):
        lr, l2, h_size, sim_h_size, batch_size = (
            trial.suggest_loguniform("lr", 0.0001, 0.1),
            trial.suggest_loguniform("l2", 1e-7, 1e-3),
            trial.suggest_int("h_size", 25, 350, 25),
            trial.suggest_int("sim_h_size", 25, 100, 25),
            trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]),
        )
        results = []
        for step, (train_idx, valid_idx) in enumerate(
            KFold(3, shuffle=True, random_state=42).split(train_valid_a)
        ):
            train_a, valid_a = train_valid_a[train_idx], train_valid_a[valid_idx]
            train_b, valid_b = train_valid_b[train_idx], train_valid_b[valid_idx]
            train_sim, valid_sim = (
                train_valid_sim[train_idx],
                train_valid_sim[valid_idx],
            )

            intermediate_result = train_regressor(
                train_a,
                train_b,
                train_sim,
                valid_a,
                valid_b,
                valid_sim,
                model_type,
                embeddings,
                lr,
                l2,
                h_size,
                sim_h_size,
                batch_size,
                2,
                5,
                10,
                device,
            )
            results.append(intermediate_result.cpu().detach().numpy())
        return np.mean(results)

    return obj


if __name__ == "__main__":
    global_device = "cuda:0" if th.cuda.is_available() else "cpu"
    print(f"Using device type: {global_device}")

    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    for model_type in config["model_types"]:
        for embeddings in config["embeddings"]:
            with open(
                f"{NUM_DATA_DIR}/sick_constituency_train_{embeddings}.pkl", "rb"
            ) as train_fd:
                train_a, train_b, train_sim = [
                    shuffle(arr, random_state=42) for arr in pickle.load(train_fd)
                ]

            with open(
                f"{NUM_DATA_DIR}/sick_constituency_valid_{embeddings}.pkl", "rb"
            ) as valid_fd:
                valid_a, valid_b, valid_sim = [
                    shuffle(arr, random_state=42) for arr in pickle.load(valid_fd)
                ]

            with open(
                f"{NUM_DATA_DIR}/sick_constituency_test_{embeddings}.pkl", "rb"
            ) as test_fd:
                test_a, test_b, test_sim = [
                    shuffle(arr, random_state=42) for arr in pickle.load(test_fd)
                ]
            train_valid_test_a = np.array(train_a + valid_a + test_a)
            train_valid_test_b = np.array(train_b + valid_b + test_b)
            train_valid_test_sim = np.array(train_sim + valid_sim + test_sim)

            for train_valid_idx, test_idx in KFold(
                5, shuffle=True, random_state=42
            ).split(train_valid_test_a):
                train_valid_a, test_a = (
                    train_valid_test_a[train_valid_idx],
                    train_valid_test_a[test_idx],
                )
                train_valid_b, test_b = (
                    train_valid_test_b[train_valid_idx],
                    train_valid_test_b[test_idx],
                )
                train_valid_sim, test_sim = (
                    train_valid_test_sim[train_valid_idx],
                    train_valid_test_sim[test_idx],
                )

                objective = objective_factory(
                    train_valid_a,
                    train_valid_b,
                    train_valid_sim,
                    model_type,
                    embeddings,
                    global_device,
                )
                sampler = optuna.samplers.TPESampler(42)
                study = optuna.create_study(
                    sampler=sampler,
                    study_name=f"sick_{model_type}_{embeddings}_{str(randint(0, 1000))}",
                    load_if_exists=True,
                )
                study.optimize(objective, n_trials=25)

                print(
                    "Evaluation for:\nmodel type: {}\nlr: {}\nl2: {}\nh_size: {}\nsim_h_size: {}\nbatch_size: {}\nembeddings: {}\n".format(
                        model_type,
                        study.best_params["lr"],
                        study.best_params["l2"],
                        study.best_params["h_size"],
                        study.best_params["sim_h_size"],
                        study.best_params["batch_size"],
                        embeddings,
                    )
                )

                test_mse = train_regressor(
                    train_valid_a,
                    train_valid_b,
                    train_valid_sim,
                    test_a,
                    test_b,
                    test_sim,
                    model_type,
                    embeddings,
                    study.best_params["lr"],
                    study.best_params["l2"],
                    study.best_params["h_size"],
                    study.best_params["sim_h_size"],
                    study.best_params["batch_size"],
                    2,
                    5,
                    10,
                    global_device,
                )
                print("Test MSE: {:.4f}".format(test_mse))