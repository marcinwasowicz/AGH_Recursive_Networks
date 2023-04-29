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
from networks import TreeNetClassifier

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
    return DataLoader(dataset, batch_size, True, collate_fn=lambda x: dgl.batch(x))


def evaluate_classifier(classifier, dataset_split, batch_size, device):
    total_correct = 0
    total = 0
    classifier.eval()
    with th.no_grad():
        for graph in make_data_loader(dataset_split, batch_size):
            graph = graph.to(device)
            response = classifier(graph, "x", "y")
            pred = th.argmax(response, 1)
            total_correct += float(
                th.sum(
                    th.eq(
                        graph.ndata["y"][graph.out_degrees() == 0],
                        pred[graph.out_degrees() == 0],
                    )
                )
            )
            total += graph.batch_size
    return total_correct / total


def train_classifier(
    train,
    valid,
    model_type,
    embeddings,
    lr,
    l2,
    h_size,
    batch_size,
    n_ary,
    num_classes,
    epochs,
    device,
):
    embedding_layer = nn.Embedding.from_pretrained(
        th.load(f"{NUM_DATA_DIR}/sst_constituency_{embeddings}_embeddings.pt")
    )
    cell = CELLS[model_type](embedding_layer.embedding_dim, h_size, n_ary)
    classifier = TreeNetClassifier(embedding_layer, cell, h_size, num_classes)
    classifier.to(device)
    optimizer = th.optim.Adam(classifier.parameters(), lr=lr, weight_decay=l2)

    best_acc = 0.0
    batch_size = batch_size

    for epoch in range(epochs):
        classifier.train()
        for graph in make_data_loader(train, batch_size):
            graph = graph.to(device)
            response = classifier(graph, "x", "y")
            probabilities = F.log_softmax(response, 1)
            loss = F.nll_loss(
                probabilities[graph.ndata["y"] != -1],
                graph.ndata["y"][graph.ndata["y"] != -1],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_classifier(classifier, valid, batch_size, device)
        print("Epoch {:05d} | Val Acc {:.4f}".format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
    return best_acc


def objective_factory(train_valid, model_type, embeddings, device):
    def obj(trial):
        lr, l2, h_size, batch_size = (
            trial.suggest_loguniform("lr", 0.0001, 0.1),
            trial.suggest_loguniform("l2", 1e-7, 1e-3),
            trial.suggest_int("h_size", 50, 300, 50),
            trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        )
        results = []

        for step, (train_idx, valid_idx) in enumerate(
            KFold(3, shuffle=True, random_state=42).split(train_valid)
        ):
            train = train_valid[train_idx]
            valid = train_valid[valid_idx]
            intermediate_result = train_classifier(
                train,
                valid,
                model_type,
                embeddings,
                lr,
                l2,
                h_size,
                batch_size,
                2,
                5,
                10,
                device,
            )
            results.append(intermediate_result)
        return np.mean(results)

    return obj


if __name__ == "__main__":
    th.manual_seed = 42
    global_device = "cuda:0" if th.cuda.is_available() else "cpu"
    print(f"Using device type: {global_device}")

    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    for model_type in config["model_types"]:
        for embeddings in config["embeddings"]:
            with open(
                f"{NUM_DATA_DIR}/sst_constituency_train_{embeddings}.pkl", "rb"
            ) as train_fd:
                train = pickle.load(train_fd)

            with open(
                f"{NUM_DATA_DIR}/sst_constituency_valid_{embeddings}.pkl", "rb"
            ) as valid_fd:
                valid = pickle.load(valid_fd)

            with open(
                f"{NUM_DATA_DIR}/sst_constituency_test_{embeddings}.pkl", "rb"
            ) as test_fd:
                test = pickle.load(test_fd)

            train_valid_test = np.array(train + valid + test)
            for train_valid_idx, test_idx in KFold(
                5, shuffle=True, random_state=42
            ).split(train_valid_test):
                train_valid = train_valid_test[train_valid_idx]
                test = train_valid_test[test_idx]

                objective = objective_factory(
                    train_valid, model_type, embeddings, global_device
                )
                sampler = optuna.samplers.TPESampler(42)
                study = optuna.create_study(
                    storage=f"sqlite:///{NUM_DATA_DIR}/sst_{model_type}_{embeddings}.db",
                    sampler=sampler,
                    study_name=f"sst_{model_type}_{embeddings}_{str(randint(0, 1000))}",
                    direction="maximize",
                    load_if_exists=True,
                )
                study.optimize(objective, n_trials=15)

                print(
                    "Evaluation for:\nmodel type: {}\nlr: {}\nl2: {}\nh_size: {}\nbatch size: {}\nembeddings: {}\n".format(
                        model_type,
                        study.best_params["lr"],
                        study.best_params["l2"],
                        study.best_params["h_size"],
                        study.best_params["batch_size"],
                        embeddings,
                    )
                )

                test_acc = train_classifier(
                    train_valid,
                    test,
                    model_type,
                    embeddings,
                    study.best_params["lr"],
                    study.best_params["l2"],
                    study.best_params["h_size"],
                    study.best_params["batch_size"],
                    2,
                    5,
                    10,
                    global_device,
                )
                print("Test accuracy: {:.4f}".format(test_acc))
