from copy import deepcopy
import json
import pickle
import sys

import dgl
from sklearn.utils import shuffle
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.data import DataLoader

sys.path.insert(0, "./")

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
):
    total_mse = 0
    total_size = 0
    with th.no_grad():
        for graph_a, graph_b, similarity in zip(
            make_data_loader(dataset_split_a, batch_size),
            make_data_loader(dataset_split_b, batch_size),
            make_primitive_data_loader(similarity_split, batch_size),
        ):
            response = regressor(graph_a, graph_b, "x")
            pred = th.sum(th.range(1, num_classes) * response, dim=1)
            total_mse += th.sum((similarity - pred) ** 2)
            total_size += graph_a.batch_size
    return total_mse / total_size


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    with open(
        f"data/sick_constituency_train_{config['embeddings']}.pkl", "rb"
    ) as train_fd:
        train_a, train_b, train_sim = [
            shuffle(arr, random_state=42) for arr in pickle.load(train_fd)
        ]

    with open(
        f"data/sick_constituency_valid_{config['embeddings']}.pkl", "rb"
    ) as valid_fd:
        valid_a, valid_b, valid_sim = [
            shuffle(arr, random_state=42) for arr in pickle.load(valid_fd)
        ]

    with open(
        f"data/sick_constituency_test_{config['embeddings']}.pkl", "rb"
    ) as test_fd:
        test_a, test_b, test_sim = [
            shuffle(arr, random_state=42) for arr in pickle.load(test_fd)
        ]

    embedding_layer = nn.Embedding.from_pretrained(
        th.load(f"embeddings/sick_constituency_{config['embeddings']}_embeddings.pt")
    )
    cell = CELLS[config["type"]](
        embedding_layer.embedding_dim, config["h_size"], config["n_ary"]
    )
    regressor = TreeNetSimilarityRegressor(
        embedding_layer,
        cell,
        config["h_size"],
        config["similarity_h_size"],
        config["num_classes"],
    )
    optimizer = th.optim.Adagrad(
        regressor.parameters(), lr=config["lr"], weight_decay=1e-4
    )

    best_mse = 0.0
    best_model = None
    batch_size = config["batch_size"]

    for epoch in range(config["epochs"]):
        total_loss = 0
        for graph_a, graph_b, similarity in zip(
            make_data_loader(train_a, batch_size),
            make_data_loader(train_b, batch_size),
            make_primitive_data_loader(train_sim, batch_size),
        ):
            response = regressor(graph_a, graph_b, "x")
            target_response = th.zeros((len(similarity), config["num_classes"]))
            for idx, sim in enumerate(similarity):
                if th.floor(sim).int() == th.ceil(sim).int():
                    target_response[idx, sim.int() - 1] = 1
                    continue
                target_response[idx, th.floor(sim).int() - 1] = sim - th.floor(sim)
                target_response[idx, th.ceil(sim).int() - 1] = th.ceil(sim) - sim

            loss = F.kl_div(response.log(), target_response)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mse = evaluate_regressor(
            regressor, valid_a, valid_b, valid_sim, batch_size, config["num_classes"]
        )
        print(
            "Epoch {:05d} | Train Loss {:.4f} | Val MSE {:.4f}".format(
                epoch, total_loss / batch_size, mse
            )
        )
        if mse > best_mse:
            best_mse = mse
            best_model = deepcopy(regressor)
            th.save(best_model.state_dict(), config["save_path"])

    acc = evaluate_regressor(
        best_model, test_a, test_b, test_sim, batch_size, config["num_classes"]
    )
    print("Test MSE {:.4f}".format(acc))