from copy import deepcopy
import json
import pickle
import sys

import dgl
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
from networks import TreeNetClassifier


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


def evaluate_classifier(classifier, dataset_split, batch_size):
    total_correct = 0
    total = 0
    with th.no_grad():
        for graph in make_data_loader(dataset_split, batch_size):
            root_idx = 0
            assert graph.out_degrees(root_idx) == 0
            assert graph.in_degrees(root_idx) != 0

            response = classifier(graph, "x", "y")
            pred = th.argmax(response, 1)
            total_correct += float(
                th.sum(th.eq(graph.ndata["y"][root_idx], pred[root_idx]))
            )
            total += 1
    return total_correct / total


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    with open(f"data/sst_train_{config['embeddings']}.pkl", "rb") as train_fd:
        train = pickle.load(train_fd)

    with open(f"data/sst_valid_{config['embeddings']}.pkl", "rb") as valid_fd:
        valid = pickle.load(valid_fd)

    with open(f"data/sst_test_{config['embeddings']}.pkl", "rb") as test_fd:
        test = pickle.load(test_fd)

    embedding_layer = nn.Embedding.from_pretrained(
        th.load(f"embeddings/sst_{config['embeddings']}_embeddings.pt")
    )
    cell = CELLS[config["type"]](
        embedding_layer.embedding_dim, config["h_size"], config["n_ary"]
    )
    classifier = TreeNetClassifier(
        embedding_layer, cell, config["h_size"], train.num_classes
    )
    optimizer = th.optim.Adagrad(
        classifier.parameters(), lr=config["lr"], weight_decay=1e-4
    )

    best_acc = 0.0
    best_model = None
    batch_size = config["batch_size"]

    for epoch in range(config["epochs"]):
        total_train_loss = 0
        for graph in make_data_loader(train, batch_size):
            response = classifier(graph, "x", "y")
            probabilities = F.log_softmax(response, 1)
            loss = F.nll_loss(probabilities, graph.ndata["y"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = evaluate_classifier(classifier, valid, batch_size)
        print("Epoch {:05d} | Val Acc {:.4f}".format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(classifier)
            th.save(best_model.state_dict(), config["save_path"])

    acc = evaluate_classifier(best_model, test, batch_size)
    print("Test accuracy {:.4f}".format(acc))
