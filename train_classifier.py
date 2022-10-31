from copy import deepcopy
import json
import sys

import dgl
from dgl.data.tree import SSTDataset
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.utils.data import DataLoader

from cells import NTreeGRU, NTreeLSTM, NTreeMGU, ChildSumTreeLSTM, ChildSumTreeGRU
from networks import TreeNetClassifier

config_path = sys.argv[1]
with open(config_path, "r") as fd:
    CONFIG = json.load(fd)

X_SIZE = 300
CELLS = {
    "lstm": NTreeLSTM,
    "mgu": NTreeMGU,
    "gru": NTreeGRU,
    "child_sum_lstm": ChildSumTreeLSTM,
    "child_sum_gru": ChildSumTreeGRU,
}


def make_data_loader(dataset):
    return DataLoader(
        dataset, CONFIG["batch_size"], True, collate_fn=lambda x: dgl.batch(x)
    )


if __name__ == "__main__":
    train = SSTDataset(mode="train")
    val = SSTDataset(mode="dev")
    test = SSTDataset(mode="test")

    embedding_layer = nn.Embedding(train.vocab_size, X_SIZE)
    cell = CELLS[CONFIG["type"]](X_SIZE, CONFIG["h_size"], CONFIG["n_ary"])
    classifier = TreeNetClassifier(
        embedding_layer, cell, CONFIG["h_size"], train.num_classes
    )

    optimizer = th.optim.Adagrad(cell.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    best_acc = 0.0
    best_model = None

    for epoch in range(CONFIG["epochs"]):
        total_train_loss = 0
        for graph in make_data_loader(train):
            response = classifier(graph, "x", "y", "mask")
            probabilities = F.log_softmax(response, 1)
            loss = F.nll_loss(probabilities, graph.ndata["y"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_correct = 0
        total = 0
        with th.no_grad():
            for graph in make_data_loader(val):
                response = classifier(graph, "x", "y", "mask")
                pred = th.argmax(response, 1)
                total_correct += float(th.sum(th.eq(graph.ndata["y"], pred)))
                total += len(graph.ndata["y"])

        acc = total_correct / total
        print("Epoch {:05d} | Val Acc {:.4f}".format(epoch, acc))
        if acc > best_acc:
            best_acc = acc
            best_model = deepcopy(classifier)
            th.save(best_model.state_dict(), CONFIG["save_path"])

    test_total_correct = 0
    test_total = 0

    with th.no_grad():
        for graph in make_data_loader(test):
            response = best_model(graph, "x", "y", "mask")
            pred = th.argmax(response, 1)
            test_total_correct += float(th.sum(th.eq(graph.ndata["y"], pred)))
            test_total += len(graph.ndata["y"])

    acc = test_total_correct / test_total
    print("Test accuracy {:.4f}".format(acc))
