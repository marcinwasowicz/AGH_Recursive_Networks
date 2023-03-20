from copy import deepcopy
import json
import pickle
import sys
import warnings
import os

import dgl
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
    model_type,
    embeddings,
    lr,
    h_size,
    batch_size,
    n_ary,
    num_classes,
    epochs,
    repeat,
    device,
):
    print(
        "Training process for the following config:\nmodel type: {}\nlearning rate: {}\nhidden size: {}\nbatch size: {}\nembeddings: {}\nrepeat :{}".format(
            model_type, lr, h_size, batch_size, embeddings, repeat
        )
    )
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

    embedding_layer = nn.Embedding.from_pretrained(
        th.load(f"{NUM_DATA_DIR}/sst_constituency_{embeddings}_embeddings.pt")
    )
    cell = CELLS[model_type](embedding_layer.embedding_dim, h_size, n_ary)
    classifier = TreeNetClassifier(embedding_layer, cell, h_size, num_classes)
    classifier.to(device)
    optimizer = th.optim.Adagrad(classifier.parameters(), lr=lr, weight_decay=1e-4)

    best_acc = 0.0
    best_model = None
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
            best_model = deepcopy(classifier)

    acc = evaluate_classifier(best_model, test, batch_size, device)
    print("Test accuracy {:.4f}".format(acc))


if __name__ == "__main__":
    th.manual_seed = 42
    global_device = "cuda:0" if th.cuda.is_available() else "cpu"
    print(f"Using device type: {global_device}")
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    for model_type in config["model_types"]:
        for lr in config["lrs"]:
            for h_size in config["h_sizes"]:
                for batch_size in config["batch_sizes"]:
                    for embeddings in config["embeddings"]:
                        for repeat in range(int(config["repeats"])):
                            train_classifier(
                                model_type,
                                embeddings,
                                lr,
                                h_size,
                                batch_size,
                                config["n_ary"],
                                config["num_classes"],
                                config["epochs"],
                                repeat,
                                global_device,
                            )
