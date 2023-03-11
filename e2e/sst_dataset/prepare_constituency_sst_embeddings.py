import json
import sys

import dgl
import gensim.downloader
import numpy as np
import pickle
import torch as th


def update_embeddings(glove_embeddings, vocabulary):
    current_vocabulary_count = len(glove_embeddings.index_to_key)
    new_word_count = 0

    for word in vocabulary:
        if word not in glove_embeddings.key_to_index:
            new_word_count += 1
            current_vocabulary_count += 1
            glove_embeddings.index_to_key.append(word)
            glove_embeddings.key_to_index[word] = current_vocabulary_count - 1
    return new_word_count


def read_and_split_lines(file_path):
    with open(file_path, "r") as f:
        lines = [line.strip().replace("\xa0", " ").split(" ") for line in f]
    return lines


def process_data(parents, tokens, labels, embeddings):
    sources = [i for i, p in enumerate(parents) if p != 0]
    destinations = [p - 1 for p in parents if p != 0]
    labels = [l + 2 for l in labels]
    graph = dgl.graph((sources, destinations), num_nodes=len(parents))
    no_input_embedding_idx = len(embeddings.index_to_key)

    iter = 0
    idx = []

    for deg in graph.in_degrees().tolist():
        if deg != 0:
            idx.append(no_input_embedding_idx)
            continue
        idx.append(embeddings.key_to_index[tokens[iter]])
        iter += 1
    graph.ndata["x"] = th.tensor(idx)
    graph.ndata["y"] = th.tensor(labels)
    return graph


def create_split(
    path_parents,
    path_tokens,
    path_labels,
    embeddings,
):
    parents = [[int(j) for j in i] for i in read_and_split_lines(path_parents)]
    tokens = read_and_split_lines(path_tokens)
    labels = [
        [int(j) if j != "None" else -1 for j in i]
        for i in read_and_split_lines(path_labels)
    ]
    graphs = [
        process_data(p, t, l, embeddings) for p, t, l in zip(parents, tokens, labels)
    ]
    return graphs


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    raw_embeddings = gensim.downloader.load(config["embeddings"])
    vocabulary = [i[0] for i in read_and_split_lines("data/sst/vocab-cased.txt")]

    new_word_count = update_embeddings(raw_embeddings, vocabulary)
    print(f"Encountered {new_word_count} unknown words out of {len(vocabulary)}")

    train = create_split(
        "data/sst/train/parents.txt",
        "data/sst/train/sents.toks",
        "data/sst/train/labels.txt",
        raw_embeddings,
    )
    valid = create_split(
        "data/sst/dev/parents.txt",
        "data/sst/dev/sents.toks",
        "data/sst/dev/labels.txt",
        raw_embeddings,
    )
    test = create_split(
        "data/sst/test/parents.txt",
        "data/sst/test/sents.toks",
        "data/sst/test/labels.txt",
        raw_embeddings,
    )

    new_embeddings = np.random.uniform(
        -1 / np.sqrt(raw_embeddings.vector_size),
        1 / np.sqrt(raw_embeddings.vector_size),
        size=(new_word_count, raw_embeddings.vector_size),
    )
    no_input_embedding = np.zeros((1, raw_embeddings.vector_size))
    embeddings = np.concatenate(
        [raw_embeddings.vectors, new_embeddings, no_input_embedding], axis=0
    )
    embeddings = th.from_numpy(embeddings).float()
    th.save(
        embeddings,
        f"embeddings/sst_constituency_{config['embeddings']}_embeddings.pt",
    )

    with open(
        f"data/sst_constituency_train_{config['embeddings']}.pkl", "wb+"
    ) as train_fd:
        pickle.dump(train, train_fd)

    with open(
        f"data/sst_constituency_valid_{config['embeddings']}.pkl", "wb+"
    ) as valid_fd:
        pickle.dump(valid, valid_fd)

    with open(
        f"data/sst_constituency_test_{config['embeddings']}.pkl", "wb+"
    ) as test_fd:
        pickle.dump(test, test_fd)
