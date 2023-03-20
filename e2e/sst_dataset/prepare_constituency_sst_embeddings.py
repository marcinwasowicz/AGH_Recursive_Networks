import json
import sys
import os

import dgl
import gensim.downloader
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import KeyedVectors
import numpy as np
import pickle
import torch as th

NUM_DATA_DIR = os.path.expandvars("$SCRATCH")


def update_glove_embeddings(glove_embeddings, vocabulary):
    current_vocabulary_count = len(glove_embeddings.index_to_key)
    new_word_count = 0

    for word in vocabulary:
        if word not in glove_embeddings.key_to_index:
            new_word_count += 1
            current_vocabulary_count += 1
            glove_embeddings.index_to_key.append(word)
            glove_embeddings.key_to_index[word] = current_vocabulary_count - 1
    return new_word_count


def update_fasttext_embeddings(fasttext_embeddings, vocabulary):
    current_vocabulary_count = len(fasttext_embeddings.index_to_key)
    extrapolated_word_count = 0
    extrapolated_embeddings = []

    for word in vocabulary:
        if word not in fasttext_embeddings.key_to_index:
            extrapolated_word_count += 1
            current_vocabulary_count += 1
            extrapolated_embedding = fasttext_embeddings[word]
            extrapolated_embeddings.append(extrapolated_embedding)

            fasttext_embeddings.index_to_key.append(word)
            fasttext_embeddings.key_to_index[word] = current_vocabulary_count - 1
    extrapolated_embeddings = np.array(extrapolated_embeddings)
    return extrapolated_embeddings, extrapolated_word_count


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

    for embeddings_name in config["embeddings"]:
        vocabulary = [i[0] for i in read_and_split_lines("data/sst/vocab-cased.txt")]
        if embeddings_name.startswith("glove"):
            if embeddings_name.startswith("glove.840B.300d"):
                raw_embeddings = KeyedVectors.load_word2vec_format(
                    f"{NUM_DATA_DIR}/{embeddings_name}.txt",
                    binary=False,
                    no_header=True,
                )
            else:
                raw_embeddings = gensim.downloader.load(embeddings_name)
            new_word_count = update_glove_embeddings(raw_embeddings, vocabulary)
            print(
                f"Encountered {new_word_count} unknown words out of {len(vocabulary)} for {embeddings_name}"
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
        elif embeddings_name.startswith("fasttext"):
            raw_embeddings = load_facebook_vectors(
                f"{NUM_DATA_DIR}/{embeddings_name}.bin"
            )
            (
                extrapolated_embeddings,
                extrapolated_word_count,
            ) = update_fasttext_embeddings(raw_embeddings, vocabulary)
            print(
                f"Extrapolated embeddings for {extrapolated_word_count} words out of {len(vocabulary)} for {embeddings_name}"
            )
            no_input_embedding = np.zeros((1, raw_embeddings.vector_size))
            embeddings = np.concatenate(
                [raw_embeddings.vectors, extrapolated_embeddings, no_input_embedding],
                axis=0,
            )

        embeddings = th.from_numpy(embeddings).float()
        th.save(
            embeddings,
            f"{NUM_DATA_DIR}/sst_constituency_{embeddings_name}_embeddings.pt",
        )

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

        with open(
            f"{NUM_DATA_DIR}/sst_constituency_train_{embeddings_name}.pkl", "wb+"
        ) as train_fd:
            pickle.dump(train, train_fd)

        with open(
            f"{NUM_DATA_DIR}/sst_constituency_valid_{embeddings_name}.pkl", "wb+"
        ) as valid_fd:
            pickle.dump(valid, valid_fd)

        with open(
            f"{NUM_DATA_DIR}/sst_constituency_test_{embeddings_name}.pkl", "wb+"
        ) as test_fd:
            pickle.dump(test, test_fd)
