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
        lines = [line.strip().split(" ") for line in f]
    return lines


def process_data(parents, tokens, embeddings):
    sources = [i for i, p in enumerate(parents) if p != 0]
    destinations = [p - 1 for p in parents if p != 0]

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
    return graph


def create_split(
    path_parents_a,
    path_tokens_a,
    path_parents_b,
    path_tokens_b,
    path_similarities,
    embeddings,
):
    parents_a = [[int(j) for j in i] for i in read_and_split_lines(path_parents_a)]
    tokens_a = read_and_split_lines(path_tokens_a)

    parents_b = [[int(j) for j in i] for i in read_and_split_lines(path_parents_b)]
    tokens_b = read_and_split_lines(path_tokens_b)

    similarities = [float(i[0]) for i in read_and_split_lines(path_similarities)]
    graphs_a = [process_data(p, t, embeddings) for p, t in zip(parents_a, tokens_a)]
    graphs_b = [process_data(p, t, embeddings) for p, t in zip(parents_b, tokens_b)]

    return (graphs_a, graphs_b, similarities)


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    raw_embeddings = gensim.downloader.load(config["embeddings"])
    vocabulary = [i[0] for i in read_and_split_lines("data/sick/vocab-cased.txt")]

    new_word_count = update_embeddings(raw_embeddings, vocabulary)
    print(f"Encountered {new_word_count} unknown words out of {len(vocabulary)}")

    train = create_split(
        "data/sick/train/a.cparents",
        "data/sick/train/a.toks",
        "data/sick/train/b.cparents",
        "data/sick/train/b.toks",
        "data/sick/train/sim.txt",
        raw_embeddings,
    )
    valid = create_split(
        "data/sick/dev/a.cparents",
        "data/sick/dev/a.toks",
        "data/sick/dev/b.cparents",
        "data/sick/dev/b.toks",
        "data/sick/dev/sim.txt",
        raw_embeddings,
    )
    test = create_split(
        "data/sick/test/a.cparents",
        "data/sick/test/a.toks",
        "data/sick/test/b.cparents",
        "data/sick/test/b.toks",
        "data/sick/test/sim.txt",
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
        f"embeddings/sick_constituency_{config['embeddings']}_embeddings.pt",
    )

    with open(
        f"data/sick_constituency_train_{config['embeddings']}.pkl", "wb+"
    ) as train_fd:
        pickle.dump(train, train_fd)

    with open(
        f"data/sick_constituency_valid_{config['embeddings']}.pkl", "wb+"
    ) as valid_fd:
        pickle.dump(valid, valid_fd)

    with open(
        f"data/sick_constituency_test_{config['embeddings']}.pkl", "wb+"
    ) as test_fd:
        pickle.dump(test, test_fd)
