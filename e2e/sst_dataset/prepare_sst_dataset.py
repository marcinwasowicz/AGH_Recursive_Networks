import json
import pickle
import sys

from dgl.data.tree import SSTDataset
import gensim.downloader
import numpy as np
import torch as th


def adjust_raw_embeddings(raw_embeddings, sst_dataset_split):
    current_vocabulary_count = len(raw_embeddings.index_to_key)
    new_word_count = 0
    idx_mapping = {}

    for word, idx in sst_dataset_split.vocab.items():
        if word not in raw_embeddings.key_to_index:
            new_word_count += 1
            current_vocabulary_count += 1
            raw_embeddings.index_to_key.append(word)
            raw_embeddings.key_to_index[word] = current_vocabulary_count - 1
        idx_mapping[idx] = raw_embeddings.key_to_index[word]
    return new_word_count, idx_mapping


def remap_embeddings(sst_dataset_split, idx_mapping, no_input_embedding_idx):
    for graph in sst_dataset_split:
        for i in range(len(graph.ndata["x"])):
            graph.ndata["x"][i] = (
                idx_mapping[int(graph.ndata["x"][i])]
                if graph.ndata["x"][i] != -1
                else no_input_embedding_idx
            )


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    raw_embeddings = gensim.downloader.load(config["embeddings"])

    train = SSTDataset(mode="train")
    valid = SSTDataset(mode="dev")
    test = SSTDataset(mode="test")

    train_new_word_count, train_index_mapping = adjust_raw_embeddings(
        raw_embeddings, train
    )
    valid_new_word_count, valid_index_mapping = adjust_raw_embeddings(
        raw_embeddings, valid
    )
    test_new_word_count, test_index_mapping = adjust_raw_embeddings(
        raw_embeddings, test
    )

    new_word_count = train_new_word_count + valid_new_word_count + test_new_word_count
    print(f"Encountered {new_word_count} unknown words out of {train.vocab_size}")

    new_embeddings = np.array(th.nn.Embedding(new_word_count, raw_embeddings.vector_size).weight.data)
    no_input_embedding = np.zeros((1, raw_embeddings.vector_size))
    embeddings = np.concatenate(
        [raw_embeddings.vectors, new_embeddings, no_input_embedding], axis=0
    )
    embeddings = th.from_numpy(embeddings).float()
    th.save(
        embeddings,
        f"embeddings/sst_{config['embeddings']}_embeddings.pt",
    )

    no_input_embedding_idx = len(embeddings) - 1
    remap_embeddings(train, train_index_mapping, no_input_embedding_idx)
    remap_embeddings(test, test_index_mapping, no_input_embedding_idx)
    remap_embeddings(valid, valid_index_mapping, no_input_embedding_idx)

    with open(f"data/sst_train_{config['embeddings']}.pkl", "wb+") as train_fd:
        pickle.dump(train, train_fd)

    with open(f"data/sst_valid_{config['embeddings']}.pkl", "wb+") as valid_fd:
        pickle.dump(valid, valid_fd)

    with open(f"data/sst_test_{config['embeddings']}.pkl", "wb+") as test_fd:
        pickle.dump(test, test_fd)
