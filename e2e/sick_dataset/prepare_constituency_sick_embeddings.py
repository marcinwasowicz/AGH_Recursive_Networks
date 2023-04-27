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
from transformers import DistilBertTokenizerFast, DistilBertModel

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


def process_data_bert(parents, tokens, embeddings, bert_model, bert_tokenizer):
    sources = [i for i, p in enumerate(parents) if p != 0]
    destinations = [p - 1 for p in parents if p != 0]

    graph = dgl.graph((sources, destinations), num_nodes=len(parents))

    tokenizer_output = bert_tokenizer(
        tokens, return_offsets_mapping=True, is_split_into_words=True
    )
    bert_token_ids, offset_mapping = (
        tokenizer_output["input_ids"],
        tokenizer_output["offset_mapping"][1:-1],
    )
    bert_token_embeddings = bert_model(th.tensor([bert_token_ids]))[
        "last_hidden_state"
    ][0][1:-1]

    split_word_count = 1
    start_token_idx = len(embeddings) + 1

    for bert_token_id, bert_token_offset in enumerate(offset_mapping):
        bert_token_embedding = (
            bert_token_embeddings[bert_token_id].cpu().detach().numpy()
        )
        if bert_token_offset[0] == 0:
            embeddings.append(bert_token_embedding)
            split_word_count = 1
            continue
        embeddings[-1] = embeddings[-1] * split_word_count + bert_token_embedding
        split_word_count += 1
        embeddings[-1] = embeddings[-1] / split_word_count

    idx = []
    token_idx = start_token_idx

    for deg in graph.in_degrees().tolist():
        if deg != 0:
            idx.append(0)
            continue
        idx.append(token_idx)
        token_idx += 1
    graph.ndata["x"] = th.tensor(idx)
    return graph


def create_split(
    path_parents_a,
    path_tokens_a,
    path_parents_b,
    path_tokens_b,
    path_similarities,
    embeddings,
    bert_tokenizer=None,
    bert_model=None,
):
    parents_a = [[int(j) for j in i] for i in read_and_split_lines(path_parents_a)]
    tokens_a = read_and_split_lines(path_tokens_a)

    parents_b = [[int(j) for j in i] for i in read_and_split_lines(path_parents_b)]
    tokens_b = read_and_split_lines(path_tokens_b)

    similarities = [float(i[0]) for i in read_and_split_lines(path_similarities)]
    if bert_model is not None and bert_tokenizer is not None:
        graphs_a = [
            process_data_bert(p, t, embeddings, bert_model, bert_tokenizer)
            for p, t in zip(parents_a, tokens_a)
        ]
        graphs_b = [
            process_data_bert(p, t, embeddings, bert_model, bert_tokenizer)
            for p, t in zip(parents_b, tokens_b)
        ]
    else:
        graphs_a = [process_data(p, t, embeddings) for p, t in zip(parents_a, tokens_a)]
        graphs_b = [process_data(p, t, embeddings) for p, t in zip(parents_b, tokens_b)]

    return (graphs_a, graphs_b, similarities)


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    for embeddings_name in config["embeddings"]:
        vocabulary = [i[0] for i in read_and_split_lines("data/sick/vocab-cased.txt")]
        bert_model = None
        bert_tokenizer = None

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
        elif embeddings_name.startswith("distilbert"):
            bert_tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased"
            )
            bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            raw_embeddings = []
            embeddings = raw_embeddings

        train = create_split(
            "data/sick/train/a.cparents",
            "data/sick/train/a.toks",
            "data/sick/train/b.cparents",
            "data/sick/train/b.toks",
            "data/sick/train/sim.txt",
            raw_embeddings,
            bert_tokenizer,
            bert_model,
        )
        valid = create_split(
            "data/sick/dev/a.cparents",
            "data/sick/dev/a.toks",
            "data/sick/dev/b.cparents",
            "data/sick/dev/b.toks",
            "data/sick/dev/sim.txt",
            raw_embeddings,
            bert_tokenizer,
            bert_model,
        )
        test = create_split(
            "data/sick/test/a.cparents",
            "data/sick/test/a.toks",
            "data/sick/test/b.cparents",
            "data/sick/test/b.toks",
            "data/sick/test/sim.txt",
            raw_embeddings,
            bert_tokenizer,
            bert_model,
        )

        if embeddings_name.startswith("distilbert"):
            embeddings = np.array([np.zeros_like(raw_embeddings[0])] + raw_embeddings)

        embeddings = th.from_numpy(embeddings).float()
        th.save(
            embeddings,
            f"{NUM_DATA_DIR}/sick_constituency_{embeddings_name}_embeddings.pt",
        )

        with open(
            f"{NUM_DATA_DIR}/sick_constituency_train_{embeddings_name}.pkl", "wb+"
        ) as train_fd:
            pickle.dump(train, train_fd)

        with open(
            f"{NUM_DATA_DIR}/sick_constituency_valid_{embeddings_name}.pkl", "wb+"
        ) as valid_fd:
            pickle.dump(valid, valid_fd)

        with open(
            f"{NUM_DATA_DIR}/sick_constituency_test_{embeddings_name}.pkl", "wb+"
        ) as test_fd:
            pickle.dump(test, test_fd)
