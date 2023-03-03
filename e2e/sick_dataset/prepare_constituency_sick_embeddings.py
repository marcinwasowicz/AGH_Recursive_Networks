import json
import sys

import dgl
import gensim
import torch as th


def update_embeddings(glove_embeddings, vocabulary):
    # read the vocabulary and create extended embeddings
    pass


def read_and_split_lines(file_path):
    # read lines and split by spaces
    pass


def process_data(parents, tokens):
    # basically create a list of graphs representing each sentence
    # graph should just contain key "x" holding embeddings
    pass


def create_split(
    path_parents_a, path_tokens_a, path_parents_b, path_tokens_b, path_similarities
):
    # use functions above to create tuple of three lists:
    # (graphs of a, graphs of b, similarities)
    pass


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config_fd:
        config = json.load(config_fd)

    # use config to create three splits and save them as files
    pass
