from torch import embedding
import torch.nn as nn

from cells import RecursiveCellInput


class TreeNetClassifier(nn.Module):
    def __init__(self, embedding, cell, cell_h_size, num_classes) -> None:
        super(TreeNetClassifier, self).__init__()
        self.embedding = embedding
        self.cell = cell
        self.dropout = nn.Dropout()
        self.response_layer = nn.Linear(cell_h_size, num_classes)

    def forward(self, graph, word_key, label_key, mask_key):
        embeddings = self.embedding(graph.ndata[word_key] * graph.ndata[mask_key])
        labels = graph.ndata[label_key]
        cell_input = RecursiveCellInput(graph, embeddings, labels)
        return self.response_layer(self.dropout(self.cell(cell_input)))
