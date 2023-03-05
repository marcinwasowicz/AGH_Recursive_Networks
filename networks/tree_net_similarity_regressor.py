import torch as th
import torch.nn as nn
import torch.nn.functional as F

from cells import RecursiveCellInput


class TreeNetSimilarityRegressor(nn.Module):
    def __init__(self, embedding, cell, cell_h_size, similarity_h_size, num_classes):
        super(TreeNetSimilarityRegressor, self).__init__()
        self.embedding = embedding
        self.cell = cell
        self.similarity_layer = nn.Linear(2 * cell_h_size, similarity_h_size)
        self.response_layer = nn.Linear(similarity_h_size, num_classes)

    def forward(self, graph_a, graph_b, word_key):
        embeddings_a = self.embedding(graph_a.ndata[word_key])
        cell_input_a = RecursiveCellInput(graph_a, embeddings_a)

        embeddings_b = self.embedding(graph_b.ndata[word_key])
        cell_input_b = RecursiveCellInput(graph_b, embeddings_b)

        final_state_a = self.cell(cell_input_a)[graph_a.out_degrees() == 0]
        final_state_b = self.cell(cell_input_b)[graph_b.out_degrees() == 0]

        angle = final_state_a * final_state_b
        distance = th.abs(final_state_a - final_state_b)

        similarity_state = th.sigmoid(
            self.similarity_layer(th.concat((angle, distance), dim=1))
        )
        return F.softmax(self.response_layer(similarity_state), dim=-1)
