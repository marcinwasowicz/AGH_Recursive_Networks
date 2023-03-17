import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class ChildSumTreeGRU(nn.Module):
    def __init__(self, x_size, h_size, *args) -> None:
        super(ChildSumTreeGRU, self).__init__()
        self._h_size = h_size

        self.W = nn.Linear(x_size, h_size)

        self.U_r = nn.Linear(h_size, h_size)
        self.U_h_candidate = nn.Linear(h_size, h_size)
        self.U_z = nn.Linear(h_size, h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        h_sum = th.sum(nodes.mailbox["h"], 1)
        r = th.sigmoid(self.U_r(h_sum))
        h_candidate = th.tanh(self.U_h_candidate(r * h_sum))
        z = th.sigmoid(self.U_z(nodes.mailbox["h"]))
        z_sum = th.sum(z, 1)
        h = th.sum(z * nodes.mailbox["h"], 1)
        h = h + (th.ones(*z_sum.size()) - z_sum) * h_candidate
        return {"h": h}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())
        if th.cuda.is_available():
            nodes_generator = map(lambda x: x.to("cuda:0"), nodes_generator)

        initial_state = th.tanh(self.W(x))
        input.get_graph().ndata["h"] = initial_state

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
        )

        return input.get_graph().ndata["h"]
