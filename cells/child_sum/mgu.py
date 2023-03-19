import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class ChildSumTreeMGU(nn.Module):
    def __init__(self, x_size, h_size, *args) -> None:
        super(ChildSumTreeMGU, self).__init__()
        self._h_size = h_size

        self.W = nn.Linear(x_size, h_size)

        self.U_h_candidate = nn.Linear(h_size, h_size)
        self.U_f = nn.Linear(h_size, h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        f_dot_h = f * nodes.mailbox["h"]
        h_candidate = th.tanh(th.sum(self.U_h_candidate(f_dot_h), 1))
        f_sum = th.sum(f, 1)
        h = th.sum(f_dot_h, 1) + (th.ones_like(f_sum) - f_sum) * h_candidate
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
