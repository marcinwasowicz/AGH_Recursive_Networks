import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class SingleForgetGateTreeGRU(nn.Module):
    def __init__(self, x_size, h_size, n_ary) -> None:
        super(SingleForgetGateTreeGRU, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, h_size)

        self.U_zr = nn.Linear(n_ary * h_size, 2 * h_size)
        self.U_h_candidate = nn.Linear(n_ary * h_size, h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)
        z, r = th.tensor_split(self.U_zr(h_cat), [self._h_size], 1)
        z, r = th.sigmoid(z), th.sigmoid(r)
        h_candidate = self.U_h_candidate(r.repeat(1, self._n_ary) * h_cat)
        h_candidate = th.tanh(h_candidate)
        h = th.sum(nodes.mailbox["h"], 1) * z + (th.ones(*z.size()) - z) * h_candidate
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
