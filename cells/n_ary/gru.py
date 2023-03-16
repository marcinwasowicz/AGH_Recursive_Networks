import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class NTreeGRU(nn.Module):
    def __init__(self, x_size, h_size, n_ary):
        super(NTreeGRU, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, h_size)

        self.U_r = nn.Linear(n_ary * h_size, h_size)
        self.U_h_candidate = nn.Linear(n_ary * h_size, h_size)
        self.U_z = nn.Linear(n_ary * h_size, n_ary * h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)
        r = th.sigmoid(self.U_r(h_cat))
        h_candidate = th.tanh(self.U_h_candidate(r.repeat(1, self._n_ary) * h_cat))
        z = self.U_z(h_cat)
        z = th.sigmoid(z).view(z.size(0), self._n_ary, self._h_size)
        z_sum = th.sum(z, 1)
        h = h_cat.view(h_cat.size(0), self._n_ary, self._h_size)
        h = th.sum(h * z, 1) + (th.ones(*z_sum.size()) - z_sum) * h_candidate
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
