import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class NTreeMGU(nn.Module):
    def __init__(self, x_size, h_size, n_ary):
        super(NTreeMGU, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, h_size)

        self.U_h_candidate = nn.Linear(n_ary * h_size, h_size)
        self.U_f = nn.Linear(n_ary * h_size, n_ary * h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)
        f = th.sigmoid(self.U_f(h_cat))
        h_candidate = th.tanh(self.U_h_candidate(f * h_cat))
        f = f.view(f.size(0), self._n_ary, self._h_size)
        h = h_cat.view(h_cat.size(0), self._n_ary, self._h_size)
        f_sum = th.sum(f, 1)
        h = (th.ones(*f_sum.size()) - f_sum) * h_candidate + th.sum(f * h, 1)
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
