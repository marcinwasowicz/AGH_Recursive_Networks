import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class NTreeGRU(nn.Module):
    def __init__(self, x_size, h_size, n_ary):
        super(NTreeGRU, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, (2 + n_ary) * h_size)

        self.U_r = nn.Linear(n_ary * h_size, h_size, bias=False)
        self.U_h_candidate = nn.Linear(n_ary * h_size, h_size, bias=False)
        self.U_z = nn.Linear(n_ary * h_size, n_ary * h_size, bias=False)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)
        wx = nodes.data["wx"]
        w_r_x, _w_h_candidate_x, w_z_x = th.tensor_split(
            wx, [self._h_size, 2 * self._h_size], 1
        )
        r = th.sigmoid(w_r_x + self.U_r(h_cat))
        h_candidate = self.U_h_candidate(r.repeat(1, self._n_ary) * h_cat)
        wx[:, self._h_size : 2 * self._h_size] += h_candidate
        z = self.U_z(h_cat)
        return {"wx": wx, "z": z, "h_cat": h_cat}

    def _update_function(self, nodes):
        wx = nodes.data["wx"]
        _w_r_x, w_h_candidate_x, w_z_x = th.tensor_split(
            wx, [self._h_size, 2 * self._h_size], 1
        )
        z = th.sigmoid(nodes.data["z"] + w_z_x).view(
            nodes.data["z"].size(0), self._n_ary, self._h_size
        )
        h_candidate = th.tanh(w_h_candidate_x)
        z_sum = th.sum(z, 1)
        h = nodes.data["h_cat"].view(
            nodes.data["h_cat"].size(0), self._n_ary, self._h_size
        )
        h = th.sum(h * z, 1) + (th.ones(*z_sum.size()) - z_sum) * h_candidate
        return {"h": h}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        n = input.get_graph().number_of_nodes()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())

        input.get_graph().ndata["wx"] = self.W(x)
        input.get_graph().ndata["h"] = th.zeros((n, self._h_size))
        input.get_graph().ndata["z"] = th.zeros((n, self._n_ary * self._h_size))
        input.get_graph().ndata["h_cat"] = th.zeros((n, self._n_ary * self._h_size))

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
            apply_node_func=self._update_function,
        )

        return input.get_graph().ndata["h"]
