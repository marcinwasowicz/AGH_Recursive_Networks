import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class NTreeMGU(nn.Module):
    def __init__(self, x_size, h_size, n_ary):
        super(NTreeMGU, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, (1 + n_ary) * h_size)

        self.U_h_candidate = nn.Linear(n_ary * h_size, h_size, bias=False)
        self.U_f = nn.Linear(n_ary * h_size, n_ary * h_size, bias=False)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)
        f = self.U_f(h_cat)
        wx = nodes.data["wx"]
        h_candidate = self.U_h_candidate(f * h)
        wx[:, 0 : self._h_size] += h_candidate
        return {
            "wx": wx,
            "f": f,
            "h": th.sum(
                (f * h).view(nodes.data["h"].size(0), self._n_ary, self._h_size), 1
            ),
        }

    def _update_function(self, nodes):
        wx = nodes.data["wx"]
        w_h_candidate_x, w_f_x = th.tensor_split(wx, [self._h_size], 1)
        f = th.sigmoid(nodes.data["f"] + w_f_x).view(
            nodes.data["h"].size(0), self._n_ary, self._h_size
        )
        h_candidate = th.tanh(w_h_candidate_x)
        f_sum = th.sum(f, 1)
        h = nodes.data["h"] + (th.ones(*f_sum.size()) - f_sum) * h_candidate
        return {"h": h}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        n = input.get_graph().number_of_nodes()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())

        input.get_graph().ndata["wx"] = self.W(x)
        input.get_graph().ndata["h"] = th.zeros((n, self._h_size))
        input.get_graph().ndata["f"] = th.zeros((n, self._n_ary * self._h_size))

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
            apply_node_func=self._update_function,
        )

        return input.get_graph().ndata["h"]
