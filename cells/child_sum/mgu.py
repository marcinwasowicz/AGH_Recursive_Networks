import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class ChildSumTreeMGU(nn.Module):
    def __init__(self, x_size, h_size, *args) -> None:
        super(ChildSumTreeMGU, self).__init__()
        self._h_size = h_size

        self.W = nn.Linear(x_size, 2 * h_size)

        self.U_h_candidate = nn.Linear(h_size, h_size, bias=False)
        self.U_f = nn.Linear(h_size, h_size, bias=False)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        f = self.U_f(nodes.mailbox["h"])
        h_sum = th.sum(nodes.mailbox["h"], 1)
        wx = nodes.data["wx"]
        h_candidate = th.sum(self.U_h_candidate(f * nodes.mailbox["h"]), 1)
        wx[:, 0 : self._h_size] += h_candidate
        return {"wx": wx, "f": f, "h": th.sum(f * nodes.mailbox["h"], 1)}

    def _update_function(self, nodes):
        wx = nodes.data["wx"]
        w_h_candidate_x, w_f_x = th.tensor_split(wx, [self._h_size], 1)
        f = nodes.data["f"]
        f = th.sigmoid(f + w_f_x.unsqueeze(1).repeat(1, f.size(1), 1))
        h_candidate = th.tanh(w_h_candidate_x)
        f_sum = th.sum(f, 1)
        h = nodes.data["h"] + (th.ones(*f_sum.size()) - f_sum) * h_candidate
        return {"h": h}

    def forward(self, input):
        x = input.get_embeddings()
        n = input.get_graph().number_of_nodes()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())

        input.get_graph().ndata["wx"] = self.W(x)
        input.get_graph().ndata["h"] = th.zeros((n, self._h_size))
        input.get_graph().ndata["f"] = th.zeros((n, 1, self._h_size))

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
            apply_node_func=self._update_function,
        )

        return input.get_graph().ndata["h"]
