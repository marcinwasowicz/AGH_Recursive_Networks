import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class ChildSumTreeGRU(nn.Module):
    def __init__(self, x_size, h_size, *args) -> None:
        super(ChildSumTreeGRU, self).__init__()
        self._h_size = h_size

        self.W = nn.Linear(x_size, 3 * h_size)

        self.U_r = nn.Linear(h_size, h_size, bias=False)
        self.U_h_candidate = nn.Linear(h_size, h_size, bias=False)
        self.U_z = nn.Linear(h_size, h_size, bias=False)

    def _message_function(self, edges):
        return {"h": edges.src["h"]}

    def _reduce_function(self, nodes):
        wx = nodes.data["wx"]
        h_sum = th.sum(nodes.mailbox["h"], 1)
        w_r_x, _w_h_candidate_x, _w_z_x = th.tensor_split(
            wx, [self._h_size, 2 * self._h_size], 1
        )
        r = th.sigmoid(w_r_x + self.U_r(h_sum))
        h_candidate = self.U_h_candidate(r * h_sum)
        wx[:, self._h_size : 2 * self._h_size] += h_candidate
        z = self.U_z(nodes.mailbox["h"])
        return {"wx": wx, "z": z, "h": th.sum(z * nodes.mailbox["h"], 1)}

    def _update_function(self, nodes):
        wx = nodes.data["wx"]
        _w_r_x, w_h_candidate_x, w_z_x = th.tensor_split(
            wx, [self._h_size, 2 * self._h_size], 1
        )
        z = nodes.data["z"]
        z = th.sigmoid(z + w_z_x.unsqueeze(1).repeat(1, z.size(1), 1))
        h_candidate = th.tanh(w_h_candidate_x)
        z_sum = th.sum(z, 1)
        h = nodes.data["h"] + (th.ones(*z_sum.size()) - z_sum) * h_candidate
        return {"h": h}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        n = input.get_graph().number_of_nodes()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())

        input.get_graph().ndata["wx"] = self.W(x)
        input.get_graph().ndata["h"] = th.zeros((n, self._h_size))
        input.get_graph().ndata["z"] = th.zeros((n, 1, self._h_size))

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
            apply_node_func=self._update_function,
        )

        return input.get_graph().ndata["h"]
