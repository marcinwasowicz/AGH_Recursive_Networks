import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, *args) -> None:
        super(ChildSumTreeLSTM, self).__init__()
        self._h_size = h_size

        self.W = nn.Linear(x_size, 4 * h_size)

        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.U_f = nn.Linear(h_size, h_size, bias=False)

    def _message_function(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def _reduce_function(self, nodes):
        wx = nodes.data["wx"]
        w_iou_x, wx_f = th.tensor_split(wx, [3 * self._h_size], dim=1)

        f = self.U_f(nodes.mailbox["h"])
        f = th.sigmoid(f + wx_f.unsqueeze(1).repeat(1, f.size(1), 1))
        c = th.sum(f * nodes.mailbox["c"], 1)

        h_sum = th.sum(nodes.mailbox["h"], 1)

        return {"wx": self.U_iou(h_sum) + w_iou_x, "c": c}

    def _update_function(self, nodes):
        wx = nodes.data["wx"]
        w_i_x, w_o_x, w_u_x, _w_f_x = th.tensor_split(
            wx, [self._h_size, 2 * self._h_size, 3 * self._h_size], 1
        )
        i, o, u = th.sigmoid(w_i_x), th.sigmoid(w_o_x), th.tanh(w_u_x)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        n = input.get_graph().number_of_nodes()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())

        input.get_graph().ndata["wx"] = self.W(x)
        input.get_graph().ndata["h"] = th.zeros((n, self._h_size))
        input.get_graph().ndata["c"] = th.zeros((n, self._h_size))

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
            apply_node_func=self._update_function,
        )

        return input.get_graph().ndata["h"]
