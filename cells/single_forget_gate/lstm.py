import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class SingleForgetGateTreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, n_ary) -> None:
        super(SingleForgetGateTreeLSTM, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, 4 * h_size)

        self.U = nn.Linear(n_ary * h_size, 4 * h_size, bias=False)

    def _message_function(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)

        wx = nodes.data["wx"]
        wx += self.U(h_cat)
        _w_iou_x, f_x = th.tensor_split(wx, [3 * self._h_size], dim=1)
        f = th.sigmoid(f_x)
        c_padding_size = self._n_ary - nodes.mailbox["c"].size(1)
        c_padding = h_cat.new_zeros(
            size=(nodes.mailbox["c"].size(0), c_padding_size, self._h_size)
        )
        c = th.cat((nodes.mailbox["c"], c_padding), dim=1)
        c = f * th.sum(c, 1)
        return {"wx": wx, "c": c}

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
