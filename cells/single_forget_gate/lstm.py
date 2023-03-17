import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class SingleForgetGateTreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, n_ary) -> None:
        super(SingleForgetGateTreeLSTM, self).__init__()
        self._n_ary = n_ary
        self._h_size = h_size

        self.W = nn.Linear(x_size, 2 * h_size)

        self.U = nn.Linear(n_ary * h_size, 4 * h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def _reduce_function(self, nodes):
        h = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
        h_padding_size = self._n_ary - nodes.mailbox["h"].size(1)
        h_padding = h.new_zeros(
            size=(nodes.mailbox["h"].size(0), h_padding_size * self._h_size)
        )
        h_cat = th.cat((h, h_padding), dim=1)

        i, o, u, f = th.tensor_split(
            self.U(h_cat), [self._h_size, 2 * self._h_size, 3 * self._h_size], 1
        )
        i, o, u, f = th.sigmoid(i), th.sigmoid(o), th.tanh(u), th.sigmoid(f)
        c_padding_size = self._n_ary - nodes.mailbox["c"].size(1)
        c_padding = h_cat.new_zeros(
            size=(nodes.mailbox["c"].size(0), c_padding_size, self._h_size)
        )
        c = th.cat((nodes.mailbox["c"], c_padding), dim=1)
        c = i * u + f * th.sum(c, 1)
        h = o * th.tanh(c)
        return {"h": h, "c": c}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())
        if th.cuda.is_available():
            nodes_generator = map(lambda x: x.to("cuda:0"), nodes_generator)

        initial_state = th.tanh(self.W(x))
        h, c = th.tensor_split(initial_state, [self._h_size], 1)
        input.get_graph().ndata["h"] = h
        input.get_graph().ndata["c"] = c

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
        )

        return input.get_graph().ndata["h"]
