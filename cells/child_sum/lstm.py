import dgl
import torch as th
import torch.nn as nn

from cells.cell_input import RecursiveCellInput


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, x_size, h_size, *args) -> None:
        super(ChildSumTreeLSTM, self).__init__()
        self._h_size = h_size

        self.W = nn.Linear(x_size, 3 * h_size)

        self.U_iou = nn.Linear(h_size, 3 * h_size)
        self.U_f = nn.Linear(h_size, h_size)

    def _message_function(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def _reduce_function(self, nodes):
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        h_sum = th.sum(nodes.mailbox["h"], 1)
        i, o, u = th.tensor_split(
            self.U_iou(h_sum), [self._h_size, 2 * self._h_size], 1
        )
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + th.sum(f * nodes.mailbox["c"], 1)
        h = o * th.tanh(c)
        return {"h": h, "c": c}

    def forward(self, input: RecursiveCellInput):
        x = input.get_embeddings()
        nodes_generator = dgl.topological_nodes_generator(input.get_graph())
        if th.cuda.is_available():
            nodes_generator = map(lambda x: x.to("cuda:0"), nodes_generator)

        initial_state = self.W(x)
        i, o, u = th.tensor_split(initial_state, [self._h_size, 2 * self._h_size], 1)
        i = th.sigmoid(i)
        o = th.sigmoid(o)
        u = th.tanh(u)
        c = i * u
        h = o * th.tanh(c)

        input.get_graph().ndata["h"] = h
        input.get_graph().ndata["c"] = c

        input.get_graph().prop_nodes(
            nodes_generator=nodes_generator,
            message_func=self._message_function,
            reduce_func=self._reduce_function,
        )

        return input.get_graph().ndata["h"]
