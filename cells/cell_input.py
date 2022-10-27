class RecursiveCellInput:
    def __init__(self, graph, per_node_embeddings, per_node_outputs):
        self.graph = graph
        graph.ndata["RecursiveCellInput.x"] = per_node_embeddings
        graph.ndata["RecursiveCellInput.y"] = per_node_outputs

    def get_embeddings(self):
        return self.graph.ndata["RecursiveCellInput.x"]

    def get_outputs(self):
        return self.graph.ndata["RecursiveCellInput.y"]

    def get_graph(self):
        return self.graph
