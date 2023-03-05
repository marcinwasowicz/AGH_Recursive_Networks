class RecursiveCellInput:
    def __init__(self, graph, per_node_embeddings, per_node_outputs=None):
        self.graph = graph
        graph.ndata["RecursiveCellInput.x"] = per_node_embeddings
        if per_node_outputs is not None:
            graph.ndata["RecursiveCellInput.y"] = per_node_outputs

    def get_embeddings(self):
        return self.graph.ndata["RecursiveCellInput.x"]

    def get_outputs(self):
        try:
            return self.graph.ndata["RecursiveCellInput.y"]
        except KeyError:
            raise Exception("Per node outputs has not been set for input.")

    def get_graph(self):
        return self.graph
