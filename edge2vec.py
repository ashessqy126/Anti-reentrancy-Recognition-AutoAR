from dependence_graph.ControlEdge import ControlEdge
from dependence_graph.DataEdge import DataEdge
from dependence_graph.VEdge import VEdge
import numpy as np


class Edge2Vec:
    def __init__(self) -> None:
        self.edge_vec_mapping = {}
        self.edge_type = [VEdge.ControlNormal, VEdge.ControlInvoke, VEdge.ControlReturn,
                          VEdge.ControlTrue, VEdge.ControlFalse, VEdge.ControlRevert,
                          VEdge.DataFlowNormal, VEdge.DataFlowReturn, VEdge.DataFlowInvoke,
                          VEdge.DataFlowAlias, VEdge.DataFlowGlobal]
        # self.dim = 5
        self.dim = 4

    def edge2vec(self, edge: ControlEdge or DataEdge):
        if edge.type in self.edge_type:
            vec = np.zeros(self.dim)
            vec[edge.type] = 1
            return vec
        return None
