from dependence_graph.VNode import VNode
from dependence_graph.ControlEdge import ControlEdge
from typing import Set, Union, Dict, Tuple
from copy import copy
import solidity
from slither.core.variables.variable import Variable
from slither.core.declarations import (
    SolidityVariable
)
from slither.slither import Slither
from dependence_graph.cfg import CFG
from dependence_graph.PostDominatorTree import TreeNode, postDominatorTree
from slither.core.declarations import (
    Contract,
    Enum,
    Function,
    SolidityFunction,
    SolidityVariable,
    SolidityVariableComposed,
    Structure,
)

Variable_types = Union[Variable, SolidityVariable]

class ControlDependence():
    def __init__(self, cfg: CFG) -> None:
        # self.pDomTree = pDomTree
        self.cfg = cfg
        # self.cfg.func_cfg_nodes
        self.all_cfgEdges = cfg.all_cfg_edges
        self.all_cfgEdges: Set[ControlEdge]
        self.all_cfgNodes = cfg.all_cfg_nodes
        self.all_cfgNodes: Set[VNode]
        # self.func_control_dependency = {}
        # self.func_control_dependency: Dict[Function, Dict[VNode, Set[VNode]]]
        # self.func_cfgNodes = cfg.func_cfg_nodes
        # self.func_cfgNodes: Dict[Function, Set[VNode]]
        # self.func_cfgEdges = cfg.func_cfg_edges
        # self.func_cfgEdges: Dict[Function, Set[ControlEdge]]
        self.all_control_dependency = dict()
        self.all_control_dependency: Dict[VNode, Set[Tuple]]
        self.func_control_dependency_nodes = dict()
        self.func_control_dependency_nodes: Dict[Function, Set[VNode]]
        self.all_control_dependency_nodes = set()
        self.all_control_dependency_nodes: Set[VNode]
        self.func_entry = {}
        self.contract_start = None
        self.contract_end = None
        self.all_control_dependency_edges = set()
        self.all_control_dependency_edges: Set[ControlEdge]
        self.func_control_dependency_edges = dict()
        self.func_control_dependency_edges: Dict[Function, Set[ControlEdge]]
        self.cNode2cdNode = dict()
        self.init_mapping()

    def init_mapping(self):
        for cNode in self.all_cfgNodes:
            cdNode = copy(cNode)
            self.cNode2cdNode[cNode] = cdNode
            if cdNode.is_entry:
                self.func_entry[cdNode.function] = cdNode
            if cdNode.is_start_contract:
                self.contract_start = cdNode
            if cdNode.is_end_contract:
                self.contract_end = cdNode
            if cdNode.function is not None:
                self.func_control_dependency_nodes[cdNode.function] = \
                    self.func_control_dependency_nodes.get(cdNode.function, set()) | {cdNode}
            self.all_control_dependency_nodes.add(cdNode)

    def compute_control_dependency(self):
        S = set()
        S: Set[ControlEdge]
        # cNode2cdNode = self.func_mapping(function)
        pDomTree = postDominatorTree(self.all_cfgNodes, self.contract_end)
        pDomTree.build_pdom_tree()

        for vedge in self.all_cfgEdges:
            a = vedge.tail
            b = vedge.head
            if b not in pDomTree.dom_mapping[a]:
                S.add((a, b, vedge.type, vedge.call_site_id))

        for a, b, _type, _call_id in S:
            treeNode_a = pDomTree.cNode2Tnode[a]
            treeNode_b = pDomTree.cNode2Tnode[b]
            cdNode_a = self.cNode2cdNode[a]
            L = pDomTree.LCA(treeNode_a, treeNode_b)
            if L is None:
                print('test', treeNode_a, treeNode_b)
                continue
            curr = treeNode_b
            curr: TreeNode
            while curr and curr != L:
                cNode = curr.node
                self.all_control_dependency[cdNode_a] = \
                    self.all_control_dependency.get(cdNode_a, set()) | \
                    {(self.cNode2cdNode[cNode], _type, _call_id)}
                curr = curr.father
            if L == treeNode_a:
                self.all_control_dependency[cdNode_a] = \
                    self.all_control_dependency.get(cdNode_a, set()) | \
                    {(self.cNode2cdNode[L.node], _type, _call_id)}

    def build_control_dependency_graph(self):
        self.compute_control_dependency()
        cdNode: VNode
        depends: Set[VNode]
        for cdNode, depends in self.all_control_dependency.items():
            for d, _type, _call_id in depends:
                cedge = ControlEdge(d, cdNode, _type)
                cedge.set_call_site_id(_call_id)
                cdNode.add_to_edge(cedge)
                d.add_in_edge(cedge)
                if cdNode.function is not None:
                    self.func_control_dependency_edges[cdNode.function] = \
                        self.func_control_dependency_edges.get(cdNode.function, set()) | {cedge}
                self.all_control_dependency_edges.add(cedge)

    def print_nodes(self):
        for function, nodes in self.func_control_dependency_nodes.items():
            print(f'-------------{function}----------------')
            for node in nodes:
                print(node, '->sons:')
                if not node.sons:
                    print('\tNULL')
                else:
                    for son in node.sons:
                        print('\t-', son)

        print('--------------CONTRACT_START--------------')
        if not self.contract_start.sons:
            print('\tNULL')
        else:
            for son in self.contract_start.sons:
                print('\t-', son)

if __name__ == '__main__':
    solc_v = solidity.get_solc('../test.sol')
    sl = Slither('../test.sol', solc=solc_v)
    contract = sl.get_contract_from_name('GoblinRareApepeYC')[0]
    c = CFG(contract)
    c.build_cfg_graph()
    c.augment_cfg()
    CD = ControlDependence(c)
    CD.build_control_dependency_graph()
    CD.print_nodes()
