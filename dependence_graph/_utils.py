from VNode import VEdge, VNode
from typing import Set, Dict, Union
from queue import Queue
from copy import copy
from slither.core.variables.variable import Variable
from slither.core.variables.state_variable import StateVariable
from slither.core.variables.local_variable import LocalVariable
from slither.slithir.operations import Index, OperationWithLValue, InternalCall, Operation, Return
from slither.slithir.variables import (
    Constant,
    LocalIRVariable,
    ReferenceVariable,
    ReferenceVariableSSA,
    StateIRVariable,
    TemporaryVariableSSA,
    TupleVariableSSA,
)
from slither.core.cfg.node import Node, NodeType
from slither.core.declarations import (
    Contract,
    Enum,
    Function,
    SolidityFunction,
    SolidityVariable,
    SolidityVariableComposed,
    Structure,
)
from slither.core.solidity_types.type import Type
from slither.core.declarations.solidity_import_placeholder import SolidityImportPlaceHolder
from slither.core.variables.top_level_variable import TopLevelVariable
from slither.slither import Slither

Variable_types = Union[Variable, SolidityVariable]


class TreeNode():
    def __init__(self, _node) -> None:
        self._node = _node
        self._sons = []
        self._father = None

    def set_node(self, _node):
        self._node = _node

    def set_sons(self, _sons):
        self._sons = _sons

    def add_son(self, node):
        self._sons.append(node)

    def set_father(self, node):
        self._father = node

    @property
    def sons(self):
        return self._sons

    @property
    def node(self):
        return self._node

    @property
    def father(self):
        return self._father

    def __str__(self) -> str:
        return str(self._node)


class postDominatorTree():
    def __init__(self, all_nodes: Set[VNode], exit_node: VNode) -> None:
        self.dom_mapping = {}
        self.all_nodes = all_nodes
        self.exit_node = exit_node
        self.entry_node = None
        self.all_tree_nodes = []
        self.root = None
        self.cNode2Tnode = {}
        self._mapping()

    def _mapping(self):
        for cnode in self.all_nodes:
            # if cnode not in self.cNode2Tnode.keys():
            # print('test', cnode)
            TNode = TreeNode(cnode)
            self.cNode2Tnode[cnode] = TNode
            if cnode.is_entry:
                self.entry_node = cnode
            if cnode.is_start:
                self.entry_node = cnode

    def intersect_immediate_post(self, n: VNode):
        temp = set()
        i = 0
        for son in n.sons:
            if i == 0:
                temp = copy(self.dom_mapping.get(son, set()))
                i = 1
            else:
                temp = temp & self.dom_mapping.get(son, set())
        return temp

    def compute_post_dominator(self):
        self.dom_mapping[self.exit_node] = {self.exit_node}
        for n in self.all_nodes - {self.exit_node}:
            self.dom_mapping[n] = self.all_nodes
        changed = True
        while changed:
            changed = False
            for n in self.all_nodes - {self.exit_node}:
                # print('iterate', n)
                new_set = self.intersect_immediate_post(n).union({n})
                # print('>>>update', self.dom_mapping[n])
                if new_set != self.dom_mapping[n]:
                    # print('>>>>>>', original_dom, self.dom_mapping[n])
                    self.dom_mapping[n] = new_set
                    changed = True
        # return self.dom_mapping

    def build_pdom_tree(self):
        Q = Queue()
        self.compute_post_dominator()
        # print(self.dom_mapping)
        n0 = self.cNode2Tnode[self.exit_node]
        self.cNode2Tnode[self.exit_node] = n0
        self.all_tree_nodes.append(n0)
        Q.put(n0)
        # n0.set_sons(set())
        self.root = n0

        for n in self.all_nodes:
            self.dom_mapping[n] = self.dom_mapping[n] - {n}

        while not Q.empty():
            m = Q.get()
            for n in self.all_nodes:
                if not self.dom_mapping[n]:
                    continue
                if m.node in self.dom_mapping[n]:
                    self.dom_mapping[n] = self.dom_mapping[n] - {m.node}
                    if not self.dom_mapping[n]:
                        # print('m add son:', n)
                        child = self.cNode2Tnode[n]
                        # print('\t-', m.sons)
                        child.set_father(m)
                        m.add_son(child)
                        self.all_tree_nodes.append(child)
                        Q.put(child)

    def _preOrderVist(self, root: TreeNode, result: set):
        if root is None:
            return
        result.add(root)
        for son in root.sons:
            self._preOrderVist(son, result)

    def preOrderVist(self):
        result = set()
        self._preOrderVist(self.root, result)
        return result

    def print(self):
        for tnode in self.all_tree_nodes:
            print(tnode.node, '-> sons')
            # print('\t')
            for tson in tnode.sons:
                print('\t-', tson.node)

    def LCA(self, a: TreeNode, b: TreeNode):
        visited = set()
        curr = b
        while curr:
            visited.add(curr)
            curr = curr.father
        curr = a
        while curr:
            if curr in visited:
                return curr
            curr = curr.father
        return curr


class ControlDependence():
    def __init__(self, pDomTree: postDominatorTree, cfgEdges: Set[VEdge]) -> None:
        self.pDomTree = pDomTree
        self.cfgEdges = cfgEdges
        self.control_dependency = {}
        self.all_control_dependency_nodes = set()
        self.entry = None
        self.exit = None
        self.cNode2cdNode = {}
        self.all_cdNodes = set()
        self.all_cdEdges = set()
        self._mapping()

    def _mapping(self):
        for vedge in self.cfgEdges:
            tail = vedge.tail
            head = vedge.head
            cdNode: VNode
            if tail not in self.cNode2cdNode.keys():
                cdNode = copy(tail)
                if cdNode.is_start:
                    self.entry = cdNode
                if cdNode.is_exit:
                    self.exit = cdNode
                self.cNode2cdNode[tail] = cdNode
                self.all_cdNodes.add(cdNode)
            if head not in self.cNode2cdNode.keys():
                cdNode = copy(head)
                if cdNode.is_start:
                    self.entry = cdNode
                if cdNode.is_exit:
                    self.exit = cdNode
                self.cNode2cdNode[head] = cdNode
                self.all_cdNodes.add(cdNode)

    def compute_control_dependency(self):
        S = set()
        S: Set[VEdge]
        for vedge in self.cfgEdges:
            a = vedge.tail
            b = vedge.head
            if b not in self.pDomTree.dom_mapping[a]:
                S.add((a, b))

        for a, b in S:
            # for key, value in self.pDomTree.cNode2Tnode
            treeNode_a = self.pDomTree.cNode2Tnode[a]
            treeNode_b = self.pDomTree.cNode2Tnode[b]
            cdNode_a = self.cNode2cdNode[a]
            L = self.pDomTree.LCA(treeNode_a, treeNode_b)
            if L is None:
                print('test', treeNode_a, treeNode_b)
                continue
            curr = treeNode_b
            curr: TreeNode
            while curr and curr != L:
                cNode = curr.node
                self.control_dependency[cdNode_a] = self.control_dependency.get(cdNode_a, set()) | {
                    self.cNode2cdNode[cNode]}
                curr = curr.father
            if L == treeNode_a:
                self.control_dependency[cdNode_a] = self.control_dependency.get(cdNode_a, set()) | {
                    self.cNode2cdNode[L.node]}

    def build_control_dependency_graph(self):
        self.compute_control_dependency()
        cdNode: VNode
        depends: Set[VNode]
        # self.all_cdNodes = set(self.control_dependency.keys())
        for cdNode, depends in self.control_dependency.items():
            for d in depends:
                cedge = VEdge(d, cdNode, ControlType)
                cdNode.add_to_edge(cedge)
                d.add_in_edge(cedge)
                self.all_cdEdges.add(cedge)


class DataDependency():
    def __init__(self, all_cdNodes: Dict[Function, Set[VNode]]) -> None:
        self.all_cdNodes = all_cdNodes
        self.func_entry = {}
        self.func_returns = {}
        self.func_returns: Dict[Function, Set[VNode]]
        self.func_dependency = {}
        self.func_dependency: Dict[Function, Dict[VNode, VNode]]
        self.func_var2node_ssa = {}
        self.func_var2node_ssa: Dict[Function, Dict[Variable_types, VNode]]
        self.func_params = {}
        self.state2node = {}
        self.state2node: Dict[StateVariable, Set[VNode]]
        self.init()

    def init(self):
        for function, cdNodes in self.all_cdNodes.items():
            self.func_dependency[function] = dict()
            self.func_var2node_ssa[function] = dict()
            self.func_params[function] = set()
            for n in cdNodes:
                if n.is_exit:
                    continue
                if n.is_start:
                    self.func_entry[function] = n
                    continue
                if n.node.type == NodeType.OTHER_ENTRYPOINT:
                    irs = n.node.irs
                else:
                    irs = n.node.irs_ssa
                for ir in irs:
                    if isinstance(ir, OperationWithLValue) and ir.lvalue:
                        if isinstance(ir.lvalue, LocalIRVariable) and ir.lvalue.is_storage:
                            continue
                        if isinstance(ir.lvalue, StateIRVariable):
                            lvalue = self.convert_variable_to_non_ssa(ir.lvalue)
                            if lvalue:
                                self.state2node[lvalue] = self.state2node.get(lvalue, set()) | {n}
                        elif isinstance(ir.lvalue, StateVariable):
                            self.state2node[ir.lvalue] = self.state2node.get(ir.lvalue, set()) | {n}
                        else:
                            self.func_var2node_ssa[function][ir.lvalue] = n
                    elif isinstance(ir, Return):
                        self.func_returns[function] = self.func_returns.get(function, set()) | {n}

            for p in function.parameters_ssa:
                if isinstance(p, Constant):
                    continue
                p_no_ssa = self.convert_variable_to_non_ssa(p)
                if p_no_ssa:
                    self.func_params[function].add(p_no_ssa)

    def convert_variable_to_non_ssa(self, v: Variable_types) -> Variable_types:
        if isinstance(
                v,
                (
                        LocalIRVariable,
                        StateIRVariable,
                        TemporaryVariableSSA,
                        ReferenceVariableSSA,
                        TupleVariableSSA,
                ),
        ):
            return v.non_ssa_version
        assert isinstance(
            v,
            (
                Constant,
                SolidityVariable,
                Contract,
                Enum,
                SolidityFunction,
                Structure,
                Function,
                Type,
                SolidityImportPlaceHolder,
                TopLevelVariable,
            ),
        )
        return v

    def find_var_nodes(self, v: Variable, f: Function) -> Set[VNode]:
        if isinstance(v, StateIRVariable):
            # _node = self.func_var2node_ssa[f].get(v, None)
            # if _node is None:
            v_no_ssa = self.convert_variable_to_non_ssa(v)
            return self.state2node.get(v_no_ssa, set())
        else:
            _node = self.func_var2node_ssa[f].get(v, None)
            if _node is None:
                print(v, type(v))
                v_no_ssa = self.convert_variable_to_non_ssa(v)
                if v_no_ssa in self.func_params[f]:
                    return {self.func_entry[f]}
                else:
                    return set()
            else:
                return {_node}

    def compute_data_dependency(self):
        for function, cdNodes in self.all_cdNodes.items():
            for n in cdNodes:
                if n.is_start or n.is_exit:
                    continue
                for ir in n.node.irs_ssa:
                    if isinstance(ir, OperationWithLValue) and ir.lvalue:
                        if isinstance(ir.lvalue, LocalIRVariable) and ir.lvalue.is_storage:
                            continue
                        self.add_in_dependency(function, ir, n)
                    if isinstance(ir, Return):
                        self.add_in_dependency(function, ir, n)
                    if isinstance(ir, InternalCall):
                        self.add_to_dependency(function, ir, n)

    def add_to_dependency(self, function: Function, ir: Operation, cd_node: VNode):
        if isinstance(ir, InternalCall):
            args = ir.arguments
            target_entry = self.func_entry.get(ir.function, None)
            if target_entry is None:
                return
            for arg in args:
                if isinstance(arg, Constant):
                    continue
                arg_node = self.find_var_nodes(arg, function)
                # print(ir.function)
                if arg_node:
                    self.func_dependency[ir.function][target_entry] = self.func_dependency[ir.function].get(
                        target_entry, set()) | arg_node

    def add_in_dependency(self, function: Function, ir: Operation, cd_node: VNode) -> None:
        # print('ir', ir, 'function', function)
        if isinstance(ir, OperationWithLValue) and isinstance(ir.lvalue, ReferenceVariable):
            target = ir.lvalue.points_to
            if target is not None:
                target_node = self.find_var_nodes(target, function)
                temp = self.func_dependency[function].get(cd_node, set())
                for n in target_node:
                    if n != cd_node:
                        temp.add(n)
                self.func_dependency[function][cd_node] = temp

        if isinstance(ir, InternalCall):
            return_nodes = self.func_returns.get(ir.function, set())
            for rn in return_nodes:
                self.func_dependency[function][cd_node] = self.func_dependency[function].get(cd_node, set()) | {rn}
        else:
            if isinstance(ir, Index):
                read = [ir.variable_left]
            else:
                read = ir.read
            for v in read:
                if isinstance(v, Constant):
                    continue
                v_node = self.find_var_nodes(v, function)
                if v_node:
                    temp = self.func_dependency[function].get(cd_node, set())
                    for n in v_node:
                        if n != cd_node:
                            temp.add(n)
                    self.func_dependency[function][cd_node] = temp

def print_doms(cfg_nodes, EXIT_NODE):
    p = postDominatorTree(cfg_nodes, EXIT_NODE)
    p.compute_post_dominator()
    for node, doms in p.dom_mapping.items():
        print(node, '-> doms:')
        if not doms:
            print('\tNULL')
            continue
        for dom in doms:
            print('\t-', dom)

def print_tree_nodes(cfg_nodes, EXIT_NODE):
    p = postDominatorTree(cfg_nodes, EXIT_NODE)
    p.build_pdom_tree()
    for tnode in p.all_tree_nodes:
        print(tnode)
        if not tnode.sons:
            print('\tNULL')
            continue
        for son in tnode.sons:
            print('\t-', son)

if __name__ == '__main__':
    sl = Slither("test.sol")
    contract = sl.get_contract_from_name('GoblinRareApepeYC')[0]
    func = contract.get_function_from_signature('_ownershipOf(uint256)')
    EXIT_NODE = VNode(None, 1, function=func)
    cfg_nodes = set()
    cfg_edges = set()
    cfg_nodes = cfg_nodes | {EXIT_NODE}
    func_entry = None
    node_mapping = {}
    for node in func.nodes:
        vnode = VNode(node)
        if node.type == NodeType.ENTRYPOINT:
            func_entry = vnode
        node_mapping[node] = vnode
        cfg_nodes = cfg_nodes | {vnode}

    for node in func.nodes:
        vnode = node_mapping.get(node, None)
        if vnode is not None:
            for father in node.fathers:
                vfather = node_mapping.get(father, None)
                if vfather is not None:
                    edge = VEdge(vnode, vfather, ControlType)
                    if len(father.sons) == 2 and father.sons[0] == node:
                        edge.set_weight('T')
                    elif len(father.sons) == 2 and father.sons[1] == node:
                        edge.set_weight('F')
                    vnode.add_in_edge(edge)
                    vfather.add_to_edge(edge)
                    cfg_edges = cfg_edges | {edge}

            if not node.sons:
                edge = VEdge(EXIT_NODE, vnode, ControlType)
                vnode.add_to_edge(edge)
                EXIT_NODE.add_in_edge(edge)
                cfg_edges = cfg_edges | {edge}

            for son in node.sons:
                vson = node_mapping.get(son, None)
                if vson is not None:
                    edge = VEdge(vson, vnode, ControlType)
                    if len(node.sons) == 2 and node.sons[0] == son:
                        # print(node, son, 'T')
                        edge.set_weight('T')
                    elif len(node.sons) == 2 and node.sons[1] == son:
                        # print(node, son, 'F')
                        edge.set_weight('F')
                    vnode.add_to_edge(edge)
                    vson.add_in_edge(edge)
                    cfg_edges = cfg_edges | {edge}
    print_tree_nodes(cfg_nodes, EXIT_NODE)


