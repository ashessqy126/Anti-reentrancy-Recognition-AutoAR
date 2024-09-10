import sys
from collections import defaultdict
import solidity
from slither.core.declarations import Contract, Function
from slither.slither import Slither
from dependence_graph.cfg import CFG
from dependence_graph.DataDependency import DataDependency
from dependence_graph.ControlDependency import ControlDependence
from typing import Set, Dict, List
from dependence_graph.VNode import VNode
from slither.slithir.operations import InternalCall
from dependence_graph.ControlEdge import ControlEdge
from dependence_graph.VEdge import VEdge
from dependence_graph.DataEdge import DataEdge
from copy import copy
from queue import Queue
import numpy as np
import time
from slither.slithir.operations import Index, OperationWithLValue, InternalCall, Operation, Return, Call
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
from scan import reentrancy_call
from dependence_graph.CallGraph import CallGraph
# class QNode:
#     symbolic_path = ''
#     visited_nodes = set()


class IVFG:
    def __init__(self, ct: Contract) -> None:
        self.cfg = None
        self.cd = None
        self.dd = None
        self.dd: DataDependency
        self.contract = ct
        self.ivfg_nodes = None
        self.ivfg_nodes: List[VNode]
        self.ivfg_control_edges = None
        self.ivfg_control_edges: Set[ControlEdge]
        self.ivfg_data_edges = None
        self.ivfg_data_edges: Set[DataEdge]
        self.ivfg_func_nodes = None
        self.ivfg_func_nodes: Dict[Function, Set[VNode]]
        self.var2node = None
        self.node2id = {}
        # self.control_adj_matrix = None
        # self.data_adj_matrix = None

    def build_cfg_graph(self):
        cfg = CFG(self.contract)
        cfg.build_cfg_graph()
        self.cfg = cfg

    def build_IVFG(self):
        self.build_cfg_graph()
        self.cfg.augment_cfg()
        self.cd = ControlDependence(self.cfg)
        self.cd.build_control_dependency_graph()
        self.ivfg_control_edges = self.cd.all_control_dependency_edges
        self.dd = DataDependency(self.cd.all_control_dependency_nodes)
        self.dd.compute_data_dependency()
        self.ivfg_nodes = list(self.dd.all_dataNodes)
        self.ivfg_data_edges = self.dd.all_dataEdges
        self.ivfg_func_nodes = self.dd.func_nodes

    # def print_nodes(self):
    #     for function, cdNodes in self.ivfg_func_nodes.items():
    #         print(f'--------------{function}---------------')
    #         for node in cdNodes:
    #             print(node, '->sons:')
    #             if not node.sons:
    #                 print('\tNULL')
    #             else:
    #                 for edge in node.edges:
    #                     print(f'\t{edge}')

    def print_nodes(self):
        if self.ivfg_func_nodes is not None:
            for function, nodes in self.ivfg_func_nodes.items():
                print(f'---------------{function}---------------')
                for n in nodes:
                    print(n, '->sons:')
                    if not n.sons:
                        print('\tNULL')
                    else:
                        for e in n.to_edges:
                            print('\t', e)

    @staticmethod
    def check_current_path(visited_call_control, visited_call_data, edge: VEdge):
        visited_call_control = copy(visited_call_control)
        visited_call_data = copy(visited_call_data)
        if edge.type not in [VEdge.ControlReturn, VEdge.ControlInvoke, VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
            return visited_call_control, visited_call_data
        if edge.type == VEdge.ControlReturn:
            visited_call_control.append(edge)
        if edge.type == VEdge.DataFlowReturn:
            visited_call_data.append(edge)
        if edge.type == VEdge.ControlInvoke:
            if len(visited_call_control) == 0:
                visited_call_control.append(edge)
                return visited_call_control, visited_call_data
            top = visited_call_control[-1]
            top: VEdge
            if top.type == VEdge.ControlReturn:
                if top.call_site_id == edge.call_site_id:
                    visited_call_control.pop()
                else:
                    return None, None
            if top.type == VEdge.ControlInvoke:
                visited_call_control.append(edge)

        if edge.type == VEdge.DataFlowInvoke:
            if len(visited_call_data) == 0:
                visited_call_data.append(edge)
                return visited_call_control, visited_call_data
            top = visited_call_data[-1]
            top: VEdge
            if top.type == VEdge.DataFlowReturn:
                if top.call_site_id == edge.call_site_id:
                    visited_call_data.pop()
                else:
                    return None, None
            if top.type == VEdge.DataFlowInvoke:
                visited_call_data.append(edge)

        return visited_call_control, visited_call_data

    def dfs_reverse_visit(self, node: VNode, visited):
        if node in visited:
            return
        visited.append(node)
        for edge in node.in_edges:
            # if isinstance(edge, DataEdge):
            #     continue
            self.dfs_reverse_visit(edge.tail, visited)

    @staticmethod
    def check(path: str):
        symbols = path.split()
        stack = []
        for s in symbols:
            if s in stack:
                return None
            if s[0] == '(':
                if len(stack) > 0:
                    top = stack[-1]
                    if top[0] == ')' and top[1:] == s[1:]:
                        return None
                stack.append(s)
            elif s[0] == ')':
                if len(stack) == 0:
                    stack.append(s)
                    continue
                top = stack[-1]
                if top[0] == '(':
                    if top[1:] == s[1:]:
                        stack.pop()
                    else:
                        return None
                else:
                    # if top[1:] != s[1:]:
                    stack.append(s)
        ret = ''.join([s + ' ' for s in stack])
        return ret

    def _find_outest_var(self, var, node_dep, outest_var):
        var_deps = node_dep.get(var, None)
        if isinstance(var, StateIRVariable):
            outest_var.add(DataDependency.convert_variable_to_non_ssa(var))
            return
        if var_deps is None:
            var = DataDependency.convert_variable_to_non_ssa(var)
            outest_var.add(var)
            return
        for v in var_deps:
            self._find_outest_var(v, node_dep, outest_var)

    def _node_depend(self, var, node: VNode):
        node_dep = {}
        if node.node.type == NodeType.OTHER_ENTRYPOINT:
            irs = node.node.irs
        else:
            irs = node.node.irs_ssa
        for ir in irs:
            if isinstance(ir, OperationWithLValue) and ir.lvalue:
                if isinstance(ir.lvalue, LocalIRVariable) and ir.lvalue.is_storage:
                    continue
                if isinstance(ir, Index):
                    read = [ir.variable_left]
                else:
                    read = ir.read
                for v in read:
                    if isinstance(v, Constant):
                        continue
                    if v != ir.lvalue:
                        node_dep[ir.lvalue] = node_dep.get(ir.lvalue, set()) | {v}
        outest_vars = set()
        self._find_outest_var(var, node_dep, outest_vars)
        return outest_vars

    def match_ivfg_nodes(self, all_ivfg_nodes, node: Node):
        for n in all_ivfg_nodes:
            if n.node == node:
                return n
        return None

    def extract_subgraph_nodes(self, var, node: VNode or Node, timeout=300):
        if isinstance(node, Node):
            node = self.match_ivfg_nodes(self.ivfg_nodes, node)
        prev_sum_path_vector, external_data_deps = self.extract_local_subgraph_nodes(var, node, dict(), timeout)
        # visited_external_data_deps = set()
        subgraph_nodes = set(prev_sum_path_vector.keys())
        total_paths = 0
        for k, v in prev_sum_path_vector.items():
            total_paths += len(v)
        return subgraph_nodes, external_data_deps

    def extract_local_subgraph_nodes(self, var, node: VNode, prev_sum_path_vector, timeout=300):
        start = time.time()
        if isinstance(node, Node):
            node = self.match_ivfg_nodes(self.ivfg_nodes, node)
        self.dd: DataDependency
        node_num = len(self.ivfg_nodes)
        if var:
            outest_vars = self._node_depend(var, node)
        else:
            outest_vars = None
        next_edges = set()
        for edge in node.in_edges:
            if isinstance(edge, DataEdge) and outest_vars is not None and edge.depend_var not in outest_vars:
                continue
            if edge.type in [VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
                continue
            next_edges.add(edge)
        sum_path_vector = {node: {1}}
        path_vector = {node: {1}}
        external_data_deps = set()
        for i in range(node_num - 1):
            if not next_edges:
                break
            path_vector, next_edges = self._update_path_vec(path_vector, next_edges,
                                                            external_data_deps, prev_sum_path_vector)
            sum_path_vector = self.union_dict(sum_path_vector, path_vector)
            end = time.time()
            if end - start > timeout:
                break
        return sum_path_vector, external_data_deps

    @staticmethod
    def union_dict(d1: Dict, d2: Dict):
        # d3 = {k: d1.get(k, set()) | d2.get(k, set()) for k in set(list(d1.keys()) + list(d2.keys()))}
        d3 = {}
        for k in set(list(d1.keys()) + list(d2.keys())):
            v = d1.get(k, set()) | d2.get(k, set())
            if 1 in v:
                d3[k] = {1}
            else:
                d3[k] = v
        return defaultdict(set, d3)

    def _update_path_vec(self, path_vector, next_edges: Set[VEdge],
                         external_data_deps, prev_sum_path_vector):
        if not next_edges:
            return dict()
        next_path_vector = dict()
        next_nodes = set()
        for edge in next_edges:
            if edge.type in [VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
                continue
            next_node = edge.tail
            s_node = edge.head
            s_paths = path_vector.get(s_node, set())
            next_paths = next_path_vector.get(next_node, set())
            # if any([fp == 1 for fp in path_vector.get(next_node, set()) | next_paths]):
            if 1 in prev_sum_path_vector.get(next_node, set()):
                continue
            if any([fp == 1 for fp in next_paths]):
                next_path_vector[next_node] = {1}
                next_nodes.add(next_node)
                continue
            if edge.type == VEdge.DataFlowGlobal:
                external_data_deps.add(edge)
                continue
            assert s_paths is not None, f'impossible paths for {next_node}'
            if edge.type not in [VEdge.ControlInvoke, VEdge.ControlReturn]:
                next_paths = s_paths - prev_sum_path_vector.get(next_node, set())
                if next_paths:
                    next_path_vector[next_node] = next_paths
                    next_nodes.add(next_node)
                continue
            for prev_path in s_paths:
                if isinstance(prev_path, int) and edge.type not in [VEdge.ControlReturn, VEdge.ControlInvoke]:
                    next_paths = {1}
                    break
                if isinstance(prev_path, int):
                    prev_path = ''
                if edge.type == VEdge.ControlInvoke:
                    n_path = prev_path + ')' + str(edge.call_site_id) + ' '
                elif edge.type == VEdge.ControlReturn:
                    n_path = prev_path + '(' + str(edge.call_site_id) + ' '
                reduced_path = self.check(n_path)
                if reduced_path == '':
                    next_paths.add(1)
                elif reduced_path is not None:
                    next_paths.add(reduced_path)
            next_paths = next_paths - prev_sum_path_vector.get(next_node, set())
            if next_paths:
                next_path_vector[next_node] = next_paths
                next_nodes.add(next_node)
        next_edges = set()
        for n in next_nodes:
            for e in n.in_edges:
                if e.type in [VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
                    continue
                next_edges.add(e)
        return next_path_vector, next_edges

    def init_adj_matrix(self):
        nid = 0
        for n in self.ivfg_nodes:
            self.node2id[n] = nid
            nid += 1
        dim = len(self.ivfg_nodes)
        adj_matrix = np.zeros((dim, dim), dtype='int')
        return adj_matrix

    @staticmethod
    def extract_subgraph_edges(subgraph_nodes):
        subgraph_edges = set()
        for n in subgraph_nodes:
            for edge in n.in_edges | n.to_edges:
                from_node = edge.tail
                to_node = edge.head
                if from_node in subgraph_nodes and to_node in subgraph_nodes:
                    subgraph_edges.add(edge)
        return subgraph_edges

    def build_subgraph_adj_matrix(self, subgraph_nodes):
        adj_matrix = self.init_adj_matrix()
        DATA_TYPE = 1
        CONTROL_TYPE = 2
        subgraph_edges = self.extract_subgraph_edges(subgraph_nodes)
        for edge in subgraph_edges:
            from_node = edge.tail
            to_node = edge.head
            from_id = self.node2id[from_node]
            to_id = self.node2id[to_node]
            if isinstance(edge, DataEdge):
                adj_matrix[from_id][to_id] = DATA_TYPE
            else:
                adj_matrix[from_id][to_id] = CONTROL_TYPE
        return adj_matrix

    @staticmethod
    def extract_destination(node: Node):
        if node.type == NodeType.OTHER_ENTRYPOINT:
            irs = node.irs
        else:
            irs = node.irs_ssa
        for ir in irs:
            if isinstance(ir, Call) and ir.function is not None and ir.can_reenter():
                return ir.destination
        return None

def match_ivfg_nodes(all_ivfg_nodes, node):
    for n in all_ivfg_nodes:
        if n.node == node:
            return n
    return None


def extract_destination(vnode: VNode):
    if vnode.node.type == NodeType.OTHER_ENTRYPOINT:
        irs = vnode.node.irs
    else:
        irs = vnode.node.irs_ssa
    for ir in irs:
        if isinstance(ir, Call) and ir.function is not None and ir.can_reenter():
            return ir.destination
    return None


def print_matrix(adj_matrix):
    x, y = adj_matrix.shape
    k = 0
    for i in range(x):
        no_item = 0
        for j in range(y):
            if adj_matrix[i][j] != 0:
                no_item = 1
                k += 1
                print(f'({i}, {j}): {adj_matrix[i][j]}', end=' ')
        if no_item == 1:
            print('')


def test1():
    solc_version = solidity.get_solc('../test.sol')
    if solc_version is None:
        print('cannot get a correct solc version')
        sys.exit(-1)
    try:
        sl = Slither('../test.sol', solc=solc_version)
    except Exception as e:
        print(f'compilation for ../test.sol failed')
        sys.exit(-1)
    contract = sl.get_contract_from_name('GoblinRareApepeYC')[0]
    ivg = IVFG(contract)
    ivg.build_IVFG()
    # ivg.print_nodes()
    for func in contract.functions:
        if func.name == '_checkContractOnERC721Received':
            function = func
    # function = contract.get_function_from_signature('_checkContractOnERC721Received(address,address,uint256,bytes)')
    find = None
    for node in function.nodes:
        if 'TRY IERC721Receiver' in str(node):
            find = node
            break
    find = match_ivfg_nodes(ivg.ivfg_nodes, find)

    var = extract_destination(find)
    t1 = time.time()
    subgraph_nodes, _ = ivg.extract_subgraph_nodes(var, find)
    t2 = time.time()
    print('duration: ', t2 - t1)
    subgraph_node_list = []
    ivg.dfs_reverse_visit(find, subgraph_node_list)
    # for n in subgraph_nodes:
    #     # if isinstance(n.node, Node):
    #     print(n)
    print(f'len {len(subgraph_nodes)}')
    print(f'dfs: {len(subgraph_node_list)}')
    total_num = 0
    for func in contract.functions_and_modifiers:
        total_num += len(func.nodes)
    print('total_num:', total_num)


def test2():
    solc_version = solidity.get_solc('../etherscan/0xd6e597034423c930132ca58f5993771f5e56dbb4/0xd6e597034423c930132ca58f5993771f5e56dbb4.sol')
    graph_num = -1
    if solc_version is None:
        print(f'cannot get a correct solc version for ../test.sol')
        return
    try:
        sl = Slither('test.sol', solc=solc_version)
    except Exception as e:
        print(f'compilation for test.sol failed')
        return
    callgraph = CallGraph(sl)
    callgraph.build_call_graph()
    scan_reentrancy = reentrancy_call(sl)
    call_nodes = scan_reentrancy.extract()
    c = call_nodes[0]
    contained_func = contained_functions[0]
    paths = callgraph.extract_func_call_list(c.function, contained_func)
    ivg = IVFG(c.function.contract)
    ivg.build_IVFG()
    c = match_ivfg_nodes(ivg.ivfg_nodes, c)
    var = extract_destination(c)
    t1 = time.time()
    subgraph_nodes, external_data_deps = ivg.extract_subgraph_nodes(var, c)
    t2 = time.time()
    print('duration: ', t2 - t1)
    for path in paths:
        print('function call list: ', end='')
        for f in path:
            print(f, end=' ')
        print('')
        scope_nodes = callgraph.extract_path_graph_nodes(path)
        filtered_nodes = subgraph_nodes & scope_nodes
        for e in external_data_deps:
            if e.head not in scope_nodes:
                continue
            filtered_nodes.add(e.tail)
        print(f'total_filtered_nodes: {len(filtered_nodes)}')


if __name__ == '__main__':
    # slither = Slither('../test.sol')
    # test1()
    test1()


