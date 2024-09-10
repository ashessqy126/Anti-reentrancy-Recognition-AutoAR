import os.path
import sys
from collections import defaultdict
import solidity
from slither.core.declarations import Contract, Function
from slither.slither import Slither
from dependence_graph.cfg import CFG
from dependence_graph.DataDependency import DataDependency
from dependence_graph.ControlDependency import ControlDependence
from typing import Set, Dict, List, Tuple
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
        # if edge.type in [VEdge.ControlReturn, VEdge.DataFlowReturn]:
        #     # visited_call_edge.append(edge)
        #     return True
        # if edge.type in [VEdge.ControlInvoke, VEdge.DataFlowInvoke]:
        #     if len(visited_call_edge) == 0:
        #         return True
        #     top = visited_call_edge[-1]
        #     top: VEdge
        #     if top.type in [VEdge.ControlInvoke, VEdge.DataFlowInvoke]:
        #         return True
        #     elif top.type in [VEdge.ControlReturn, VEdge.DataFlowReturn] and top.call_site_id == edge.call_site_id:
        #         return True
        #     else:
        #         return False
        # return False

    # def reverse_dfs_visit(self, edge, node_visited: set, edge_visited: set,
    #                       to_match_call_control: list, to_match_call_data: list):
    #     if edge in edge_visited:
    #         return
    #
    #     _node = edge.tail
    #     edge_visited = edge_visited | {edge}
    #     node_visited.add(_node)
    #
    #     print(len(edge_visited))
    #     for edge in _node.in_edges:
    #         edge: VEdge
    #         # if edge.type in [VEdge.ControlReturn, VEdge.DataFlowReturn, VEdge.ControlInvoke, VEdge.DataFlowInvoke]:
    #         #     # history_edges = copy(visited_call_edge)
    #         #     _visited_call_control, _visited_call_data = self.check_current_path(visited_call_control,
    #         #                                                                       visited_call_data,
    #         #                                                                       edge)
    #         #     if _visited_call_control is None or _visited_call_data is None:
    #         #         continue
    #             # if len(visited_call_edge) == 0:
    #             #     visited_call_edge = visited_call_edge + [edge]
    #             # elif visited_call_edge[-1].type in [VEdge.ControlInvoke, VEdge.DataFlowInvoke]:
    #             #     visited_call_edge = visited_call_edge + [edge]
    #             # elif visited_call_edge[-1].type in [VEdge.ControlReturn, VEdge.DataFlowReturn]:
    #             #     visited_call_edge = visited_call_edge[:-1]
    #             # visited_call_control = _visited_call_control
    #             # visited_call_data = _visited_call_data
    #
    #         # if edge.tail not in fathers:
    #         #     fathers.add(edge.tail)
    #         #     # print(visited_call_data, visited_call_control)
    #         self.reverse_dfs_visit(edge, node_visited, edge_visited, to_match_call_control, to_match_call_data)

    def dfs_reverse_visit(self, node: VNode, visited):
        if node in visited:
            return
        visited.append(node)
        for edge in node.in_edges:
            # if isinstance(edge, DataEdge):
            #     continue
            self.dfs_reverse_visit(edge.tail, visited)
        # for father in node.fathers:
        #     self.dfs_reverse_visit(father, visited)

    # def node_mapping(self):
    #     id = 0
    #     for n in self.ivfg_nodes:
    #         self.node2id[n] = id
    #         self.id2node[id] = n
    #         id += 1
    #
    # def build_control_adj_matrix(self):
    #     dim = len(self.ivfg_nodes)
    #     self.control_adj_matrix = np.zeros((dim, dim), dtype='str')
    #     for edge in self.ivfg_control_edges:
    #         tail = self.node2id[edge.tail]
    #         head = self.node2id[edge.head]
    #         if edge.type == VEdge.ControlInvoke:
    #             self.control_adj_matrix[tail][head] = '(' + str(edge.call_site_id)
    #         elif edge.type == VEdge.ControlReturn:
    #             self.control_adj_matrix[tail][head] = ')' + str(edge.call_site_id)
    #         else:
    #             # print(dim)
    #             self.control_adj_matrix[tail][head] = 1
    #
    # def check(self):
    #     return True

    # def symbol_mul(self, a, b):
    #     a_symbols = a.split()
    #     b_symbols = b.split()
    #     ret = ''
    #     for x in a_symbols:
    #         for y in b_symbols:
    #             if x.isdigit() and y.isdigit():
    #                 ret += ' ' + str(int(x) * int(y))
    #             else:
    #                 self.check()
    #                 ret += ' ' + x + y
    #
    #     return ret
    #
    # @staticmethod
    # def symbol_add(a, b):
    #     if a.isdigit() and b.isdigit():
    #         return str(int(a) + int(b))
    #     else:
    #         return a + ' ' + b
    #
    # def matrix_mul(self, A, B):
    #     dim = A.shape[0]
    #     ret = np.zeros((dim, dim), dtype='str')
    #     for i in range(dim):
    #         for j in range(dim):
    #             print((i, j))
    #             ret[i][j] = ''
    #             for k in range(dim):
    #                 ret[i][j] = self.symbol_add(ret[i][j], self.symbol_mul(A[i][k], B[k][j]))
    #     return ret
    #
    # def extract_subgraph(self, var, node: VNode):
    #     self.node_mapping()
    #     self.build_control_adj_matrix()
    #     dim = len(self.ivfg_nodes)
    #     start_matrix = np.zeros((dim, dim), dtype='str')
    #     start_id = self.node2id[node]
    #     for edge in node.in_edges:
    #         fid = self.node2id[edge.tail]
    #         if edge.type == VEdge.ControlInvoke:
    #             start_matrix[fid][start_id] = '(' + str(edge.call_site_id)
    #         elif edge.type == VEdge.ControlReturn:
    #             start_matrix[fid][start_id] = ')' + str(edge.call_site_id)
    #         else:
    #             start_matrix[fid][start_id] = 1
    #     t1 = time.time()
    #     print('start')
    #     print(self.matrix_mul(start_matrix, self.control_adj_matrix))
    #     t2 = time.time()
    #     print('end:', t2 - t1)

        # node_visited = []
        # self.dfs_reverse_visit(node, node_visited)
        # return node_visited
        # visit_queue = Queue()
        # node_visited = [node]
        # to_match = False
        # for edge in node.in_edges:
        #     visit_queue.put((set(), to_match, edge))
        #
        # while not visit_queue.empty():
        #     edge_visited, to_match, edge = visit_queue.get()
        #     # print(edge_visited)
        #     if edge in edge_visited:
        #         continue
        #     print('path len:', len(edge_visited))
        #     next_node = edge.tail
        #     if next_node not in node_visited:
        #         node_visited.append(next_node)
        #     if to_match:
        #         edge_visited.add(edge)
        #     else:
        #         edge_visited = edge_visited | {edge}
        #     for next_edge in next_node.in_edges:
        #         next_edge: VEdge
        #         if next_edge.type in [VEdge.ControlReturn, VEdge.ControlInvoke,
        #                               VEdge.DataFlowReturn, VEdge.DataFlowInvoke]:
        #             to_match = True
        #             visit_queue.put((edge_visited, to_match, next_edge))
        #
        # return node_visited

        # for edge in node.in_edges:
        #     self.reverse_dfs_visit(edge, node_visited, edge_visited, [], [])
        # return node_visited

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

    def extract_subgraph_nodes(self, var, node: VNode or Node):
        if isinstance(node, Node):
            node = self.match_ivfg_nodes(self.ivfg_nodes, node)
        prev_sum_path_vector, external_data_deps = self.extract_local_subgraph_nodes(var, node, dict())
        visited_external_data_deps = set()
        subgraph_nodes = set(prev_sum_path_vector.keys())
        while external_data_deps:
            new_external_data_deps = set()
            for external_node in external_data_deps:
                if external_node in visited_external_data_deps:
                    continue
                visited_external_data_deps.add(external_node)
                s = time.time()
                sn, ed = self.extract_local_subgraph_nodes(None, external_node, prev_sum_path_vector)
                prev_sum_path_vector = self.union_dict(prev_sum_path_vector, sn)
                e = time.time()
                print('test:', e - s)
                subgraph_nodes |= sn.keys()
                new_external_data_deps |= ed
            external_data_deps = new_external_data_deps
        return subgraph_nodes

    def extract_local_subgraph_nodes(self, var, node: VNode):
        if isinstance(node, Node):
            node = self.match_ivfg_nodes(self.ivfg_nodes, node)
        self.dd: DataDependency
        node_num = len(self.ivfg_nodes)
        if var:
            outest_vars = self._node_depend(var, node)
        else:
            outest_vars = None
        next_edges = dict()
        for edge in node.in_edges:
            if isinstance(edge, DataEdge) and outest_vars is not None and edge.depend_var not in outest_vars:
                continue
            if edge.type in [VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
                continue
            next_edges[edge] = 1
        sum_path_vector = {node: {1}}
        path_vector = {node: {1}}
        while next_edges:
            path_vector, next_edges = self._update_path_vec(path_vector, next_edges)
            sum_path_vector = self.union_dict(sum_path_vector, path_vector)
        return sum_path_vector

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

    def _update_path_vec(self, path_vector, next_edges: Dict[VNode, int]):
        if not next_edges:
            return dict(), dict()
        next_path_vector = dict()
        next_nodes = dict()
        for edge, p_len in next_edges.items():
            if edge.type in [VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
                continue
            next_node = edge.tail
            s_node = edge.head
            s_paths = path_vector.get(s_node, set())
            next_paths = next_path_vector.get(next_node, set())
            if any([fp == 1 for fp in next_paths]):
                next_path_vector[next_node] = {1}
                if p_len + 1 < next_nodes.get(next_node, 9999999999999):
                    next_nodes[next_node] = p_len + 1
                # next_nodes[next_node] = p_len +
                # next_nodes.add((next_node, p_len + 1))
                continue
            if edge.type == VEdge.DataFlowGlobal:
                next_path_vector[next_node] = {1}
                next_nodes[next_node] = 1
                # next_nodes.add((next_node, 1))
                continue
            assert s_paths is not None, f'impossible paths for {next_node}'
            if edge.type not in [VEdge.ControlInvoke, VEdge.ControlReturn]:
                next_paths = s_paths
                next_path_vector[next_node] = next_paths
                # next_nodes.add((next_node, p_len + 1))
                # next_nodes[next_node] = p_len + 1
                if p_len + 1 < next_nodes.get(next_node, 9999999999999):
                    next_nodes[next_node] = p_len + 1
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
            if next_paths:
                next_path_vector[next_node] = next_paths
                if p_len + 1 < next_nodes.get(next_node, 9999999999999):
                    next_nodes[next_node] = p_len + 1
                # next_nodes.add((next_node, p_len + 1))
        next_edges = dict()
        for n, p_len in next_nodes.items():
            if p_len > len(self.ivfg_nodes) - 1:
                continue
            for e in n.in_edges:
                if e.type in [VEdge.DataFlowInvoke, VEdge.DataFlowReturn]:
                    continue
                next_edges[e] = p_len
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


if __name__ == '__main__':
    # slither = Slither('../test.sol')
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
    t1 = time.time()
    var = extract_destination(find)
    subgraph_nodes = ivg.extract_local_subgraph_nodes(var, find)
    # subgraph_adj_matrix = ivg.build_subgraph_adj_matrix(subgraph_nodes)
    t2 = time.time()
    print('duration: ', t2 - t1)
    subgraph_node_list = []
    ivg.dfs_reverse_visit(find, subgraph_node_list)
    for n in subgraph_nodes:
        # if isinstance(n.node, Node):
        print(n)
    print(f'len {len(subgraph_nodes)}')
    total_num = 0
    for func in contract.functions_and_modifiers:
        total_num += len(func.nodes)
    print('total_num:', total_num)
