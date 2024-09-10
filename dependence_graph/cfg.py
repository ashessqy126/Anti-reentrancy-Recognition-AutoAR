from slither.core.declarations import (
    Contract,
    Enum,
    Function,
    SolidityFunction,
    SolidityVariable,
    SolidityVariableComposed,
    Structure,
)
from slither.slither import Slither
from dependence_graph.VNode import VNode
from dependence_graph.ControlEdge import ControlEdge
from dependence_graph.VEdge import VEdge
from slither.core.cfg.node import Node, NodeType
from slither.slithir.operations.solidity_call import SolidityCall
from slither.slithir.operations import InternalCall
from copy import copy
class CFG():
    def __init__(self, contract: Contract):
        self.contract = contract
        self.func_entry = {}
        self.func_returns = {}
        self.func_cfg_nodes = {}
        self.func_cfg_edges = {}
        self.all_cfg_nodes = set()
        self.all_cfg_edges = set()
        self.node_mapping = {}
        self.contract_start = VNode(None, 2, None, self.contract)
        self.all_cfg_nodes.add(self.contract_start)
        self.contract_end = VNode(None, 3, None, self.contract)
        self.all_cfg_nodes.add(self.contract_end)

    def _mapping(self, func: Function):
        self.node_mapping[func] = {}
        for node in func.nodes:
            vnode = VNode(node)
            # if func.is_constructor_variables and node == func.entry_point:
            if func.entry_point == node:
                self.func_entry[func] = vnode
            # if not func.is_constructor_variables and node == func.entry_point:
            #     continue
            # if func.is_constructor_variables and node == func.entry_point:
            #     self.func_entry[func] = vnode
            # if not func.is_constructor_variables and node == func.entry_point.sons[0]:
            #     self.func_entry[func] = vnode
            if not node.sons:
                self.func_returns[func] = self.func_returns.get(func, set()) | {vnode}
            self.node_mapping[func][node] = vnode
            self.func_cfg_nodes[func] = self.func_cfg_nodes.get(func, set()) | {vnode}
            self.all_cfg_nodes.add(vnode)
            self.call_site_id = -1

    def build_cfg_graph(self):
        for func in self.contract.functions_and_modifiers:
            self._mapping(func)
        for func in self.contract.functions_and_modifiers:
            if not (func.is_constructor_variables
                    or (func.is_implemented and not func.is_empty)
            ):
                continue
            self.build_function_cfg_graph(func)

    def build_function_cfg_graph(self, func: Function):
        # EXIT_NODE = VNode(None, 1, function=func)
        # self.func_cfg_nodes[func] = self.func_cfg_nodes.get(func, set()) | {EXIT_NODE}
        # self.func_exit[func] = EXIT_NODE
        # if func.visibility in ['public', 'external'] and not func.is_shadowed:
        entry_node = self.func_entry[func]
        # print('entry_node', entry_node)
        edge = ControlEdge(entry_node, self.contract_start)
        self.contract_start.add_to_edge(edge)
        entry_node.add_in_edge(edge)
        self.all_cfg_edges.add(edge)
        self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}

        for node in func.nodes:
            vnode = self.node_mapping[func].get(node, None)
            if vnode is None:
                continue

            if self.is_revert(node):
                edge = ControlEdge(self.contract_end, vnode, VEdge.ControlRevert)
                self.contract_end.add_in_edge(edge)
                vnode.add_to_edge(edge)
                self.all_cfg_edges.add(edge)
                self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}
                continue

            for father in node.fathers:
                vfather = self.node_mapping[func].get(father, None)
                if vfather is not None:
                    edge = ControlEdge(vnode, vfather)
                    if len(father.sons) == 2 and father.sons[0] == node:
                        edge.set_type(VEdge.ControlTrue)
                    elif len(father.sons) == 2 and father.sons[1] == node:
                        edge.set_type(VEdge.ControlFalse)
                    vnode.add_in_edge(edge)
                    vfather.add_to_edge(edge)
                    self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}
                    self.all_cfg_edges.add(edge)

            if not node.sons:
                edge = ControlEdge(self.contract_end, vnode)
                vnode.add_to_edge(edge)
                self.contract_end.add_in_edge(edge)
                self.all_cfg_edges.add(edge)
                self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}

            for son in node.sons:
                vson = self.node_mapping[func].get(son, None)
                if vson is not None:
                    edge = ControlEdge(vson, vnode)
                    if len(node.sons) == 2 and node.sons[0] == son:
                        edge.set_type(VEdge.ControlTrue)
                    elif len(node.sons) == 2 and node.sons[1] == son:
                        edge.set_type(VEdge.ControlFalse)
                    vnode.add_to_edge(edge)
                    vson.add_in_edge(edge)
                    self.all_cfg_edges.add(edge)
                    self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}

            if self.is_require(node):
                edge = ControlEdge(self.contract_end, vnode, VEdge.ControlRevert)
                vnode.add_to_edge(edge)
                self.contract_end.add_in_edge(edge)
                self.all_cfg_edges.add(edge)
                self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}

            # invoked_funcs = self.extract_invoked(node)
            for f in vnode.calls:
                self.call_site_id += 1
                vnode.set_call_site_id(f, self.call_site_id)
                target_entry = self.func_entry.get(f, None)
                if target_entry is None:
                    continue
                edge = ControlEdge(target_entry, vnode, VEdge.ControlInvoke)
                edge.set_call_site_id(self.call_site_id)
                vnode.add_to_edge(edge)
                target_entry.add_in_edge(edge)
                self.all_cfg_edges.add(edge)
                self.func_cfg_edges[f] = self.func_cfg_edges.get(f, set()) | {edge}
                for ret_node in self.func_returns[f]:
                    edge = ControlEdge(vnode, ret_node, VEdge.ControlReturn)
                    edge.set_call_site_id(self.call_site_id)
                    vnode.add_in_edge(edge)
                    ret_node.add_to_edge(edge)
                    self.all_cfg_edges.add(edge)
                    self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}

    @staticmethod
    def extract_invoked(node: Node):
        invoked_func = set()
        for ir in node.irs:
            if isinstance(ir, InternalCall):
                invoked_func.add(ir.function)
        return invoked_func

    @staticmethod
    def is_require(node: Node):
        for ir in node.irs:
            if isinstance(ir, SolidityCall) and ('require(' in str(ir.expression) or 'assert(' in str(ir.expression)):
                return True
        return False

    @staticmethod
    def is_revert(node: Node):
        if node.type == NodeType.THROW:
            return True
        for ir in node.irs:
            if isinstance(ir, SolidityCall) and 'revert' in str(ir.expression):
                return True
        return False

    def print(self):
        for function, edges in self.func_cfg_edges.items():
            print(f'-----------{function}--------------')
            for e in edges:
                print(e)

    def print_all_edges(self):
        for edge in self.all_cfg_edges:
            print(edge)

    def augment_cfg(self):
        edge = ControlEdge(self.contract_end, self.contract_start)
        self.contract_start.add_to_edge(edge)
        self.contract_end.add_in_edge(edge)
        self.all_cfg_edges.add(edge)
        for func in self.func_entry.keys():
            entry_node = self.func_entry[func]
            # for end_node in self.func_returns[func]:
            edge = ControlEdge(self.contract_end, entry_node)
            entry_node.add_to_edge(edge)
            self.contract_end.add_in_edge(edge)
            self.all_cfg_edges.add(edge)
            self.func_cfg_edges[func] = self.func_cfg_edges.get(func, set()) | {edge}

    def print_all_nodes(self):
        for function, nodes in self.func_cfg_nodes.items():
            print(f'-------------{function}-----------------')
            for n in nodes:
                print(n, '->sons:')
                if not n.sons:
                    print('\tNULL')
                    continue
                for son in n.sons:
                    print('\t', son)
                for call, cid in n.call2id.items():
                    print(f'\tcall_site: {call}, id: {cid}')
        print(f'---------{self.contract}-----------')
        print(self.contract_start, '->sons:')
        if not self.contract_start.sons:
            print('\tNULL')
        else:
            for son in self.contract_start.sons:
                print('\t', son)
        print(self.contract_end, '->sons:')
        if not self.contract_end.sons:
            print('\tNULL')
        else:
            for son in self.contract_end.sons:
                print('\t', son)

if __name__ == '__main__':
    sl = Slither('../test.sol')
    contract = sl.get_contract_from_name('GoblinRareApepeYC')[0]
    c = CFG(contract)
    c.build_cfg_graph()
    c.augment_cfg()
    c.print_all_nodes()
