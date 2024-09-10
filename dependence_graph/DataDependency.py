from dependence_graph.VNode import VNode
from dependence_graph.DataEdge import DataEdge
from dependence_graph.VEdge import VEdge
from typing import Set, Dict, Union
from slither.core.variables.variable import Variable
from slither.core.variables.state_variable import StateVariable
from slither.slithir.operations import Index, OperationWithLValue, InternalCall, Operation, Return, Phi, \
    HighLevelCall, LowLevelCall
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

Variable_types = Union[Variable, SolidityVariable]


class DataDependency():
    def __init__(self, all_cdNodes: Set[VNode]) -> None:
        self.all_dataNodes = all_cdNodes
        self.all_dataEdges = set()
        self.all_dataEdges: Set[DataEdge]
        self.contract_start = None
        self.contract_end = None
        self.func_entry = {}
        self.func_returns: Dict[Function, Set[VNode]]
        self.func_returns = {}
        self.func_dependency = {}
        self.func_edges = {}
        self.func_edges: Dict[Function, Set[DataEdge]]
        self.func_nodes = {}
        self.func_nodes: Dict[Function, Set[VNode]]
        self.func_dependency: Dict[Function, Dict[VNode, VNode]]
        self.func_var2node_ssa = {}
        self.func_var2node_ssa: Dict[Function, Dict[Variable_types, VNode]]
        self.func_params = {}
        self.state2node = {}
        self.state2node: Dict[StateVariable, Set[VNode]]
        # self.call_site_id = -1
        self.init()

    def init(self):
        functions = set()
        for cdNode in self.all_dataNodes:
            if cdNode.is_start_contract:
                self.contract_start = cdNode
                continue
            if cdNode.is_end_contract:
                self.contract_end = cdNode
                continue
            if cdNode.is_entry:
                self.func_entry[cdNode.function] = cdNode
            self.func_nodes[cdNode.function] = self.func_nodes.get(cdNode.function, set()) | {cdNode}
            if cdNode.function is not None:
                functions.add(cdNode.function)

        for function in functions:
            if function.is_constructor_variables:
                continue
            self.func_params[function] = []
            self.func_dependency[function] = {}
            self.func_var2node_ssa[function] = {}
            self.func_edges[function] = set()
            for p in function.parameters_ssa:
                if isinstance(p, Constant):
                    continue
                p_no_ssa = self.convert_variable_to_non_ssa(p)
                if p_no_ssa:
                    self.func_params[function].append(p_no_ssa)

        for cdNode in self.all_dataNodes:
            if cdNode.is_start_contract or cdNode.is_end_contract:
                continue
            if cdNode.is_entry and not cdNode.function.is_constructor_variables:
                continue
            if cdNode.node.type == NodeType.OTHER_ENTRYPOINT:
                continue
            irs = cdNode.node.irs_ssa
            for ir in irs:
                if isinstance(ir, OperationWithLValue) and ir.lvalue:
                    if isinstance(ir.lvalue, LocalIRVariable) and ir.lvalue.is_storage:
                        continue
                    if isinstance(ir.lvalue, StateIRVariable):
                        if isinstance(ir, Phi):
                            continue
                        lvalue = self.convert_variable_to_non_ssa(ir.lvalue)
                        if lvalue:
                            self.state2node[lvalue] = self.state2node.get(lvalue, set()) | {cdNode}
                    elif isinstance(ir.lvalue, StateVariable):
                        if isinstance(ir, Phi):
                            continue
                        self.state2node[ir.lvalue] = self.state2node.get(ir.lvalue, set()) | {cdNode}
                    else:
                        self.func_var2node_ssa[cdNode.function][ir.lvalue] = cdNode
                elif isinstance(ir, Return):
                    self.func_returns[cdNode.function] = self.func_returns.get(cdNode.function, set()) | {cdNode}

    @staticmethod
    def convert_variable_to_non_ssa(v: Variable_types) -> Variable_types:
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
        return v

    def find_var_nodes(self, v: Variable, f: Function) -> Set[VNode]:
        if isinstance(v, StateIRVariable):
            v_no_ssa = self.convert_variable_to_non_ssa(v)
            return self.state2node.get(v_no_ssa, set())
        else:
            _node = self.func_var2node_ssa[f].get(v, None)
            if _node is None:
                v_no_ssa = self.convert_variable_to_non_ssa(v)
                if v_no_ssa in self.func_params[f]:
                    return {self.func_entry[f]}
                else:
                    return set()
            else:
                return {_node}

    def compute_data_dependency(self):
        for n in self.all_dataNodes:
            if n.is_start_contract or n.is_end_contract or n.is_exit:
                continue
            if n.is_entry and not n.function.is_constructor_variables:
                continue
            if n.node.type == NodeType.OTHER_ENTRYPOINT:
                continue
            irs = n.node.irs_ssa
            function = n.function
            for ir in irs:
                self.add_in_dependency(function, ir, n)
                if isinstance(ir, InternalCall):
                    self.add_to_dependency(function, ir, n)

    def add_to_dependency(self, function: Function, ir: Operation, cd_node: VNode):
        if isinstance(ir, InternalCall):
            args = ir.arguments
            target_function = ir.function
            call_site_id = cd_node.get_call_site_id(target_function)
            target_entry = self.func_entry.get(target_function, None)
            if target_entry is None:
                return
            arg_len = len(args)
            for i in range(arg_len):
                if isinstance(args[i], Constant):
                    continue
                if isinstance(args[i], list):
                    continue
                arg_nodes = self.find_var_nodes(args[i], function)
                if arg_nodes:
                    temp = self.func_dependency[target_function].get(target_entry, set())
                    for n in arg_nodes:
                        if n == cd_node:
                            continue
                        temp.add(n)
                        depend_var = self.convert_variable_to_non_ssa(args[i])
                        data_flow = DataEdge(target_entry, n, VEdge.DataFlowInvoke)
                        data_flow.set_call_site_id(call_site_id)
                        data_flow.store_flow(depend_var)
                        self.all_dataEdges.add(data_flow)
                        self.func_edges[target_function] = \
                            self.func_edges.get(target_function, set()) | {data_flow}
                        n.add_to_edge(data_flow)
                        target_entry.add_in_edge(data_flow)
                    self.func_dependency[target_function][target_entry] = temp

    def add_in_dependency(self, function: Function, ir: Operation, cd_node: VNode) -> None:
        # if isinstance(ir, OperationWithLValue) and isinstance(ir.lvalue, ReferenceVariable):
        #     alias = ir.lvalue.points_to
        #     if isinstance(alias, Constant):
        #         return
        #     if alias is not None:
        #         alias_nodes = self.find_var_nodes(alias, function)
        #         temp = self.func_dependency[function].get(cd_node, set())
        #         for n in alias_nodes:
        #             if n == cd_node:
        #                 continue
        #             temp.add(n)
        #             source_var = self.convert_variable_to_non_ssa(alias)
        #             target_var = self.convert_variable_to_non_ssa(target_var)
        #             data_flow = DataEdge(cd_node, n, VEdge.DataFlowAlias)
        #             data_flow.store_flow(source_var, target_var)
        #             self.all_dataEdges.add(data_flow)
        #             self.func_edges[function] = self.func_edges.get(function, set()) | {data_flow}
        #             n.add_to_edge(data_flow)
        #             cd_node.add_in_edge(data_flow)
        #         self.func_dependency[function][cd_node] = temp
        if isinstance(ir, InternalCall):
            # self.call_site_id += 1
            call_site_id = cd_node.get_call_site_id(ir.function)
            return_nodes = self.func_returns.get(ir.function, set())
            for rn in return_nodes:
                ret_var = self.return_var(rn)
                if isinstance(ret_var, Constant):
                    continue
                depend_var = self.convert_variable_to_non_ssa(ret_var)
                # target_var = self.convert_variable_to_non_ssa(target_var)
                data_flow = DataEdge(cd_node, rn, VEdge.DataFlowReturn)
                data_flow.set_call_site_id(call_site_id)
                data_flow.store_flow(depend_var)
                self.all_dataEdges.add(data_flow)
                self.func_edges[function] = self.func_edges.get(function, set()) | {data_flow}
                rn.add_to_edge(data_flow)
                cd_node.add_in_edge(data_flow)
                self.func_dependency[function][cd_node] = self.func_dependency[function].get(cd_node, set()) | {rn}
        # else:
        # if 'require' in str(ir):
        #     print('hahah')
        if isinstance(ir, HighLevelCall) or isinstance(ir, LowLevelCall):
            read = [ir.destination]
            if ir.call_value is not None:
                read.append(ir.call_value)
        # elif isinstance(ir, Index):
        #     read = [ir.variable_left]
        else:
            read = ir.read
        for v in read:
            if isinstance(v, Constant):
                continue
            if isinstance(v, ReferenceVariable):
                alias = v.points_to
                if isinstance(v, Constant):
                    continue
                if alias is None:
                    continue
                v_nodes = self.find_var_nodes(alias, function)
            else:
                v_nodes = self.find_var_nodes(v, function)

            if v_nodes:
                temp = self.func_dependency[function].get(cd_node, set())
                for n in v_nodes:
                    if n == cd_node:
                        continue
                    temp.add(n)
                    depend_var = self.convert_variable_to_non_ssa(v)
                    if n.function == cd_node.function:
                        data_flow = DataEdge(cd_node, n, VEdge.DataFlowNormal)
                    else:
                        data_flow = DataEdge(cd_node, n, VEdge.DataFlowGlobal)
                    data_flow.store_flow(depend_var)
                    self.all_dataEdges.add(data_flow)
                    self.func_edges[function] = self.func_edges.get(function, set()) | {data_flow}
                    n.add_to_edge(data_flow)
                    cd_node.add_in_edge(data_flow)
                self.func_dependency[function][cd_node] = temp

    @staticmethod
    def return_var(node: VNode):
        for ir in node.node.irs_ssa:
            if isinstance(ir, Return):
                return ir.read[0]

    def print_dependency(self):
        for function, dependencies in self.func_dependency.items():
            print(f'----------------{function}-------------')
            for node, depend_nodes in dependencies.items():
                print(node, '->depend:', end=' ')
                if not depend_nodes:
                    print('NULL')
                else:
                    print('')
                    for d in depend_nodes:
                        print('\t', d)

    def print_data_edges(self):
        for function, data_flows in self.func_edges.items():
            print(f'--------------{function}---------------')
            for flow in data_flows:
                print(flow)

    def print_nodes(self):
        for function, cdNodes in self.func_nodes.items():
            print(f'--------------{function}---------------')
            for node in cdNodes:
                print(node, '->sons:')
                if not node.sons:
                    print('\tNULL')
                else:
                    for n in node.sons:
                        print(f'\t{n}')