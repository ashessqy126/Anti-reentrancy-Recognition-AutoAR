from slither.core.cfg.node import Node, NodeType
from slither.slither import Slither
from slither.slithir.operations import Call
from slither.core.declarations import Contract
from VNode import VNode
from slither.analyses.data_dependency.data_dependency import get_dependencies, get_dependencies_ssa, pprint_dependency
from slither.core.variables.local_variable import LocalVariable
from slither.core.variables.state_variable import StateVariable
from slither.slithir.variables.temporary import TemporaryVariable

def extract_cfg_node(node: Node, visited:set()):
    if node in visited:
        return None
    visited.add(node)
    for father in node.fathers:
        extract_cfg_node(father, visited)

def extract_control_flow_graph(start_node: Node):
    visited = set()
    node_mapping = dict()
    extract_cfg_node(start_node, visited)
    for n in visited:
        wrapped_node = GNode(n)
        node_mapping[n] = wrapped_node

    ENTRY_NODE = None
    for n in visited:
        wrapped_node = node_mapping[n]
        for son in n.sons:
            if son in visited:
                wrapped_node.add_son(node_mapping[son])
        for father in n.fathers:
            if father in visited:
                wrapped_node.add_father(node_mapping[father])
        if str(n) == 'ENTRY_POINT':
            ENTRY_NODE = wrapped_node
    return ENTRY_NODE

def extract_data_flow_graph(var, contract):
    print('original_var', var)
    # pprint_dependency(context)
    node: Node
    depend_vars = get_dependencies_ssa(var, contract)
    depend_nodes = []
    for f in contract.functions:
        for node in f.nodes:
            find = False
            for v in node.variables_written:
                if v in depend_vars:
                    find = True
            if find:
                depend_nodes.append(node)

    for n in depend_nodes:
        print(n)


def contain_call(node:Node):
    for ir in node.all_slithir_operations():
        if isinstance(ir, Call) and ir.can_reenter():
            return True
    return False

def extract_dest(node: Node):
    for ir in node.all_slithir_operations():
        if isinstance(ir, Call) and ir.can_reenter():
            return ir.destination
    return None

def test():
    sl = Slither('test.sol')
    contract = sl.get_contract_from_name('GoblinRareApepeYC')
    contract = contract[0]
    func = None
    call_node = None
    for f in contract.functions:
        print('----function', f)
        if f.name == 'mint':
            func = f
    for node in func.nodes:
        if contain_call(node):
            call_node = node
            break
    dest = extract_dest(call_node)
    extract_data_flow_graph(dest, contract)

def explore(gnode: GNode, visited:set()):
    if gnode in visited:
        return
    visited.add(gnode)
    print(gnode.node)
    for son in gnode.sons:
        explore(son, visited)

if __name__ == '__main__':
    test()