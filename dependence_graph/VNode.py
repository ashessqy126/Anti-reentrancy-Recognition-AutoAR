from slither.core.cfg.node import Node, NodeType
from slither.slither import Slither
from slither.core.declarations import Function, Contract
from dependence_graph.VEdge import VEdge
from dependence_graph.ControlEdge import ControlEdge
from slither.slithir.operations import InternalCall, LibraryCall
from copy import copy

class VNode():
    def __init__(self, _node: Node, _status=0, function: Function = None,
                 ct: Contract = None, _calls=None, _call2id=None) -> None:
        self._node = _node
        self._to_edges = set()
        self._in_edges = set()
        self.is_entry = False
        self.is_exit = _status == 1
        self.is_start_contract = _status == 2
        self.is_end_contract = _status == 3
        self.is_pre_call = _status == 4
        self.is_post_call = _status == 5
        self.function = function
        self.contract = ct
        self._status = _status
        if _calls is None:
            _calls = set()
        if _call2id is None:
            _call2id = {}
        self._calls = _calls
        self._call2id = _call2id

        if isinstance(_node, Node):
            self.function = _node.function
            self.contract = _node.function.contract
            for ir in _node.irs:
                if isinstance(ir, InternalCall):
                    self._calls.add(ir.function)
                # if isinstance(ir, LibraryCall):
                #     self._calls.add(ir.function)
        if function is not None and _node == function.entry_point:
            self.is_entry = True
            self.is_exit = False
            self.is_start_contract = False
            self.is_end_contract = False
            self._status = 0
            # if not function.is_constructor_variables and _node == function.entry_point.sons[0]:
            #     self.is_entry = True
            #     self.is_exit = False
            #     self.is_start_contract = False
            #     self.is_end_contract = False
            #     self._status = 0
        if _status == 1:
            assert function is not None, 'please specify the function when VNode is EXIT type'
            self.function = function
            self.contract = function.contract
            self._node = f'{function.contract}.{function}.EXIT_POINT'
        if _status == 2:
            assert ct is not None, 'please specify the contract when VNode is START_of_CONTRACT type'
            self._node = f'{ct}.START_CONTRACT'
        if _status == 3:
            assert ct is not None, 'please specify the contract when VNode is END_of_CONTRACT type'
            self._node = f'{ct}.END_CONTRACT'
    
    def set_node(self, _node):
        self._node = _node
    
    def set_status(self, _status):
        self._status = _status
        if _status == 1:
            self.is_exit = True
            self.is_entry = False
            self.is_start_contract = False
            self.is_end_contract = False
            self.is_pre_call = False
            self.is_post_call = False
            self._node = f'{self.function.contract}.{self.function}.EXIT_POINT'
        elif _status == 2:
            self.is_start_contract = True
            self.is_exit = False
            self.is_entry = False
            self.is_end_contract = False
            self.is_pre_call = False
            self.is_post_call = False
            self._node = f'{self.function.contract}.{self.function}.START_POINT'
        elif _status == 3:
            self.is_start_contract = False
            self.is_exit = False
            self.is_entry = False
            self.is_end_contract = True
            self.is_pre_call = False
            self.is_post_call = False
            self._node = f'{self.contract}.END_CONTRACT'
        elif _status == 4:
            self.is_start_contract = False
            self.is_exit = False
            self.is_entry = False
            self.is_end_contract = False
            self.is_pre_call = True
            self.is_post_call = False
        elif _status == 5:
            self.is_start_contract = False
            self.is_exit = False
            self.is_entry = False
            self.is_end_contract = False
            self.is_pre_call = False
            self.is_post_call = True
    
    def set_to_edges(self, to_edges):
        self._to_edges = to_edges
    
    def set_in_edges(self, in_edges):
        self._in_edges = in_edges

    def add_to_edge(self, edge:VEdge):
        self._to_edges.add(edge)
    
    def add_in_edge(self, edge:VEdge):
        self._in_edges.add(edge)

    @property
    def call2id(self):
        return self._call2id

    @property
    def calls(self):
        return self._calls

    @property
    def in_edges(self):
        return self._in_edges

    @property
    def to_edges(self):
        return self._to_edges

    @property
    def node(self):
        return self._node
    
    @property
    def sons(self):
        _sons = set()
        for to_edge in self._to_edges:
            _sons.add(to_edge.head)
        return _sons

    @property
    def fathers(self):
        _fathers = set()
        for in_edge in self._in_edges:
            _fathers.add(in_edge.tail)
        return _fathers

    @property
    def status(self):
        return self._status

    def get_call_site_id(self, invoked_func):
        return self._call2id.get(invoked_func, None)

    def set_call_site_id(self, invoked_func, call_id):
        self._call2id[invoked_func] = call_id

    def __hash__(self) -> int:
        return hash((self._node, self.function, self.contract, self._status))

    def __str__(self):
        if self.is_exit:
            return f'{self.function.contract}.{self.function}.EXIT_POINT'
        elif self.is_start_contract:
            return f'{self.contract}.START_CONTRACT'
        elif self.is_end_contract:
            return f'{self.contract}.END_CONTRACT'
        else:
            return str(self._node) + f' ({self.function})'

    def __eq__(self, __o: object) -> bool:
        return (self._node == __o.node) and (self._in_edges == self._in_edges) and \
               (self._to_edges == self._to_edges) and (self.function == __o.function) and \
               (self.contract == __o.contract) and (self.status == __o.status)

    def __copy__(self):
        new_calls = copy(self._calls)
        new_call2id = copy(self._call2id)
        return VNode(self._node, self._status, self.function, self.contract, new_calls, new_call2id)


if __name__ == '__main__':
    sl = Slither('test.sol')
    contract = sl.get_contract_from_name('GoblinRareApepeYC')[0]
    func = contract.get_function_from_signature('tokenURI(uint256)')
    node = func.nodes[0]

    a = VNode(node)
    b = VNode(node.sons[0])
    edge = ControlEdge(VNode(node), VNode(node.sons[0]), ControlEdge.ControlNormal)
    a.add_in_edge(edge)
    b.add_to_edge(edge)
    s = {a, b}
    # s.add(VNode(node.sons[0]))
    print(f'first {len(s)}')
    a = VNode(node)
    a.add_in_edge(edge)
    s.add(a)
    print(f'second {len(s)}')
    for item in s:
        print(item)