
class VEdge():
    ControlNormal = 0
    ControlTrue = 0
    ControlFalse = 0
    ControlInvoke = 1
    ControlReturn = 1
    ControlRevert = 0
    DataFlowNormal = 2
    DataFlowAlias = 2
    DataFlowInvoke = 2
    DataFlowReturn = 2
    DataFlowGlobal = 3

    def __init__(self, _head, _tail, _type=None, _context: dict = None) -> None:
        self._tail = _tail
        self._head = _head
        if _context is None:
            _context = dict()
        self._context = _context
        # self._target_var = _target_var
        # self._source_var = _source_var
        self._type = _type
        self._call_site_id = None

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        return self._tail

    @property
    def weight(self):
        return self._weight

    @property
    def type(self):
        return self._type

    @property
    def call_site_id(self):
        return self._call_site_id

    def set_head(self, head):
        self._head = head

    def set_tail(self, tail):
        self._tail = tail

    def set_type(self, t):
        self._type = t

    def set_call_site_id(self, cid):
        self._call_site_id = cid

    def set_context(self, _context):
        self._context = _context

    def is_control(self):
        if self._type in [VEdge.ControlNormal,
                          VEdge.ControlTrue,
                          VEdge.ControlFalse,
                          VEdge.ControlReturn,
                          VEdge.ControlInvoke]:
            return True
        return False

    def is_data(self):
        if self._type in [VEdge.DataFlowInvoke,
                          VEdge.DataFlowReturn,
                          VEdge.DataFlowNormal,
                          VEdge.DataFlowAlias]:
            return True
        return False

    def __hash__(self) -> int:
        return hash((self._head.node, self._tail.node, self._type))

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, VEdge):
            return self.head.node == __o.head.node and self._tail.node == __o.tail.node and self.type == __o.type
        return False

    def __str__(self):
        return f'{self._tail.node} -> {self._head.node}'
