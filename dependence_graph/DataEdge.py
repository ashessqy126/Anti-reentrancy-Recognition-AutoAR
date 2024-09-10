from dependence_graph.VEdge import VEdge
from dependence_graph.VNode import VNode

class DataEdge(VEdge):
    def __int__(self, _head: VNode, _tail: VNode, _type=VEdge.DataFlowNormal, _context: dict = None):
        '''
        :param _head:
        :param _tail:
        :param _type:
        :param _context:
        :return:
        '''
        super().__init__(_head, _tail, _type, _context)
        if _type is None:
            self._type = VEdge.DataFlowNormal
        self._depend_var = None

    @property
    def type(self):
        return self._type

    @property
    def depend_var(self):
        return self._depend_var

    def store_flow(self, var):
        self._depend_var = var

    def __str__(self):
        return f'{self._tail.node} -> {self._head.node} ({self.depend_var})'

