from dependence_graph.VEdge import VEdge

class ControlEdge(VEdge):
    def __init__(self, _head, _tail, _type=VEdge.ControlNormal, _context: dict = None) -> None:
        '''
        :param _head:
        :param _tail:
        :param _type:
        :param _context:
        :return:
        '''

        super().__init__(_head, _tail, _type, _context)
        if _type is None:
            self._type = VEdge.ControlNormal

