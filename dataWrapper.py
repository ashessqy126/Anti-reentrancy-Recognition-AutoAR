import torch
from torch_geometric.data import Data
import json


class DataWrapper:
    def __init__(self, root):
        self.root = root

    def wrap_data(self, raw_file_path):
        with open(raw_file_path, 'r') as f:
            json_data = json.load(f)
        if not json_data:
            return None
        x = torch.tensor(json_data['nodes'], dtype=torch.float)
        edge_index = torch.tensor(json_data['edge_index'])
        edge_attr = torch.tensor(json_data['edges'])
        file = json_data['file']
        call_expression = json_data['call_expression']
        source_mapping = json_data['source_mapping']
        external_node = json_data['external_node']
        start_node = json_data['start_node']
        modifier_nodes = json_data['modifier_calls']
        function_call_list = json_data['function_call_list']
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=None,
                    file=file, call_expression=call_expression, source_mapping=source_mapping,
                    external_node=external_node, start_node=start_node, modifier_nodes=modifier_nodes,
                    function_call_list=function_call_list)


if __name__ == '__main__':
    dw = DataWrapper('raw_reentrancy_graph')
    # data = dw.wrap_data('raw_reentrancy_graph/0x0a36d9abd535cd863b6c228039e29cca9c07334a.sol/graph_0.json')
    print(dw)
