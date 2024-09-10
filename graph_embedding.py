from torch_geometric.loader import DataLoader
from myDataSet import MyOwnDataset
from Graph_AE.utils.train_utils import train_cp
import torch
import os
import json
import numpy as np
import sys

class GraphEmbedding:
    def __init__(self, device, epochs, hidden_layers: list, model_name='MIAGAE',
                 model_dir='data/model1/', c_rate=0.75, kernel=2, learning_rate=1e-3, pooling_ratio=0.7):
        self.device = device
        self.epochs = epochs
        self.shapes = hidden_layers
        self.hidden_layers = hidden_layers
        self.model_name = model_name
        self.kernel = kernel
        self.depth = len(self.hidden_layers)
        self.c_rate = c_rate
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.pooling_ratio = 0.7

    def train(self, data_name, batch_size, train_per=0.75):
        data_set = MyOwnDataset(data_name)
        data_set.data_name = data_name
        input_size = data_set.num_features
        n_train = int(train_per * len(data_set))
        train_set = DataLoader(data_set[:n_train], batch_size=batch_size, shuffle=True, num_workers=0)
        test_set = DataLoader(data_set[n_train:], batch_size=batch_size, shuffle=False, num_workers=0)

        if self.model_name == "MIAGAE":
            from Graph_AE.classification.Graph_AE import Net
            model = Net(input_size, self.kernel, self.depth,
                        [self.c_rate] * self.depth, self.shapes, self.device).to(self.device)
        elif self.model_name == "UNet":
            from Graph_AE.classification.UNet import Net
            model = Net(input_size, self.depth, self.c_rate, self.shapes, self.device).to(self.device)
        elif self.model_name == "Gpool":
            from Graph_AE.classification.Gpool_model import Net
            model = Net(input_size, self.depth, self.c_rate, self.shapes, self.device).to(self.device)
        elif self.model_name == "SAGpool":
            from Graph_AE.classification.SAG_model import Net
            model = Net(input_size, self.depth, [self.c_rate] * self.depth, self.shapes, self.device).to(self.device)
        else:
            print("model not found")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        print('start training......')
        train_cp(model, optimizer, self.device, train_set, test_set, self.epochs, self.model_dir, self.model_name)

    def infer(self, data_name, target_dir):
        data_set = MyOwnDataset(data_name)
        for f in os.listdir(data_name):
            if f.endswith('.sol'):
                os.makedirs(os.path.join(target_dir, f), exist_ok=True)
        data_set.data_name = data_name
        input_size = data_set.num_features
        if self.model_name == "MIAGAE":
            from Graph_AE.classification.Graph_AE import Net
            model = Net(input_size, self.kernel, self.depth,
                        [self.c_rate] * self.depth, self.shapes, self.device).to(self.device)
        elif self.model_name == "UNet":
            from Graph_AE.classification.UNet import Net
            model = Net(input_size, self.depth, self.c_rate, self.shapes, self.device).to(self.device)
        elif self.model_name == "Gpool":
            from Graph_AE.classification.Gpool_model import Net
            model = Net(input_size, self.depth, self.c_rate, self.shapes, self.device).to(self.device)
        elif self.model_name == "SAGpool":
            from Graph_AE.classification.SAG_model import Net
            model = Net(input_size, self.depth, [self.c_rate] * self.depth, self.shapes, self.device).to(self.device)
        else:
            print("model not found")
            return
        model_path = os.path.join(self.model_dir, f'{self.model_name}.ckpt')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
        infer_set = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0)
        i = 0
        for data in infer_set:
            # _, node_embed, edge_embed, batch, node_scores, start_nodes, external_nodes = model(data)
            _, node_embed, edge_embed, batch, node_scores, external_nodes, modifier_nodes = model(data)
            avg_embed = node_embed.sum(dim=0) / node_embed.size(0)
            en_id = external_nodes[0]
            en_embed = node_embed[en_id]
            sort_inds = torch.argsort(-node_scores)
            k = node_embed.size(0) * (self.pooling_ratio ** 3)
            k = round(k)
            # print(k)
            if len(sort_inds) >= k:
                top_k_inds = sort_inds[:k]
            else:
                top_k_inds = sort_inds
            top_k_embed = node_embed[top_k_inds].sum(dim=0)/node_embed[top_k_inds].size(0)
            me_list = []
            m_scores = []
            modifier = ''
            for modifier_list in modifier_nodes[0]:
                me = torch.zeros(en_embed.shape[0])
                ms = 0
                nlen = 0
                for m_id in modifier_list:
                    me += node_embed[m_id]
                    ms += node_scores[m_id]
                    nlen += 1
                m_scores.append(ms)
                if nlen > 0:
                    me = me / nlen
                me_list.append(me)
            if len(m_scores) == 0:
                mn_embed = torch.zeros(en_embed.shape[0])
            else:
                max_ind = torch.argmax(torch.Tensor(m_scores))
                mn_embed = me_list[max_ind]
            graph_embed = torch.cat((en_embed, mn_embed, top_k_embed))
            graph_data = {"source_mapping": data.source_mapping[0],
                          "file": data.file[0], "graph_embedding": graph_embed.tolist(),
                          'modifier_call_exps': data.modifier_call_exps[0],
                          'function_call_list': data.function_call_list[0]}
            if i % 100 == 0:
                print(f'processed {i}/{len(infer_set)} files.....')
            dir_name = data.file[0].split('/')[-1]
            os.makedirs(os.path.join(target_dir, dir_name, data.source_mapping[0].split('#')[-1]), exist_ok=True)
            file_name = ''
            for f in data.function_call_list[0]:
                file_name += f + '-'
            if len(file_name) > 0:
                file_name = file_name[:-1]
            with open(os.path.join(target_dir, dir_name,
                                   data.source_mapping[0].split('#')[-1], f'{file_name}.json'), 'w') as f:
                json.dump(graph_data, f)
            i += 1


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('please input correct parms')
        sys.exit()
    dataset_name = sys.argv[1]
    target_dir = sys.argv[4]
    down_sampling_ratio = float(sys.argv[3])
    pooling_ratio = 0.7
    c1 = round(160 * down_sampling_ratio)
    c2 = round(160 * (down_sampling_ratio**2))
    c3 = round(160 * (down_sampling_ratio**3))
    graph_e = GraphEmbedding("cpu", 60,  [c1, c2, c3], model_name="MIAGAE", learning_rate=1e-3, pooling_ratio=pooling_ratio)
    if sys.argv[2] == 'train':
        graph_e.train(dataset_name, 256)
    else:
        print(target_dir)
        graph_e.infer(dataset_name, target_dir=target_dir)