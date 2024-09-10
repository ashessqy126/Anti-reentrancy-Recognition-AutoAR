import os

import torch
from torch_geometric.data import InMemoryDataset, Data
from dataWrapper import DataWrapper
from torch_geometric.loader import DataLoader
from time import time
import multiprocessing as mp

class MyOwnDataset(InMemoryDataset):
    def __init__(self, raw_data, transform=None, pre_transform=None):
        self.raw_data = raw_data
        super().__init__(raw_data, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_files = []
        for root, dirs, files in os.walk(self.raw_data):
            for file in files:
                if not file.endswith('.json'):
                    continue
                raw_files.append(os.path.join(root, file))
        return raw_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        dw = DataWrapper(self.raw_data)
        data_list = []
        for raw_file in self.raw_file_names:
            data = dw.wrap_data(raw_file)
            # if data.edge_index.dtype != torch.long:
            #     print('hahah')
            if data is None:
                continue
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # def len(self) -> int:
    #     return len(self.raw_file_names)

    # def get(self, idx: int) -> Data:
    #     return self.data[idx]


if __name__ == '__main__':
    train_set = MyOwnDataset('test_non_vulnerable_graphs_1')
    for num_workers in range(0, mp.cpu_count(), 2):
        train_loader = DataLoader(train_set, shuffle=True, num_workers=num_workers, batch_size=128, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            print(f'epoch: {epoch}')
            for data in train_loader:
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
