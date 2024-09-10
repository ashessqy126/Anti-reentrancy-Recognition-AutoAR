import os
import json
import numpy as np
from scipy.spatial.distance import pdist, mahalanobis
from graph_embedding import GraphEmbedding
import sys

def identification(dir_name):
    threshold = 2.3
    fileNum = 0
    total_positive = 0
    centroids = np.load('centroids.npy', allow_pickle=True)
    for file in os.listdir(dir_name):
        if not file.endswith('.sol'):
            continue
        fileNum += 1
        file_flag = True
        graph_paths = []
        for root, dirs, files in os.walk(os.path.join(dir_name, file)):
            for f in files:
                if not f.endswith('.json'):
                    continue
                graph_paths.append(os.path.join(root, f))

        for graph_path in graph_paths:
            flag = False
            with open(graph_path, 'r') as f:
                json_data = json.load(f)
                gembed = json_data['graph_embedding']
                for c, std_dev, d_mean, d_std in centroids:
                    X = np.vstack((gembed, c))
                    dist = pdist(X, metric='seuclidean', V=std_dev)[0]
                    if dist < d_mean + d_std * threshold:
                        flag = True
                        break
            if not flag:
                # ddist[mind_mean] = ddist.get(mind_mean, []) + [[min_dist, mind_std]]
                file_flag = False
                break

        if file_flag:
            total_positive += 1
    print(f'>>>anti-reentrancy detected: {total_positive}/{fileNum}')
    return total_positive, fileNum


def output_embed(dataset_name, down_sampling_ratio, pooling_ratio, embed_dir):
    c1 = round(160 * down_sampling_ratio)
    c2 = round(160 * (down_sampling_ratio ** 2))
    c3 = round(160 * (down_sampling_ratio ** 3))
    graph_e = GraphEmbedding("cpu", None, [c1, c2, c3], model_name="MIAGAE",
                             learning_rate=None, pooling_ratio=pooling_ratio)
    graph_e.infer(dataset_name, target_dir=embed_dir)


def sys_recognition(source_dataset_name, down_sampling_ratio, pooling_ratio, embed_dir):
    output_embed(source_dataset_name, down_sampling_ratio, pooling_ratio, embed_dir)
    return identification(embed_dir)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('incorrect parameter numbers')
    source_graph_data = sys.argv[1]
    embed_dir = sys.argv[2]
    sys_recognition(source_graph_data, 0.7, 0.7, embed_dir)
