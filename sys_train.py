import sys

from graph_embedding import GraphEmbedding
from test_clustering import *

def train_graph_embed(dataset_name, down_sampling_ratio, pooling_ratio):
    c1 = round(160 * down_sampling_ratio)
    c2 = round(160 * (down_sampling_ratio ** 2))
    c3 = round(160 * (down_sampling_ratio ** 3))
    graph_e = GraphEmbedding("cpu", 60, [c1, c2, c3], model_name="MIAGAE",
                             learning_rate=1e-3, pooling_ratio=pooling_ratio)
    graph_e.train(dataset_name, 256)

def output_embed(dataset_name, down_sampling_ratio, pooling_ratio, embed_dir):
    c1 = round(160 * down_sampling_ratio)
    c2 = round(160 * (down_sampling_ratio ** 2))
    c3 = round(160 * (down_sampling_ratio ** 3))
    graph_e = GraphEmbedding("cpu", None, [c1, c2, c3], model_name="MIAGAE",
                             learning_rate=None, pooling_ratio=pooling_ratio)
    graph_e.infer(dataset_name, target_dir=embed_dir)

def clustering(embed_dir):
    graph_data = {'file': [], 'embedding': [], 'source_mapping': []}
    for root, dirs, files in os.walk(embed_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['file'].append(json_data['file'])
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['source_mapping'].append(json_data['source_mapping'])
    data = np.array(graph_data['embedding'])
    estimator = AgglomerativeClustering(n_clusters=12,
                                        linkage='ward')
    estimator.fit(data)
    labels = estimator.labels_
    extract_cluster_centroids(data, labels)


def train_sys(source_dataset_name, down_sampling_ratio, pooling_ratio, embed_dir):
    train_graph_embed(source_dataset_name, down_sampling_ratio, pooling_ratio)
    output_embed(source_dataset_name, down_sampling_ratio, pooling_ratio, embed_dir)
    clustering(embed_dir)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('incorrect parameter numbers')
    down_sampling_ratio = float(sys.argv[1])
    pooling_ratio = float(sys.argv[2])
    source_graph_data = sys.argv[3]
    embed_dir = sys.argv[4]
    train_sys(source_graph_data, down_sampling_ratio, pooling_ratio, embed_dir)