from sklearn.cluster import (KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, SpectralClustering,
                             estimate_bandwidth, AgglomerativeClustering, DBSCAN, OPTICS, Birch)
from scipy.cluster.hierarchy import linkage, dendrogram
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
import pandas as pd
from sklearn import metrics
import plotly.io as pio
from scipy.spatial.distance import pdist, mahalanobis

def kmean_clustering(dir_name: str):
    # read two-dimensional input data 'Target'
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    SSE = []
    for k in range(1, 20):
        print(f'n_clusters: {k}')
        estimator = KMeans(n_clusters=k, random_state=0, init="k-means++")
        estimator.fit(data)
        SSE.append(estimator.inertia_)
    # 8
    X = range(1, 20)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


def mini_kmeans_clustering(dir_name: str):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    SSE = []
    for k in range(1, 20):
        print(f'n_clusters: {k}')
        estimator = MiniBatchKMeans(n_clusters=k, random_state=0, init="k-means++", batch_size=128)
        estimator.fit(data)
        SSE.append(estimator.inertia_)
    # 8
    X = range(1, 20)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()

def mini_kmeans_display(dir_name: str, n_clusters):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, init="k-means++", batch_size=128)
    km.fit(data)
    labels = km.labels_
    X_tsne = TSNE(learning_rate=200, perplexity=30).fit_transform(data)
    df = pd.DataFrame(columns=['x', 'y', 'labels'])
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['labels'] = np.zeros(X_tsne.shape[0], dtype=int)
    # cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'magenta', 7: 'yellow'}
    # colour scheme
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'labels'] = np.array([l for _ in range(n_ix)], dtype=int)
    fig = px.scatter(df, x='x', y='y', color='labels', hover_name=graph_data['text'],
                     color_continuous_scale="rainbow")
    fig.show()


def kmeans_display(dir_name: str, n_clusters):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + json_data['source_mapping'].split('/')[-1])
    data = np.array(graph_data['embedding'])
    km = KMeans(n_clusters=8, random_state=0, init="k-means++")
    km.fit(data)
    labels = km.labels_
    X_tsne = TSNE(learning_rate=200, perplexity=30).fit_transform(data)
    df = pd.DataFrame(columns=['x', 'y', 'labels'])
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['labels'] = np.zeros(X_tsne.shape[0], dtype=int)
    # cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'magenta', 7: 'yellow'}
    # colour scheme
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'labels'] = np.array([l for _ in range(n_ix)], dtype=int)
    fig = px.scatter(df, x='x', y='y', color='labels', hover_name=graph_data['text'],
                         color_continuous_scale="rainbow")
    fig.show()


def AP_clustering(dir_name):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    af = AffinityPropagation(preference=-800, random_state=0).fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    # SSE = []
    # for k in range(1, 20):
    #     print(f'n_clusters: {k}')
    #     estimator = MiniBatchKMeans(n_clusters=k, random_state=0, init="k-means++", batch_size=128)
    #     estimator.fit(data)
    #     SSE.append(estimator.inertia_)
    # 8
    print("Estimated number of clusters: %d" % n_clusters_)
    print(
        "Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(data, labels, metric="sqeuclidean")
    )


def AP_display(dir_name: str, n_clusters):
    pass

def Mean_shift_clustering(dir_name: str):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    bandwidth = estimate_bandwidth(data, quantile=0.1, n_samples=5000)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(data)
    labels = ms.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("Estimated number of clusters: %d" % n_clusters_)
    print(
        "Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(data, labels, metric="sqeuclidean")
    )

def Mean_shift_display(dir_name):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    bandwidth = estimate_bandwidth(data, quantile=0.1, n_samples=5000)
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(data)
    labels = ms.labels_
    n_clusters = len(np.unique(labels))
    X_tsne = TSNE(learning_rate=200, perplexity=20).fit_transform(data)
    df = pd.DataFrame(columns=['x', 'y', 'labels'])
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['labels'] = np.zeros(X_tsne.shape[0], dtype=int)
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'labels'] = np.array([l for _ in range(n_ix)], dtype=int)
    fig = px.scatter(df, x='x', y='y', color='labels', hover_name=graph_data['text'],
                     color_continuous_scale="rainbow")
    fig.show()

def spectural_clustering_test(dir_name: str):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    SSE = []
    for k in range(1, 20):
        print(f'n_clusters: {k}')
        estimator = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=0)
        estimator.fit(data)
        SSE.append(estimator.inertia_)
    # 8
    X = range(1, 20)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()


def spectural_display(dir_name, n_clusters):
    #time-consuming
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    # bandwidth = estimate_bandwidth(data, quantile=0.1, n_samples=5000)
    sc = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    sc.fit(data)
    labels = sc.labels_
    n_clusters = len(np.unique(labels))
    X_tsne = TSNE(learning_rate=200, perplexity=20).fit_transform(data)
    df = pd.DataFrame(columns=['x', 'y', 'labels'])
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['labels'] = np.zeros(X_tsne.shape[0], dtype=int)
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'labels'] = np.array([l for _ in range(n_ix)], dtype=int)
    fig = px.scatter(df, x='x', y='y', color='labels', hover_name=graph_data['text'],
                     color_continuous_scale="rainbow")
    fig.show()


def agglomerative_clustering(dir_name: str):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'])
    data = np.array(graph_data['embedding'])
    SSE = []
    for k in range(5, 20):
        print(f'clusters: {k}')
        estimator = AgglomerativeClustering(linkage='ward', n_clusters=k)
        estimator.fit(data)
        labels = estimator.labels_
        # print(f'distance_theshold: {estimator.distances_}')
        # print(f'n_clusters: {len(np.unique(labels))}')
        SSE.append(metrics.silhouette_score(data, labels, metric='sqeuclidean'))
        # SSE.append(metrics.davies_bouldin_score(data, labels))
        # 8
    with open('sc_data.txt', 'a') as f:
        for i in range(5, 20):
            f.write(str(i) + ' ')
        f.write('\n')
        for i in SSE:
            f.write(str(i) + ' ')
        f.write('\n')
    X = range(5, 20)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()
    # plt.savefig('silhouette_score_curve.pdf')


def agglomerative_display(dir_name: str, n_clusters=11):
    graph_data = {'embedding': [], 'text': [], 'source_mapping': []}
    callerisUser = []
    verifys = []
    transfers = []
    recovers = []
    idx = 0
    source = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + str(json_data['modifier_call_exps']))
                if 'callerIsUser' in json_data['modifier_call_exps']:
                    callerisUser.append(idx)
                    source.append(json_data['source_mapping'])
                if json_data['contain_verify']:
                    verifys.append(idx)
                    # source.append(json_data['source_mapping'])
                if json_data['contain_transfer']:
                    transfers.append(idx)
                if json_data['contain_recover']:
                    recovers.append(idx)
                idx += 1
    data = np.array(graph_data['embedding'])
    print(source)
    estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    estimator.fit(data)
    labels = estimator.labels_

    n_clusters = len(np.unique(labels))
    X_tsne = TSNE(learning_rate=30, n_components=2).fit_transform(data)
    df = pd.DataFrame(columns=['t-SNE Dim 1', 't-SNE Dim 2', 'Cluster'])
    df['t-SNE Dim 1'] = X_tsne[:, 0]
    df['t-SNE Dim 2'] = X_tsne[:, 1]
    df['Cluster'] = np.zeros(X_tsne.shape[0], dtype=int)
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'Cluster'] = np.array([l + 1 for _ in range(n_ix)], dtype=int)
    # for i in range(len(df)):
    #     if '006c58402b6e6b99101cdaffbb106b64862791a1' in graph_data['text'][i]:
    #         print(df.iloc[i]['labels'])color_continuous_scale="rainbow"
    #         print(df.iloc[i]['labels'])
    df["Cluster"] = df["Cluster"].astype(str)
    df.to_csv("backup_clusters.csv")
    fig = px.scatter(df, x='t-SNE Dim 1', y='t-SNE Dim 2', color='Cluster', hover_name=graph_data['text'],
                     color_continuous_scale="rainbow", category_orders={'Cluster': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']})
    # for ind in callerisUser:
    #     fig.add_annotation(x=X_tsne[ind, 0], y=X_tsne[ind, 1], text='caller')
    # for ind in verifys:
    #     fig.add_annotation(x=X_tsne[ind, 0], y=X_tsne[ind, 1], text='verify')
    # for ind in recovers:
    #     fig.add_annotation(x=X_tsne[ind, 0], y=X_tsne[ind, 1], text='recover')
    plt.rcParams["figure.figsize"] = (8, 8)
    fig.update_layout(font_family="Times New Roman", font_size=38, font_color='black', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=34)
    ))
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 1, row = 1, col = 1, mirror = True)
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 1, row = 1, col = 1, mirror = True)

    pio.write_image(fig, 'clustering.pdf', 'pdf')
    
    fig.show()
    


def DBSCAN_clustering(dir_name: str):
    graph_data = {'embedding': [], 'text': [], 'source_mapping': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + json_data['source_mapping'].split('/')[-1])
    data = np.array(graph_data['embedding'])
    db = DBSCAN(eps=7.5, min_samples=8)
    db.fit(data)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(data, labels):.3f}")


def DBSCAN_display(dir_name: str, eps=7.5):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + json_data['source_mapping'].split('/')[-1])
    data = np.array(graph_data['embedding'])
    db = DBSCAN(eps=eps, min_samples=8)
    db.fit(data)
    labels = db.labels_
    n_clusters = len(np.unique(labels))
    X_tsne = TSNE(learning_rate=200, perplexity=20).fit_transform(data)
    df = pd.DataFrame(columns=['x', 'y', 'labels'])
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['labels'] = np.zeros(X_tsne.shape[0], dtype=int)
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'labels'] = np.array([l for _ in range(n_ix)], dtype=int)
    fig = px.scatter(df, x='x', y='y', color='labels', hover_name=graph_data['text'],
                     color_continuous_scale="rainbow")
    fig.show()


def OPTICS_clustering(dir_name: str):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + json_data['source_mapping'].split('/')[-1])
    data = graph_data['embedding']
    clust = OPTICS(min_samples=60, xi=0.05, min_cluster_size=0.01)
    clust.fit(data)
    labels = clust.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(data, labels):.3f}")


def birch_clustering(dir_name: str):
    #一般
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + json_data['source_mapping'].split('/')[-1])
    data = graph_data['embedding']
    for t in [9 + 0.1 * i for i in range(10)]:
        birch_model = Birch(threshold=t, n_clusters=None)
        birch_model.fit(data)
        labels = birch_model.labels_
        centroids = birch_model.subcluster_centers_
        n_clusters = np.unique(labels).size
        print("threshold : %f" % t)
        print("n_clusters : %d" % n_clusters)
        print(f"Silhouette Coefficient: {metrics.silhouette_score(data, labels):.3f}")


def birch_display(dir_name: str, threshold=9.6):
    graph_data = {'embedding': [], 'text': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['text'].append(json_data['call_expression'] + '; '
                                          + json_data['source_mapping'].split('/')[-1])
    data = np.array(graph_data['embedding'])
    birch_model = Birch(threshold=threshold, n_clusters=None)
    birch_model.fit(data)
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    X_tsne = TSNE(learning_rate=200, perplexity=20).fit_transform(data)
    df = pd.DataFrame(columns=['x', 'y', 'labels'])
    df['x'] = X_tsne[:, 0]
    df['y'] = X_tsne[:, 1]
    df['labels'] = np.zeros(X_tsne.shape[0], dtype=int)
    for l in range(n_clusters):
        ix = np.where(labels == l)
        n_ix = len(ix[0])
        df.loc[ix[0], 'labels'] = np.array([l for _ in range(n_ix)], dtype=int)
    fig = px.scatter(df, x='x', y='y', color='labels', hover_name=graph_data['text'],
                     color_continuous_scale="rainbow")
    fig.show()


def extract_cluster_centroids(data, labels):
    n_clusters = len(np.unique(labels))
    cluster_centroids = []
    statistics = []
    for i in range(n_clusters):
        ix = np.where(labels == i)
        data_in_cluster = data[ix[0]]
        c = np.mean(data_in_cluster, axis=0)
        std_dev = np.std(data_in_cluster, axis=0)
        std_dev[std_dev==0] = 0.001
        # cov = np.cov(data_in_cluster, rowvar=False)
        distances = []
        for d in data_in_cluster:
            X = np.vstack((d, c))
            distances.append(pdist(X, 'seuclidean', V=std_dev)[0])
        statistics.append([c, std_dev, np.mean(distances), np.std(distances)]);
        cluster_centroids.append(c)
    statistics = np.array(statistics, dtype=object)
    np.save('centroids.npy', statistics)
    return np.array(cluster_centroids)


def extract_surrounding_samples(data, labels, cluster_centroids, n_samples=10):
    if n_samples <= 0:
        return []
    n_labels = len(np.unique(labels))
    cluster_samples = []
    for i in range(n_labels):
        ix = np.where(labels == i)
        data_in_label = data[ix[0]]
        centroid = cluster_centroids[i]
        centroid_array = np.array([centroid for _ in range(data_in_label.shape[0])])
        dist = np.linalg.norm(data_in_label-centroid_array, axis=1)
        indices = np.argsort(dist)
        ori_indices = ix[0][indices]
        cluster_samples.append(ori_indices[:n_samples])
    return np.array(cluster_samples)

def output_cluster_files(labels, graph_data):
    n_labels = len(np.unique(labels))
    cluster_files = []
    for i in range(n_labels):
        ix = np.where(labels == i)
        cluster_files.append([graph_data['file'][k] for k in ix[0]])
    return cluster_files

def rand_extract_samples(data, labels, n_samples=10):
    if n_samples <= 0:
        return []
    n_labels = len(np.unique(labels))
    cluster_samples = []
    for i in range(n_labels):
        ix = np.where(labels == i)
        if n_samples <= len(ix[0]):
            cluster_samples.append(np.random.choice(ix[0], n_samples))
        else:
            cluster_samples.append(ix[0])
    return cluster_samples

def test_cluster(dir_name, show_samples=20):
    graph_data = {'file': [], 'embedding': [], 'text': [], 'source_mapping': [], 'function_call_list': [], 'vars_written': []}
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if not file.endswith('.json'):
                continue
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                json_data = json.load(f)
                graph_data['file'].append(json_data['file'])
                graph_data['embedding'].append(json_data['graph_embedding'])
                graph_data['source_mapping'].append(json_data['source_mapping'])
                func_call_list = ''
                for c in json_data['function_call_list']:
                    func_call_list += '-' + c
                graph_data['function_call_list'].append(func_call_list)
                graph_data['vars_written'].append(json_data['vars_written'])
                graph_data['text'].append(json_data['call_expression'] + '; ' + str(json_data['modifier_call_exps']))
                                          # + json_data['source_mapping'].split('/')[-1] + '')
    data = np.array(graph_data['embedding'])
    estimator = AgglomerativeClustering(n_clusters=12,
                                        linkage='ward')
    estimator.fit(data)
    labels = estimator.labels_

    cluster_centroids = extract_cluster_centroids(data, labels)
    cluster_files = output_cluster_files(labels, graph_data)
    with open('cluster_files.json', 'w') as f:
        json.dump(cluster_files, f)


if __name__ == '__main__':
   test_cluster('training_embedding', show_samples=20)
