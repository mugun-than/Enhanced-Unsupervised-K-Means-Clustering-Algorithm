import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

centroid_count = 3
inter_cluster_distance = 0
intra_cluster_distance = 0


def initilaize():
    players = pd.read_csv(".././Dataset/players_22.csv")
    features = ["overall", "potential", "wage_eur", "value_eur", "age"]
    players = players.dropna(subset=features)
    dataset = players[features].copy()

    dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 10 + 1
    plt.ion()
    return dataset


def random_centroids(dataset, k):
    centroids = []
    for i in range(k):
        centroid = dataset.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)


def get_labels(dataset, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((dataset - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)


def new_centroids(dataset, labels):
    centroids = dataset.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids


def plot_clusters(dataset, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(dataset)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:, 0], y=data_2d[:, 1], c=labels)
    plt.scatter(x=centroids_2d[:, 0], y=centroids_2d[:, 1])
    plt.pause(0.1)
    plt.clf()


def kmeans(dataset):
    max_iterations = 100
    centroids = random_centroids(dataset, centroid_count)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids

        labels = get_labels(dataset, centroids)
        centroids = new_centroids(dataset, labels)
        plot_clusters(dataset, labels, centroids, iteration)
        iteration += 1


def end_clustering():
    plt.ioff()
    time.sleep(2)
    plt.close()


data = initilaize()
kmeans(data)
end_clustering()