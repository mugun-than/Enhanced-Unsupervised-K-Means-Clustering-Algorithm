# Average running time - 2.0585486

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
start_time = time.time()
centroid_count = 3


# Importing the iris dataset
def initilaise():
    players = pd.read_csv(".././Dataset/iris.csv")
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    players = players.dropna(subset=features)
    dataset = players[features].copy()

    dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1
    #plt.ion()
    return dataset

# Initialising centroids
def random_centroids(dataset, k):
    centroids = []
    for i in range(k):
        centroid = dataset.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

# Clustering the datapoints with initialised centroids
def get_labels(dataset, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((dataset - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

# Updating centroids
def new_centroids(dataset, labels):
    centroids = dataset.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

# Plotting the clusters
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

# Traditional K-Means clustering algorithm
def kmeans(dataset):
    max_iterations = 100
    centroids = random_centroids(dataset, centroid_count)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids

        labels = get_labels(dataset, centroids)
        centroids = new_centroids(dataset, labels)
        #plot_clusters(dataset, labels, centroids, iteration)
        iteration += 1

# Terminating the clustering process
def end_clustering():
    plt.ioff()
    time.sleep(2)
    plt.close()
    print(f"--- {time.time() - start_time:.6f} seconds ---")

# Main method
data = initilaise()
kmeans(data)
end_clustering()

