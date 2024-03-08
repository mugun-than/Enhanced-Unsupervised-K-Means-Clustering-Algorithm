import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

iteration = 1

# Importing iris dataset
def initilaise():
    players = pd.read_csv(".././Dataset/iris.csv")
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    players = players.dropna(subset=features)
    dataset = players[features].copy()

    dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1
    return dataset

# Intialising centroids
def random_centroids(dataset, k):
    centroids = []
    for i in range(k):
        centroid = dataset.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

# Clustering the data points with initialised centroids
def get_labels(dataset, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((dataset - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

# Updating centroids
def new_centroids(dataset, labels):
    centroids = dataset.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T
    return centroids

# Plotting clusters
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
def kmeans(dataset, centroid_count):
    global iteration
    max_iterations = 100
    centroids = random_centroids(dataset, centroid_count)
    old_centroids = pd.DataFrame()

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids

        labels = get_labels(dataset, centroids)
        centroids = new_centroids(dataset, labels)
        plot_clusters(dataset, labels, centroids, iteration)
        iteration += 1

    inertia = 0
    for centroid in centroids.columns:
        cluster_points = dataset[labels == centroid]
        inertia += np.sum((cluster_points - centroids[centroid]) ** 2)

    return inertia.sum()

# Finding optimal number of clusters (k)
def start_clustering(data):
    wcss = []
    for i in range(1, 11):
        inertia = kmeans(data, i)
        wcss.append(inertia)
    return wcss

# Terminating the process
def end_clustering():
    plt.ioff()
    time.sleep(2)
    plt.close()

# Plotting elbow method graph
def plot_elbow(WCSS):
    plt.plot(range(1, len(WCSS) + 1), WCSS)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    end_clustering()

# Main method
data = initilaise()
plot_elbow(start_clustering(data))
