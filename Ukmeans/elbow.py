import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#Load either one of the dataset from the two given below
# Load the Iris dataset

"""
flowers = pd.read_csv(".././Dataset/iris.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
flowers = flowers.dropna(subset=features)
dataset = flowers[features].copy()
dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1

wine = pd.read_csv(".././Dataset/WineQT.csv")
features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
wine = wine.dropna(subset=features)
dataset = wine[features].copy()
dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1
"""
glass = pd.read_csv(".././Dataset/glass.csv")
features = ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]
glass = glass.dropna(subset=features)
dataset = glass[features].copy()
dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1

# Traditional elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Glass')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
