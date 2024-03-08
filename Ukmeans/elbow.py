import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
""" Load either one of the dataset from the two given below
# Load the Iris dataset
customers = pd.read_csv(".././Dataset/Wholesale_customers.csv")
features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
customers = customers.dropna(subset=features)
dataset = customers[features].copy()
dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1


flowers = pd.read_csv(".././Dataset/iris.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
flowers = flowers.dropna(subset=features)
dataset = flowers[features].copy()
dataset = ((dataset - dataset.min()) / (dataset.max() - dataset.min())) * 9 + 1
"""

# Traditional elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Enhanced elbow method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
    if i > 2:
        x = wcss[0] - wcss[1]
        u = wcss[i - 2] - wcss[i - 1]
        if u < x / 2.5:
            break

plt.plot(range(1, len(wcss)+1), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
print(wcss[len(wcss) - 1])
