# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
# y = dataset.iloc[:, 3].values

# using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
     kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
     kmeans.fit(X)
     wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.Xlable('no. of clusters')
plt.ylable('WCSS')
plt.show()     

# applying kmeans to the mall dataset
kmeas = KMeans(n_clusters = 5, init = 'k-means++' , n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)