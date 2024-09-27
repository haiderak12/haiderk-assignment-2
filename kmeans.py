import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=3, init_method="random", max_iters=100):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iters = max_iters
        self.centroids = None

    def initialize_centroids(self, data):
        if self.init_method == "random":
            self.centroids = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]
        elif self.init_method == "farthest_first":
            self.centroids = [data[np.random.randint(data.shape[0])]]
            for _ in range(1, self.n_clusters):
                dist_sq = np.min([np.sum((data - centroid) ** 2, axis=1) for centroid in self.centroids], axis=0)
                next_centroid = data[np.argmax(dist_sq)]
                self.centroids.append(next_centroid)
            self.centroids = np.array(self.centroids)
        # Add k-means++ and manual later based on the UI

    def assign_clusters(self, data):
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def update_centroids(self, data, labels):
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, data):
        self.initialize_centroids(data)
        for i in range(self.max_iters):
            labels = self.assign_clusters(data)
            new_centroids = self.update_centroids(data, labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return self.centroids, labels
