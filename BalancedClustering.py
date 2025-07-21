
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    """Randomly initialize k centroids from the data points."""
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]

def assign_clusters(data, centroids, max_capacity):
    """Assign data points to the nearest centroid while respecting max capacity."""
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    cluster_assignments = np.full(data.shape[0], -1, dtype=int) # -1 indicates unassigned
    cluster_sizes = np.zeros(centroids.shape[0], dtype=int)
    
    for i in range(data.shape[0]):
        for idx in np.argsort(distances[:, i]):
            if cluster_sizes[idx] < max_capacity:
                cluster_assignments[i] = idx
                cluster_sizes[idx] += 1
                break
    return cluster_assignments

def update_centroids(data, assignments, k):
    """Recalculate centroids as the mean of assigned data points."""
    centroids = np.array([data[assignments == i].mean(axis=0) for i in range(k)])
    return centroids

def k_means_with_capacity(data, k, max_capacity, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        assignments = assign_clusters(data, centroids, max_capacity)
        new_centroids = update_centroids(data, assignments, k)
        
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return assignments, centroids