from kmeans_gpu import KMeans
import torch

# Config
batch_size = 128
feature_dim = 1024
pts_dim = 3
num_pts = 256
num_cluster = 15

# Create data
features = torch.randn(batch_size, feature_dim, num_pts)
points = torch.randn(batch_size, num_pts, pts_dim)

# Create KMeans Module
kmeans = KMeans(
    n_clusters=num_cluster,
    max_iter=10,
    tolerance=1e-4,
    distance='cosine',
    sub_sampling=128,
    max_neighbors=15,
)

# Forward
centroids, features = kmeans(points, features, centroids=torch.randn(batch_size, num_cluster, pts_dim))

print(centroids.shape, features.shape)