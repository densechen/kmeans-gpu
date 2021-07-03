# kmeans-gpu

kmeans-gpu with pytorch (batch version). It is faster than sklearn.cluster.KMeans.

You can easily use `KMeans` as a `nn.Module`, and embed into your network structure.

## Install

1. From Git:

```bash
git clone git@github.com:densechen/kmeans-gpu.git
cd kmeans-gpu
pip install -r requirements.txt
python setup.py install

# check installation
python -c "import kmeans_gpu; print(kmeans_gpu.__version__)"
```

2. From PyPI:

```bash
pip install kmeans-gpu

# check installation
python -c "import kmeans_gpu; print(kmeans_gpu.__version__)"
```

## Demo

```python
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
# Pay attention to the different dimension order between features and points.
points = torch.randn(batch_size, num_pts, pts_dim)

# Create KMeans Module
kmeans = KMeans(
    n_clusters=num_cluster,
    max_iter=100,
    tolerance=1e-4,
    distance='euclidean',
    sub_sampling=None,
    max_neighbors=15,
)

# Forward
centroids, features = kmeans(points, features)

print(centroids.shape, features.shape)
# output: 
# >>> torch.Size([128, 15, 3]) torch.Size([128, 1024, 15])
```