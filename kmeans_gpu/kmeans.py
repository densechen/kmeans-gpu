# MIT License

# Copyright (c) 2021 densechen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeans(nn.Module):
    r"""KMeans module with PyTorch support.

    Args:
        n_clusters: Number of clusters.
        max_iter: Maximum number of iterations.
        tolerance: Tolerance for error/distance.

        distance: `euclidean` or `cosine`.
        sub_sampling: The number of points used in KMeans.
            If None, use all points to do KMeans.
        max_neighbors: The number of neighbors to use for aggregating features.
    """

    def __init__(self,
                 n_clusters: int,
                 max_iter: int = 100,
                 tolerance: float = 1e-4,
                 distance: str = 'euclidean',
                 sub_sampling: int = None,
                 max_neighbors: int = 15):
        super().__init__()
        assert distance in ['euclidean', 'cosine']
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.distance = distance
        self.sub_sampling = sub_sampling
        self.max_neighbors = max_neighbors

    @classmethod
    def cos_sim(cls, vec_a, vec_b):
        """Compute Cosine Similarity between vec_a and vec_b.
        Args:
            vec_a: m x d
            vec_b: n x d

        Returns:
            m x n
        """
        vec_a = vec_a.unsqueeze(1).expand(vec_a.shape[0], vec_b.shape[0], -1)
        vec_b = vec_b.unsqueeze(0).expand_as(vec_a)
        return F.cosine_similarity(vec_a, vec_b, dim=-1)

    @classmethod
    def euc_sim(cls, vec_a, vec_b):
        r"""Compute Euclidean Distance between vec_a and vec_b.
        Args:
            vec_a: m x d
            vec_b: n x d

        Returns:
            m x n
        """
        # (vec_a - vec_b)^2 = vec_a^2 + vec_b.T^2 - 2 vec_a @ vec_b.T
        return 2 * vec_a @ vec_b.T - (vec_a**2).sum(dim=1, keepdim=True) - (vec_b.T**2).sum(dim=0, keepdim=True)

    @classmethod
    def max_sim(cls, vec_a, vec_b, distance):
        """Compute maximum similarity (or minimum distance) of each vector in vec_a with all of the vectors in vec_b.

        Args:
            vec_a: m x d
            vec_b: n x d
        Returns:
            [value, indices]: m
        """
        sim_score = KMeans.cos_sim(
            vec_a, vec_b) if distance == "cosine" else KMeans.euc_sim(vec_a, vec_b)
        return sim_score.max(dim=-1)

    @classmethod
    def predict(cls, X, centroids, distance):
        """Predict the closest cluster each sample in X belongs to.
        Args:
            X: n x d
            centroids: m x d
            distance:

        Returns:
            labels: n
        """
        return cls.max_sim(vec_a=X, vec_b=centroids, distance=distance)[1]

    @torch.no_grad()
    def fit_predict(self, X, centroids=None):
        """Combination of fit() and predict() methods.
        Args:
            X: torch.Tensor, shape: [n_samples, n_features]
            centroids: {torch.Tensor, None}, default: None
                If given, centroids will be initialized with given tensor
                If None, centroids will be randomly chosen from X
            Return:
                labels: n_samples
                centroids: n_samples x 3
        """
        pts, _ = X.shape
        device = X.device
        if centroids is None:
            centroids = X[np.random.choice(
                pts, size=[self.n_clusters], replace=False)]

        num_points_in_clusters = torch.ones(self.n_clusters, device=device)
        for _ in range(self.max_iter):
            # 1. Data propare
            if not self.sub_sampling:
                x = X
            else:
                # Sampling a subset to speedup KMeans
                x = X[np.random.choice(
                    pts, size=[self.sub_sampling], replace=False)]

            # 2. Similarity
            closest = KMeans.max_sim(
                vec_a=x, vec_b=centroids, distance=self.distance)[1]

            matched_clusters, counts = closest.unique(return_counts=True)

            c_grad = torch.zeros_like(centroids)
            matched_clusters_ = torch.arange(
                self.n_clusters, device=device) if not self.sub_sampling else matched_clusters
            expanded_closest = closest.unsqueeze(
                0).expand(len(matched_clusters_), -1)
            mask = (expanded_closest == matched_clusters_[:, None]).float()
            c_grad[matched_clusters_] = mask @ x / \
                (mask.sum(-1, keepdim=True) + 1e-8)

            error = (c_grad - centroids).pow(2).sum()
            lr = (
                0.9 / (num_points_in_clusters[:, None] + 1e-8) + 0.1) if self.sub_sampling else 1

            num_points_in_clusters[matched_clusters] += counts

            centroids = centroids * (1 - lr) + c_grad * lr
            if error <= self.tolerance:
                break
        if self.sub_sampling:
            closest = KMeans.predict(X, centroids, distance=self.distance)
        return closest, centroids

    def forward(self, points, features=None, centroids=None):
        r"""KMeans on points and then do an average aggregation on neighborhood points to get the feature for each cluster.
        Args:
            points: bz x n x 3
            features: bz x f x n, if features is given, we will aggregate the feature at the same time. 
            centroids: bz x m x 3, the initial centroids points.

        Returns:
            cluster centroids: bz x cc x 3
            cluster features: bz x f x cc
        """
        def single_batch(pts, ft, ct):
            closest, centroids = self.fit_predict(pts, ct)

            if ft is not None:
                cluster_features = []
                for cls in range(self.n_clusters):
                    cf = ft[:, closest == cls]
                    cp = pts[closest == cls]

                    # Compute distance to center points
                    sim_score = KMeans.cos_sim(
                        cp, centroids[cls:cls+1]) if self.distance == "cosine" else KMeans.euc_sim(cp, centroids[cls:cls+1])
                    sim_score = sim_score.reshape(-1)
                    score, index = torch.topk(sim_score, k=min(
                        self.max_neighbors, len(cp)), largest=True)

                    # Select features
                    cf = cf[:, index]
                    score = F.softmax(score, dim=0).reshape(1, -1)
                    cluster_features.append(
                        torch.sum(cf * score, dim=1, keepdim=True))
                cluster_features = torch.cat(cluster_features, dim=1)

                return centroids, cluster_features
            else:
                return centroids, None

        features = features if features is not None else [
            None for _ in range(len(points))]
        r_points, r_features = [], []
        for i, (pts, ft) in enumerate(zip(points, features)):
            if centroids is not None:
                ct = centroids[i]
            else:
                ct = None
            pts, ft = single_batch(pts, ft, ct)
            r_points.append(pts)
            r_features.append(ft)
        if features[0] is not None:
            return torch.stack(r_points, dim=0), torch.stack(r_features, dim=0)
        else:
            return torch.stack(r_points, dim=0)


if __name__ == '__main__':
    # bz x fd x n
    features = torch.randn(128, 512, 256)
    # bz x n x 3
    points = torch.randn(128, 256, 3)

    kmeans = KMeans(
        n_clusters=15,
        max_iter=100,
        tolerance=1e-4,
        distance='euclidean',
        sub_sampling=None,
        max_neighbors=15
    )

    centroids, features = kmeans(points, features)

    print(centroids.shape, features.shape)
