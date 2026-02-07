import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os

# ---------------- Distributed KMeans ---------------- #
def worker(data, centroids, K, queue):
    distances = np.linalg.norm(data[:, None] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    sums = np.zeros_like(centroids)
    counts = np.zeros(K)

    for k in range(K):
        points = data[labels == k] # Selecting only the points which are assigned to the cluster
        if len(points) > 0:
            sums[k] = points.sum(axis=0)
            counts[k] = len(points)

    queue.put((sums, counts, labels))


def distributed_kmeans(X, K, workers, iters=10):
    n = len(X)
    chunks = np.array_split(X, workers)
    centroids = X[np.random.choice(n, K, replace=False)]

    for _ in range(iters):
        queue = mp.Queue()
        procs = []

        for i in range(workers):
            p = mp.Process(target=worker, args=(chunks[i], centroids, K, queue))
            p.start()
            procs.append(p)

        total_sum = np.zeros_like(centroids)
        total_count = np.zeros(K)
        all_labels = []
 
        for _ in range(workers):
            s, c, labels = queue.get()
            total_sum += s
            total_count += c
            all_labels.append(labels)

        for k in range(K):
            if total_count[k] > 0:
                centroids[k] = total_sum[k] / total_count[k]

        for p in procs:
            p.join()

    return centroids, np.concatenate(all_labels)


# ---------------- Sequential KMeans ---------------- #
def sequential_kmeans(X, K, iters=10):
    n, d = X.shape
    centroids = X[np.random.choice(n, K, replace=False)]

    for _ in range(iters):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        for k in range(K):
            points = X[labels == k]
            if len(points) > 0:
                centroids[k] = points.mean(axis=0)

    return centroids, labels


# ---------------- Benchmark Runner ---------------- #
def run_benchmark():
    os.makedirs("plots", exist_ok=False)

    sizes = [1_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000, 60_000_000]
    seq_times = []
    dist_times = []

    K = 5
    workers = 4

    for n_samples in sizes:
        print(f"\nRunning for {n_samples:,} samples")

        X, _ = make_blobs(n_samples=n_samples, centers=K, n_features=2, cluster_std=5)

        # ---------------- Sequential ----------------
        start = time.time()
        seq_centroids, seq_labels = sequential_kmeans(X, K)
        seq_time = time.time() - start
        seq_times.append(seq_time)

        # ---- Sequential clustering plot ----
        plt.figure(figsize=(4, 4))
        plt.scatter(X[:, 0], X[:, 1], c=seq_labels, s=1, cmap="viridis", alpha=0.4)
        plt.scatter(seq_centroids[:, 0], seq_centroids[:, 1], c="red", marker="x", s=80)
        plt.title(f"Sequential KMeans ({n_samples//1_000_000}M Data Points)")
        plt.tight_layout()
        plt.savefig(f"plots/seq_kmeans_{n_samples//1_000_000}M.png")
        plt.close()

        # ---------------- Distributed ----------------
        start = time.time()
        dist_centroids, dist_labels = distributed_kmeans(X, K, workers)
        dist_time = time.time() - start
        dist_times.append(dist_time)

        # ---- Distributed clustering plot ----
        plt.figure(figsize=(4, 4))
        plt.scatter(X[:, 0], X[:, 1], c=dist_labels, s=1, cmap="viridis", alpha=0.4)
        plt.scatter(dist_centroids[:, 0], dist_centroids[:, 1], c="red", marker="x", s=80)
        plt.title(f"Distributed KMeans ({n_samples//1_000_000}M Data Points)")
        plt.tight_layout()
        plt.savefig(f"plots/dist_kmeans_{n_samples//1_000_000}M.png")
        plt.close()

        # ---------------- Time Comparison (per size) ----------------
        plt.figure(figsize=(4, 3))
        plt.bar(["Sequential", "Distributed"],[seq_time, dist_time])
        plt.ylabel("Time (in seconds)")
        plt.title(f"Time Comparison ({n_samples//1_000_000}M Data Points)")
        plt.tight_layout()
        plt.savefig(f"plots/time_compare_{n_samples//1_000_000}M.png")
        plt.close()

        print(f"Sequential: {seq_time:.2f}s | " f"Distributed: {dist_time:.2f}s")

    # ---------------- Final Comparison Plot ----------------
    plt.figure(figsize=(6, 4))
    plt.plot([s // 1_000_000 for s in sizes], seq_times, marker="x", label="Sequential")
    plt.plot([s // 1_000_000 for s in sizes], dist_times, marker="x", label="Distributed")
    plt.xlabel("Number of Samples (in Millions)")
    plt.ylabel("Time (in seconds)")
    plt.title("Overall Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/kmeans_comparison_all_sizes.png")

# ---------------- Main ---------------- #
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_benchmark()
