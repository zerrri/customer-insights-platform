from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal",
    2: "At Risk",
    3: "Hibernating",
}

def segment_rfm(df: pd.DataFrame, k: int = 4, random_state: int = 42) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    work = df.copy()
    features = work[["Recency", "Frequency", "Monetary"]].fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    clusters = km.fit_predict(X)

    work["SegmentID"] = clusters
    # Map informative names by rough heuristics:
    # Lower Recency + higher Frequency/Monetary => better segment.
    # We’ll reorder labels by centroid quality rank.
    centroids = pd.DataFrame(km.cluster_centers_, columns=["R", "F", "M"])
    centroids["score"] = (-centroids["R"]) + centroids["F"] + centroids["M"]
    order = centroids["score"].rank(ascending=False, method="first").astype(int) - 1
    id_to_rank = dict(zip(centroids.index, order))

    inv_map = {rank: cid for cid, rank in id_to_rank.items()}
    label_map = {}
    for rank in range(k):
        cid = inv_map.get(rank, rank)
        label_map[cid] = SEGMENT_LABELS.get(rank, f"Segment-{rank}")

    work["Segment"] = work["SegmentID"].map(label_map)
    return work, scaler, km
