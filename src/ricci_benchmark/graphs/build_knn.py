from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def load_pointclouds(dirpath: Path):
    # Load all point clouds (npz) from directory
    clouds = {}
    for f in dirpath.glob("*.npz"):
        data = np.load(f)
        X = data["X"]
        clouds[f.stem] = X
    return clouds


def build_knn_graph(X: np.ndarray, k: int):
    # Build symmetric k-NN graph (sparse distance matrix)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    dist, ind = dist[:, 1:], ind[:, 1:]

    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = ind.ravel()
    dval = dist.ravel()

    D = sp.csr_matrix((dval, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    D = D.maximum(D.T)

    r = dist[:, -1]
    h = r.mean()
    return D, h, r


def save_sparse_npz(path: Path, **arrays):
    # Save sparse matrix and arrays to compressed npz
    np.savez_compressed(
        path, **{k: v if isinstance(v, np.ndarray) else v.A for k, v in arrays.items()}
    )


if __name__ == "__main__":
    PTS_DIR   = Path("data/points")
    GRAPH_DIR = Path("data/graphs")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    clouds = load_pointclouds(PTS_DIR)
    k = 10

    for stem, X in clouds.items():
        D, h, r = build_knn_graph(X, k)
        save_sparse_npz(GRAPH_DIR / f"{stem}.npz", D=D, h=np.array([h]), r=r)
