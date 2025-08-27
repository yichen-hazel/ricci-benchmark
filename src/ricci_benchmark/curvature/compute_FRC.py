import networkx as nx
import numpy as np
import scipy.sparse as sp
from pathlib import Path


def load_sparse_graph(npz_path: Path) -> nx.Graph:
    # Load sparse distance matrix and build weighted undirected graph
    data = np.load(npz_path)
    D = sp.csr_matrix(data["D"])
    G = nx.from_scipy_sparse_array(D)
    for (u, v, w) in zip(*sp.find(D)):
        if u < v:
            G[u][v]["weight"] = float(w)
    return G


def weighted_frc_edge_curvature(G: nx.Graph) -> dict:
    # Compute weighted Forman–Ricci curvature on edges
    eps = 1e-12
    F = {}
    for u, v, d in G.edges(data=True):
        w_uv = max(d.get("weight", 1.0), eps)

        term_vertex = 2.0 / w_uv

        sum_u = 0.0
        for x in G.neighbors(u):
            if x == v:
                continue
            w_ux = max(G[u][x].get("weight", 1.0), eps)
            sum_u += 1.0 / np.sqrt(w_uv * w_ux)

        sum_v = 0.0
        for y in G.neighbors(v):
            if y == u:
                continue
            w_vy = max(G[v][y].get("weight", 1.0), eps)
            sum_v += 1.0 / np.sqrt(w_uv * w_vy)

        F_uv = w_uv * (term_vertex - (sum_u + sum_v))
        F[(u, v)] = F_uv
        F[(v, u)] = F_uv
    return F


def node_weighted_mean_from_edges(G: nx.Graph, edge_value: dict) -> np.ndarray:
    # Aggregate edge curvature to nodes (weighted mean by edge weight)
    n = G.number_of_nodes()
    node_val = np.zeros(n, dtype=float)
    w_sum = np.zeros(n, dtype=float)

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        val = edge_value[(u, v)]
        node_val[u] += w * val
        node_val[v] += w * val
        w_sum[u] += w
        w_sum[v] += w

    with np.errstate(divide="ignore", invalid="ignore"):
        node_val = np.divide(node_val, w_sum, out=np.zeros_like(node_val), where=w_sum > 0)
    return node_val


def compute_and_save_frc_standard(graph_dir: Path):
    # Compute standard FRC for all graphs and save results
    out_dir = Path("data/frc")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in graph_dir.glob("*.npz"):
        if "forman" in f.stem:
            continue

        G = load_sparse_graph(f)
        F_edge = weighted_frc_edge_curvature(G)
        F_node = node_weighted_mean_from_edges(G, F_edge)

        edges = list(G.edges())
        edge_u = np.array([u for u, v in edges], dtype=int)
        edge_v = np.array([v for u, v in edges], dtype=int)
        w_edge = np.array([G[u][v].get("weight", 1.0) for u, v in edges], dtype=float)
        F_edge_arr = np.array([F_edge[(u, v)] for u, v in edges], dtype=float)

        out = out_dir / f"{f.stem}_forman.npz"
        np.savez_compressed(
            out,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_weight=w_edge,
            forman_edge=F_edge_arr,
            forman_node=F_node,
        )


if __name__ == "__main__":
    GRAPH_DIR = Path("data/graphs")
    compute_and_save_frc_standard(GRAPH_DIR)

