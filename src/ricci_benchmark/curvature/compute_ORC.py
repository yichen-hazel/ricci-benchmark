from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import numpy as np
import scipy.sparse as sp
from pathlib import Path


def load_sparse_graph(npz_path: Path) -> nx.Graph:
    # Load sparse distance matrix and build weighted graph
    data = np.load(npz_path)
    D = sp.csr_matrix(data["D"])
    G = nx.from_scipy_sparse_array(D)
    return G


def compute_and_save_orc(graph_dir: Path, alpha=0.5, method="OTD"):
    # Compute Ollivier-Ricci curvature on graphs and save edge/node results
    orc_dir = Path("data/orc")
    orc_dir.mkdir(parents=True, exist_ok=True)

    for f in graph_dir.glob("*.npz"):
        if "ricci" in f.stem:
            continue

        G = load_sparse_graph(f)
        orc = OllivierRicci(G, alpha=alpha, method=method, verbose="ERROR")
        ok = orc.compute_ricci_curvature()

        if not ok or any("ricciCurvature" not in d for _, _, d in orc.G.edges(data=True)):
            continue

        # Collect edge-level curvature
        edge_u, edge_v, k_edge = [], [], []
        for u, v, d in orc.G.edges(data=True):
            edge_u.append(u)
            edge_v.append(v)
            k_edge.append(d["ricciCurvature"])
        edge_u = np.array(edge_u, dtype=int)
        edge_v = np.array(edge_v, dtype=int)
        k_edge = np.array(k_edge)

        # Aggregate to node-level curvature (weighted mean by 1/length)
        k_node = np.zeros(G.number_of_nodes())
        weight_sum = np.zeros_like(k_node)
        for u, v, d in orc.G.edges(data=True):
            k = d.get("ricciCurvature", None)
            length = d.get("weight", 1.0)
            w = 1.0 / length
            if k is None or np.isnan(k):
                continue
            k_node[u] += w * k
            k_node[v] += w * k
            weight_sum[u] += w
            weight_sum[v] += w
        k_node = np.divide(k_node, weight_sum, out=np.zeros_like(k_node), where=weight_sum > 0)

        # Save edge and node curvature
        out_path = orc_dir / f"{f.stem}_orc.npz"
        np.savez_compressed(
            out_path,
            edge_u=edge_u,
            edge_v=edge_v,
            ricci_edge=k_edge,
            ricci_node=k_node,
        )


if __name__ == "__main__":
    GRAPH_DIR = Path("data/graphs")
    compute_and_save_orc(GRAPH_DIR, alpha=0.5, method="OTD")

