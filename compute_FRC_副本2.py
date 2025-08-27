import networkx as nx
import numpy as np
import scipy.sparse as sp
from pathlib import Path


def load_sparse_graph(npz_path: Path) -> nx.Graph:
    """
    读取稀疏距离矩阵 D（对称，非负），构建无向图；
    边权 edge['weight'] = 距离。
    """
    data = np.load(npz_path)
    D = sp.csr_matrix(data["D"])
    G = nx.from_scipy_sparse_array(D)  # 无向
    for (u, v, w) in zip(*sp.find(D)):
        if u < v:
            G[u][v]["weight"] = float(w)
    return G


def weighted_frc_edge_curvature(G: nx.Graph) -> dict:
    """
    标准加权 FRC（节点权=1）：
    F(e) = w_uv * ( 2/w_uv
                    - sum_{x~u, x!=v} 1/sqrt(w_uv*w_ux)
                    - sum_{y~v, y!=u} 1/sqrt(w_uv*w_vy) )
    无权图可化为 4 - deg(u) - deg(v)。
    """
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
        F[(v, u)] = F_uv  # 无向复用
    return F


def node_weighted_mean_from_edges(G: nx.Graph, edge_value: dict) -> np.ndarray:
    """
    节点层：以“边距离权”做加权均值：
    F_node(v) = (sum_{e~v} w_e * F(e)) / (sum_{e~v} w_e)
    """
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
    out_dir = Path("./frc")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in graph_dir.glob("*.npz"):
        if "forman" in f.stem:   # 只跳过已有 *_forman.npz
            continue

        G = load_sparse_graph(f)

        # 1) 标准加权 FRC（节点权=1）
        F_edge = weighted_frc_edge_curvature(G)

        # 2) 节点层加权均值
        F_node = node_weighted_mean_from_edges(G, F_edge)

        # 3) 保存为旧命名：*_forman.npz，字段为 forman_edge/forman_node
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
        print(f"[✔] Saved standard FRC to: {out.name}")


if __name__ == "__main__":
    GRAPH_DIR = Path("./graphs")   # 你的 kNN 图目录
    compute_and_save_frc_standard(GRAPH_DIR)


