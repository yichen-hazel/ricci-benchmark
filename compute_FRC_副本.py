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

    # NetworkX ≥ 3.0：from_scipy_sparse_array 会把矩阵条目放在 edge 'weight'
    G = nx.from_scipy_sparse_array(D)  # 无向
    # 明确写一遍，确保属性存在且只保留一次（u<v）
    for (u, v, w) in zip(*sp.find(D)):
        if u < v:
            G[u][v]["weight"] = float(w)
    return G


# --------------------------
# 核心：加权 FRC（节点权 = 1）
# --------------------------
def weighted_frc_edge_curvature(G: nx.Graph) -> dict:
    """
    F(e) = w_uv * ( (1/w_uv + 1/w_uv)
                    - sum_{x~u, x!=v} 1/sqrt(w_uv*w_ux)
                    - sum_{y~v, y!=u} 1/sqrt(w_uv*w_vy) )
    即文中公式 (1)（节点权取 1），无权时可化为 4 - deg(u) - deg(v)。
    """
    F = {}
    for u, v, d in G.edges(data=True):
        w_uv = d.get("weight", 1.0)
        # 顶点项（w_u = w_v = 1）
        term_vertex = 2.0 / w_uv

        # u 端邻边项（除去 (u,v) 本身）
        sum_u = 0.0
        for x in G.neighbors(u):
            if x == v:
                continue
            w_ux = G[u][x].get("weight", 1.0)
            sum_u += 1.0 / np.sqrt(w_uv * w_ux)

        # v 端邻边项
        sum_v = 0.0
        for y in G.neighbors(v):
            if y == u:
                continue
            w_vy = G[v][y].get("weight", 1.0)
            sum_v += 1.0 / np.sqrt(w_uv * w_vy)

        F_uv = w_uv * (term_vertex - (sum_u + sum_v))
        F[(u, v)] = F_uv
        F[(v, u)] = F_uv  # 无向复用
        # 可选：写回属性
        G[u][v]["FRC_weighted"] = float(F_uv)
    return F


# ----------------------------------------
# 三角形-AFRC：固定增益 +3 * T_e（推荐主结果）
# ----------------------------------------
def afrc_triangle_fixed(G: nx.Graph, F_edge: dict) -> dict:
    """
    F_triangle(e) = F(e) + 3 * T_e
    其中 T_e = |N(u) ∩ N(v)|
    """
    F_tri = {}
    for u, v in G.edges():
        # 统计三角形个数（公共邻居）
        t = len(set(G.neighbors(u)).intersection(G.neighbors(v)))
        F_uv = F_edge[(u, v)] + 3.0 * t
        F_tri[(u, v)] = F_uv
        F_tri[(v, u)] = F_uv
        # 写回属性（便于可视化/调试）
        G[u][v]["AFRC_triangle"] = float(F_uv)
        G[u][v]["triangles_on_edge"] = int(t)
    return F_tri


# ----------------------------------------
# 节点层：以“边距离权”做加权均值
# ----------------------------------------
def node_weighted_mean_from_edges(G: nx.Graph, edge_value: dict, key_name: str) -> np.ndarray:
    """
    对每个节点 v：  sum_{e~v} w_e * value(e) / sum_{e~v} w_e
    其中 w_e = edge['weight']（距离）。
    """
    n = G.number_of_nodes()
    node_val = np.zeros(n, dtype=float)
    w_sum = np.zeros(n, dtype=float)

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        val = edge_value[(u, v)]  # 无向时 (u,v) 已复用
        node_val[u] += w * val
        node_val[v] += w * val
        w_sum[u] += w
        w_sum[v] += w

    # 加权均值
    with np.errstate(divide="ignore", invalid="ignore"):
        node_val = np.divide(node_val, w_sum, out=np.zeros_like(node_val), where=w_sum > 0)

    # 可选：写回每个节点的属性
    for n_id in G.nodes():
        G.nodes[n_id][key_name] = float(node_val[n_id])

    return node_val


def compute_and_save_afrc_triangle(graph_dir: Path):
    out_dir = Path("./frc")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in graph_dir.glob("*.npz"):
        if "forman" in f.stem:   # 只跳过已有 *_forman.npz
            continue

        G = load_sparse_graph(f)

        # 1) 加权 FRC
        F_edge = weighted_frc_edge_curvature(G)

        # 2) 三角形-AFRC（固定 +3）
        F_tri_edge = afrc_triangle_fixed(G, F_edge)

        # 3) 节点层（以边距离权加权均值）
        Ftri_node = node_weighted_mean_from_edges(
            G, F_tri_edge, key_name="AFRC_triangle_node"
        )

        # 打包保存 —— 覆盖旧命名：*_forman.npz
        edges = list(G.edges())
        edge_u = np.array([u for u, v in edges], dtype=int)
        edge_v = np.array([v for u, v in edges], dtype=int)
        w_edge = np.array([G[u][v].get("weight", 1.0) for u, v in edges], dtype=float)
        Ftri_edge_arr = np.array([F_tri_edge[(u, v)] for u, v in edges], dtype=float)

        out = out_dir / f"{f.stem}_forman.npz"
        np.savez_compressed(
            out,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_weight=w_edge,
            forman_edge=Ftri_edge_arr,   # 直接覆盖为 AFRC 边结果
            forman_node=Ftri_node,       # 直接覆盖为 AFRC 节点结果
        )
        print(f"[✔] Saved AFRC(triangle) to: {out.name}")



if __name__ == "__main__":
    GRAPH_DIR = Path("./graphs")   # 你的 kNN 图目录
    compute_and_save_afrc_triangle(GRAPH_DIR)

