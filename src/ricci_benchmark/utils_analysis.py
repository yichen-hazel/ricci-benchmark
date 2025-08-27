
from __future__ import annotations
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==== Enrichment Ratio ====

def select_topk_from_df(df: pd.DataFrame, manifold: str, proxy: str = "orc", k: int = 10) -> pd.DataFrame:
    """Select top-k (n, noise) settings by mean Spearman for given proxy."""
    proxy = str(proxy).lower()
    if proxy == "orc":
        metric_col = "orc_s"
        out_col = "mean_orc_s"
    elif proxy == "frc":
        metric_col = "frc_s"
        out_col = "mean_frc_s"
    else:
        raise ValueError("proxy must be 'orc' or 'frc'.")

    df_mean = (
        df[df["manifold"] == manifold]
        .groupby(["manifold", "n", "noise"], as_index=False)
        .agg({metric_col: "mean"})
        .rename(columns={metric_col: out_col})
    )
    return (
        df_mean.sort_values(out_col, ascending=False)
               .head(k)[["manifold", "n", "noise"]]
               .reset_index(drop=True)
    )


def compute_er_pos_simple(
    manifold: str,
    n: int,
    noise: float,
    rep: int,
    *,
    proxy: str = "orc",
    q_tail: float = 5.0,
    alpha: float = 0.30,
    load_points,
    load_orc,
    load_frc
) -> dict:
    """Compute ER for high/low curvature regions for one run."""
    tag = f"{manifold}_n{n}_noise{noise:.3f}_rep{rep}"
    _, R_true = load_points(tag)

    if proxy.lower() == "orc":
        proxyv, _, _, _ = load_orc(tag)
    elif proxy.lower() == "frc":
        proxyv, _, _, _ = load_frc(tag)
        proxyv = -proxyv
    else:
        raise ValueError("proxy must be 'orc' or 'frc'.")

    if R_true is None or len(R_true) != len(proxyv):
        raise FileNotFoundError(f"Missing or mismatched R_true for {tag}")

    hi_thr = np.quantile(R_true, 1 - alpha)
    lo_thr = np.quantile(R_true, alpha)
    high_region = R_true >= hi_thr
    low_region  = R_true <= lo_thr
    p_high = high_region.mean()
    p_low  = low_region.mean()

    lo = np.percentile(proxyv, q_tail)
    hi = np.percentile(proxyv, 100 - q_tail)
    lower = proxyv < lo
    upper = proxyv > hi
    lower_cnt = int(lower.sum())
    upper_cnt = int(upper.sum())

    er_high_upper = ((upper & high_region).sum() / upper_cnt) / p_high if (upper_cnt and p_high) else np.nan
    er_low_lower  = ((lower & low_region).sum() / lower_cnt) / p_low  if (lower_cnt and p_low)  else np.nan
    return {"ER_high_upper": er_high_upper, "ER_low_lower": er_low_lower}


def run_manifold_for_proxy(
    df: pd.DataFrame,
    *,
    manifold: str = "saddle",
    proxy: str = "orc",
    k: int = 10,
    reps = range(5),
    q_tail: float = 10.0,
    alpha: float = 0.30,
    load_points=None,
    load_orc=None,
    load_frc=None
) -> pd.DataFrame:
    """Run ER computation across top-k settings and reps for one proxy."""
    if (load_points is None) or (load_orc is None) or (load_frc is None):
        raise ValueError("Must provide loaders.")

    rows = []
    topk = select_topk_from_df(df, manifold=manifold, proxy=proxy, k=k)
    for _, r in topk.iterrows():
        m, n, noise = r.get("manifold", manifold), int(r["n"]), float(r["noise"])
        for rep in reps:
            try:
                met = compute_er_pos_simple(
                    m, n, noise, rep,
                    proxy=proxy, q_tail=q_tail, alpha=alpha,
                    load_points=load_points, load_orc=load_orc, load_frc=load_frc
                )
                rows.append({"manifold": m, "n": n, "noise": noise, "proxy": proxy, **met})
            except FileNotFoundError:
                continue

    if not rows:
        return pd.DataFrame(columns=[
            "manifold","n","noise","proxy","ER_high_upper_mean","ER_low_lower_mean"
        ])

    g = (
        pd.DataFrame(rows)
          .groupby(["manifold","n","noise","proxy"], as_index=False)[["ER_high_upper","ER_low_lower"]]
          .mean()
          .rename(columns={"ER_high_upper":"ER_high_upper_mean",
                           "ER_low_lower":"ER_low_lower_mean"})
    )
    return g


def show_er_table(
    df: pd.DataFrame, *,
    manifold: str,
    proxy: str,
    k: int = 10,
    reps = range(5),
    q_tail: float = 10.0,
    alpha: float = 0.30,
    load_points=None,
    load_orc=None,
    load_frc=None,
    title: str | None = None
) -> None:
    """Print ER summary table for a given manifold/proxy."""
    tbl = run_manifold_for_proxy(
        df, manifold=manifold, proxy=proxy, k=k, reps=reps,
        q_tail=q_tail, alpha=alpha,
        load_points=load_points, load_orc=load_orc, load_frc=load_frc
    )
    cols = ["manifold","n","noise","proxy","ER_high_upper_mean","ER_low_lower_mean"]
    hdr  = title or f"{manifold.capitalize()} — {proxy.upper()} (ER)"
    print(f"=== {hdr} ===")
    if tbl.empty:
        print("(empty)")
    else:
        for c in cols:
            if c not in tbl.columns:
                tbl[c] = np.nan
        print(tbl[cols].to_string(index=False))

# ==== Outlier consensus utilities ====


YELLOW = "#FDD835"
ORANGE = "#FB8C00"
RED    = "#E53935"

def _load_mesh_center_unit(mesh_path, preprocess_mesh=None):
    mesh = trimesh.load(mesh_path, process=False)
    if preprocess_mesh is not None:
        mesh = preprocess_mesh(mesh)
    else:
        mesh = mesh.copy()
        mesh.apply_translation(-mesh.bounding_box.centroid)
        mesh.apply_scale(1.0 / mesh.bounding_box.extents.max())
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)
    return V, F

def compute_outlier_consensus_core(
    manifold: str, n: int, noise: float, reps,
    mesh_path,
    *, proxy: str = "orc", tail: str = "upper", q: float = 5.0,
    load_points=None, load_orc=None, load_frc=None, preprocess_mesh=None,
):
    if any(fn is None for fn in (load_points, load_orc, load_frc)):
        raise ValueError("Need load_points, load_orc, load_frc.")

    V, F = _load_mesh_center_unit(mesh_path, preprocess_mesh=preprocess_mesh)
    tree = cKDTree(V)
    votes = np.zeros(V.shape[0], dtype=float)

    for rep in reps:
        tag = f"{manifold}_n{n}_noise{noise:.3f}_rep{rep}"
        try:
            if proxy.lower() == "frc":
                proxyv, *_ = load_frc(tag); proxyv = -proxyv
            else:
                proxyv, *_ = load_orc(tag)
            X, _ = load_points(tag)
        except Exception:
            continue
        if proxyv.size == 0:
            continue

        finite = np.isfinite(proxyv)
        if not finite.any(): 
            continue
        pv, P = proxyv[finite], X[finite]

        if tail == "upper":
            thr = np.nanpercentile(pv, 100 - q); mask = pv >= thr
        else:
            thr = np.nanpercentile(pv, q);      mask = pv <= thr

        if mask.sum() == 0:
            k = max(1, int(np.ceil(q/100.0 * pv.size)))
            idx = (np.argpartition(pv, -k)[-k:] if tail=="upper" else np.argpartition(pv, k)[:k])
            mask = np.zeros_like(pv, dtype=bool); mask[idx] = True

        _, nn_idx = tree.query(P[mask])
        np.add.at(votes, nn_idx, 1.0)

    R_eff = max(1, len(list(reps)))
    presence_rate = votes / R_eff
    return V, F, presence_rate

def infer_R_from_presence(presence_rate: np.ndarray) -> int:
    nz = presence_rate[np.isfinite(presence_rate) & (presence_rate > 0)]
    if nz.size == 0: return 1
    vals = np.unique(np.round(nz, 6))
    if vals.size == 1: return int(round(1.0/vals[0]))
    diffs = np.diff(vals); step = diffs[diffs > 1e-9].min() if np.any(diffs > 1e-9) else vals.min()
    R = int(round(1.0 / max(step, 1e-9)))
    return max(1, min(R, 1000))

def _add_panel(fig, V, F, hits, R, col, marker_size):
    fig.add_trace(
        go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=F[:,0], j=F[:,1], k=F[:,2],
                  color="lightsteelblue", opacity=0.35, name="Mesh", showlegend=False),
        row=1, col=col
    )
    m1 = (hits==1); m2 = (hits>=2)&(hits<=3); m3 = (hits>=4)
    if np.any(m1):
        fig.add_trace(go.Scatter3d(x=V[m1,0], y=V[m1,1], z=V[m1,2], mode="markers",
                                   name="hits=1", showlegend=(col==1),
                                   marker=dict(size=marker_size, color=YELLOW)), row=1, col=col)
    if np.any(m2):
        fig.add_trace(go.Scatter3d(x=V[m2,0], y=V[m2,1], z=V[m2,2], mode="markers",
                                   name="hits=2–3", showlegend=(col==1),
                                   marker=dict(size=marker_size, color=ORANGE)), row=1, col=col)
    if np.any(m3):
        fig.add_trace(go.Scatter3d(x=V[m3,0], y=V[m3,1], z=V[m3,2], mode="markers",
                                   name=("hits=4–5" if R>=5 else "hits≥4"), showlegend=(col==1),
                                   marker=dict(size=marker_size, color=RED)), row=1, col=col)

def build_hits_figure(
    V_orc, F_orc, pr_orc, V_frc, F_frc, pr_frc,
    *, title_left, title_right, marker_size=3.0, pad=0.03, scene_eye=None, scene2_eye=None
):
    def _hits(pr): 
        R = infer_R_from_presence(pr); return np.rint(pr*R).astype(int), R
    hits_orc, R_orc = _hits(pr_orc)
    hits_frc, R_frc = _hits(pr_frc)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"},{"type":"scene"}]],
                        subplot_titles=[title_left, title_right], horizontal_spacing=0.02)
    _add_panel(fig, V_orc, F_orc, hits_orc, R_orc, col=1, marker_size=marker_size)
    _add_panel(fig, V_frc, F_frc, hits_frc, R_frc, col=2, marker_size=marker_size)

    def rng(V): 
        return ([V[:,0].min()-pad, V[:,0].max()+pad],
                [V[:,1].min()-pad, V[:,1].max()+pad],
                [V[:,2].min()-pad, V[:,2].max()+pad])
    xr1, yr1, zr1 = rng(V_orc); xr2, yr2, zr2 = rng(V_frc)

    fig.update_scenes(xaxis=dict(range=xr1,visible=False), yaxis=dict(range=yr1,visible=False),
                      zaxis=dict(range=zr1,visible=False), aspectmode="data", row=1,col=1)
    fig.update_scenes(xaxis=dict(range=xr2,visible=False), yaxis=dict(range=yr2,visible=False),
                      zaxis=dict(range=zr2,visible=False), aspectmode="data", row=1,col=2)

    lay = dict(margin=dict(l=0,r=0,t=48,b=0),
               scene=dict(xaxis_visible=False,yaxis_visible=False,zaxis_visible=False),
               scene2=dict(xaxis_visible=False,yaxis_visible=False,zaxis_visible=False),
               showlegend=True)
    if scene_eye:  lay["scene_camera"]  = dict(eye=scene_eye)
    if scene2_eye: lay["scene2_camera"] = dict(eye=scene2_eye)
    fig.update_layout(**lay)
    return fig

