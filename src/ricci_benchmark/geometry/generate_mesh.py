import numpy as np
import trimesh
import potpourri3d as pp3d
from pathlib import Path
from typing import Tuple


def preprocess_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, float]:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    scale = mesh.bounding_box.extents.max()
    mesh.apply_scale(1.0 / scale)
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    mesh.fix_normals()
    return mesh, scale


def estimate_scalar_curvature(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    A = pp3d.vertex_areas(V, F)
    A[A < 1e-12] = 1e-12
    face_ang = trimesh.triangles.angles(V[F])
    idx = F.ravel()
    ang = face_ang.ravel()
    angle_sum = np.bincount(idx, weights=ang, minlength=V.shape[0])
    K = (2 * np.pi - angle_sum) / A
    R = 2 * K
    R[~np.isfinite(R)] = 0.0
    return R.astype(np.float32)


def sample_with_noise(mesh: trimesh.Trimesh, n: int, sigma: float) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts + np.random.normal(0, sigma, pts.shape)


def gen_from_mesh(mesh_path: Path, n: int, sigma: float):
    raw_mesh = trimesh.load(mesh_path, process=False)
    X_noisy = sample_with_noise(raw_mesh, n, sigma)
    mesh, scale = preprocess_mesh(raw_mesh)
    V, F = mesh.vertices, mesh.faces
    R_all = estimate_scalar_curvature(V, F)
    X_noisy = X_noisy - X_noisy.mean(0)
    X_noisy /= np.linalg.norm(X_noisy, axis=1).max()
    d2 = np.linalg.norm(V[:, None, :] - X_noisy[None, :, :], axis=2)
    idx = np.argmin(d2, axis=0)
    R_true = R_all[idx]
    return X_noisy.astype(np.float32), R_true


gen_teapot = lambda n, s: gen_from_mesh(Path("models/teapot.obj"), n, s)
gen_bunny = lambda n, s: gen_from_mesh(Path("models/bunny.obj"), n, s)


if __name__ == "__main__":
    out_dir = Path("data/points")
    out_dir.mkdir(exist_ok=True, exist_ok=True)
    sample_sizes = [500, 1000, 2000, 3000]
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    n_repeats = 5
    MODELS = {"teapot": gen_teapot, "bunny": gen_bunny}

    for name, gen_func in MODELS.items():
        for n in sample_sizes:
            for sigma in noise_levels:
                for rep in range(n_repeats):
                    np.random.seed(rep + n + int(sigma * 1000))
                    X, R_true = gen_func(n, sigma)
                    tag = f"{name}_n{n}_noise{sigma:.3f}_rep{rep}"
                    np.savez_compressed(out_dir / f"{tag}.npz", X=X, R_true=R_true)




