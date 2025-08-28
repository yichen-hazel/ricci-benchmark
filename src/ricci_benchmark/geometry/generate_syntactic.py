import numpy as np
from scipy.stats import qmc
from typing import Tuple, Union
from pathlib import Path


def normalize_points(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(axis=0)
    X = X / np.linalg.norm(X, axis=1).max()
    return X


def gen_sphere(n: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    g = qmc.Halton(d=2).random(n)
    phi = 2 * np.pi * g[:, 0]
    theta = np.arccos(1 - 2 * g[:, 1])
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    X = np.c_[x, y, z] + np.random.normal(0, sigma, (n, 3))
    X = normalize_points(X)
    R = np.full(n, 2.0, dtype=np.float32)
    return X, R


def gen_plane(n: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    g = qmc.Halton(d=2).random(n)
    x = g[:, 0]
    y = g[:, 1]
    z = np.zeros(n)
    X = np.c_[x, y, z] + np.random.normal(0, sigma, (n, 3))
    X = normalize_points(X)
    R = np.zeros(n, dtype=np.float32)
    return X, R


def gen_torus(n: int, sigma: float, R0=3.5, r0=1.5) -> Tuple[np.ndarray, np.ndarray]:
    g = qmc.Halton(d=2).random(n)
    u = 2 * np.pi * g[:, 0]
    v = 2 * np.pi * g[:, 1]
    x = (R0 + r0 * np.cos(v)) * np.cos(u)
    y = (R0 + r0 * np.cos(v)) * np.sin(u)
    z = r0 * np.sin(v)
    X = np.c_[x, y, z] + np.random.normal(0, sigma, (n, 3))
    X = normalize_points(X)
    R = 2 * np.cos(v) / (r0 * (R0 + r0 * np.cos(v)))
    return X, R.astype(np.float32)


def gen_paraboloid(n: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    g = qmc.Halton(d=2).random(n)
    r, th = g[:, 0], 2 * np.pi * g[:, 1]
    x = r * np.cos(th)
    y = r * np.sin(th)
    z = r**2
    X = np.c_[x, y, z] + np.random.normal(0, sigma, (n, 3))
    X = normalize_points(X)
    R = 8 / (1 + 4 * r**2) ** 2
    return X, R.astype(np.float32)


def gen_saddle(n: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    g = qmc.Halton(d=2).random(n)
    x = 2 * (g[:, 0] - 0.5)
    y = 2 * (g[:, 1] - 0.5)
    z = x**2 - y**2
    X = np.c_[x, y, z] + np.random.normal(0, sigma, (n, 3))
    X = normalize_points(X)
    R = -8 / (1 + 4 * x**2 + 4 * y**2) ** 2
    return X, R.astype(np.float32)


def generate_and_save(
    gen_fn,
    n_samples: int,
    noise: float,
    save_dir: Union[str, Path],
    tag: str,
) -> np.ndarray:
    out = gen_fn(n_samples, noise)
    if isinstance(out, tuple) and len(out) == 2:
        X, R_true = out
    else:
        X = out
        R_true = None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = save_dir / f"{tag}.npz"

    if R_true is not None:
        np.savez_compressed(fname, X=X, R_true=R_true)
    else:
        np.savez_compressed(fname, X=X)
    return X


if __name__ == "__main__":
    sample_sizes = [500, 1000, 2000, 3000]
    n_repeats = 5
    noise_levels = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    save_root = Path("data/points")

    generators = {
        "sphere": gen_sphere,
        "plane": gen_plane,
        "torus": gen_torus,
        "paraboloid": gen_paraboloid,
        "saddle": gen_saddle,
    }

    for n_samples in sample_sizes:
        for name, gen_fn in generators.items():
            for noise in noise_levels:
                for i in range(n_repeats):
                    tag = f"{name}_n{n_samples}_noise{noise:.3f}_rep{i}"
                    np.random.seed(i + hash(name) % 1000)
                    generate_and_save(gen_fn, n_samples, noise, save_root, tag)
