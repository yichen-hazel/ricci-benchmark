# Statistically Assessing Node-Level Ricci Curvature Proxies via Scalar Curvature

This repository contains the code and data for my MSc Statistics research project at Imperial College London.  
Itintroduces a benchmarking framework for evaluating node-level Ricci curvature proxies (Ollivier–Ricci, Forman–Ricci, etc.) against scalar curvature on both synthetic manifolds (sphere, torus, paraboloid, saddle, etc.) and mesh-based models (Utah Teapot, Stanford Bunny).

The framework systematically assesses performance across varying sample sizes and noise levels using metrics such as Spearman correlation, Precision@k, and enrichment ratios.  These results provide insight into the robustness and practical utility of different curvature proxies, with potential applications to different types of data.  The entire workflow is fully reproducible and can be readily adapted for alternative benchmarking studies or future extensions.

---

## Project Structure
- `src/` — Core source code (geometry generation, graph construction, curvature computation)  
- `analysis/` — Main Jupyter notebooks for experiments and plotting  
- `data/` — Generated data (point clouds, graphs, curvature arrays)  
- `figures/` — Benchmark plots and visualizations used in the thesis  
- `models/` — Mesh models (Utah Teapot, Stanford Bunny)  
- `requirements.txt` — Python package used  
- `README.md` — Project documentation  

---
## Reproducibility
- All figures in the thesis can be reproduced via notebooks in `analysis/`
- Data is pre-generated data in `data/`, New datasets can be generated using scripts in `src/geometry/`
- The framework can be adapted to new manifolds or alternative curvature definitions
