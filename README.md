# WaveRider

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/WaveRider)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**WaveRider** is the official PyTorch implementation of the paper *WaveRider: A Self-Similar
Wave-Skeleton Neural Operator for Transient Hydrogen Blowdown*.

## 📌 Overview

Emergency blowdown of a high-pressure hydrogen vessel drives a shock and a rarefaction wave along the
relief line. Thin moving discontinuities and multidecade state variations make the transient hard for a
conventional surrogate. WaveRider splits the field into two parts:

- the **wave skeleton** is the exact self-similar Riemann solution of the two vessel states, evaluated on
  the mesh at zero trainable cost;
- the **residual corrector** is a physics-attention network that learns only what the one-dimensional
  solution cannot express, with the state input carried in a shock-comoving coordinate.

Across 55 two-vessel cases under five-fold cross-validation the balanced relative $L^2$ error is 0.0568,
against 0.0903 for an equal-capacity Transolver, and the untrained skeleton alone accounts for over 96%
of that reduction.

## ✨ Highlights

- A self-similar Riemann solution supplies the wave skeleton without training
- A WaveRider corrector learns only the residual the skeleton cannot carry
- A two-vessel blowdown case spans pressure differences and pipe diameters
- A balanced relative $L^2$ error reaches 0.0568, 37% below Transolver's 0.0903

## 🧱 Repository Layout

```text
WaveRider/
├── config.py            # paths, constants, model configurations
├── riemann.py           # exact Riemann solver and self-similar sampling
├── data.py              # case loading, features, wave skeleton, folds, loss
├── normalization.py     # log-space transform and per-fold standardization
├── model.py             # corrector, DeepONet baseline, model factory
├── train.py             # training for one configuration and one fold
├── eval.py              # five-fold evaluation and fold-paired t test
├── infer.py             # prediction and the error metrics
├── doe_combined.csv     # case list and the five-fold split
├── README.md
└── LICENSE
```

## 🚀 Running Experiments

### Install the dependencies

```bash
pip install torch numpy
```

### Point the code at the dataset

```bash
export WR_DATA_DIR=/path/to/case/files
export WR_CKPT_DIR=/path/to/checkpoints
```

### Train one fold

```bash
python train.py --config waverider --fold 0
```

### Evaluate the folds

```bash
python eval.py --configs waverider transolver --base transolver
```

`eval.py` without arguments evaluates all seven configurations and prints the fold-paired statistics.

## 🗂️ Data

The case trajectories are not part of this repository. Each case is a `.pt` file named
`dp{dp_MPa}_D{D_mm}.pt`, as listed in `doe_combined.csv`, and holds the five fields, the node coordinates,
the region and boundary masks, the nodal areas and the time grid of the 15276-node axisymmetric mesh.

## 📊 Results

Five-fold means of the balanced relative $L^2$ error.

| configuration | state input | balanced rel. $L^2$ |
|---|---|---|
| `transolver` | initial field, fixed coordinates | 0.0903 |
| **`waverider`** | **wave skeleton, comoving coordinates** | **0.0568** |
| `fixed_coord` | wave skeleton, fixed coordinates | 0.0580 |
| `head_reinject` | wave skeleton, head re-injection | 0.0614 |
| `deeponet` | initial field, fixed coordinates | 0.1482 |
| `transolver_ar` | previous frame, autoregressive | 0.4848 |
| `deeponet_ar` | previous frame, autoregressive | 1.1717 |

## 📚 Citation

```bibtex
@article{wang2026waverider,
  author = {Shengning Wang and Yong Pang and Xinao Ma and Jiale Zuo and Kunpeng Li and Xueguan Song},
  title  = {WaveRider: A Self-Similar Wave-Skeleton Neural Operator for Transient Hydrogen Blowdown},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
