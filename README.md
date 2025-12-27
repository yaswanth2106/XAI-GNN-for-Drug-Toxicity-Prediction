# 🧪 Molecule Toxicity Prediction with Graph Neural Networks

A **self‑contained** Python project that trains a Graph Convolutional Network (GCN) on the **Tox21** molecular toxicity dataset (via the `MoleculeNet` collection) and serves an interactive web UI with **Streamlit** to explore predictions and per‑atom importance (XAI).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Training the Model](#training-the-model)
5. [Running the Web App](#running-the-web-app)
6. [File Overview](#file-overview)
7. [Future Improvements](#future-improvements)

---

## Project Structure

```
gnn/
│
├─ .venv/                # Virtual environment 
├─ __pycache__/          # Compiled Python files
│
├─ app.py                # Streamlit UI – loads the saved model & visualises molecules
├─ data.py               # Loads the MoleculeNet Tox21 dataset, splits train/test, creates DataLoaders
├─ model.py              # Core GCN definition (flexible `in_channels` argument)
├─ train.py              # Training loop, saves `gnn_tox21.pth`
├─ visualise.py          # Converts a `torch_geometric` graph to a NetworkX plot with RDKit atom labels
├─ xai.py                # Simple gradient‑based node importance (per‑atom attribution)
├─ data/                 # Will contain the downloaded MoleculeNet files after first run
└─ gnn_tox21.pth         # Saved checkpoint (generated after training)
```

---

## Installation

1. **Clone the repository** (or copy the folder to your machine).

2. **Create & activate a virtual environment** (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. **Install the dependencies**

   Follow the instructions in `req.txt`. The most reliable way on Windows is:

```powershell
pip install torch==2.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch_geometric
pip install -r req.txt
```

> **Note:** If you have a CUDA‑enabled GPU, replace the `cpu` suffix with the appropriate CUDA version (e.g., `+cu121`).

---

## Dataset

`data.py` uses **MoleculeNet** to fetch the **Tox21** dataset automatically:

```python
from torch_geometric.datasets import MoleculeNet
dataset = MoleculeNet(root="data", name="Tox21")
```

The first time you run any script that imports `data.py`, the dataset will be downloaded into `./data/`. No manual download is required.

---

## Training the Model

Run the training script:

```powershell
python train.py
```

What happens:
* The GCN (`model.GNN`) is instantiated with the correct number of node features (`in_channels` is inferred from the first training graph). 
* The model is trained for 20 epochs on the training split. 
* After training, a checkpoint `gnn_tox21.pth` is saved (contains `model_state_dict` and the `in_channels` value). 

You will see console output similar to:

```
Epoch 01 | Loss 0.6931 | Accuracy 0.5000
...
Epoch 20 | Loss 0.2123 | Accuracy 0.8421
✅ Model saved
```

---

## Running the Web App

Launch the Streamlit UI:

```powershell
streamlit run app.py
```

The UI provides:
* **Molecule selector** – slide to pick any molecule from the dataset.
* **Toxicity probability** – model’s sigmoid output (0 = non‑toxic, 1 = toxic).
* **Node importance values** – gradient‑based attribution per atom.
* **Graph visualisation** – a NetworkX plot where node colour intensity (blue→red) reflects toxicity contribution.

All interactions happen locally; no external API keys are required.

---

## File Overview

| File | Purpose |
|------|---------|
| **model.py** | Defines `GNN` (GCN) with flexible `in_channels`. Handles float conversion of node features and edge indices. |
| **data.py** | Loads the MoleculeNet Tox21 dataset, filters out graphs with missing labels, splits into train/test, creates `DataLoader`s. |
| **train.py** | Training loop, computes binary cross‑entropy loss, reports loss & accuracy, saves checkpoint. |
| **visualise.py** | Turns a `torch_geometric` data object into a NetworkX graph, draws atom symbols with RDKit, colours nodes by importance. |
| **xai.py** | Simple gradient‑based node importance (per‑atom attribution). |
| **app.py** | Streamlit front‑end: loads checkpoint, runs inference, displays probability, importance values, and the visualisation. |
| **gnn_tox21.pth** | Model checkpoint generated after training (`train.py`). |
| **data/** | Auto‑created folder that stores the raw MoleculeNet files (`.npz`, `.csv`, etc.). |

---

## Future Improvements

* **More sophisticated XAI** – integrate Integrated Gradients or GNNExplainer for richer explanations.
* **Multi‑task learning** – Tox21 actually contains 12 toxicity endpoints; extend the model to predict all simultaneously.
* **GPU acceleration** – switch to a CUDA‑enabled environment for faster training on larger batches.
* **Dockerisation** – wrap the whole stack in a Docker image for reproducible deployment.
* **Model checkpointing** – add early‑stopping and best‑model saving based on validation loss.

---

## 🎉 Quick Start Recap

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r req.txt

python train.py
streamlit run app.py
```

Enjoy experimenting with graph neural networks for molecular toxicity! 🚀
