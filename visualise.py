import matplotlib
matplotlib.use("Agg")

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from rdkit import Chem

def visualize_molecule(data, importance=None):
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    mol = Chem.MolFromSmiles(data.smiles)
    atom_labels = {
        i: atom.GetSymbol()
        for i, atom in enumerate(mol.GetAtoms())
    }

    if importance is not None:
        node_colors = importance.numpy()
    else:
        node_colors = "lightblue"

    fig, ax = plt.subplots(figsize=(5, 5))

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        cmap="coolwarm",
        node_size=700,
        edge_color="gray",
        with_labels=False
    )

    nx.draw_networkx_labels(
        G,
        pos=pos,
        labels=atom_labels,
        font_size=12,
        font_weight="bold",
        ax=ax
    )

    ax.set_title("Blue to Red color intensity indicates toxicity")
    ax.axis("off")

    return fig
