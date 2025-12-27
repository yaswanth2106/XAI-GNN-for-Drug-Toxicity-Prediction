from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import torch

dataset = MoleculeNet(root="data", name="Tox21")
dataset = [d for d in dataset if d.y.sum() >= 0]

train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Total graphs:", len(dataset))
print("Node features:", dataset[0].x.shape[1])
print(dataset[0].smiles)