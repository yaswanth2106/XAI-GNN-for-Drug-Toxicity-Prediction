import streamlit as st
import torch
from model import GNN
from data import dataset
from torch_geometric.data import Batch
from visualise import visualize_molecule

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`"
)

device = "cpu"
st.title("🧪 Molecule Toxicity Prediction (GNN)")

idx = st.slider("Select molecule", 0, len(dataset)-1, 0)
data = dataset[idx]

checkpoint = torch.load("gnn_tox21.pth", map_location=device)

model = GNN(in_channels=checkpoint["in_channels"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

batch = Batch.from_data_list([data])

with torch.no_grad():
    pred = torch.sigmoid(model(batch)).item()

st.write("Toxicity Probability:", round(pred, 5))



st.subheader("Atom Importance (XAI)")

from xai import node_importance
importance = node_importance(model, data)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Node Importance Values")
    st.write(importance.numpy())

with col2:
    st.markdown("### Molecule Graph")
    fig = visualize_molecule(data, importance)
    st.pyplot(fig)
