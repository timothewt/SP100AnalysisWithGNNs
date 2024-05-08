import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
	"""
	Simple GCN model with varaiable layers nb.
	:param in_channels: The number of input features.
	:param layer_sizes: The number of hidden units in each layer.
	"""
	def __init__(self, in_channels: int, layer_sizes: list[int] = None):
		super(GCN, self).__init__()
		layer_sizes = layer_sizes or [32, 32]
		self.convs = nn.ModuleList([
				GCNConv(in_channels, layer_sizes[0]),
			] + [
				GCNConv(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
			]
		)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
		"""
		Performs a forward pass on the GCN model.
		:param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb)
		:param edge_index: The edge index of the graph A (2, Edges_nb)
		:param edge_weight: The edge weight of the graph (Edges_nb,)
		:return: The hidden state of the GCN h_t (Nodes_nb, Hidden_size)
		"""
		for conv in self.convs:
			x = F.relu(conv(x, edge_index, edge_weight))
		return x
