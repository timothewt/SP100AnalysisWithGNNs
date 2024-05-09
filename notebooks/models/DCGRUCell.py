import torch
import torch.nn as nn
import torch.nn.functional as F

from notebooks.models.GCN import GCN


class DCGRUCell(nn.Module):
	"""
	DCRNN Cell for one timestep, from https://arxiv.org/pdf/1707.01926.
	"""

	def __init__(self, in_channels: int, hidden_size: int):
		super(DCGRUCell, self).__init__()
		self.gcn_r = GCN(in_channels + hidden_size, [hidden_size, hidden_size], bias=True)
		self.gcn_u = GCN(in_channels + hidden_size, [hidden_size, hidden_size], bias=True)
		self.gcn_c = GCN(in_channels + hidden_size, [hidden_size, hidden_size], bias=True)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor,
				h: torch.tensor) -> torch.tensor:
		"""
		Performs a forward pass on a single DCRNN cell.
		:param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb)
		:param edge_index: The edge index of the graph A (2, Edges_nb)
		:param edge_weight: The edge weight of the graph (Edges_nb,)
		:param h: The hidden state of the GRU h_{t-1} (Nodes_nb, Hidden_size)
		:return: The hidden state of the GRU h_t (Nodes_nb, Hidden_size)
		"""
		x_h = torch.cat([x, h], dim=-1)
		r = F.sigmoid(self.gcn_r(x_h, edge_index, edge_weight))
		u = F.sigmoid(self.gcn_u(x_h, edge_index, edge_weight))
		c = F.tanh(self.gcn_c(torch.cat([x, r * h], dim=-1), edge_index, edge_weight))
		return u * h + (1 - u) * c
