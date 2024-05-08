import torch
import torch.nn as nn
import torch.nn.functional as F

from notebooks.models.GCN import GCN


class TGCNCell(nn.Module):
	"""
	T-GCN Cell for one timestep, from https://arxiv.org/pdf/1811.05320.
	"""

	def __init__(self, in_channels: int, hidden_size: int):
		super(TGCNCell, self).__init__()
		self.gcn = GCN(in_channels, [hidden_size, hidden_size])
		self.lin_u = nn.Linear(2 * hidden_size, hidden_size)
		self.lin_r = nn.Linear(2 * hidden_size, hidden_size)
		self.lin_c = nn.Linear(2 * hidden_size, hidden_size)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor, h: torch.tensor) -> tuple[
		torch.tensor, torch.tensor]:
		"""
		Performs a forward pass on a single T-GCN cell (GCN + GRU).
		:param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb)
		:param edge_index: The edge index of the graph A (2, Edges_nb)
		:param edge_weight: The edge weight of the graph (Edges_nb,)
		:param h: The hidden state of the GRU h_{t-1} (Nodes_nb, Hidden_size)
		:return: The hidden state of the GRU h_t (Nodes_nb, Hidden_size)
		"""
		x = F.sigmoid(self.gcn(x, edge_index, edge_weight))  # f(A,X_t), Eq. 2
		u = F.sigmoid(self.lin_u(torch.cat([x, h], dim=-1)))  # u_t, Eq. 3
		r = F.sigmoid(self.lin_r(torch.cat([x, h], dim=-1)))  # r_t,  Eq. 4
		c = F.tanh(self.lin_c(torch.cat([x, r * h], dim=-1)))  # c_t, Eq. 5

		return u * h + (1 - u) * c  # h_t, Eq. 6
