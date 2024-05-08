import torch
from torch import nn

from notebooks.models.TGCNCell import TGCNCell


class A3TGCN(nn.Module):
	"""
	A3T-GCN model from https://arxiv.org/pdf/2006.11583.
	"""

	def __init__(self, in_channels: int, out_channels: int, hidden_size: int):
		super(A3TGCN, self).__init__()
		self.hidden_size = hidden_size
		self.cell = TGCNCell(in_channels, hidden_size)
		self.attention = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1),
			nn.Softmax(dim=1),
		)
		self.out = nn.Linear(hidden_size, out_channels)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
		"""
		Performs a forward pass on the A3T-GCN model.
		:param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb, SeqLength)
		:param edge_index: The edge index of the graph A (2, Edges_nb)
		:param edge_weight: The edge weight of the graph (Edges_nb,)
		:return: The output of the model (Nodes_nb, OutFeatures_nb)
		"""
		h = torch.zeros(x.shape[0], self.hidden_size)
		hs = torch.zeros(x.shape[0], x.shape[-1], self.hidden_size)
		for t in range(x.shape[-1]):
			h = self.cell(x[:, :, t], edge_index, edge_weight, h)
			hs[:, t, :] = h
		return self.out(torch.sum(hs * self.attention(hs), dim=1))
