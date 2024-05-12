import torch
from torch import nn

from notebooks.models.TGCNCell import TGCNCell


class A3TGCN(nn.Module):
	"""
	A3T-GCN model from https://arxiv.org/pdf/2006.11583.
	"""
	def __init__(self, in_channels: int, out_channels: int, hidden_size: int, layers_nb: int = 2, output_activation: nn.Module = None):
		super(A3TGCN, self).__init__()
		self.hidden_size = hidden_size
		self.layers_nb = max(1, layers_nb)
		self.cells = nn.ModuleList(
			[TGCNCell(in_channels, hidden_size)] + [TGCNCell(hidden_size, hidden_size) for _ in range(self.layers_nb - 1)]
		)
		self.attention = nn.Sequential(
			nn.Linear(hidden_size, 1),
			nn.Softmax(dim=1),
		)
		nn.init.uniform_(self.attention[0].weight)
		self.out = nn.Sequential(
			nn.Linear(hidden_size, out_channels),
			output_activation if output_activation is not None else nn.Identity(),
		)

	def forward(self, x: torch.tensor, edge_index: torch.tensor, edge_weight: torch.tensor) -> torch.tensor:
		"""
		Performs a forward pass on the A3T-GCN model.
		:param x: The feature matrix of the graph X_t (Nodes_nb, Features_nb, SeqLength)
		:param edge_index: The edge index of the graph A (2, Edges_nb)
		:param edge_weight: The edge weight of the graph (Edges_nb,)
		:return: The output of the model (Nodes_nb, OutFeatures_nb)
		"""
		h_prev = [
			torch.zeros(x.shape[0], self.hidden_size) for _ in range(self.layers_nb)
		]
		h_final = torch.zeros(x.shape[0], x.shape[-1], self.hidden_size)
		for t in range(x.shape[-1]):
			h = x[:, :, t]  # h is the output of the previous GRU layer (the input features for the first layer)
			for i, cell in enumerate(self.cells):
				h = cell(h, edge_index, edge_weight, h_prev[i])
				h_prev[i] = h
			h_final[:, t, :] = h
		return self.out(torch.sum(h_final * self.attention(h_final), dim=1))
