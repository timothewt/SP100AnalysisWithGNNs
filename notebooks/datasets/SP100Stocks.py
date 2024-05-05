import numpy as np
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data


class SP100Stocks(Dataset):
	"""
	Stock price data for the S&P 100 companies.
	The graph data built from the notebook is used.
	"""

	def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv",
				 adj_file_name: str = "adj.npy"):
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		super().__init__(root)

	@property
	def raw_file_names(self) -> list[str]:
		return [
			self.values_file_name, self.adj_file_name
		]

	@property
	def processed_file_names(self) -> list[str]:
		return [
			f'timestep_{idx}.pt' for idx in range(len(self))
		]

	def download(self) -> None:
		pass

	def process(self) -> None:
		values = pd.read_csv('../data/SP100/raw/values.csv').set_index(['Symbol', 'Date'])
		adj = np.load('../data/SP100/raw/adj.npy')
		nodes_nb, edge_nb = len(adj), np.count_nonzero(adj) // 2
		x = torch.tensor(
			values.drop(columns=["Close"]).to_numpy().reshape((nodes_nb, -1, values.shape[1] - 1))
		)
		x = np.swapaxes(x, 1, 2)
		close_prices = torch.tensor(
			values[["Close"]].to_numpy().reshape((nodes_nb, -1))
		)
		edge_index, edge_weight = torch.zeros((2, edge_nb)), torch.zeros((edge_nb,))
		count = 0
		for i in range(nodes_nb):
			for j in range(i + 1, nodes_nb):
				if (weight := adj[i, j]) != 0:
					edge_index[0, count], edge_index[1, count] = i, j
					edge_weight[count] = weight
					count += 1
		timestamps = [
			Data(x=x[:, :, idx], edge_index=edge_index, edge_weight=edge_weight, close_price=close_prices[:, idx]) for
			idx in range(x.shape[2])
		]
		for t, timestep in enumerate(timestamps):
			torch.save(
				timestep, osp.join(self.processed_dir, f"timestep_{t}.pt")
			)

	def len(self) -> int:
		values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
		return len(values.loc[values.index[0][0]])

	def get(self, idx: int) -> Data:
		data = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'))
		return data