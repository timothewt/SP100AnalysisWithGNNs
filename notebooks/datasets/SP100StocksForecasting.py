import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from notebooks.datasets.utils import get_graph_in_pyg_format


class SP100StocksForecasting(Dataset):
	"""
	Stock price data for the S&P 100 companies.
	The graph data built from the notebooks is used.
	"""

	def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv",
				 adj_file_name: str = "adj.npy", time_window: int = 10):
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		self.time_window = time_window
		super().__init__(root)

	@property
	def processed_dir(self) -> str:
		return osp.join(self.root, 'forecasting_processed')

	@property
	def raw_file_names(self) -> list[str]:
		return [
			self.values_file_name, self.adj_file_name
		]

	@property
	def processed_file_names(self) -> list[str]:
		return [
			f'forecasting_timestep_{idx}.pt' for idx in range(len(self))
		]

	def download(self) -> None:
		pass

	def process(self) -> None:
		x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
			values_path='../data/SP100/raw/values.csv',
			adj_path='../data/SP100/raw/adj.npy',
		)
		timestamps = [
			Data(
				x=x[idx:idx + self.time_window, :, :],
				edge_index=edge_index,
				edge_weight=edge_weight,
				close_price=close_prices[:, idx:idx + self.time_window],
				y=x[idx + self.time_window, :, 1:2],
				close_price_y=close_prices[:, idx + self.time_window:idx + self.time_window + 1],
			) for idx in range(x.shape[0] - self.time_window)
		]
		for t, timestep in enumerate(timestamps):
			torch.save(
				timestep, osp.join(self.processed_dir, f"forecasting_timestep_{t}.pt")
			)

	def len(self) -> int:
		values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
		return len(values.loc[values.index[0][0]]) - self.time_window

	def get(self, idx: int) -> Data:
		data = torch.load(osp.join(self.processed_dir, f'forecasting_timestep_{idx}.pt'))
		return data
