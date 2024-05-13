import numpy as np
import torch
from torch_geometric.data import Batch, Data


def batch_obs(obs: np.ndarray) -> Batch:
	"""
	Converts dict of a list of observations into a PyTorch Geometric Batch object
	:param obs: dict of observations {'x': (n_envs, [dim]), 'edge_index': (n_envs, 2, n_edges), 'edge_weight': (n_envs, n_edges)}
	:return: Batch object
	"""
	obs = [
		Data(
			x=torch.tensor(x, dtype=torch.float32),
			edge_index=torch.tensor(edge_index, dtype=torch.int64),
			edge_weight=torch.tensor(edge_weight, dtype=torch.float32)
		) for x, edge_index, edge_weight in zip(obs['x'], obs['edge_index'], obs['edge_weight'])
	]
	return Batch.from_data_list(obs)
