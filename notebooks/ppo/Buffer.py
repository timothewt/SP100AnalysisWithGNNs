import torch
from torch import tensor
from torch_geometric.data import Batch


class Buffer:
	"""
	Memory buffer used for PPO with PyTorch Geometric
	"""
	def __init__(
			self,
			num_envs: int,
			max_len: int = 5,
			actions_nb: int = 1,
			device: torch.device = torch.device("cpu"),
	):
		"""
		Initialization of the buffer
		:param num_envs: number of parallel environments
		:param max_len: maximum length of the buffer, typically the PPO horizon parameter
		:param actions_nb: number of possible actions (1 for discrete and n for continuous)
		:param device: device used by PyTorch
		"""
		self.states: list[Batch] = [None for _ in range(max_len)]
		self.next_states: list[Batch] = [None for _ in range(max_len)]
		self.dones = torch.empty((max_len, num_envs, 1), device=device)
		self.actions = torch.empty((max_len, num_envs, actions_nb), device=device)
		self.rewards = torch.empty((max_len, num_envs, 1), device=device)
		self.values = torch.empty((max_len, num_envs, 1), device=device)
		self.log_probs = torch.empty((max_len, num_envs, actions_nb), device=device)

		self.num_envs = num_envs
		self.max_len = max_len
		self.device = device
		self.i = 0

	def is_full(self) -> bool:
		"""
		Checks if the buffer is full
		:return: True if the buffer is full False otherwise
		"""
		return self.i == self.max_len

	def push(
			self,
			states: Batch,
			next_states: Batch,
			dones: tensor,
			actions: tensor,
			rewards: tensor,
			values: tensor,
			log_probs: tensor,
	) -> None:
		"""
		Pushes new values in the buffer of shape (num_env, data_shape)
		:param states: states of each environment
		:param next_states: next states after this step
		:param dones: if the step led to a termination
		:param actions: actions made by the agents
		:param rewards: rewards given for this action
		:param values: critic policy value
		:param log_probs: log probability of the actions
		"""
		assert self.i < self.max_len, "Buffer is full!"

		self.states[self.i] = states
		self.next_states[self.i] = next_states
		self.dones[self.i] = dones
		self.actions[self.i] = actions
		self.rewards[self.i] = rewards
		self.values[self.i] = values
		self.log_probs[self.i] = log_probs

		self.i += 1

	def get_all(self) -> tuple[list[Batch], list[Batch], tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the values of the buffer
		:return: all buffer tensors
		"""
		return self.states, self.next_states, self.dones, self.actions, self.rewards, self.values, self.log_probs

	def get_all_flattened(self) -> tuple[Batch, Batch, tensor, tensor, tensor, tensor, tensor]:
		"""
		Gives all the buffer values as flattened tensors
		:return: all buffer tensors flattened
		"""
		return Batch.from_data_list([data for state in self.states for data in state.to_data_list()]), \
			Batch.from_data_list([data for next_state in self.next_states for data in next_state.to_data_list()]), \
			self.dones.flatten(), \
			self.actions.flatten(end_dim=1), \
			self.rewards.flatten(), \
			self.values.flatten(), \
			self.log_probs.flatten(end_dim=1),

	def reset(self) -> None:
		"""
		Resets the iteration variable
		"""
		self.i = 0
		self.states = [None for _ in range(self.max_len)]
		self.next_states = [None for _ in range(self.max_len)]
