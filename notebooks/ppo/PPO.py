import glob
import os
from datetime import datetime
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from torch import nn, tensor
from torch.nn import functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, Batch

from tqdm import tqdm

from notebooks.ppo.Buffer import Buffer
from notebooks.ppo.utils import batch_obs


class PPO:
	"""
	Proximal Policy Optimization for PyTorch Geometric
	Modified from https://github.com/timothewt/DeepRL/
	"""

	def __init__(self, config: dict[str: Any]):
		"""
		:param config:
			env_fn (Callable[[], gymnasium.Env]): function returning a Gymnasium environment
			num_envs (int): number of environments in parallel

			actor_model (nn.Module): actor model
			critic_model (nn.Module): critic model

			actor_lr (float): learning rate of the actor
			critic_lr (float): learning rate of the critic
			gamma (float): discount factor
			gae_lambda (float): GAE parameter
			horizon (int): steps number between each update
			num_epochs (int): number of epochs during the policy updates
			ent_coef (float): entropy bonus coefficient
			vf_coef (float): value function loss coefficient
			eps (float): epsilon clip value
			minibatch_size (float): size of the mini-batches used to update the policy
			use_grad_clip (bool): boolean telling if gradient clipping is used
			grad_clip (float): value at which the gradients will be clipped
		"""
		# Vectorized envs

		self.env_fn = config.get("env_fn", None)
		assert self.env_fn is not None, "No environment function provided!"
		self.env: gym.Env = self.env_fn()

		assert isinstance(self.env, gym.Env), "Only gymnasium.Env is currently supported."
		self.num_envs = max(config.get("num_envs", 1), 1)
		self.envs: SyncVectorEnv = SyncVectorEnv([self.env_fn for _ in range(self.num_envs)])
		self.observation_space = self.envs.single_observation_space
		self.action_space = self.envs.single_action_space
		assert isinstance(self.action_space, spaces.Box), "Action space needs to be continuous (spaces.Box)!"

		# Stats

		self.writer = None

		# Algorithm hyperparameters

		self.actor_lr: float = config.get("actor_lr", .0001)
		self.critic_lr: float = config.get("critic_lr", .0005)
		self.gamma: float = config.get("gamma", .99)
		self.gae_lambda: float = config.get("gae_lambda", .95)
		self.horizon: int = config.get("horizon", 5)
		self.num_epochs: int = config.get("num_epochs", 5)
		self.ent_coef: float = config.get("ent_coef", .01)
		self.vf_coef = config.get("vf_coef", .5)
		self.eps = config.get("eps", .2)
		self.use_grad_clip = config.get("use_grad_clip", False)
		self.grad_clip = config.get("grad_clip", .5)

		self.batch_size = self.horizon * self.num_envs
		self.minibatch_size = config.get("minibatch_size", self.batch_size)
		assert self.batch_size % self.minibatch_size == 0
		self.minibatch_nb_per_batch = self.batch_size // self.minibatch_size

		# Policies

		self.env_flat_obs_space = gym.spaces.utils.flatten_space(self.observation_space)
		self.actions_nb = np.prod(self.action_space.shape)

		assert (actor := config.get("actor_model")) is not None, "No actor model provided!"
		assert (critic := config.get("critic_model")) is not None, "No critic model provided!"

		self.actor: nn.Module = actor
		self.actor_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

		self.critic: nn.Module = critic
		self.critic_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

	def train(self, max_steps: int, save_models: bool = False, checkpoints: bool = False, save_freq: int = 1_000,
			  task_title: str = "") -> None:
		"""
		Trains the algorithm on the chosen environment
		From https://arxiv.org/pdf/1707.06347.pdf and https://arxiv.org/pdf/2205.09123.pdf
		:param max_steps: maximum number of steps for the whole training process
		:param save_models: indicates if the models should be saved at the end of the training
		:param checkpoints: indicates if the models should be saved at regular intervals
		:param save_freq: frequency at which the models should be saved
		:param task_title: title of the task
		"""
		exp_name = f"{task_title}_PPO_{datetime.now().strftime('%d_%m_%Hh%Mm')}"
		self.writer = SummaryWriter(
			f"runs/{exp_name}",
		)
		self.writer.add_text(
			"Hyperparameters/hyperparameters",
			self.dict2mdtable({
				"num_envs": self.num_envs,
				"actor_lr": self.actor_lr,
				"critic_lr": self.critic_lr,
				"gamma": self.gamma,
				"gae_lambda": self.gae_lambda,
				"horizon": self.horizon,
				"num_epochs": self.num_epochs,
				"ent_coef": self.ent_coef,
				"vf_coef": self.vf_coef,
				"eps": self.eps,
				"minibatch_size": self.minibatch_size,
				"use_grad_clip": self.use_grad_clip,
				"grad_clip": self.grad_clip,
			})
		)

		episode = 0

		buffer = Buffer(self.num_envs, self.horizon, self.actions_nb)

		print("==== STARTING TRAINING ====")

		obs, infos = self.envs.reset()
		obs = batch_obs(obs)
		first_agent_rewards = 0

		for step in tqdm(range(max_steps), desc="PPO Training"):
			critic_output = self.critic(obs)  # value function

			means, std = self.actor(obs)
			dist = Normal(loc=means, scale=std)
			actions = dist.sample()
			actions_to_input = actions.cpu().numpy()
			log_probs = dist.log_prob(actions)

			new_obs, rewards, dones, truncateds, new_infos = self.envs.step(actions_to_input)
			dones = dones + truncateds  # done or truncate
			new_obs = batch_obs(new_obs)

			buffer.push(
				obs,
				new_obs,
				torch.from_numpy(dones).float().unsqueeze(1),
				actions,
				torch.from_numpy(rewards).float().unsqueeze(1),
				critic_output,
				log_probs,
			)

			obs = new_obs

			if buffer.is_full():
				self._update_networks(buffer)
				buffer.reset()

			first_agent_rewards += rewards[0]
			if dones[0]:
				self.writer.add_scalar("Rewards", first_agent_rewards, episode)
				first_agent_rewards = 0
				episode += 1

			if save_models and checkpoints and step % save_freq == 0:
				self.save_models(f"{exp_name}/step{step}", [("actor", self.actor), ("critic", self.critic)])

		print("==== TRAINING COMPLETE ====")
		if save_models:
			self.save_models(exp_name, [("actor", self.actor), ("critic", self.critic)])

	def _update_networks(self, buffer: Buffer) -> None:
		"""
		Updates the actor and critic networks according to the PPO paper
		:param buffer: complete buffer of experiences
		"""
		states, _, _, actions, rewards, values, old_log_probs = buffer.get_all_flattened()
		values, old_log_probs = values.detach().view(self.batch_size, 1), old_log_probs.detach()

		advantages = self._compute_advantages(buffer, self.gamma, self.gae_lambda).flatten(end_dim=1)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		returns = advantages + values

		for _ in range(self.num_epochs):
			indices = torch.randperm(self.batch_size)

			for m in range(self.minibatch_nb_per_batch):
				start = m * self.minibatch_size
				end = start + self.minibatch_size
				minibatch_indices = indices[start:end]

				means, stds = self.actor(Batch.from_data_list(states[minibatch_indices]))
				new_dist = Normal(loc=means, scale=stds)

				new_log_probs = new_dist.log_prob(actions[minibatch_indices]).view(self.minibatch_size, self.actions_nb)
				new_entropy = new_dist.entropy()
				new_values = self.critic(Batch.from_data_list(states[minibatch_indices]))

				r = torch.exp(new_log_probs - old_log_probs[minibatch_indices])  # policy ratio
				L_clip = torch.min(
					r * advantages[minibatch_indices],
					torch.clamp(r, 1 - self.eps, 1 + self.eps) * advantages[minibatch_indices]
				).mean()
				L_vf = F.mse_loss(new_values, returns[minibatch_indices])
				L_S = new_entropy.mean()

				# Updating the network
				self.actor_optimizer.zero_grad()
				self.critic_optimizer.zero_grad()
				(- L_clip + self.vf_coef * L_vf - self.ent_coef * L_S).backward()
				if self.use_grad_clip:
					nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
					nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
				self.actor_optimizer.step()
				self.critic_optimizer.step()

				self.writer.add_scalar("Loss/Actor_Loss", L_clip.item())
				self.writer.add_scalar("Loss/Critic_Loss", L_vf.item())
				self.writer.add_scalar("Loss/Entropy", L_S.item())

	def _compute_advantages(self, buffer: Buffer, gamma: float, gae_lambda: float) -> tensor:
		"""
		Computes the advantages for all steps of the buffer
		:param buffer: complete buffer of experiences
		:param gamma: rewards discount rate
		:param gae_lambda: lambda parameter of the GAE
		:return: the advantages for each timestep as a tensor
		"""
		_, next_states, dones, _, rewards, values, _ = buffer.get_all()

		next_values = values.roll(-1, dims=0)
		next_values[-1] = self.critic(next_states[-1])

		deltas = (rewards + gamma * next_values - values).detach()

		advantages = torch.zeros(deltas.shape)
		last_advantage = advantages[-1]
		next_step_terminates = dones[-1]  # should be the dones of the next step however cannot reach it
		for t in reversed(range(buffer.max_len)):
			advantages[t] = last_advantage = deltas[t] + gamma * gae_lambda * last_advantage * (
						1 - next_step_terminates)
			next_step_terminates = dones[t]

		return advantages

	def compute_single_action(self, obs: np.ndarray, infos: dict = None) -> int | np.ndarray:
		"""
		Computes one action for the given observation
		:param obs: observation to compute the action from
		:param infos: infos given by the environment
		:return: the action
		"""
		means, _ = self.actor(Data(x=torch.from_numpy(obs["x"]), edge_index=torch.from_numpy(obs["edge_index"]),
								   edge_weight=torch.from_numpy(obs["edge_weight"]), num_graphs=1))
		return means.detach().numpy().squeeze(0)

	@staticmethod
	def dict2mdtable(d: dict[str: float], key: str = 'Name', val: str = 'Value'):
		"""
		Used to log hyperparameters in tensorboard
		From https://github.com/tensorflow/tensorboard/issues/46#issuecomment-1331147757
		:param d: dict mapping name to values
		:param key: key in table header
		:param val: value in table header
		:return:
		"""
		rows = [f'| {key} | {val} |']
		rows += ['|--|--|']
		rows += [f'| {k} | {v} |' for k, v in d.items()]
		return "  \n".join(rows)

	@staticmethod
	def save_models(exp_name: str, models: list[tuple[str, nn.Module]]) -> None:
		"""
		Saves the models in the given directory
		:param models: list of tuples of the algorithm's models (name, model)
		:param exp_name: name of the experiment
		"""
		os.makedirs(f"saved_models/{exp_name}", exist_ok=True)
		for name, model in models:
			torch.save(model.state_dict(), f"saved_models/{exp_name}/{name}.pt")

	def load_models(self, dir_path: str) -> None:
		"""
		Loads the models from the given directory
		:param dir_path: directory path
		"""
		for saved_model in glob.glob(f"{dir_path}/*.pt"):
			self.__getattribute__(saved_model.split("\\")[-1].split(".")[0]).load_state_dict(torch.load(saved_model))
