from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional

import torch


def _normalize_topology_config(cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
	"""Return a plain topology config dictionary.

	Accepts either a direct topology config or a wrapper with a nested ``topology``
	entry, which keeps the factory compatible with both flat and Hydra-style
	configuration layouts.
	"""
	if cfg is None:
		return {}

	config = dict(cfg)
	nested = config.get("topology")
	if isinstance(nested, Mapping) and "type" not in config:
		config = dict(nested)

	if "type" not in config:
		if "name" in config:
			config["type"] = config["name"]
		elif "kind" in config:
			config["type"] = config["kind"]

	return config


class Topology(ABC):
	"""Base API for PSO communication topologies.

	Canonical data format is a boolean adjacency matrix ``A`` with shape
	``[num_particles, num_particles]`` where ``A[i, j]`` means particle ``j`` is
	in the neighborhood of particle ``i``.
	"""

	def __init__(self, num_particles: int, include_self: bool = True):
		if num_particles <= 0:
			raise ValueError("num_particles must be positive")
		self.num_particles = int(num_particles)
		self.include_self = bool(include_self)
		self._adjacency = torch.zeros(
			(self.num_particles, self.num_particles), dtype=torch.bool
		)
		self._neighbors_cache: Optional[List[torch.Tensor]] = None

	@abstractmethod
	def build_adjacency(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Build and return a boolean adjacency matrix with shape [N, N]."""

	def initialize(self, positions: Optional[torch.Tensor] = None) -> None:
		"""Initialize adjacency and invalidate cached neighbor lists."""
		self.set_adjacency(self.build_adjacency(positions))

	def update(self, positions: Optional[torch.Tensor] = None, step: int = 0) -> bool:
		"""Optional update hook for dynamic topologies.

		Returns True if adjacency changed, False otherwise.
		"""
		return False

	def switch(
		self,
		cfg: Optional[Mapping[str, Any]] = None,
		positions: Optional[torch.Tensor] = None,
		initialize: bool = False,
	) -> "Topology":
		"""Create a new topology instance from a config while preserving swarm size.

		This is the runtime switch point: callers can keep the current topology object
		for its metadata, then replace it with the returned instance when the topology
		type or parameters change.
		"""
		config = _normalize_topology_config(cfg)
		config.setdefault("include_self", self.include_self)
		return create_topology(
			num_particles=self.num_particles,
			cfg=config,
			positions=positions,
			initialize=initialize,
		)

	def set_adjacency(self, adjacency: torch.Tensor) -> None:
		adjacency = adjacency.to(dtype=torch.bool)
		expected_shape = (self.num_particles, self.num_particles)
		if tuple(adjacency.shape) != expected_shape:
			raise ValueError(
				f"adjacency must have shape {expected_shape}, got {tuple(adjacency.shape)}"
			)

		if self.include_self:
			adjacency = adjacency.clone()
			adjacency.fill_diagonal_(True)

		self._adjacency = adjacency
		self._neighbors_cache = None

	def get_adjacency(self, device: Optional[torch.device] = None) -> torch.Tensor:
		"""Return canonical adjacency matrix (optionally moved to ``device``)."""
		if device is None:
			return self._adjacency
		return self._adjacency.to(device=device)

	def get_neighbors(self, particle_id: int, device: Optional[torch.device] = None) -> torch.Tensor:
		"""Return cached neighbor indices for a particle."""
		idx = int(particle_id)
		if idx < 0 or idx >= self.num_particles:
			raise IndexError(
				f"particle_id out of range: {idx}, expected [0, {self.num_particles})"
			)

		if self._neighbors_cache is None:
			self._neighbors_cache = [
				torch.nonzero(self._adjacency[i], as_tuple=False).squeeze(-1)
				for i in range(self.num_particles)
			]

		neighbors = self._neighbors_cache[idx]
		if device is not None:
			neighbors = neighbors.to(device=device)
		return neighbors


class GlobalBestTopology(Topology):
	"""Fully connected topology (gBest)."""

	def build_adjacency(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
		return torch.ones((self.num_particles, self.num_particles), dtype=torch.bool)


class RingTopology(Topology):
	"""Ring local-best topology where each particle connects to +/- k neighbors."""

	def __init__(self, num_particles: int, k: int = 1, include_self: bool = True):
		super().__init__(num_particles=num_particles, include_self=include_self)
		if k <= 0:
			raise ValueError("k must be positive")
		self.k = int(k)

	def build_adjacency(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
		n = self.num_particles
		adjacency = torch.zeros((n, n), dtype=torch.bool)
		max_k = min(self.k, n - 1)

		for i in range(n):
			for offset in range(1, max_k + 1):
				adjacency[i, (i + offset) % n] = True
				adjacency[i, (i - offset) % n] = True

		return adjacency


class VonNeumannTopology(Topology):
	"""Von Neumann topology on a toroidal 2D grid."""

	def __init__(
		self,
		num_particles: int,
		rows: Optional[int] = None,
		cols: Optional[int] = None,
		include_self: bool = True,
	):
		super().__init__(num_particles=num_particles, include_self=include_self)
		if (rows is None) != (cols is None):
			raise ValueError("rows and cols must either both be set or both be omitted")

		if rows is None and cols is None:
			rows, cols = self._infer_grid_shape(num_particles)

		if rows is None or cols is None or rows <= 0 or cols <= 0:
			raise ValueError("rows and cols must be positive")

		if rows * cols != num_particles:
			raise ValueError("rows * cols must equal num_particles for Von Neumann")

		self.rows = int(rows)
		self.cols = int(cols)

	@staticmethod
	def _infer_grid_shape(num_particles: int) -> tuple[int, int]:
		rows = int(num_particles**0.5)
		while rows > 1 and num_particles % rows != 0:
			rows -= 1
		cols = num_particles // rows
		return rows, cols

	def build_adjacency(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
		adjacency = torch.zeros(
			(self.num_particles, self.num_particles), dtype=torch.bool
		)

		def idx(r: int, c: int) -> int:
			return r * self.cols + c

		for r in range(self.rows):
			for c in range(self.cols):
				i = idx(r, c)
				up = idx((r - 1) % self.rows, c)
				down = idx((r + 1) % self.rows, c)
				left = idx(r, (c - 1) % self.cols)
				right = idx(r, (c + 1) % self.cols)

				adjacency[i, up] = True
				adjacency[i, down] = True
				adjacency[i, left] = True
				adjacency[i, right] = True

		return adjacency


class KNearestTopology(Topology):
	"""Dynamic k-nearest topology from particle positions."""

	def __init__(
		self,
		num_particles: int,
		k: int = 2,
		recompute_interval: int = 1,
		symmetric: bool = True,
		include_self: bool = True,
	):
		super().__init__(num_particles=num_particles, include_self=include_self)
		if k <= 0:
			raise ValueError("k must be positive")
		if recompute_interval <= 0:
			raise ValueError("recompute_interval must be positive")

		self.k = int(k)
		self.recompute_interval = int(recompute_interval)
		self.symmetric = bool(symmetric)

	def build_adjacency(self, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
		if positions is None:
			raise ValueError("positions are required for k-nearest topology")

		if positions.dim() == 3:
			# Use the first batch element as shared topology for vectorized environments.
			positions = positions[0]

		if positions.dim() != 2 or positions.shape[0] != self.num_particles:
			raise ValueError(
				"positions must have shape [num_particles, dim] or [batch, num_particles, dim]"
			)

		n = self.num_particles
		m = min(self.k, n - 1)
		adjacency = torch.zeros((n, n), dtype=torch.bool, device=positions.device)

		if n == 1:
			return adjacency

		distances = torch.cdist(positions, positions)
		distances.fill_diagonal_(float("inf"))
		knn = torch.topk(distances, k=m, largest=False, dim=-1).indices

		row_idx = (
			torch.arange(n, device=positions.device).unsqueeze(1).expand_as(knn).reshape(-1)
		)
		adjacency[row_idx, knn.reshape(-1)] = True

		if self.symmetric:
			adjacency = adjacency | adjacency.transpose(0, 1)

		return adjacency

	def update(self, positions: Optional[torch.Tensor] = None, step: int = 0) -> bool:
		if step % self.recompute_interval != 0 and self._adjacency.any():
			return False

		new_adjacency = self.build_adjacency(positions)
		changed = not torch.equal(new_adjacency, self._adjacency)
		if changed:
			self.set_adjacency(new_adjacency)
		return changed


def create_topology(
	num_particles: int,
	cfg: Optional[Dict[str, Any]] = None,
	positions: Optional[torch.Tensor] = None,
	initialize: bool = False,
) -> Topology:
	"""Factory for topology objects from config dictionaries.

	Args:
		num_particles: Number of particles in the swarm.
		cfg: Topology config dictionary.
		positions: Optional particle positions used by dynamic topologies.
		initialize: If True, initialize the topology before returning it.
	"""
	config = _normalize_topology_config(cfg)
	topology_type = str(config.get("type", "global")).lower()
	include_self = bool(config.get("include_self", True))

	if topology_type in {"global", "gbest"}:
		topology = GlobalBestTopology(
			num_particles=num_particles,
			include_self=include_self,
		)
	elif topology_type in {"ring", "lbest"}:
		topology = RingTopology(
			num_particles=num_particles,
			k=int(config.get("k", 1)),
			include_self=include_self,
		)
	elif topology_type in {"von_neumann", "vonneumann", "grid"}:
		topology = VonNeumannTopology(
			num_particles=num_particles,
			rows=config.get("rows"),
			cols=config.get("cols"),
			include_self=include_self,
		)
	elif topology_type in {"knearest", "k_nearest", "knn"}:
		topology = KNearestTopology(
			num_particles=num_particles,
			k=int(config.get("k", 2)),
			recompute_interval=int(config.get("recompute_interval", 1)),
			symmetric=bool(config.get("symmetric", True)),
			include_self=include_self,
		)
	else:
		raise ValueError(
			f"Unknown topology type: {topology_type}. "
			"Expected one of: global, ring, von_neumann, knearest"
		)

	if initialize:
		if topology_type in {"knearest", "k_nearest", "knn"} and positions is None:
			raise ValueError("positions are required to initialize a k-nearest topology")
		topology.initialize(positions)

	return topology
