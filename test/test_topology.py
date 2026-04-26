import sys
from pathlib import Path

import torch

# Add src to path before importing project modules
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from envs.topology import (  # noqa: E402
	GlobalBestTopology,
	KNearestTopology,
	RingTopology,
	VonNeumannTopology,
	create_topology,
)


def _neighbor_set(topology, particle_id):
	return set(topology.get_neighbors(particle_id).tolist())


def test_create_topology_from_nested_config():
	topology = create_topology(
		num_particles=4,
		cfg={"topology": {"type": "ring", "k": 2, "include_self": False}},
	)

	assert isinstance(topology, RingTopology)
	assert topology.k == 2
	assert topology.include_self is False


def test_global_topology_includes_every_particle():
	topology = GlobalBestTopology(num_particles=4)
	topology.initialize()

	for particle_id in range(4):
		assert _neighbor_set(topology, particle_id) == {0, 1, 2, 3}


def test_ring_topology_neighbors_small_swarm():
	topology = RingTopology(num_particles=5, k=1)
	topology.initialize()

	assert _neighbor_set(topology, 0) == {0, 1, 4}
	assert _neighbor_set(topology, 2) == {1, 2, 3}
	assert _neighbor_set(topology, 4) == {0, 3, 4}


def test_von_neumann_topology_infers_odd_grid_shape():
	topology = VonNeumannTopology(num_particles=9)
	topology.initialize()

	assert (topology.rows, topology.cols) == (3, 3)
	assert _neighbor_set(topology, 4) == {1, 3, 4, 5, 7}


def test_ring_topology_clamps_large_k():
	topology = RingTopology(num_particles=4, k=10)
	topology.initialize()

	for particle_id in range(4):
		assert _neighbor_set(topology, particle_id) == {0, 1, 2, 3}


def test_single_particle_topology_keeps_self_neighbor():
	for topology in [
		GlobalBestTopology(num_particles=1),
		RingTopology(num_particles=1, k=1),
		VonNeumannTopology(num_particles=1),
		KNearestTopology(num_particles=1, k=1),
	]:
		if isinstance(topology, KNearestTopology):
			topology.initialize(torch.zeros((1, 2)))
		else:
			topology.initialize()

		assert topology.get_adjacency().shape == (1, 1)
		assert _neighbor_set(topology, 0) == {0}


def test_k_nearest_recomputes_on_interval():
	topology = KNearestTopology(num_particles=4, k=1, recompute_interval=2, symmetric=True)
	positions_a = torch.tensor(
		[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]], dtype=torch.float32
	)
	positions_b = torch.tensor(
		[[0.0, 0.0], [10.0, 0.0], [1.0, 0.0], [11.0, 0.0]], dtype=torch.float32
	)

	topology.initialize(positions_a)
	initial_adjacency = topology.get_adjacency().clone()

	assert topology.update(positions_b, step=1) is False
	assert torch.equal(topology.get_adjacency(), initial_adjacency)

	assert topology.update(positions_b, step=2) is True
	updated_neighbors = _neighbor_set(topology, 0)
	assert updated_neighbors == {0, 2}
	assert not torch.equal(topology.get_adjacency(), initial_adjacency)