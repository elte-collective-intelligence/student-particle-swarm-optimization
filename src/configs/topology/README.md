# Topology Config Group

This config group defines one Hydra config per topology:

- `global`
- `ring`
- `von_neumann`
- `knearest`

Each config exposes the same common fields for consistent overrides:

- `type`: topology type selector.
- `k`: neighborhood size for topologies that use it (`ring`, `knearest`).
- `recompute_interval`: dynamic neighbor recompute period (`knearest`).
- `rows`, `cols`: direct grid shape for `von_neumann` (optional).
- `grid_shape`: grid-shape rules metadata:
  - `rows`, `cols`: optional shape hints.
  - `rule`: how shape is determined when omitted (`infer_if_missing`).

Notes:

- Unused fields in a given topology are intentionally left as `null` for a stable schema.
- Current topology factory reads `type`, `k`, `recompute_interval`, `rows`, `cols`, and `include_self`.