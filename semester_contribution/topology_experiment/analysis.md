# Scientific Analysis: PSO Communication Topologies

## Research Question

How do different swarm communication topologies affect convergence, diversity, and robustness in learned Particle Swarm Optimization?

## Hypothesis

Local topologies should preserve diversity and help on multimodal or dynamic landscapes, while global topology should converge faster on simple unimodal landscapes.

## Main Findings

The results support the hypothesis in the dynamic and low-dimensional multimodal cases, but not uniformly across all dimensions.

| Landscape | 2D result | 10D result | Interpretation |
|---|---|---|---|
| `sphere` | all topologies solve it | `knearest` slightly best | topology has little practical effect on this easy unimodal task |
| `rastrigin` | `von_neumann` best, `ring` second | `global` best | local topology helps avoid local minima in 2D, but high-dimensional search favors faster information sharing |
| `dynamic_sphere` | `ring` best | `ring` best | local communication improves adaptation when the optimum moves |

Aggregated trained-policy final-score means:

| Condition | Best topology | Mean final score |
|---|---|---:|
| `sphere 2D` | effectively tied | ~0 |
| `sphere 10D` | `knearest` | `-0.1356` |
| `rastrigin 2D` | `von_neumann` | `-0.3059` |
| `rastrigin 10D` | `global` | `-14.9262` |
| `dynamic_sphere 2D` | `ring` | `-2.6138` |
| `dynamic_sphere 10D` | `ring` | `-1.7095` |

Because the objective functions are negated, higher scores are better.

## Interpretation

`global` topology gives every particle access to the best-known solution quickly. This creates strong exploitation pressure: the swarm can converge quickly, but it can also collapse prematurely around a local optimum or stale best solution.

`ring` and `von_neumann` topologies slow down information propagation. This preserves multiple semi-independent search regions, which helps on `rastrigin 2D` and both dynamic-sphere conditions. The dynamic-sphere result is especially strong: `ring` beats `global` in all seeds in both 2D and 10D.

`knearest` increases diversity but is not consistently best. This shows that diversity is useful only when paired with enough information flow to exploit good discoveries.

## Hypothesis Support

Supported:

- Local topologies preserve more diversity than global topology.
- Local topologies help on low-dimensional multimodal search.
- Local topology, especially `ring`, improves robustness on dynamic landscapes.
- Global topology is useful when fast exploitation is more important than maintaining diversity.

Partially contradicted:

- Global topology is not meaningfully better on `sphere 2D`; all topologies solve it.
- `knearest` is numerically best on `sphere 10D`, although the differences are small.
- Local topology does not help on `rastrigin 10D`; `global` performs best there.

## Exploration vs Exploitation

Topology acts as an information-pressure parameter:

- `global`: high information pressure, fast exploitation, lower diversity.
- `ring`: low information pressure, strong diversity preservation, robust adaptation.
- `von_neumann`: intermediate local structure, especially strong on `rastrigin 2D`.
- `knearest`: adaptive local structure, high diversity, less predictable performance.

The main scientific takeaway is that there is no universally best topology. The best topology depends on whether the landscape rewards fast convergence, diversity preservation, or adaptation.

## Limitations

- Only five seeds were used.
- Each condition used 10 evaluation episodes.
- PPO did not consistently outperform the random baseline, so conclusions about topology are stronger than conclusions about learned control.
- Some information-spread metrics are weak or near-zero, so convergence and diversity metrics are more reliable.
- `diversity_position_spread` can become very large in 10D and should be interpreted comparatively.
- No formal significance testing was performed.
- Results depend on the specific dynamic-sphere schedule, swarm size, training budget, and PPO setup.

## Concise Report Conclusion

Communication topology substantially changes PSO behavior by controlling how quickly information spreads through the swarm. Global topology favors rapid exploitation and performs well when fast convergence is beneficial, while local topologies preserve diversity and improve robustness on multimodal low-dimensional and dynamic landscapes. The strongest evidence is that `ring` consistently outperforms `global` on `dynamic_sphere`, and `von_neumann`/`ring` outperform `global` on `rastrigin 2D`. However, the hypothesis is only partially supported because `global` is best on `rastrigin 10D`, showing that diversity preservation can become costly in higher-dimensional search. Overall, topology should be treated as an exploration-exploitation mechanism rather than a universally optimal design choice.

