# Communication Topologies and Information Flow in Learned PSO

## Research Question & Hypothesis

This project asks how different swarm communication topologies affect convergence, diversity, and robustness in learned Particle Swarm Optimization (PSO). The central hypothesis was that local communication structures should preserve diversity and therefore help on multimodal or dynamic landscapes, while global communication should converge faster on simple unimodal landscapes.

The tested topologies were `global`, `ring`, `von_neumann`, and `knearest`. They were evaluated on `sphere`, `rastrigin`, and `dynamic_sphere` landscapes in both 2D and 10D, across five random seeds. Because the objective functions were negated, higher final scores are better.

## Implementation Summary

The original PSO experiment was extended to compare multiple communication topologies under a common learned-control setup. The implementation evaluates how each topology changes the neighborhood best information available to particles. This makes topology an explicit information-flow parameter rather than a fixed implementation detail.

The experiment pipeline produced trained-policy outputs, aggregated metric tables, and presentation-ready visualizations. The main analysis used final score, best score, convergence AUC, plateau fraction, pairwise particle distance, position spread, velocity alignment, and information-spread measures. Results were summarized in `presentation/topology_experiment/analysis/trained_topology_summary.csv` and supporting figures were generated under `presentation/topology_experiment/presentation_assets/key_figures`.

## Key Results

The results partially support the hypothesis. On the easy `sphere` landscape, topology had little practical effect: all topologies solved the 2D case, and differences in 10D were small. On `rastrigin 2D`, local communication helped: `von_neumann` achieved the best mean final score, followed by `ring`, and both beat `global` in all five seeds. This supports the idea that slower information propagation can prevent premature convergence in low-dimensional multimodal search.

![Rastrigin 2D convergence curves](presentation/topology_experiment/presentation_assets/key_figures/rastrigin_2d_seed11_convergence_curves.png)

The strongest result appeared on `dynamic_sphere`. `ring` was best in both 2D and 10D, and it beat `global` in all five seeds for both dimensions. This suggests that local communication improves robustness when the optimum moves, because the swarm avoids collapsing too quickly around stale information.

![Dynamic sphere 2D convergence curves](presentation/topology_experiment/presentation_assets/key_figures/dynamic_sphere_2d_seed11_convergence_curves.png)

However, the hypothesis did not hold uniformly. On `rastrigin 10D`, `global` performed best, with a mean final score of `-14.9262`, while local topologies were worse despite preserving more diversity. In higher-dimensional multimodal search, fast information sharing appears more valuable than maintaining broad spatial diversity.

![Dynamic sphere swarm behaviour](presentation/topology_experiment/presentation_assets/key_figures/side_by_side_dynamic_sphere.gif)

## Conclusions & Limitations

The main conclusion is that communication topology controls the exploration-exploitation balance of PSO. `global` creates high information pressure and fast exploitation, which is useful when good discoveries should spread quickly. `ring` and `von_neumann` slow down information propagation, preserve diversity, and improve adaptation on dynamic or low-dimensional multimodal problems. `knearest` often preserves diversity, but its performance payoff is less consistent.

There is no universally best topology. The best choice depends on whether the landscape rewards fast convergence, diversity preservation, or continued adaptation.

Important limitations remain. Only five seeds and ten evaluation episodes per condition were used, and no formal significance tests were performed. PPO did not consistently outperform the random baseline, so conclusions about topology are stronger than conclusions about learned control. Some information-spread metrics were weak or near-zero, making convergence and diversity metrics more reliable. Results also depend on the dynamic-sphere schedule, swarm size, training budget, PPO setup, and the limited set of tested landscapes.

## Future Work

Future work should run more seeds, more evaluation episodes, and paired statistical tests. The learned controller should be compared against stronger baselines, including fixed PSO, random control, and hand-tuned variants. Additional landscapes and dimensions would clarify whether the `rastrigin 10D` reversal is a general high-dimensional effect or specific to this benchmark. A promising extension is adaptive topology control, where the swarm switches between global and local communication depending on convergence, diversity, or environmental change.
