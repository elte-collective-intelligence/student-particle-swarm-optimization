"""
Visualization module for PSO environment.

This module handles all rendering, plotting, and GIF generation for particle swarm
optimization, including 2D and 3D swarm visualizations with landscape contours.
"""

import os
from typing import List, Callable

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


class SwarmVisualizer:
    """Handles visualization and rendering for PSO swarm optimization."""

    def __init__(self, vis_config: dict, landscape_fn: Callable, dim: int):
        """
        Initialize the visualizer.

        Args:
            vis_config: Visualization configuration dictionary
            landscape_fn: Landscape function to visualize
            dim: Dimensionality of the search space (2 or 3 for visualization)
        """
        self.vis_config = vis_config
        self.landscape_fn = landscape_fn
        self.dim = dim
        self.visualize_swarm = vis_config.get("visualize_swarm", False)
        self.visualize_landscape = vis_config.get("visualize_landscape", True)
        self.save_gif = vis_config.get("save_gif", False)
        self.save_dir = vis_config.get("save_dir", "outputs/vis/")
        self.fps = vis_config.get("fps", 10)
        self.dpi = vis_config.get("dpi", 150)

        # Storage for animation frames
        self.frames: List[dict] = []
        self.episode = 0

        # Landscape bounds (default for typical test functions)
        self.bounds = (-5.12, 5.12)

        os.makedirs(self.save_dir, exist_ok=True)

    def set_bounds(self, lower: float, upper: float):
        """Set the visualization bounds for the search space."""
        self.bounds = (lower, upper)

    def record_frame(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        personal_bests: torch.Tensor,
        global_best: torch.Tensor,
        scores: torch.Tensor,
        timestep: int,
    ):
        """
        Record a frame for animation.

        Args:
            positions: Particle positions [batch, agents, dim]
            velocities: Particle velocities [batch, agents, dim]
            personal_bests: Personal best positions [batch, agents, dim]
            global_best: Global best position [batch, dim]
            scores: Current scores [batch, agents]
            timestep: Current timestep
        """
        if not self.visualize_swarm:
            return

        # Take first batch element for visualization
        frame_data = {
            "positions": positions[0].cpu().numpy(),
            "velocities": velocities[0].cpu().numpy(),
            "personal_bests": personal_bests[0].cpu().numpy(),
            "global_best": global_best[0].cpu().numpy(),
            "scores": scores[0].cpu().numpy(),
            "timestep": timestep,
        }
        self.frames.append(frame_data)

    def reset(self, episode: int = 0):
        """Reset frame storage for a new episode."""
        self.frames = []
        self.episode = episode

    def _create_landscape_grid(self, resolution: int = 50):
        """Create a grid for landscape visualization."""
        x = np.linspace(self.bounds[0], self.bounds[1], resolution)
        y = np.linspace(self.bounds[0], self.bounds[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate landscape on grid
        points = torch.tensor(
            np.stack([X.ravel(), Y.ravel()], axis=-1), dtype=torch.float32
        )
        Z = self.landscape_fn(points).numpy().reshape(X.shape)

        return X, Y, Z

    def create_2d_animation(self, filename: str = None) -> str:
        """
        Create a 2D animation of particle movement.

        Args:
            filename: Output filename (without extension)

        Returns:
            Path to the saved GIF
        """
        if not self.frames or self.dim < 2:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create landscape contours if enabled
        if self.visualize_landscape and self.dim == 2:
            X, Y, Z = self._create_landscape_grid()
            contour = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.7)
            plt.colorbar(contour, ax=ax, label="Fitness (higher = better)")

        # Initialize scatter plots
        positions = self.frames[0]["positions"][:, :2]  # First 2 dims
        (particles,) = ax.plot(
            positions[:, 0], positions[:, 1], "ko", markersize=8, label="Particles"
        )
        (velocities,) = ax.plot([], [], "b-", alpha=0.5, linewidth=1)
        (global_best,) = ax.plot(
            [], [], "r*", markersize=15, label="Global Best", zorder=10
        )
        (personal_bests,) = ax.plot(
            [], [], "g^", markersize=6, alpha=0.5, label="Personal Bests"
        )

        ax.set_xlim(self.bounds)
        ax.set_ylim(self.bounds)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(loc="upper right")
        title = ax.set_title("PSO Swarm - Step 0")

        def update(frame_idx):
            frame = self.frames[frame_idx]
            pos = frame["positions"][:, :2]
            vel = frame["velocities"][:, :2]
            pb = frame["personal_bests"][:, :2]
            gb = frame["global_best"][:2]

            # Update particles
            particles.set_data(pos[:, 0], pos[:, 1])

            # Update velocity arrows (as lines from position)
            vel_lines_x = []
            vel_lines_y = []
            for i in range(len(pos)):
                vel_lines_x.extend([pos[i, 0], pos[i, 0] + vel[i, 0] * 0.5, np.nan])
                vel_lines_y.extend([pos[i, 1], pos[i, 1] + vel[i, 1] * 0.5, np.nan])
            velocities.set_data(vel_lines_x, vel_lines_y)

            # Update bests
            global_best.set_data([gb[0]], [gb[1]])
            personal_bests.set_data(pb[:, 0], pb[:, 1])

            title.set_text(f"PSO Swarm - Step {frame['timestep']}")

            return particles, velocities, global_best, personal_bests, title

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.frames), interval=1000 // self.fps, blit=False
        )

        # Save animation
        if filename is None:
            filename = f"swarm_2d_ep{self.episode}"

        gif_path = os.path.join(self.save_dir, f"{filename}.gif")
        ani.save(gif_path, writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(fig)

        return gif_path

    def create_3d_animation(self, filename: str = None) -> str:
        """
        Create a 3D animation of particle movement on the landscape surface.

        Args:
            filename: Output filename (without extension)

        Returns:
            Path to the saved GIF
        """
        if not self.frames or self.dim < 2:
            return None

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Create landscape surface
        X, Y, Z = self._create_landscape_grid(resolution=40)
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, linewidth=0)

        # Initialize particle scatter (will be redrawn in update)
        positions = self.frames[0]["positions"][:, :2]
        scores = self.frames[0]["scores"]
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            scores,
            c="red",
            s=50,
            marker="o",
            label="Particles",
            depthshade=True,
        )

        ax.set_xlim(self.bounds)
        ax.set_ylim(self.bounds)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("Fitness")
        ax.legend(loc="upper right")
        ax.set_title("PSO Swarm 3D - Step 0")

        def update(frame_idx):
            ax.clear()

            # Redraw surface
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5, linewidth=0)

            frame = self.frames[frame_idx]
            pos = frame["positions"][:, :2]
            scores = frame["scores"]
            gb = frame["global_best"][:2]

            # Compute global best score
            gb_tensor = torch.tensor(gb, dtype=torch.float32).unsqueeze(0)
            gb_score = self.landscape_fn(gb_tensor).numpy()[0]

            # Plot particles on surface
            ax.scatter(
                pos[:, 0],
                pos[:, 1],
                scores,
                c="red",
                s=50,
                marker="o",
                depthshade=True,
            )

            # Plot global best
            ax.scatter([gb[0]], [gb[1]], [gb_score], c="yellow", s=200, marker="*")

            ax.set_xlim(self.bounds)
            ax.set_ylim(self.bounds)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_zlabel("Fitness")
            ax.set_title(f"PSO Swarm 3D - Step {frame['timestep']}")

            # Rotate view slightly
            ax.view_init(elev=30, azim=frame_idx * 2)

        ani = animation.FuncAnimation(
            fig, update, frames=len(self.frames), interval=1000 // self.fps, blit=False
        )

        # Save animation
        if filename is None:
            filename = f"swarm_3d_ep{self.episode}"

        gif_path = os.path.join(self.save_dir, f"{filename}.gif")
        ani.save(gif_path, writer="pillow", fps=self.fps, dpi=self.dpi)
        plt.close(fig)

        return gif_path

    def create_convergence_plot(
        self, best_scores: List[float], mean_scores: List[float], filename: str = None
    ) -> str:
        """
        Create a convergence plot showing score improvement over time.

        Args:
            best_scores: List of best scores per timestep
            mean_scores: List of mean scores per timestep
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        timesteps = range(len(best_scores))
        ax.plot(timesteps, best_scores, "b-", linewidth=2, label="Global Best")
        ax.plot(timesteps, mean_scores, "g--", linewidth=1, alpha=0.7, label="Mean")

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Score (higher = better)")
        ax.set_title("PSO Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if filename is None:
            filename = f"convergence_ep{self.episode}"

        plot_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return plot_path

    def create_trajectory_plot(self, filename: str = None) -> str:
        """
        Create a 2D plot showing particle trajectories over time.

        Args:
            filename: Output filename

        Returns:
            Path to saved plot
        """
        if not self.frames or self.dim < 2:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw landscape contours
        if self.visualize_landscape and self.dim == 2:
            X, Y, Z = self._create_landscape_grid()
            ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.5)

        # Extract trajectories for each particle
        n_particles = self.frames[0]["positions"].shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, n_particles))

        for p in range(n_particles):
            traj_x = [f["positions"][p, 0] for f in self.frames]
            traj_y = [f["positions"][p, 1] for f in self.frames]

            ax.plot(
                traj_x, traj_y, "-", color=colors[p], alpha=0.6, linewidth=1, zorder=1
            )
            ax.scatter(
                traj_x[0], traj_y[0], c=[colors[p]], marker="o", s=40, zorder=2
            )  # Start
            ax.scatter(
                traj_x[-1], traj_y[-1], c=[colors[p]], marker="s", s=60, zorder=3
            )  # End

        # Mark global best
        final_gb = self.frames[-1]["global_best"][:2]
        ax.scatter(
            [final_gb[0]],
            [final_gb[1]],
            c="yellow",
            marker="*",
            s=200,
            edgecolors="black",
            zorder=10,
            label="Final Global Best",
        )

        ax.set_xlim(self.bounds)
        ax.set_ylim(self.bounds)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title("Particle Trajectories")
        ax.legend()

        if filename is None:
            filename = f"trajectories_ep{self.episode}"

        plot_path = os.path.join(self.save_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return plot_path

    def save_all_visualizations(
        self, best_scores: List = None, mean_scores: List = None
    ):
        """
        Generate and save all available visualizations.

        Args:
            best_scores: Optional list of best scores for convergence plot
            mean_scores: Optional list of mean scores for convergence plot

        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}

        if self.visualize_swarm and self.frames:
            # 2D animation
            if self.dim >= 2:
                path = self.create_2d_animation()
                if path:
                    saved_files["animation_2d"] = path
                    print(f"Saved 2D animation: {path}")

            # 3D animation (only for 2D search space)
            if self.dim == 2:
                path = self.create_3d_animation()
                if path:
                    saved_files["animation_3d"] = path
                    print(f"Saved 3D animation: {path}")

            # Trajectory plot
            path = self.create_trajectory_plot()
            if path:
                saved_files["trajectories"] = path
                print(f"Saved trajectory plot: {path}")

        # Convergence plot
        if best_scores and mean_scores:
            path = self.create_convergence_plot(best_scores, mean_scores)
            if path:
                saved_files["convergence"] = path
                print(f"Saved convergence plot: {path}")

        return saved_files
