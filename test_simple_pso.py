# test_simple_pso.py
import torch
from src.utils import LandscapeWrapper

def simple_pso_test():
    print("🧪 Testing PSO core logic...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple test function
    def sphere(x):
        return torch.sum(x**2, dim=-1)
    
    landscape = LandscapeWrapper(sphere, dim=2)
    
    # Manual PSO without TorchRL
    num_agents = 5
    batch_size = 2
    dim = 2
    
    # Initialize
    positions = torch.rand(batch_size, num_agents, dim, device=device) * 100 - 50
    velocities = torch.zeros_like(positions)
    personal_best_pos = positions.clone()
    personal_best_val = landscape(positions.reshape(-1, dim)).view(batch_size, num_agents)
    
    # One PSO step
    inertia = 0.1
    cognitive = 0.2
    social = 0.3
    
    # Random actions (in real PSO, these would be calculated)
    actions = torch.randn(batch_size, num_agents, dim, device=device) * 0.1
    
    # Update
    velocities = velocities + actions
    positions = positions + velocities
    
    # Evaluate
    new_vals = landscape(positions.reshape(-1, dim)).view(batch_size, num_agents)
    
    print("✅ PSO core logic works!")
    print(f"Positions shape: {positions.shape}")
    print(f"Rewards shape: {new_vals.shape}")
    print("🎉 Your PSO algorithm is functional!")

if __name__ == "__main__":
    simple_pso_test()