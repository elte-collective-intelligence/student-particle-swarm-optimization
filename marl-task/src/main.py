import torch
from tensordict import TensorDict
from envs import LandscapeWrapper, PSOEnv

def eggholder(x: torch.Tensor) -> torch.Tensor:
    """
    Eggholder test function generalized for even-dimensional input.
    Args:
        x (torch.Tensor): Input tensor of shape (..., d), where d is even.
    Returns:
        torch.Tensor: Function value for each input in the batch.
    """
    # Ensure input has even number of dimensions
    if x.shape[-1] % 2 != 0:
        raise ValueError("Eggholder function requires even-dimensional input.")

    # Reshape to (..., d//2, 2) to pair up variables
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_i = x_pairs[..., 0]
    x_j = x_pairs[..., 1]

    term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
    term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
    result = term1 + term2

    # Sum over all pairs for each sample
    return result.sum(dim=-1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    landscape_function = LandscapeWrapper(eggholder, dim=2)
    env = PSOEnv(landscape=landscape_function,
                 num_agents=10,
                 device=device,
                 batch_size=(32,))
    env.reset()
    action = TensorDict({
        "inertia": torch.rand((32, 10, 2), device=device),
        "cognitive": torch.rand((32, 10, 2), device=device),
        "social": torch.rand((32, 10, 2), device=device)
    }, batch_size=(32,))
    print(env.step(action))

if __name__ == "__main__":
    main()

    