import torch


def sphere(x: torch.Tensor) -> torch.Tensor:
    """Sphere: f(x) = sum(x²). Optimum 0 at origin."""
    return -torch.sum(x ** 2, dim=-1)


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    """Rosenbrock (banana): optimum 0 at (1,...,1)."""
    xi  = x[..., :-1]
    xi1 = x[..., 1:]
    return -(100 * (xi1 - xi ** 2) ** 2 + (1 - xi) ** 2).sum(dim=-1)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Rastrigin: highly multimodal."""
    A = 10
    return -(A * x.shape[-1] + torch.sum(x ** 2 - A * torch.cos(2 * 3.14159 * x), dim=-1))


def eggholder(x: torch.Tensor) -> torch.Tensor:
    """Eggholder: requires even-dimensional input."""
    if x.shape[-1] % 2 != 0:
        raise ValueError("Eggholder requires even-dimensional input.")
    x_pairs = x.view(*x.shape[:-1], -1, 2)
    x_i, x_j = x_pairs[..., 0], x_pairs[..., 1]
    term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
    term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
    return (term1 + term2).sum(dim=-1)


STATIC_FUNCTIONS = {
    "sphere":     sphere,
    "rosenbrock": rosenbrock,
    "rastrigin":  rastrigin,
    "eggholder":  eggholder,
}
