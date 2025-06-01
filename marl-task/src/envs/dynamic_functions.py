import torch
import math

class DynamicSphere:
    """
    DynamicSphere is a time-varying variant of the classic Sphere function.

    The global optimum of the function moves over time in the first two dimensions,
    following a circular trajectory determined by the `shift_speed` parameter.
    This makes the optimization landscape non-stationary, which is useful for benchmarking
    dynamic optimization algorithms such as PSO.

    f(x, t) = -sum((x - shift(t))^2)
    where shift(t) = [sin(t * shift_speed), cos(t * shift_speed), 0, ..., 0]

    Args:
        dim (int): Dimensionality of the input space.
        shift_speed (float): Speed at which the optimum moves in the first two dimensions.

    Attributes:
        dim (int): Dimensionality of the input space.
        shift_speed (float): Speed of optimum movement.
        optimum (torch.Tensor): Current optimum position (always zeros after reset).
        time (int): Current time step.

    Methods:
        __call__(x): Evaluate the function at positions x, incrementing time.
        reset(): Reset the time and optimum to initial state.
    """
    def __init__(self, dim: int, shift_speed: float = 0.1):
        self.dim = dim
        self.shift_speed = shift_speed
        self.optimum = torch.zeros(dim) 
        self.time = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the dynamic Sphere function at positions x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Function values at x, shape (...,).
        """
        self.time += 1
        shift = torch.tensor([
            math.sin(self.time * self.shift_speed),
            math.cos(self.time * self.shift_speed),
            *[0.0] * (self.dim - 2) 
        ], dtype=x.dtype, device=x.device)
        
        shifted_x = x - shift
        return -torch.sum(shifted_x ** 2, dim=-1) 

    def reset(self):
        """
        Reset the function's time and optimum to their initial state.
        """
        self.time = 0
        self.optimum = torch.zeros(self.dim)


class DynamicRastrigin:
    """
    DynamicRastrigin is a time-varying variant of the classic Rastrigin function.

    The amplitude of the cosine modulation oscillates over time, making the landscape
    more or less rugged as time progresses. This is controlled by the `frequency` parameter.

    f(x, t) = -[A(t) * dim + sum(x^2 - A(t) * cos(2*pi*x))]
    where A(t) = 10 * (1 + 0.5 * sin(t * frequency))

    Args:
        dim (int): Dimensionality of the input space.
        frequency (float): Frequency of the amplitude oscillation.

    Attributes:
        dim (int): Dimensionality of the input space.
        frequency (float): Frequency of amplitude oscillation.
        time (int): Current time step.

    Methods:
        __call__(x): Evaluate the function at positions x, incrementing time.
        reset(): Reset the time to initial state.
    """
    def __init__(self, dim: int, frequency: float = 0.05):
        self.dim = dim
        self.frequency = frequency
        self.time = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the dynamic Rastrigin function at positions x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).

        Returns:
            torch.Tensor: Function values at x, shape (...,).
        """
        self.time += 1
        A = 10 * (1 + 0.5 * math.sin(self.time * self.frequency))  # Time-varying amplitude
        
        return -(A * self.dim + torch.sum(x**2 - A * torch.cos(2 * math.pi * x), dim=-1))

    def reset(self):
        """
        Reset the function's time to its initial state.
        """
        self.time = 0


class DynamicEggHolder:
    """
    DynamicEggHolder is a time-varying variant of the Eggholder function in 2D.

    The coordinate system is rotated at each time step, making the landscape
    non-stationary and more challenging for optimization algorithms.
    The rotation speed is controlled by the `rotation_speed` parameter.

    Only works for 2D input.

    f(x, t) = sum( -[x_j + 47] * sin(sqrt(|x_j + x_i/2 + 47|))
                   - x_i * sin(sqrt(|x_i - (x_j + 47)|)) )
    where [x_i, x_j] are the rotated coordinates.

    Args:
        dim (int): Dimensionality of the input space (must be 2).
        rotation_speed (float): Speed of coordinate system rotation.

    Attributes:
        dim (int): Dimensionality of the input space (always 2).
        rotation_speed (float): Speed of rotation.
        time (int): Current time step.

    Methods:
        __call__(x): Evaluate the function at positions x, incrementing time.
        reset(): Reset the time to initial state.
    """
    def __init__(self, dim: int = 2, rotation_speed: float = 0.01):
        assert dim == 2, "DynamicEggholder only works in 2D"
        self.dim = dim
        self.rotation_speed = rotation_speed
        self.time = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the dynamic Eggholder function at positions x.

        Args:
            x (torch.Tensor): Input tensor of shape (..., 2).

        Returns:
            torch.Tensor: Function values at x, shape (...,).
        """
        self.time += 1
        theta = self.time * self.rotation_speed
        
        x_rot = x.clone()
        x_rot[..., 0] = x[..., 0] * math.cos(theta) - x[..., 1] * math.sin(theta)
        x_rot[..., 1] = x[..., 0] * math.sin(theta) + x[..., 1] * math.cos(theta)
        
        x_pairs = x_rot.view(*x_rot.shape[:-1], -1, 2)
        x_i, x_j = x_pairs[..., 0], x_pairs[..., 1]
        
        term1 = -(x_j + 47) * torch.sin(torch.sqrt(torch.abs(x_j + x_i / 2 + 47)))
        term2 = -x_i * torch.sin(torch.sqrt(torch.abs(x_i - (x_j + 47))))
        return (term1 + term2).sum(dim=-1)

    def reset(self):
        """
        Reset the function's time to its initial state.
        """
        self.time = 0