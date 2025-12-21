# src/utils.py
import torch
import torch.nn as nn
from tensordict import TensorDict


class LandscapeWrapper:
    """
    Wrap a callable function f(x: (..., dim)) -> (...). Optionally support dynamic changes by implementing reset(step=...).
    """

    def __init__(self, function, dim, dynamic=False):
        self.function = function
        self.dim = dim
        self.dynamic = dynamic
        # example internal state for moving_optimum
        self.step = 0
        self.offset = torch.zeros(self.dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        # call underlying function; the function should be implemented to accept x and use self.offset if needed
        return self.function(x)

    def reset(self, step: int = 0):
        self.step = step
        # If the function supports dynamic changes, the user-defined function may check LandscapeWrapper.step or offset
        # You can update offset here for dynamic behaviors
        # Example: self.offset = 5.0 * torch.sin(step / 10)
        return


class PSOObservationWrapper(nn.Module):
    """
    Input: tensordict with ("agents","obs") -> (B,N,obs_dim)
    Output: ("agents","agent_input") -> flattened vector per agent (B*N, input_dim) or keep (B,N,input_dim)
    For compatibility we will produce (B,N,input_dim) - further modules can flatten as needed.
    """

    def __init__(self):
        super().__init__()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if ("agents", "obs") not in tensordict:
            raise KeyError("PSOObservationWrapper expects ('agents','obs') key in TensorDict")
        obs = tensordict.get(("agents", "obs"))  # (B,N,obs_dim)
        # pass-through for now; actor may expect flattened
        tensordict.set(("agents", "agent_input"), obs)
        return tensordict


class PSOActionExtractor(nn.Module):
    """
    After NormalParamExtractor we usually have loc, scale tensors.
    This extractor will sample or convert params to action coefficients and put them into ("agents","action").
    If ProbabilisticActor + CompositeDistribution is used, ensure the actor writes ("params", <name>, <loc/scale>) keys.
    For simpler pipeline we accept a pair (loc, scale) and produce action = loc (deterministic) or sampled.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # dimensionality of the search space (e.g. 2)
        # But for action coefficients we only need 3 scalars per agent (inertia, cognitive, social)
        # So we expect loc/scale to have shape (B,N,3)
        # This extractor converts them into ("agents","action") shape (B,N,3)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # This extractor expects that upstream NormalParamExtractor or network placed loc/scale under ("params","loc") etc.
        # But to be robust, accept a direct key ("agents","raw_action_params") as well.
        if ("agents", "raw_action_params") in tensordict:
            raw = tensordict.get(("agents", "raw_action_params"))
            tensordict.set(("agents", "action"), raw)
            return tensordict

        # Try to read ("params","loc") and ("params","scale")
        if ("params", "loc") in tensordict and ("params", "scale") in tensordict:
            loc = tensordict.get(("params", "loc"))
            # if loc is (B,N,3) -> set as action
            tensordict.set(("agents", "action"), loc)
            return tensordict

        # Fallback: do nothing
        raise KeyError("PSOActionExtractor: couldn't find input params. Expected ('agents','raw_action_params') or ('params','loc').")
