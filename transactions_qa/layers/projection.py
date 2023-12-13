import torch.nn as nn
from torch import TensorType
from collections import OrderedDict

from romashka.logging_handler import get_logger

logger = get_logger(
    name="Projection"
)


class IdentityProjection(nn.Module):
    """
    Identity projection (no changes to input tensor).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Identity()

    def forward(self, x: TensorType):
        return self.proj(x)


class LinearProjection(nn.Module):
    """
    Linear projection.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: TensorType):
        return self.proj(x)


class MLPProjection(nn.Module):
    """
    MLP projection.
    """
    def __init__(self, in_dim: int, hidden_size: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, out_dim, bias=False),
            )

    def forward(self, x: TensorType):
        return self.proj(x)


PROJECTIONS_TYPES = [
    ("IDENTITY", IdentityProjection),
    ("LINEAR", LinearProjection),
    ("MLP", MLPProjection)
]
PROJECTIONS_TYPES = OrderedDict(PROJECTIONS_TYPES)


class ProjectionsType:
    """
    Selector class for specific projection types.
    """
    @classmethod
    def get(cls, projection_type_name: str, **kwargs):
        try:
            return PROJECTIONS_TYPES[projection_type_name](**kwargs)
        except Exception as e:
            logger.error(f"Error during PoolerType creation with `projection_type_name`-`{projection_type_name}`\n:{e}")
            raise ValueError(f"Error during PoolerType creation with `projection_type_name`-`{projection_type_name}`\n:{e}")

    @classmethod
    def get_available_names(cls):
        """
        Returns a list of available enumeration name.
        """
        return [member for member in PROJECTIONS_TYPES.keys()]

    @classmethod
    def to_str(cls):
        s = " / ".join([member for member in PROJECTIONS_TYPES.keys()])
        return s