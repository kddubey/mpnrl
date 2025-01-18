"""
Multiple Positives and Negatives Ranking Loss
"""

__version__ = "0.1.0"

from . import data_collator
from . import losses

__all__ = [
    "data_collator",
    "losses",
]
