"""
Physical helpers package.
Keep this __init__ lightweight to avoid import-time failures.
"""

from .werner_utils import (
    secret_fraction_nat,
    binary_entropy_nat,
    swap_output_werner)

__all__ = [
    "secret_fraction_nat",
    "binary_entropy_nat",
    "swap_output_werner",
]