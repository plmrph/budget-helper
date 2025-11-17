"""
PXBlendSC-RF ML Strategy implementation.

This module provides backward compatibility by importing from the refactored module structure.
The actual implementation is now in the pxblendsc_strategy package.
"""

# Import the main strategy class from the refactored module
from .pxblendsc_strategy import PXBlendSCStrategy

__all__ = ["PXBlendSCStrategy"]
