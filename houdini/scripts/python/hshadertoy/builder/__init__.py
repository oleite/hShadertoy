"""
hShadertoy Builder - Because manually wiring COP networks is for masochists.

This module creates and configures hShadertoy HDA nodes from Shadertoy API JSON.
It's like IKEA furniture but the instructions actually make sense.
"""

from .builder import build_shadertoy_hda

__all__ = ['build_shadertoy_hda']
