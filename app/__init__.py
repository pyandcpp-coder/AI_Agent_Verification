"""
Verification Agent Package

This package contains the core agents for document verification:
- EntityAgent: Extracts and validates entity information from documents
- FaceAgent: Handles face similarity and matching
- DocAgent: Detects document tampering and forgery
- GenderPipeline: Gender detection and verification
"""

__version__ = "1.0.0"

__all__ = [
    "EntityAgent",
    "FaceAgent", 
    "DocAgent",
    "GenderPipeline",
]
