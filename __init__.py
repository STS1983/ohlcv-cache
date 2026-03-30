"""
🚀 OHLCV Cache – Production-Ready Compression
==============================================

High-performance OHLCV data compression for backtesting.

Features:
- 14x compression with delta encoding
- <0.05% reconstruction error
- Persistent storage (survives restarts)
- Query API (get exactly what you need)
- Incremental updates (O(1) append)
- Full test suite

Author: M.C.B. (for EvilStief's Evolution Trading)
Date: 2026-03-30
"""

from .compressor import OHLCVCompressor, DeltaEncoder
from .cache import OHLCVCache
from .config import CacheConfig
from .query import QueryBuilder

__version__ = '1.0.0'
__all__ = [
    'OHLCVCompressor',
    'DeltaEncoder',
    'OHLCVCache',
    'CacheConfig',
    'QueryBuilder'
]
