"""
🚀 OHLCV Cache – Production-Ready Compression
==============================================

High-performance OHLCV data compression for backtesting.

Features:
- Anchor-based delta encoding (bounded error)
- Multi-bit quantization (8-16 bits per column)
- Adaptive bit-depth based on volatility
- Persistent storage (survives restarts)
- Query API (get exactly what you need)
- Incremental updates (O(1) append)
- Full test suite

Author: M.C.B. (for EvilStief's Evolution Trading)
Date: 2026-03-30
Updated: 2026-10-10 — Fixed 4 critical bugs
"""

from .compressor import OHLCVCompressor, DeltaEncoder, AnchorDeltaEncoder, CompressionResult
from .cache import OHLCVCache, CacheBlock
from .config import CacheConfig, CompressionConfig, DEFAULT_CONFIG
from .query import QueryBuilder

__version__ = '2.0.0'
__all__ = [
    'OHLCVCompressor',
    'DeltaEncoder',
    'AnchorDeltaEncoder',
    'CompressionResult',
    'OHLCVCache',
    'CacheBlock',
    'CacheConfig',
    'CompressionConfig',
    'DEFAULT_CONFIG',
    'QueryBuilder',
]