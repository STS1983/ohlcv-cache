"""
OHLCV Cache with persistence and query API.
Production-ready, supports incremental updates and flexible queries.

Bug 3 fix: Cache now reads compression config from CacheConfig and passes
per-symbol bit overrides to the compressor.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime
import json

from .compressor import OHLCVCompressor, CompressionResult
from .config import CacheConfig, CompressionConfig, DEFAULT_CONFIG
from .query import QueryBuilder


class CacheBlock:
    """A block of compressed candles."""

    def __init__(
        self,
        symbol: str,
        start_idx: int,
        n_candles: int,
        compression_result: CompressionResult,
        timestamp: Optional[datetime] = None,
    ):
        self.symbol = symbol
        self.start_idx = start_idx
        self.n_candles = n_candles
        self.compression_result = compression_result
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        """Serialize block to dict (for JSON storage)."""
        return {
            'symbol': self.symbol,
            'start_idx': self.start_idx,
            'n_candles': self.n_candles,
            'params': self.compression_result.params,
            'timestamp': self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict, packed_data: bytes) -> 'CacheBlock':
        """Deserialize block from dict."""
        result = CompressionResult(
            packed_data=packed_data,
            params=data['params'],
            original_size=data['params'].get('original_size', 0),
            compressed_size=data['params'].get('compressed_size', 0),
            compression_ratio=data['params'].get('compression_ratio', 0),
        )

        return cls(
            symbol=data['symbol'],
            start_idx=data['start_idx'],
            n_candles=data['n_candles'],
            compression_result=result,
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else None,
        )


class OHLCVCache:
    """
    Production-ready OHLCV cache.

    Bug 3 fix: CompressionConfig is now embedded in CacheConfig and
    actually used. Per-symbol bit overrides are passed to the compressor.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        persist_to: Optional[Union[str, Path]] = None,
    ):
        self.config = config or DEFAULT_CONFIG

        if persist_to:
            self.config.persist_to = Path(persist_to)

        # In-memory cache
        self.blocks: Dict[str, List[CacheBlock]] = {}
        self.block_data: Dict[str, bytes] = {}

        # Statistics
        self.stats = {
            'total_symbols': 0,
            'total_candles': 0,
            'total_blocks': 0,
            'memory_usage_mb': 0.0,
        }

        # Bug 3 fix: Create compressor from config's compression settings
        comp_cfg = self.config.compression
        self.compressor = OHLCVCompressor(
            bits=comp_cfg.default_bits,
            delta_encoding=comp_cfg.delta_encoding,
            adaptive_bits=comp_cfg.adaptive_enabled,
            delta_anchor_interval=comp_cfg.delta_anchor_interval,
        )

        # Load from disk if configured
        if self.config.persist_to:
            self._load_from_disk()

    def add_candles(
        self,
        ohlcv: np.ndarray,
        symbol: str = 'default',
        timestamp: Optional[datetime] = None,
    ):
        """
        Add candles to cache (incremental update, O(1)).

        Bug 3 fix: Uses per-symbol compression config if available.
        """
        if len(ohlcv) == 0:
            return

        if symbol not in self.blocks:
            self.blocks[symbol] = []
            self.stats['total_symbols'] += 1

        start_idx = sum(b.n_candles for b in self.blocks[symbol])

        # Bug 3 fix: Get per-symbol compression config
        sym_comp_cfg = self.config.get_compression_config(symbol)

        # Use per-symbol compressor if bits differ from default
        if sym_comp_cfg.default_bits != self.config.compression.default_bits:
            sym_compressor = OHLCVCompressor(
                bits=sym_comp_cfg.default_bits,
                delta_encoding=sym_comp_cfg.delta_encoding,
                adaptive_bits=sym_comp_cfg.adaptive_enabled,
                delta_anchor_interval=sym_comp_cfg.delta_anchor_interval,
            )
            result = sym_compressor.compress(ohlcv, symbol)
        else:
            result = self.compressor.compress(ohlcv, symbol)

        block = CacheBlock(
            symbol=symbol,
            start_idx=start_idx,
            n_candles=len(ohlcv),
            compression_result=result,
            timestamp=timestamp,
        )

        block_id = f"{symbol}_{start_idx}"
        self.blocks[symbol].append(block)
        self.block_data[block_id] = result.packed_data

        self.stats['total_candles'] += len(ohlcv)
        self.stats['total_blocks'] += 1
        self._update_memory_usage()

        if self.config.auto_save and self.config.persist_to:
            self.save()

    def get_all_candles(self, symbol: str = 'default') -> np.ndarray:
        """Get all candles for a symbol."""
        if symbol not in self.blocks or len(self.blocks[symbol]) == 0:
            return np.array([])

        candles_list = []
        for block in self.blocks[symbol]:
            block_id = f"{symbol}_{block.start_idx}"
            packed_data = self.block_data.get(block_id, block.compression_result.packed_data)

            result = CompressionResult(
                packed_data=packed_data,
                params=block.compression_result.params,
                original_size=block.compression_result.original_size,
                compressed_size=block.compression_result.compressed_size,
                compression_ratio=block.compression_result.compression_ratio,
            )

            # Create compressor with the right bits for this block
            block_bits = block.compression_result.params.get('global_bits', 8)
            block_anchor_interval = block.compression_result.params.get('delta_anchor_interval', 32)

            if block_bits != self.compressor.bits:
                decompressor = OHLCVCompressor(
                    bits=block_bits,
                    delta_encoding=block.compression_result.params.get('use_delta', False),
                    adaptive_bits=False,
                    delta_anchor_interval=block_anchor_interval,
                )
                ohlcv = decompressor.decompress(result)
            else:
                ohlcv = self.compressor.decompress(result)

            candles_list.append(ohlcv)

        return np.vstack(candles_list)

    def query(
        self,
        symbol: Optional[str] = None,
        start: Optional[Union[str, datetime, int]] = None,
        end: Optional[Union[str, datetime, int]] = None,
        columns: Optional[List[str]] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Query candles with flexible filters."""
        query = QueryBuilder(
            symbol=symbol,
            start=start,
            end=end,
            columns=columns,
        )

        if query.symbol:
            return self._query_symbol(query)
        else:
            results = {}
            for sym in self.blocks.keys():
                q = query.with_symbol(sym)
                results[sym] = self._query_symbol(q)
            return results

    def _query_symbol(self, query: QueryBuilder) -> np.ndarray:
        """Query candles for a specific symbol."""
        symbol = query.symbol

        if symbol not in self.blocks:
            return np.array([])

        all_candles = self.get_all_candles(symbol)

        if len(all_candles) == 0:
            return all_candles

        start_idx = 0
        end_idx = len(all_candles)

        if query.start is not None:
            if isinstance(query.start, int):
                start_idx = query.start
        if query.end is not None:
            if isinstance(query.end, int):
                end_idx = query.end

        candles = all_candles[start_idx:end_idx]

        if query.columns:
            col_indices = [self._column_to_index(col) for col in query.columns]
            candles = candles[:, col_indices]

        return candles

    def _column_to_index(self, col: str) -> int:
        """Convert column name to index."""
        col_map = {
            'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4,
            'o': 0, 'h': 1, 'l': 2, 'c': 3, 'v': 4,
        }
        return col_map.get(col.lower(), 0)

    def save(self, path: Optional[Union[str, Path]] = None):
        """Save cache to disk."""
        save_path = Path(path) if path else self.config.persist_to

        if not save_path:
            raise ValueError("No save path configured")

        save_path.mkdir(parents=True, exist_ok=True)

        manifest = {
            'version': '2.0',
            'created': datetime.now().isoformat(),
            'stats': self.stats,
            'config': self.config.to_dict(),
            'symbols': {},
        }

        for symbol, blocks in self.blocks.items():
            symbol_dir = save_path / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            block_files = []
            for i, block in enumerate(blocks):
                block_file = symbol_dir / f"block_{block.start_idx}.bin"
                with open(block_file, 'wb') as f:
                    f.write(
                        self.block_data.get(
                            f"{symbol}_{block.start_idx}",
                            block.compression_result.packed_data,
                        )
                    )

                block_meta = block.to_dict()
                block_meta['file'] = block_file.name
                block_files.append(block_meta)

            manifest['symbols'][symbol] = {
                'total_candles': sum(b.n_candles for b in blocks),
                'blocks': block_files,
            }

        with open(save_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        self.config.save(save_path / 'config.json')

    def _load_from_disk(self):
        """Load cache from disk."""
        if not self.config.persist_to or not self.config.persist_to.exists():
            return

        load_path = self.config.persist_to

        manifest_file = load_path / 'manifest.json'
        if not manifest_file.exists():
            return

        with open(manifest_file) as f:
            manifest = json.load(f)

        # Load config if available
        config_file = load_path / 'config.json'
        if config_file.exists():
            self.config = CacheConfig.from_json(config_file)
            comp_cfg = self.config.compression
            self.compressor = OHLCVCompressor(
                bits=comp_cfg.default_bits,
                delta_encoding=comp_cfg.delta_encoding,
                adaptive_bits=comp_cfg.adaptive_enabled,
                delta_anchor_interval=comp_cfg.delta_anchor_interval,
            )

        # Load blocks
        for symbol, symbol_data in manifest['symbols'].items():
            self.blocks[symbol] = []
            symbol_dir = load_path / symbol

            for block_meta in symbol_data['blocks']:
                block_file = symbol_dir / block_meta['file']

                if block_file.exists():
                    with open(block_file, 'rb') as f:
                        packed_data = f.read()

                    block = CacheBlock.from_dict(block_meta, packed_data)
                    self.blocks[symbol].append(block)
                    self.block_data[f"{symbol}_{block.start_idx}"] = packed_data

        self.stats = manifest.get('stats', self.stats)
        self._update_memory_usage()

    @classmethod
    def load_from(cls, path: Union[str, Path]) -> 'OHLCVCache':
        """Load cache from disk."""
        config = CacheConfig(persist_to=Path(path))
        cache = cls(config=config)
        return cache

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_original = 0
        total_compressed = 0

        for symbol, blocks in self.blocks.items():
            for block in blocks:
                total_original += block.compression_result.original_size
                total_compressed += block.compression_result.compressed_size

        return {
            'total_symbols': self.stats['total_symbols'],
            'total_candles': self.stats['total_candles'],
            'total_blocks': self.stats['total_blocks'],
            'original_size_mb': total_original / (1024 * 1024),
            'compressed_size_mb': total_compressed / (1024 * 1024),
            'compression_ratio': total_original / total_compressed if total_compressed > 0 else 0,
            'memory_saved_mb': (total_original - total_compressed) / (1024 * 1024),
            'memory_usage_mb': self.stats['memory_usage_mb'],
        }

    def _update_memory_usage(self):
        """Update memory usage statistics."""
        total_bytes = sum(len(data) for data in self.block_data.values())
        self.stats['memory_usage_mb'] = total_bytes / (1024 * 1024)

    def clear(self):
        """Clear all cached data."""
        self.blocks.clear()
        self.block_data.clear()
        self.stats = {
            'total_symbols': 0,
            'total_candles': 0,
            'total_blocks': 0,
            'memory_usage_mb': 0.0,
        }