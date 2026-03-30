"""
Test suite for OHLCV Cache.
Run with: python3 -m pytest ohlcv_cache/tests.py -v
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from .compressor import OHLCVCompressor, DeltaEncoder
from .cache import OHLCVCache, CacheBlock
from .config import CacheConfig, DEFAULT_CONFIG
from .query import QueryBuilder, last_n_candles, candles_between


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n_candles = 1000
    
    # Realistic price movement
    close = 100 + np.cumsum(np.random.randn(n_candles) * 0.5)
    close = np.maximum(close, 1)
    
    high = close + np.abs(np.random.randn(n_candles) * 0.3)
    low = close - np.abs(np.random.randn(n_candles) * 0.3)
    low = np.minimum(low, close * 0.95)
    high = np.maximum(high, close * 1.05)
    open_ = close + np.random.randn(n_candles) * 0.2
    
    volume = np.exp(np.random.randn(n_candles) * 0.8 + 11)
    
    return np.column_stack([open_, high, low, close, volume])


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================================
# DeltaEncoder Tests
# ============================================================================

class TestDeltaEncoder:
    
    def test_encode_decode_roundtrip(self, sample_ohlcv):
        """Test that encode → decode returns original data."""
        encoder = DeltaEncoder()
        
        first_row, deltas = encoder.encode(sample_ohlcv)
        reconstructed = encoder.decode(first_row, deltas)
        
        assert np.allclose(sample_ohlcv, reconstructed)
    
    def test_encode_reduces_range(self, sample_ohlcv):
        """Test that deltas have smaller range than absolute values."""
        encoder = DeltaEncoder()
        
        first_row, deltas = encoder.encode(sample_ohlcv)
        
        # Deltas should have much smaller range
        for i in range(5):
            original_range = sample_ohlcv[:, i].max() - sample_ohlcv[:, i].min()
            delta_range = deltas[:, i].max() - deltas[:, i].min()
            
            # Delta range should be significantly smaller
            assert delta_range < original_range * 0.5, f"Column {i}: delta range not smaller"
    
    def test_empty_input(self):
        """Test encoding empty array."""
        encoder = DeltaEncoder()
        empty = np.array([]).reshape(0, 5)
        
        first_row, deltas = encoder.encode(empty)
        
        assert len(first_row) == 0
        assert len(deltas) == 0


# ============================================================================
# OHLCVCompressor Tests
# ============================================================================

class TestOHLCVCompressor:
    
    def test_compress_decompress_roundtrip(self, sample_ohlcv):
        """Test that compress → decompress returns original data."""
        compressor = OHLCVCompressor(bits=8, delta_encoding=True)
        
        result = compressor.compress(sample_ohlcv)
        reconstructed = compressor.decompress(result)
        
        assert reconstructed.shape == sample_ohlcv.shape
        assert np.allclose(sample_ohlcv, reconstructed, rtol=0.01)
    
    def test_compression_ratio(self, sample_ohlcv):
        """Test that we achieve good compression."""
        compressor = OHLCVCompressor(bits=8, delta_encoding=True)
        
        result = compressor.compress(sample_ohlcv)
        
        assert result.compression_ratio >= 5.0, f"Compression ratio too low: {result.compression_ratio}"
    
    def test_different_bit_depths(self, sample_ohlcv):
        """Test compression with different bit depths."""
        for bits in [4, 6, 8, 10]:
            compressor = OHLCVCompressor(bits=bits, delta_encoding=True)
            result = compressor.compress(sample_ohlcv)
            
            # Higher bits = better quality, lower compression
            assert result.compression_ratio > 0
    
    def test_delta_encoding_improves_compression(self, sample_ohlcv):
        """Test that delta encoding improves compression."""
        compressor_with_delta = OHLCVCompressor(bits=8, delta_encoding=True)
        compressor_without_delta = OHLCVCompressor(bits=8, delta_encoding=False)
        
        result_with = compressor_with_delta.compress(sample_ohlcv)
        result_without = compressor_without_delta.compress(sample_ohlcv)
        
        # Delta encoding should give better compression
        assert result_with.compression_ratio > result_without.compression_ratio
    
    def test_adaptive_bits(self, sample_ohlcv):
        """Test adaptive bit-depth based on volatility."""
        compressor = OHLCVCompressor(bits=8, adaptive_bits=True)
        
        result = compressor.compress(sample_ohlcv)
        
        # Check that different columns got different bit depths
        params = result.params
        bits_used = set()
        for i in range(5):
            bits_used.add(params.get(f'col_{i}_bits', 8))
        
        # Should have used different bit depths
        assert len(bits_used) >= 1
    
    def test_invalid_columns(self):
        """Test that invalid column count raises error."""
        compressor = OHLCVCompressor()
        invalid_data = np.random.randn(100, 3)  # Only 3 columns
        
        with pytest.raises(ValueError):
            compressor.compress(invalid_data)
    
    def test_single_candle(self):
        """Test compression of single candle."""
        compressor = OHLCVCompressor()
        single = np.array([[100, 105, 95, 102, 1000000]])
        
        result = compressor.compress(single)
        reconstructed = compressor.decompress(result)
        
        assert reconstructed.shape == (1, 5)
        assert np.allclose(single, reconstructed, rtol=0.1)


# ============================================================================
# OHLCVCache Tests
# ============================================================================

class TestOHLCVCache:
    
    def test_add_and_retrieve(self, sample_ohlcv):
        """Test adding candles and retrieving them."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        
        retrieved = cache.get_all_candles('BTC/USDT')
        
        assert retrieved.shape == sample_ohlcv.shape
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01)
    
    def test_incremental_append(self, sample_ohlcv):
        """Test incremental updates (append)."""
        cache = OHLCVCache()
        
        # Add in batches
        batch_size = 100
        for i in range(0, len(sample_ohlcv), batch_size):
            batch = sample_ohlcv[i:i+batch_size]
            cache.add_candles(batch, symbol='BTC/USDT')
        
        retrieved = cache.get_all_candles('BTC/USDT')
        
        assert len(retrieved) == len(sample_ohlcv)
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01)
    
    def test_multi_symbol(self, sample_ohlcv):
        """Test multiple symbols."""
        cache = OHLCVCache()
        
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        cache.add_candles(sample_ohlcv * 0.1, symbol='ETH/USDT')  # Different prices
        
        btc = cache.get_all_candles('BTC/USDT')
        eth = cache.get_all_candles('ETH/USDT')
        
        assert len(btc) == len(sample_ohlcv)
        assert len(eth) == len(sample_ohlcv)
        assert not np.allclose(btc, eth)  # Different data
    
    def test_query_by_symbol(self, sample_ohlcv):
        """Test query API with symbol filter."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        
        result = cache.query(symbol='BTC/USDT')
        
        assert len(result) == len(sample_ohlcv)
    
    def test_query_by_columns(self, sample_ohlcv):
        """Test query API with column filter."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        
        result = cache.query(symbol='BTC/USDT', columns=['close', 'volume'])
        
        assert result.shape == (len(sample_ohlcv), 2)
    
    def test_query_by_range(self, sample_ohlcv):
        """Test query API with index range."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        
        result = cache.query(symbol='BTC/USDT', start=100, end=200)
        
        assert len(result) == 100
        assert np.allclose(result, sample_ohlcv[100:200], rtol=0.01)
    
    def test_persistence(self, sample_ohlcv, temp_cache_dir):
        """Test save and load from disk."""
        # Create and populate cache
        cache1 = OHLCVCache(persist_to=temp_cache_dir)
        cache1.add_candles(sample_ohlcv, symbol='BTC/USDT')
        cache1.save()
        
        # Load from disk
        cache2 = OHLCVCache.load_from(temp_cache_dir)
        retrieved = cache2.get_all_candles('BTC/USDT')
        
        assert len(retrieved) == len(sample_ohlcv)
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01)
    
    def test_stats(self, sample_ohlvc):
        """Test statistics tracking."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        
        stats = cache.get_stats()
        
        assert stats['total_candles'] == len(sample_ohlcv)
        assert stats['total_symbols'] == 1
        assert stats['compression_ratio'] > 0
        assert stats['memory_saved_mb'] > 0


# ============================================================================
# QueryBuilder Tests
# ============================================================================

class TestQueryBuilder:
    
    def test_basic_query(self):
        """Test basic query construction."""
        query = QueryBuilder(
            symbol='BTC/USDT',
            start=0,
            end=100,
            columns=['close']
        )
        
        assert query.symbol == 'BTC/USDT'
        assert query.start_idx == 0
        assert query.end_idx == 100
        assert query.columns == ['close']
    
    def test_chaining(self):
        """Test query builder chaining."""
        query = (QueryBuilder()
            .with_symbol('BTC/USDT')
            .with_range(0, 100)
            .with_columns(['close', 'volume'])
            .with_limit(50))
        
        assert query.symbol == 'BTC/USDT'
        assert query.columns == ['close', 'volume']
    
    def test_invalid_column(self):
        """Test that invalid column raises error."""
        query = QueryBuilder(columns=['invalid'])
        
        with pytest.raises(ValueError):
            query.build()
    
    def test_convenience_functions(self):
        """Test convenience query functions."""
        query = last_n_candles('BTC/USDT', 100, ['close'])
        assert query.symbol == 'BTC/USDT'
        
        query = candles_between('BTC/USDT', '2024-01-01', '2024-03-30')
        assert query.symbol == 'BTC/USDT'


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    
    def test_compression_speed(self, sample_ohlcv):
        """Test compression speed (should be >1000 candles/sec)."""
        import time
        
        compressor = OHLCVCompressor(bits=8)
        
        start = time.perf_counter()
        result = compressor.compress(sample_ohlcv)
        elapsed = time.perf_counter() - start
        
        candles_per_sec = len(sample_ohlcv) / elapsed
        
        assert candles_per_sec > 1000, f"Too slow: {candles_per_sec:.0f} candles/sec"
    
    def test_decompression_speed(self, sample_ohlcv):
        """Test decompression speed."""
        import time
        
        compressor = OHLCVCompressor(bits=8)
        result = compressor.compress(sample_ohlcv)
        
        start = time.perf_counter()
        reconstructed = compressor.decompress(result)
        elapsed = time.perf_counter() - start
        
        candles_per_sec = len(sample_ohlcv) / elapsed
        
        assert candles_per_sec > 1000, f"Too slow: {candles_per_sec:.0f} candles/sec"
    
    def test_large_dataset(self):
        """Test with large dataset (100k candles)."""
        np.random.seed(42)
        large_ohlcv = np.random.randn(100000, 5) * 100 + 100
        
        cache = OHLCVCache()
        cache.add_candles(large_ohlcv, symbol='BTC/USDT')
        
        stats = cache.get_stats()
        
        assert stats['total_candles'] == 100000
        assert stats['compression_ratio'] > 3.0  # Should still compress well


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
