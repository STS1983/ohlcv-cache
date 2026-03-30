#!/usr/bin/env python3
"""
Simple test runner without pytest dependency.
Run with: python3 ohlcv_cache/run_tests.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ohlcv_cache.compressor import OHLCVCompressor, DeltaEncoder
from ohlcv_cache.cache import OHLCVCache
from ohlcv_cache.config import CacheConfig
from ohlcv_cache.query import QueryBuilder


def generate_sample_ohlcv(n_candles=1000, seed=42):
    """Generate realistic OHLCV data."""
    np.random.seed(seed)
    
    close = 100 + np.cumsum(np.random.randn(n_candles) * 0.5)
    close = np.maximum(close, 1)
    
    high = close + np.abs(np.random.randn(n_candles) * 0.3)
    low = close - np.abs(np.random.randn(n_candles) * 0.3)
    low = np.minimum(low, close * 0.95)
    high = np.maximum(high, close * 1.05)
    open_ = close + np.random.randn(n_candles) * 0.2
    
    volume = np.exp(np.random.randn(n_candles) * 0.8 + 11)
    
    return np.column_stack([open_, high, low, close, volume])


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test(self, name, func):
        """Run a test function."""
        try:
            func()
            self.passed += 1
            print(f"  ✅ {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ❌ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, f"{type(e).__name__}: {e}"))
            print(f"  💥 {name}: {type(e).__name__}: {e}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed")
        
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        
        return self.failed == 0


def test_delta_encoder_roundtrip():
    """Test that encode → decode returns original data."""
    encoder = DeltaEncoder()
    ohlcv = generate_sample_ohlcv(1000)
    
    first_row, deltas = encoder.encode(ohlcv)
    reconstructed = encoder.decode(first_row, deltas)
    
    assert np.allclose(ohlcv, reconstructed), "Delta encoder roundtrip failed"


def test_compress_decompress_roundtrip():
    """Test that compress → decompress returns original data."""
    compressor = OHLCVCompressor(bits=8, delta_encoding=False)
    ohlcv = generate_sample_ohlcv(1000)
    
    result = compressor.compress(ohlcv)
    reconstructed = compressor.decompress(result)
    
    assert reconstructed.shape == ohlcv.shape, f"Shape mismatch: {reconstructed.shape} vs {ohlcv.shape}"
    
    # Check relative error per column (more meaningful than absolute)
    for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
        col_range = ohlcv[:, i].max() - ohlcv[:, i].min()
        max_diff = np.max(np.abs(ohlcv[:, i] - reconstructed[:, i]))
        rel_error = (max_diff / col_range * 100) if col_range > 0 else 0
        assert rel_error < 1.0, f"{col}: relative error {rel_error:.3f}% > 1%"


def test_compression_ratio():
    """Test that we achieve good compression."""
    compressor = OHLCVCompressor(bits=8)
    ohlcv = generate_sample_ohlcv(1000)
    
    result = compressor.compress(ohlcv)
    
    assert result.compression_ratio >= 5.0, f"Compression ratio too low: {result.compression_ratio:.2f}x"


def test_reconstruction_error():
    """Test that reconstruction error is <0.5%."""
    compressor = OHLCVCompressor(bits=8)
    ohlcv = generate_sample_ohlcv(1000)
    
    result = compressor.compress(ohlcv)
    reconstructed = compressor.decompress(result)
    
    # Check close price error (most important)
    close_rmse = np.sqrt(np.mean((ohlcv[:, 3] - reconstructed[:, 3]) ** 2))
    close_range = ohlcv[:, 3].max() - ohlcv[:, 3].min()
    rel_error = (close_rmse / close_range * 100) if close_range > 0 else 0
    
    assert rel_error < 0.5, f"Reconstruction error too high: {rel_error:.3f}%"


def test_different_bit_depths():
    """Test compression with different bit depths."""
    ohlcv = generate_sample_ohlcv(1000)
    
    for bits in [4, 6, 8, 10]:
        compressor = OHLCVCompressor(bits=bits)
        result = compressor.compress(ohlcv)
        
        assert result.compression_ratio > 0, f"Bits={bits}: No compression"


def test_cache_add_retrieve():
    """Test adding candles and retrieving them."""
    cache = OHLCVCache()
    ohlcv = generate_sample_ohlcv(1000)
    
    cache.add_candles(ohlcv, symbol='BTC/USDT')
    retrieved = cache.get_all_candles('BTC/USDT')
    
    assert retrieved.shape == ohlcv.shape, f"Shape mismatch: {retrieved.shape} vs {ohlcv.shape}"
    
    # Check relative error (volume can be large, so use relative)
    close_range = ohlcv[:, 3].max() - ohlcv[:, 3].min()
    max_diff = np.max(np.abs(ohlcv[:, 3] - retrieved[:, 3]))
    rel_error = (max_diff / close_range * 100) if close_range > 0 else 0
    assert rel_error < 1.0, f"Relative error {rel_error:.3f}% > 1%"


def test_cache_incremental_append():
    """Test incremental updates (append)."""
    cache = OHLCVCache()
    ohlcv = generate_sample_ohlcv(1000)
    
    # Add in batches
    batch_size = 100
    for i in range(0, len(ohlcv), batch_size):
        batch = ohlcv[i:i+batch_size]
        cache.add_candles(batch, symbol='BTC/USDT')
    
    retrieved = cache.get_all_candles('BTC/USDT')
    
    assert len(retrieved) == len(ohlcv), f"Length mismatch: {len(retrieved)} vs {len(ohlcv)}"
    
    # Check relative error
    close_range = ohlcv[:, 3].max() - ohlcv[:, 3].min()
    max_diff = np.max(np.abs(ohlcv[:, 3] - retrieved[:, 3]))
    rel_error = (max_diff / close_range * 100) if close_range > 0 else 0
    assert rel_error < 1.0, f"Relative error {rel_error:.3f}% > 1%"


def test_cache_multi_symbol():
    """Test multiple symbols."""
    cache = OHLCVCache()
    ohlcv1 = generate_sample_ohlcv(1000, seed=42)
    ohlcv2 = generate_sample_ohlcv(1000, seed=123)
    
    cache.add_candles(ohlcv1, symbol='BTC/USDT')
    cache.add_candles(ohlcv2, symbol='ETH/USDT')
    
    btc = cache.get_all_candles('BTC/USDT')
    eth = cache.get_all_candles('ETH/USDT')
    
    assert len(btc) == 1000, f"BTC length: {len(btc)}"
    assert len(eth) == 1000, f"ETH length: {len(eth)}"
    assert not np.allclose(btc, eth), "BTC and ETH should be different"


def test_cache_query_range():
    """Test query API with index range."""
    cache = OHLCVCache()
    ohlcv = generate_sample_ohlcv(1000)
    
    cache.add_candles(ohlcv, symbol='BTC/USDT')
    
    result = cache.query(symbol='BTC/USDT', start=100, end=200)
    
    assert len(result) == 100, f"Expected 100 candles, got {len(result)}"
    
    # Check relative error (2% for small slices due to quantization)
    close_range = ohlcv[100:200, 3].max() - ohlcv[100:200, 3].min()
    max_diff = np.max(np.abs(ohlcv[100:200, 3] - result[:, 3]))
    rel_error = (max_diff / close_range * 100) if close_range > 0 else 0
    assert rel_error < 2.0, f"Relative error {rel_error:.3f}% > 2%"


def test_cache_query_columns():
    """Test query API with column filter."""
    cache = OHLCVCache()
    ohlcv = generate_sample_ohlcv(1000)
    
    cache.add_candles(ohlcv, symbol='BTC/USDT')
    
    result = cache.query(symbol='BTC/USDT', columns=['close', 'volume'])
    
    assert result.shape == (1000, 2), f"Expected shape (1000, 2), got {result.shape}"


def test_cache_stats():
    """Test statistics tracking."""
    cache = OHLCVCache()
    ohlcv = generate_sample_ohlcv(1000)
    
    cache.add_candles(ohlcv, symbol='BTC/USDT')
    
    stats = cache.get_stats()
    
    assert stats['total_candles'] == 1000, f"Expected 1000 candles, got {stats['total_candles']}"
    assert stats['compression_ratio'] > 0, "Compression ratio should be > 0"
    assert stats['memory_saved_mb'] > 0, "Memory saved should be > 0"


def test_performance_compression():
    """Test compression speed (>1000 candles/sec)."""
    import time
    
    compressor = OHLCVCompressor(bits=8)
    ohlcv = generate_sample_ohlcv(10000)
    
    start = time.perf_counter()
    result = compressor.compress(ohlcv)
    elapsed = time.perf_counter() - start
    
    candles_per_sec = len(ohlcv) / elapsed
    
    assert candles_per_sec > 1000, f"Too slow: {candles_per_sec:.0f} candles/sec"


def test_performance_decompression():
    """Test decompression speed."""
    import time
    
    compressor = OHLCVCompressor(bits=8)
    ohlcv = generate_sample_ohlcv(10000)
    result = compressor.compress(ohlcv)
    
    start = time.perf_counter()
    reconstructed = compressor.decompress(result)
    elapsed = time.perf_counter() - start
    
    candles_per_sec = len(ohlcv) / elapsed
    
    assert candles_per_sec > 1000, f"Too slow: {candles_per_sec:.0f} candles/sec"


def test_large_dataset():
    """Test with large dataset (100k candles)."""
    np.random.seed(42)
    large_ohlcv = np.random.randn(100000, 5) * 100 + 100
    
    cache = OHLCVCache()
    cache.add_candles(large_ohlcv, symbol='BTC/USDT')
    
    stats = cache.get_stats()
    
    assert stats['total_candles'] == 100000, f"Expected 100k candles, got {stats['total_candles']}"
    assert stats['compression_ratio'] > 3.0, f"Compression too low: {stats['compression_ratio']:.2f}x"


def main():
    print("="*60)
    print("🧪 OHLCV CACHE TEST SUITE")
    print("="*60)
    
    runner = TestRunner()
    
    print("\n📦 DeltaEncoder Tests:")
    runner.test("roundtrip", test_delta_encoder_roundtrip)
    
    print("\n📦 OHLCVCompressor Tests:")
    runner.test("roundtrip", test_compress_decompress_roundtrip)
    runner.test("compression_ratio", test_compression_ratio)
    runner.test("reconstruction_error", test_reconstruction_error)
    runner.test("different_bit_depths", test_different_bit_depths)
    
    print("\n📦 OHLCVCache Tests:")
    runner.test("add_retrieve", test_cache_add_retrieve)
    runner.test("incremental_append", test_cache_incremental_append)
    runner.test("multi_symbol", test_cache_multi_symbol)
    runner.test("query_range", test_cache_query_range)
    runner.test("query_columns", test_cache_query_columns)
    runner.test("stats", test_cache_stats)
    
    print("\n📦 Performance Tests:")
    runner.test("compression_speed", test_performance_compression)
    runner.test("decompression_speed", test_performance_decompression)
    runner.test("large_dataset", test_large_dataset)
    
    # Summary
    success = runner.summary()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
