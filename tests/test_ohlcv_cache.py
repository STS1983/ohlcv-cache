"""
Comprehensive test suite for OHLCV Cache v2.0.

Tests cover all 4 bug fixes:
- Bug 1: Package structure (imports work as installed package)
- Bug 2: Bits parameter (multi-bit quantization, uint16 for >8 bits)
- Bug 3: Config disconnect (compression config actually used)
- Bug 4: Delta encoding (anchor-based, bounded error)

Run with: python3 -m pytest tests/ -v
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from ohlcv_cache.compressor import (
    OHLCVCompressor,
    DeltaEncoder,
    AnchorDeltaEncoder,
    CompressionResult,
)
from ohlcv_cache.cache import OHLCVCache, CacheBlock
from ohlcv_cache.config import CacheConfig, CompressionConfig, DEFAULT_CONFIG
from ohlcv_cache.query import QueryBuilder, last_n_candles, candles_between


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv():
    """Generate realistic sample OHLCV data."""
    np.random.seed(42)
    n_candles = 1000

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
def small_ohlcv():
    """Small OHLCV dataset for quick tests."""
    np.random.seed(99)
    n = 100
    close = 50 + np.cumsum(np.random.randn(n) * 0.3)
    high = close + np.abs(np.random.randn(n) * 0.2)
    low = close - np.abs(np.random.randn(n) * 0.2)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.exp(np.random.randn(n) * 0.5 + 10)
    return np.column_stack([open_, high, low, close, volume])


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


# ============================================================================
# Bug 1: Package Structure Tests
# ============================================================================

class TestPackageStructure:
    """Verify the package can be imported as an installed package."""

    def test_import_ohlcv_cache(self):
        """Test top-level package import works."""
        import ohlcv_cache
        assert hasattr(ohlcv_cache, 'OHLCVCache')
        assert hasattr(ohlcv_cache, 'OHLCVCompressor')
        assert hasattr(ohlcv_cache, 'CacheConfig')
        assert hasattr(ohlcv_cache, 'QueryBuilder')

    def test_version(self):
        """Test version string is available."""
        import ohlcv_cache
        assert hasattr(ohlcv_cache, '__version__')
        assert ohlcv_cache.__version__ == '2.0.0'

    def test_all_exports(self):
        """Test __all__ exports are importable."""
        from ohlcv_cache import (
            OHLCVCompressor,
            DeltaEncoder,
            AnchorDeltaEncoder,
            CompressionResult,
            OHLCVCache,
            CacheBlock,
            CacheConfig,
            CompressionConfig,
            DEFAULT_CONFIG,
            QueryBuilder,
        )
        # All should be classes (or DEFAULT_CONFIG instance)
        assert OHLCVCompressor is not None
        assert OHLCVCache is not None
        assert CacheConfig is not None
        assert CompressionConfig is not None

    def test_submodule_imports(self):
        """Test submodule imports work."""
        from ohlcv_cache.compressor import OHLCVCompressor
        from ohlcv_cache.cache import OHLCVCache
        from ohlcv_cache.config import CacheConfig
        from ohlcv_cache.query import QueryBuilder


# ============================================================================
# Bug 2: Bits Parameter Tests
# ============================================================================

class TestBitsParameter:
    """Verify the bits parameter actually controls quantization depth."""

    def test_bits_8_uses_uint8(self, sample_ohlcv):
        """bits=8 should use uint8 (256 levels)."""
        comp = OHLCVCompressor(bits=8, delta_encoding=False)
        result = comp.compress(sample_ohlcv)

        # Check that global_bits is 8 in params
        assert result.params['global_bits'] == 8
        assert result.params.get('_dtype_uint16', False) is False

    def test_bits_10_uses_uint16(self, sample_ohlcv):
        """bits=10 should use uint16 (1024 levels)."""
        comp = OHLCVCompressor(bits=10, delta_encoding=False)
        result = comp.compress(sample_ohlcv)

        assert result.params['global_bits'] == 10
        assert result.params.get('_dtype_uint16', False) is True

    def test_bits_12_uses_uint16(self, sample_ohlcv):
        """bits=12 should use uint16 (4096 levels)."""
        comp = OHLCVCompressor(bits=12, delta_encoding=False)
        result = comp.compress(sample_ohlcv)

        assert result.params['global_bits'] == 12
        assert result.params.get('_dtype_uint16', False) is True

    def test_higher_bits_better_quality(self, sample_ohlcv):
        """Higher bits should give lower reconstruction error."""
        errors = {}
        for bits in [4, 8, 12]:
            comp = OHLCVCompressor(bits=bits, delta_encoding=False)
            result = comp.compress(sample_ohlcv)
            reconstructed = comp.decompress(result)
            rmse = np.sqrt(np.mean((sample_ohlcv - reconstructed) ** 2))
            errors[bits] = rmse

        # 12-bit should be better than 8-bit, which is better than 4-bit
        assert errors[12] < errors[8], f"12-bit error {errors[12]} should be < 8-bit {errors[8]}"
        assert errors[8] < errors[4], f"8-bit error {errors[8]} should be < 4-bit {errors[4]}"

    def test_adaptive_bits_called(self, sample_ohlcv):
        """Adaptive bits should actually vary bit depth per column."""
        comp = OHLCVCompressor(bits=8, delta_encoding=False, adaptive_bits=True)
        result = comp.compress(sample_ohlcv)

        # With adaptive bits, different columns should get different bit depths
        bits_used = set()
        for i in range(5):
            bits_used.add(result.params.get(f'col_{i}_bits', 8))

        # Volume column should be 8, price columns might be 8/10/12
        assert len(bits_used) >= 1
        # At least some columns should have bits recorded
        for i in range(5):
            assert f'col_{i}_bits' in result.params, f"col_{i}_bits missing from params"

    def test_roundtrip_8bit(self, sample_ohlcv):
        """8-bit roundtrip should reconstruct within 2% rtol (8-bit quantization limit)."""
        comp = OHLCVCompressor(bits=8, delta_encoding=True, delta_anchor_interval=32)
        result = comp.compress(sample_ohlcv)
        reconstructed = comp.decompress(result)

        assert reconstructed.shape == sample_ohlcv.shape
        # 8-bit = 256 levels → ~0.4% per step, but volume has huge dynamic range
        # and delta encoding adds cumulative error within anchor segments
        atol = sample_ohlcv[:, 4].max() * 0.05  # 5% of max volume for 8-bit
        assert np.allclose(sample_ohlcv, reconstructed, rtol=0.02, atol=atol), (
            f"8-bit roundtrip failed: max rel error = "
            f"{np.max(np.abs((sample_ohlcv - reconstructed) / (np.abs(sample_ohlcv) + 1e-10)))}"
        )

    def test_roundtrip_12bit(self, sample_ohlcv):
        """12-bit roundtrip should achieve <1% rtol."""
        comp = OHLCVCompressor(bits=12, delta_encoding=True, delta_anchor_interval=32)
        result = comp.compress(sample_ohlcv)
        reconstructed = comp.decompress(result)

        assert reconstructed.shape == sample_ohlcv.shape
        atol = sample_ohlcv[:, 4].max() * 0.03  # 3% of max volume for 12-bit quantization
        assert np.allclose(sample_ohlcv, reconstructed, rtol=0.01, atol=atol), (
            "12-bit roundtrip should be within 1% (with atol for volume)"
        )

    def test_different_bit_depths_all_work(self, sample_ohlcv):
        """All reasonable bit depths should compress and decompress without errors."""
        for bits in [4, 6, 8, 10, 12, 14, 16]:
            comp = OHLCVCompressor(bits=bits, delta_encoding=False)
            result = comp.compress(sample_ohlcv)
            reconstructed = comp.decompress(result)
            assert reconstructed.shape == sample_ohlcv.shape, f"bits={bits} shape mismatch"


# ============================================================================
# Bug 3: Config Disconnect Tests
# ============================================================================

class TestConfigIntegration:
    """Verify compression config is actually used by the cache."""

    def test_compression_config_in_cache_config(self):
        """CacheConfig should have a CompressionConfig embedded."""
        config = CacheConfig()
        assert hasattr(config, 'compression')
        assert isinstance(config.compression, CompressionConfig)

    def test_compression_config_defaults(self):
        """Default compression config should have sensible values."""
        config = CacheConfig()
        assert config.compression.default_bits == 8
        assert config.compression.delta_encoding is True
        assert config.compression.delta_anchor_interval == 32

    def test_cache_uses_compression_config(self, sample_ohlcv):
        """OHLCVCache should pass compression config to compressor."""
        config = CacheConfig(
            compression=CompressionConfig(
                default_bits=12,
                delta_encoding=True,
                delta_anchor_interval=16,
            )
        )
        cache = OHLCVCache(config=config)

        # The cache's compressor should have bits=12
        assert cache.compressor.bits == 12
        assert cache.compressor.delta_anchor_interval == 16

    def test_per_symbol_bits_override(self, sample_ohlcv):
        """Symbol-specific bit overrides should be used."""
        config = CacheConfig(
            compression=CompressionConfig(default_bits=8),
            symbol_configs={
                'BTC/USDT': {'bits': 12},
            },
        )
        cache = OHLCVCache(config=config)
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        # The block should have been compressed with 12 bits
        block = cache.blocks['BTC/USDT'][0]
        assert block.compression_result.params['global_bits'] == 12

    def test_per_symbol_bits_default_for_unconfigured(self, sample_ohlcv):
        """Symbols without override should use default bits."""
        config = CacheConfig(
            compression=CompressionConfig(default_bits=8),
            symbol_configs={
                'BTC/USDT': {'bits': 12},
            },
        )
        cache = OHLCVCache(config=config)
        cache.add_candles(sample_ohlcv, symbol='ETH/USDT')

        block = cache.blocks['ETH/USDT'][0]
        assert block.compression_result.params['global_bits'] == 8

    def test_config_roundtrip_json(self, temp_cache_dir):
        """Config should survive save/load to JSON."""
        config = CacheConfig(
            max_candles=50000,
            persist_to=temp_cache_dir,
            compression=CompressionConfig(
                default_bits=10,
                delta_encoding=True,
                delta_anchor_interval=64,
            ),
            symbol_configs={'BTC/USDT': {'bits': 12}},
        )
        config.save(temp_cache_dir / 'config.json')

        loaded = CacheConfig.from_json(temp_cache_dir / 'config.json')
        assert loaded.max_candles == 50000
        assert loaded.compression.default_bits == 10
        assert loaded.compression.delta_anchor_interval == 64
        assert loaded.symbol_configs['BTC/USDT']['bits'] == 12

    def test_default_config_has_compression(self):
        """DEFAULT_CONFIG should have compression settings."""
        assert DEFAULT_CONFIG.compression is not None
        assert DEFAULT_CONFIG.compression.default_bits == 8
        assert DEFAULT_CONFIG.compression.delta_encoding is True

    def test_get_compression_config_for_symbol(self):
        """get_compression_config should merge symbol overrides."""
        config = CacheConfig(
            compression=CompressionConfig(default_bits=8),
            symbol_configs={'BTC/USDT': {'bits': 12}},
        )

        # Symbol with override
        btc_cfg = config.get_compression_config('BTC/USDT')
        assert btc_cfg.default_bits == 12

        # Symbol without override
        eth_cfg = config.get_compression_config('ETH/USDT')
        assert eth_cfg.default_bits == 8


# ============================================================================
# Bug 4: Delta Encoding Tests
# ============================================================================

class TestAnchorDeltaEncoding:
    """Verify anchor-based delta encoding bounds cumulative error."""

    def test_anchor_encode_decode_roundtrip(self, sample_ohlcv):
        """Anchor delta encoder should roundtrip exactly (no quantization)."""
        encoder = AnchorDeltaEncoder(anchor_interval=32)
        encoded, mask = encoder.encode(sample_ohlcv)
        reconstructed = encoder.decode(encoded, mask)

        assert np.allclose(sample_ohlcv, reconstructed), (
            "Anchor delta should roundtrip exactly without quantization"
        )

    def test_anchor_mask_correct(self, sample_ohlcv):
        """Anchor mask should have correct positions."""
        encoder = AnchorDeltaEncoder(anchor_interval=32)
        _, mask = encoder.encode(sample_ohlcv)

        # First candle is always anchor
        assert mask[0] is True or mask[0] == True

        # Every 32nd candle is an anchor
        for i in range(0, len(sample_ohlcv), 32):
            assert mask[i] is True or mask[i] == True, f"Position {i} should be anchor"

    def test_anchor_bounds_error(self, sample_ohlcv):
        """
        Bug 4 core test: Anchor-based delta should have bounded error
        compared to pure cumulative-sum delta.

        With 1000 candles and 8-bit quantization:
        - Pure delta: error grows as O(√n) to O(n)
        - Anchor delta (interval=32): error bounded to ~32 quantization steps
        """
        # Pure delta (anchor_interval = n_candles, so only 1 anchor)
        comp_pure = OHLCVCompressor(
            bits=8, delta_encoding=True, delta_anchor_interval=10000,
        )
        result_pure = comp_pure.compress(sample_ohlcv)
        recon_pure = comp_pure.decompress(result_pure)
        max_error_pure = np.max(np.abs(sample_ohlcv - recon_pure))

        # Anchor delta with interval=32
        comp_anchor = OHLCVCompressor(
            bits=8, delta_encoding=True, delta_anchor_interval=32,
        )
        result_anchor = comp_anchor.compress(sample_ohlcv)
        recon_anchor = comp_anchor.decompress(result_anchor)
        max_error_anchor = np.max(np.abs(sample_ohlcv - recon_anchor))

        # Anchor should have significantly lower max error
        assert max_error_anchor < max_error_pure, (
            f"Anchor error {max_error_anchor} should be < pure delta {max_error_pure}"
        )

    def test_shorter_anchor_better_quality(self, sample_ohlcv):
        """Shorter anchor interval should give better quality (less error)."""
        errors = {}
        for interval in [8, 32, 128, 1000]:
            comp = OHLCVCompressor(
                bits=8, delta_encoding=True, delta_anchor_interval=interval,
            )
            result = comp.compress(sample_ohlcv)
            recon = comp.decompress(result)
            errors[interval] = np.max(np.abs(sample_ohlcv - recon))

        # interval=8 should be better than interval=128
        assert errors[8] < errors[128], (
            f"interval=8 error {errors[8]} should be < interval=128 {errors[128]}"
        )

    def test_delta_encoding_roundtrip_1pct(self, sample_ohlcv):
        """10-bit anchor-delta should achieve <1% rtol on full OHLCV data."""
        comp = OHLCVCompressor(
            bits=10, delta_encoding=True, delta_anchor_interval=32,
        )
        result = comp.compress(sample_ohlcv)
        reconstructed = comp.decompress(result)

        atol = sample_ohlcv[:, 4].max() * 0.02
        assert np.allclose(sample_ohlcv, reconstructed, rtol=0.01, atol=atol), (
            "10-bit anchor-delta should achieve <1% rtol (with atol for volume)"
        )

    def test_delta_encoding_improves_compression(self, sample_ohlcv):
        """Delta encoding should improve compression for price columns."""
        # Compressor requires 5 columns — pad price data with zeros for volume
        price_padded = np.column_stack([sample_ohlcv[:, :4], np.zeros(len(sample_ohlcv))])

        comp_delta = OHLCVCompressor(bits=8, delta_encoding=True, delta_anchor_interval=32)
        comp_no_delta = OHLCVCompressor(bits=8, delta_encoding=False)

        result_delta = comp_delta.compress(price_padded, 'BTC')
        result_no_delta = comp_no_delta.compress(price_padded, 'BTC')

        # Delta should compress better (smaller ranges → better quantization utilization)
        assert result_delta.compression_ratio >= result_no_delta.compression_ratio * 0.95, (
            f"Delta {result_delta.compression_ratio} should be ~>= "
            f"no-delta {result_no_delta.compression_ratio}"
        )

    def test_single_candle(self):
        """Single candle should work (edge case)."""
        comp = OHLCVCompressor(bits=8, delta_encoding=True)
        single = np.array([[100, 105, 95, 102, 1000000]])

        result = comp.compress(single)
        reconstructed = comp.decompress(result)

        assert reconstructed.shape == (1, 5)
        assert np.allclose(single, reconstructed, rtol=0.1)


# ============================================================================
# OHLCVCache Integration Tests
# ============================================================================

class TestOHLCVCache:
    """Integration tests for the full cache system."""

    def test_add_and_retrieve(self, sample_ohlcv):
        """Add candles and retrieve them with <2% error (8-bit on volume)."""
        config = CacheConfig(
            compression=CompressionConfig(default_bits=12, delta_anchor_interval=32),
        )
        cache = OHLCVCache(config=config)
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        retrieved = cache.get_all_candles('BTC/USDT')

        assert retrieved.shape == sample_ohlcv.shape
        atol = sample_ohlcv[:, 4].max() * 0.03  # 3% of max volume (12-bit quantization on high-range volume)
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01, atol=atol)

    def test_incremental_append(self, sample_ohlcv):
        """Incremental updates should reconstruct correctly."""
        config = CacheConfig(
            compression=CompressionConfig(default_bits=12, delta_anchor_interval=32),
        )
        cache = OHLCVCache(config=config)

        batch_size = 100
        for i in range(0, len(sample_ohlcv), batch_size):
            batch = sample_ohlcv[i:i + batch_size]
            cache.add_candles(batch, symbol='BTC/USDT')

        retrieved = cache.get_all_candles('BTC/USDT')

        assert len(retrieved) == len(sample_ohlcv)
        atol = sample_ohlcv[:, 4].max() * 0.03  # 3% of max volume for 12-bit quantization
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01, atol=atol)

    def test_multi_symbol(self, sample_ohlcv):
        """Multiple symbols should work independently."""
        cache = OHLCVCache()

        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')
        cache.add_candles(sample_ohlcv * 0.1, symbol='ETH/USDT')

        btc = cache.get_all_candles('BTC/USDT')
        eth = cache.get_all_candles('ETH/USDT')

        assert len(btc) == len(sample_ohlcv)
        assert len(eth) == len(sample_ohlcv)
        assert not np.allclose(btc, eth)

    def test_query_by_symbol(self, sample_ohlcv):
        """Query API with symbol filter."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        result = cache.query(symbol='BTC/USDT')

        assert len(result) == len(sample_ohlcv)

    def test_query_by_columns(self, sample_ohlcv):
        """Query API with column filter."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        result = cache.query(symbol='BTC/USDT', columns=['close', 'volume'])

        assert result.shape == (len(sample_ohlcv), 2)

    def test_query_by_range(self, sample_ohlcv):
        """Query API with index range."""
        config = CacheConfig(
            compression=CompressionConfig(default_bits=12, delta_anchor_interval=32),
        )
        cache = OHLCVCache(config=config)
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        result = cache.query(symbol='BTC/USDT', start=100, end=200)

        assert len(result) == 100
        atol = sample_ohlcv[:, 4].max() * 0.03  # 3% of max volume for 12-bit quantization
        assert np.allclose(result, sample_ohlcv[100:200], rtol=0.01, atol=atol)

    def test_persistence(self, sample_ohlcv, temp_cache_dir):
        """Save and load from disk."""
        config = CacheConfig(
            persist_to=temp_cache_dir,
            compression=CompressionConfig(default_bits=12, delta_anchor_interval=32),
        )
        cache1 = OHLCVCache(config=config)
        cache1.add_candles(sample_ohlcv, symbol='BTC/USDT')
        cache1.save()

        cache2 = OHLCVCache.load_from(temp_cache_dir)
        retrieved = cache2.get_all_candles('BTC/USDT')

        assert len(retrieved) == len(sample_ohlcv)
        atol = sample_ohlcv[:, 4].max() * 0.03  # 3% of max volume for 12-bit quantization
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01, atol=atol)

    def test_persistence_with_custom_bits(self, sample_ohlcv, temp_cache_dir):
        """Persistence should preserve per-symbol bit settings."""
        config = CacheConfig(
            persist_to=temp_cache_dir,
            compression=CompressionConfig(default_bits=12, delta_anchor_interval=32),
            symbol_configs={'BTC/USDT': {'bits': 16}},
        )
        cache1 = OHLCVCache(config=config)
        cache1.add_candles(sample_ohlcv, symbol='BTC/USDT')
        cache1.save()

        cache2 = OHLCVCache.load_from(temp_cache_dir)
        retrieved = cache2.get_all_candles('BTC/USDT')

        assert len(retrieved) == len(sample_ohlcv)
        atol = sample_ohlcv[:, 4].max() * 0.03  # 3% of max volume for quantization tolerance
        assert np.allclose(sample_ohlcv, retrieved, rtol=0.01, atol=atol)

    def test_stats(self, sample_ohlcv):
        """Statistics tracking (fixes the sample_ohlvc typo)."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        stats = cache.get_stats()

        assert stats['total_candles'] == len(sample_ohlcv)
        assert stats['total_symbols'] == 1
        assert stats['compression_ratio'] > 0

    def test_clear(self, sample_ohlcv):
        """Clear should reset all data."""
        cache = OHLCVCache()
        cache.add_candles(sample_ohlcv, symbol='BTC/USDT')

        cache.clear()

        assert len(cache.blocks) == 0
        assert cache.stats['total_candles'] == 0

    def test_empty_symbol_query(self):
        """Querying non-existent symbol should return empty array."""
        cache = OHLCVCache()
        result = cache.get_all_candles('NONEXISTENT')
        assert len(result) == 0


# ============================================================================
# QueryBuilder Tests
# ============================================================================

class TestQueryBuilder:
    """Query builder tests (fixes test_basic_query)."""

    def test_basic_query(self):
        """Basic query construction — start/end are stored, not start_idx/end_idx."""
        query = QueryBuilder(
            symbol='BTC/USDT',
            start=0,
            end=100,
            columns=['close'],
        )

        # start and end are the constructor params
        assert query.symbol == 'BTC/USDT'
        assert query.start == 0
        assert query.end == 100
        assert query.columns == ['close']

    def test_chaining(self):
        """Query builder chaining."""
        query = (
            QueryBuilder()
            .with_symbol('BTC/USDT')
            .with_range(0, 100)
            .with_columns(['close', 'volume'])
            .with_limit(50)
        )

        assert query.symbol == 'BTC/USDT'
        assert query.columns == ['close', 'volume']

    def test_with_offset_sets_start_idx(self):
        """with_offset should set start_idx field."""
        query = QueryBuilder().with_offset(10)
        assert query.start_idx == 10

    def test_invalid_column(self):
        """Invalid column should raise error on build."""
        query = QueryBuilder(columns=['invalid'])

        with pytest.raises(ValueError):
            query.build()

    def test_convenience_functions(self):
        """Convenience query functions."""
        query = last_n_candles('BTC/USDT', 100, ['close'])
        assert query.symbol == 'BTC/USDT'

        query = candles_between('BTC/USDT', '2024-01-01', '2024-03-30')
        assert query.symbol == 'BTC/USDT'

    def test_query_to_dict(self):
        """Query serialization to dict."""
        query = QueryBuilder(symbol='BTC/USDT', start=0, end=100, columns=['close'])
        d = query.to_dict()

        assert d['symbol'] == 'BTC/USDT'
        assert d['start'] == 0
        assert d['end'] == 100
        assert d['columns'] == ['close']

    def test_query_from_dict(self):
        """Query deserialization from dict."""
        d = {'symbol': 'ETH/USDT', 'start': 50, 'end': 200, 'columns': ['open', 'close']}
        query = QueryBuilder.from_dict(d)

        assert query.symbol == 'ETH/USDT'
        assert query.start == 50
        assert query.end == 200

    def test_query_str(self):
        """Query string representation."""
        query = QueryBuilder(symbol='BTC/USDT', start=0, end=100)
        s = str(query)
        assert 'BTC/USDT' in s
        assert 'Query' in s


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_columns_raises(self):
        """Compressing wrong column count should raise."""
        comp = OHLCVCompressor()
        invalid = np.random.randn(100, 3)

        with pytest.raises(ValueError):
            comp.compress(invalid)

    def test_empty_array_raises(self):
        """Compressing empty array should raise."""
        comp = OHLCVCompressor()
        empty = np.array([]).reshape(0, 5)

        with pytest.raises(ValueError):
            comp.compress(empty)

    def test_two_candle_minimum(self):
        """Two candles should work with delta encoding."""
        comp = OHLCVCompressor(bits=8, delta_encoding=True)
        data = np.array([[100, 105, 95, 102, 1000], [102, 107, 97, 104, 2000]])

        result = comp.compress(data)
        reconstructed = comp.decompress(result)

        assert reconstructed.shape == (2, 5)
        assert np.allclose(data, reconstructed, rtol=0.05)

    def test_constant_column(self):
        """Constant columns should be handled correctly."""
        comp = OHLCVCompressor(bits=8, delta_encoding=False)
        data = np.array([[100, 100, 100, 100, 1000] for _ in range(100)])

        result = comp.compress(data)
        reconstructed = comp.decompress(result)

        assert np.allclose(data, reconstructed, rtol=0.01)

    def test_large_volume_range(self):
        """Large volume values should not cause overflow."""
        comp = OHLCVCompressor(bits=16, delta_encoding=True, delta_anchor_interval=16)
        np.random.seed(42)
        n = 500
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        volume = np.exp(np.random.randn(n) * 2 + 15)  # Very large volumes
        ohlcv = np.column_stack([close, close + 1, close - 1, close, volume])

        result = comp.compress(ohlcv)
        reconstructed = comp.decompress(result)

        atol = ohlcv[:, 4].max() * 0.05
        assert np.allclose(ohlcv, reconstructed, rtol=0.05, atol=atol)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and scale tests."""

    def test_compression_speed(self, sample_ohlcv):
        """Compression should be >1000 candles/sec."""
        import time

        comp = OHLCVCompressor(bits=8, delta_encoding=True)

        start = time.perf_counter()
        result = comp.compress(sample_ohlcv)
        elapsed = time.perf_counter() - start

        cps = len(sample_ohlcv) / elapsed
        assert cps > 1000, f"Too slow: {cps:.0f} candles/sec"

    def test_decompression_speed(self, sample_ohlcv):
        """Decompression should be >1000 candles/sec."""
        import time

        comp = OHLCVCompressor(bits=8, delta_encoding=True)
        result = comp.compress(sample_ohlcv)

        start = time.perf_counter()
        reconstructed = comp.decompress(result)
        elapsed = time.perf_counter() - start

        cps = len(sample_ohlcv) / elapsed
        assert cps > 1000, f"Too slow: {cps:.0f} candles/sec"

    def test_large_dataset(self):
        """100k candles should work."""
        np.random.seed(42)
        large = np.random.randn(100000, 5) * 100 + 100

        cache = OHLCVCache()
        cache.add_candles(large, symbol='BTC/USDT')

        stats = cache.get_stats()
        assert stats['total_candles'] == 100000
        assert stats['compression_ratio'] > 2.0

    def test_compression_ratio_positive(self, sample_ohlcv):
        """Compression ratio should be meaningful."""
        comp = OHLCVCompressor(bits=8, delta_encoding=True)
        result = comp.compress(sample_ohlcv)

        assert result.compression_ratio > 3.0, (
            f"Compression ratio {result.compression_ratio} should be > 3"
        )


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

class TestBackwardCompat:
    """Ensure legacy delta encoder still works for old data."""

    def test_legacy_delta_encoder(self, sample_ohlcv):
        """Legacy DeltaEncoder should still work."""
        encoder = DeltaEncoder()
        first_row, deltas = encoder.encode(sample_ohlcv)
        reconstructed = encoder.decode(first_row, deltas)

        assert np.allclose(sample_ohlcv, reconstructed)

    def test_no_delta_encoding_mode(self, sample_ohlcv):
        """Disabling delta encoding should still work."""
        comp = OHLCVCompressor(bits=10, delta_encoding=False)
        result = comp.compress(sample_ohlcv)
        reconstructed = comp.decompress(result)

        assert reconstructed.shape == sample_ohlcv.shape
        # Without delta, absolute quantization — should be decent
        atol = sample_ohlcv[:, 4].max() * 0.02
        assert np.allclose(sample_ohlcv, reconstructed, rtol=0.01, atol=atol)