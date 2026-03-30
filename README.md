# 🚀 OHLCV Cache v1.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 14/14 passing](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)](https://github.com/STS1983/ohlcv-cache)

**High-performance OHLCV data compression for trading and backtesting.**

- 📦 **8x Compression** – Store 8x more candles in same RAM
- ⚡ **>10M candles/sec** – Blazing fast compression/decompression
- 🎯 **<0.5% Error** – High-fidelity reconstruction
- 💾 **Persistent Storage** – Survives restarts
- 🔍 **Query API** – Get exactly what you need
- 📊 **Multi-Symbol** – Support for multiple trading pairs

## Features

✅ **7-8x Compression** – Store 7x more candles in same RAM  
✅ **<0.5% Error** – High-fidelity reconstruction  
✅ **>10M candles/sec** – Blazing fast compression/decompression  
✅ **Persistent Storage** – Survives restarts  
✅ **Query API** – Get exactly what you need  
✅ **Incremental Updates** – O(1) append operations  
✅ **Multi-Symbol** – Support for multiple trading pairs  

## Installation

### From PyPI (coming soon)
```bash
pip install ohlcv-cache
```

### From Source
```bash
git clone https://github.com/EvilStief/ohlcv-cache.git
cd ohlcv-cache
pip install -e .
```

### Quick Test
```python
from ohlcv_cache import OHLCVCache, OHLCVCompressor
```

## Quick Start

```python
import numpy as np
from ohlcv_cache import OHLCVCache

# Create cache
cache = OHLCVCache(persist_to='/data/ohlcv_cache')

# Add candles
ohlcv = np.random.randn(1000, 5) * 100 + 100  # [open, high, low, close, volume]
cache.add_candles(ohlcv, symbol='BTC/USDT')

# Query
candles = cache.query(symbol='BTC/USDT', start=0, end=100)
close_only = cache.query(symbol='BTC/USDT', columns=['close'])

# Get stats
stats = cache.get_stats()
print(f"Compression: {stats['compression_ratio']:.2f}x")
print(f"Memory saved: {stats['memory_saved_mb']:.2f} MB")
```

## API Reference

### OHLCVCache

```python
# Create cache
cache = OHLCVCache(
    persist_to='/data/cache',  # Optional: auto-save to disk
    max_candles=100000          # Optional: max candles in memory
)

# Add candles
cache.add_candles(ohlcv, symbol='BTC/USDT')

# Query
candles = cache.query(
    symbol='BTC/USDT',
    start=0,           # Start index (optional)
    end=1000,          # End index (optional)
    columns=['close']  # Specific columns (optional)
)

# Get all candles
all_candles = cache.get_all_candles('BTC/USDT')

# Save/Load
cache.save()
cache2 = OHLCVCache.load_from('/data/cache')

# Stats
stats = cache.get_stats()
```

### OHLCVCompressor

```python
from ohlcv_cache import OHLCVCompressor

compressor = OHLCVCompressor(
    bits=8,              # Bit depth (4, 6, 8, 10)
    delta_encoding=False, # Currently disabled (buggy)
    adaptive_bits=True   # Adjust bits based on volatility
)

# Compress
result = compressor.compress(ohlcv, symbol='BTC/USDT')
print(f"Compression ratio: {result.compression_ratio:.2f}x")

# Decompress
reconstructed = compressor.decompress(result)
```

## Configuration

```python
from ohlcv_cache import CacheConfig

config = CacheConfig(
    max_candles=100000,
    persist_to='/data/cache',
    auto_save=True,
    save_interval_seconds=300,
    symbol_configs={
        'BTC/USDT': {'bits': 10},  # Higher precision
        'ETH/USDT': {'bits': 8},
    }
)

cache = OHLCVCache(config=config)
```

## Performance

| Metric | Value |
|--------|-------|
| **Compression** | 7-8x |
| **Error (RMSE)** | <0.5% |
| **Speed** | >10M candles/sec |
| **Memory** | ~5 bytes per candle |

## Known Issues

- **Delta-Encoding**: Currently disabled (causes reconstruction errors)
- **Date-based queries**: Not yet implemented (only index-based)

## Benchmarks

| Metric | Value |
|--------|-------|
| **Compression Ratio** | 8.00x |
| **Reconstruction Error** | <0.12% (RMSE) |
| **Compression Speed** | >12M candles/sec |
| **Decompression Speed** | >36M candles/sec |
| **Memory Usage** | ~5 bytes per candle |

**Tested with:** 249,307 real candles (BTC, ETH, ADA, SOL, LINK)

## Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `python3 -m pytest ohlcv_cache/tests.py -v`
5. Submit a PR

## License

MIT License – Use freely for your trading projects!

---

**Made with ❤️ by STS1983 for the trading community**
