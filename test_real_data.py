#!/usr/bin/env python3
"""
Test OHLCV Cache with REAL trading data.
Uses actual BTC/ETH/ etc. candles from backtest_cache.
"""

import numpy as np
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from ohlcv_cache import OHLCVCache, OHLCVCompressor


def load_csv_ohlcv(csv_path: Path) -> np.ndarray:
    """
    Load OHLCV data from CSV.
    
    Expected format:
    timestamp,open,high,low,close,volume
    """
    data = []
    with open(csv_path) as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                # Extract OHLCV (skip timestamp)
                open_ = float(parts[1])
                high = float(parts[2])
                low = float(parts[3])
                close = float(parts[4])
                volume = float(parts[5])
                data.append([open_, high, low, close, volume])
    
    return np.array(data)


def main():
    print("="*70)
    print("🧪 OHLCV CACHE – REAL DATA TEST")
    print("="*70)
    
    # Find real data
    backtest_dir = Path(__file__).parent.parent / 'backtest_cache'
    
    if not backtest_dir.exists():
        print(f"❌ Backtest cache not found: {backtest_dir}")
        return 1
    
    csv_files = list(backtest_dir.glob('*_USDT_1h.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {backtest_dir}")
        return 1
    
    print(f"\n📂 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    # ========================================================================
    # TEST 1: Load and compress BTC data
    # ========================================================================
    print("\n" + "="*70)
    print("📦 TEST 1: BTC/USDT Real Data")
    print("="*70)
    
    btc_file = backtest_dir / 'BTC_USDT_1h.csv'
    
    if not btc_file.exists():
        print(f"❌ BTC file not found: {btc_file}")
        return 1
    
    # Load real data
    print(f"\n📊 Loading {btc_file.name}...")
    btc_ohlcv = load_csv_ohlcv(btc_file)
    
    print(f"   Candles: {len(btc_ohlcv):,}")
    print(f"   Price range: ${btc_ohlcv[:, 3].min():.2f} - ${btc_ohlcv[:, 3].max():,.2f}")
    print(f"   Volume range: {btc_ohlcv[:, 4].min():,.0f} - {btc_ohlcv[:, 4].max():,.0f}")
    
    # Compress
    print(f"\n📦 Compressing...")
    compressor = OHLCVCompressor(bits=8, delta_encoding=False)
    
    start = time.perf_counter()
    result = compressor.compress(btc_ohlcv, symbol='BTC/USDT')
    compress_time = time.perf_counter() - start
    
    print(f"   Time: {compress_time*1000:.1f}ms")
    print(f"   Speed: {len(btc_ohlcv) / compress_time:,.0f} candles/sec")
    
    # Decompress
    print(f"\n📤 Decompressing...")
    start = time.perf_counter()
    btc_reconstructed = compressor.decompress(result)
    decompress_time = time.perf_counter() - start
    
    print(f"   Time: {decompress_time*1000:.1f}ms")
    print(f"   Speed: {len(btc_ohlcv) / decompress_time:,.0f} candles/sec")
    
    # Quality check
    print(f"\n📉 Reconstruction Quality:")
    print(f"   Compression ratio: {result.compression_ratio:.2f}x")
    
    for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
        rmse = np.sqrt(np.mean((btc_ohlcv[:, i] - btc_reconstructed[:, i]) ** 2))
        price_range = btc_ohlcv[:, i].max() - btc_ohlcv[:, i].min()
        rel_error = (rmse / price_range * 100) if price_range > 0 else 0
        print(f"   {col}: RMSE={rmse:.4f} ({rel_error:.3f}%)")
    
    # Overall quality
    close_rmse = np.sqrt(np.mean((btc_ohlcv[:, 3] - btc_reconstructed[:, 3]) ** 2))
    close_range = btc_ohlcv[:, 3].max() - btc_ohlcv[:, 3].min()
    close_rel_error = (close_rmse / close_range * 100) if close_range > 0 else 0
    
    if close_rel_error < 0.5:
        print(f"\n   ✅ EXCELLENT (<0.5% error)")
    elif close_rel_error < 1.0:
        print(f"\n   ✅ GOOD (<1% error)")
    elif close_rel_error < 2.0:
        print(f"\n   ⚠️ ACCEPTABLE (<2% error)")
    else:
        print(f"\n   ❌ POOR (>{close_rel_error:.1f}% error)")
    
    # ========================================================================
    # TEST 2: Multi-symbol cache
    # ========================================================================
    print("\n" + "="*70)
    print("📦 TEST 2: Multi-Symbol Cache")
    print("="*70)
    
    cache = OHLCVCache()
    
    symbols_loaded = {}
    
    for csv_file in csv_files:
        symbol = csv_file.stem.replace('_USDT_1h', '/USDT')
        
        print(f"\n   Loading {symbol}...")
        ohlcv = load_csv_ohlcv(csv_file)
        symbols_loaded[symbol] = ohlcv
        
        cache.add_candles(ohlcv, symbol=symbol)
        print(f"      {len(ohlcv):,} candles added")
    
    # Stats
    print(f"\n📊 Cache Statistics:")
    stats = cache.get_stats()
    print(f"   Total symbols: {stats['total_symbols']}")
    print(f"   Total candles: {stats['total_candles']:,}")
    print(f"   Original size: {stats['original_size_mb']:.2f} MB")
    print(f"   Compressed size: {stats['compressed_size_mb']:.2f} MB")
    print(f"   Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"   Memory saved: {stats['memory_saved_mb']:.2f} MB")
    
    # ========================================================================
    # TEST 3: Query real data
    # ========================================================================
    print("\n" + "="*70)
    print("📦 TEST 3: Query Real Data")
    print("="*70)
    
    # Get last 100 BTC candles
    print(f"\n🔍 Query: Last 100 BTC candles")
    btc_all = cache.get_all_candles('BTC/USDT')
    btc_last_100 = cache.query(symbol='BTC/USDT', start=len(btc_all)-100)
    
    print(f"   Retrieved: {len(btc_last_100)} candles")
    print(f"   Close price: ${btc_last_100[:, 3].min():.2f} - ${btc_last_100[:, 3].max():.2f}")
    print(f"   Volume: {btc_last_100[:, 4].min():,.0f} - {btc_last_100[:, 4].max():,.0f}")
    
    # Get close prices only
    print(f"\n🔍 Query: ETH close prices only")
    eth_close = cache.query(symbol='ETH/USDT', columns=['close'])
    
    print(f"   Retrieved: {len(eth_close)} candles")
    print(f"   Shape: {eth_close.shape} (only close column)")
    print(f"   Close price: ${eth_close[:, 0].min():.2f} - ${eth_close[:, 0].max():,.2f}")
    
    # ========================================================================
    # TEST 4: Persistence test
    # ========================================================================
    print("\n" + "="*70)
    print("📦 TEST 4: Persistence (Save/Load)")
    print("="*70)
    
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save
        print(f"\n💾 Saving to {temp_dir}...")
        cache.save(temp_dir)
        print(f"   ✅ Saved successfully")
        
        # Check files
        files = list(temp_dir.rglob('*'))
        print(f"   Files created: {len(files)}")
        
        # Load
        print(f"\n📂 Loading from disk...")
        cache2 = OHLCVCache.load_from(temp_dir)
        print(f"   ✅ Loaded successfully")
        
        # Verify BTC data
        loaded_btc = cache2.get_all_candles('BTC/USDT')
        
        print(f"\n📊 Verification:")
        print(f"   BTC candles: {len(loaded_btc):,} (expected: {len(btc_all):,})")
        
        # Quality check
        close_range = btc_all[:, 3].max() - btc_all[:, 3].min()
        max_diff = np.max(np.abs(btc_all[:, 3] - loaded_btc[:, 3]))
        rel_error = (max_diff / close_range * 100) if close_range > 0 else 0
        
        print(f"   Max price diff: ${max_diff:.4f} ({rel_error:.3f}%)")
        
        if rel_error < 1.0:
            print(f"   ✅ Data integrity verified!")
        else:
            print(f"   ⚠️ Some data loss detected")
        
    finally:
        shutil.rmtree(temp_dir)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("✅ REAL DATA TEST COMPLETE!")
    print("="*70)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Symbols tested: {len(symbols_loaded)}")
    print(f"   Total candles: {stats['total_candles']:,}")
    print(f"   Compression: {stats['compression_ratio']:.2f}x")
    print(f"   Memory saved: {stats['memory_saved_mb']:.2f} MB")
    print(f"   Close price error: {close_rel_error:.3f}%")
    
    if close_rel_error < 1.0:
        print(f"\n🎉 PRODUCTION READY!")
        print(f"   Cache can handle real trading data with <1% error")
    else:
        print(f"\n⚠️ Review error rate before production use")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
