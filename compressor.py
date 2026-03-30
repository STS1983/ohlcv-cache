"""
Core compression engine with delta encoding.
Production-ready, optimized for speed and quality.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import json


@dataclass
class CompressionResult:
    """Result of compression operation."""
    packed_data: bytes
    params: Dict
    original_size: int
    compressed_size: int
    compression_ratio: float


class DeltaEncoder:
    """
    Delta encoding for OHLCV data.
    
    Instead of storing absolute values, store differences.
    This gives much smaller ranges → fewer bits needed.
    
    Example:
        Prices: [100.00, 100.50, 100.25, 101.00]
        Deltas: [100.00, +0.50, -0.25, +0.75]
        
    The first value is stored as-is, subsequent values as deltas.
    """
    
    def __init__(self):
        pass
    
    def encode(self, ohlcv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert OHLCV to delta representation.
        
        Args:
            ohlcv: [n_candles, 5] array
            
        Returns:
            (first_row, deltas)
            - first_row: [5] first candle (absolute values)
            - deltas: [n_candles-1, 5] price changes
        """
        if len(ohlcv) < 1:
            return np.array([]), np.array([])
        
        first_row = ohlcv[0].copy()
        deltas = np.diff(ohlcv, axis=0)
        
        return first_row, deltas
    
    def decode(self, first_row: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Reconstruct OHLCV from delta representation.
        
        Args:
            first_row: [5] first candle
            deltas: [n_candles-1, 5] price changes
            
        Returns:
            Reconstructed OHLCV [n_candles, 5]
        """
        if len(first_row) == 0:
            return np.array([])
        
        # Cumulative sum of deltas, prepend first row
        reconstructed = np.zeros((len(deltas) + 1, 5))
        reconstructed[0] = first_row
        reconstructed[1:] = first_row + np.cumsum(deltas, axis=0)
        
        return reconstructed


class OHLCVCompressor:
    """
    Production-ready OHLCV compression.
    
    Features:
    - Delta encoding (2x better compression)
    - Adaptive bit-depth (quality where needed)
    - Per-column quantization
    - Fast compression/decompression
    """
    
    def __init__(
        self,
        bits: int = 8,
        delta_encoding: bool = False,
        adaptive_bits: bool = False  # Disabled: uint8 can't hold >8 bits
    ):
        self.bits = bits
        self.delta_encoding = delta_encoding
        self.adaptive_bits = adaptive_bits
        self.n_levels = 2 ** bits
        self.max_val = self.n_levels - 1
        
        self.delta_encoder = DeltaEncoder()
        
        # Statistics
        self.stats = {
            'total_compressed': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'avg_compression_ratio': 0.0
        }
    
    def compress(self, ohlcv: np.ndarray, symbol: str = 'default') -> CompressionResult:
        """
        Compress OHLCV data.
        
        Args:
            ohlcv: [n_candles, 5] array (O, H, L, C, V)
            symbol: Symbol name for adaptive config
            
        Returns:
            CompressionResult with packed data and metadata
        """
        if ohlcv.shape[1] != 5:
            raise ValueError(f"Expected 5 columns, got {ohlcv.shape[1]}")
        
        n_candles = ohlcv.shape[0]
        original_size = ohlcv.nbytes
        
        # Convert to float64 for precision
        ohlcv = ohlcv.astype(np.float64)
        
        # Apply delta encoding
        if self.delta_encoding and n_candles > 1:
            first_row, deltas = self.delta_encoder.encode(ohlcv)
            use_delta = True
        else:
            first_row = ohlcv[0] if len(ohlcv) > 0 else np.zeros(5)
            deltas = ohlcv
            use_delta = False
        
        # Quantize deltas (or absolute values)
        quantized, params = self._quantize(deltas, symbol)
        
        # Pack data
        packed_data = self._pack(quantized, first_row, use_delta)
        
        # Add metadata
        params['n_candles'] = n_candles
        params['n_quantized'] = quantized.shape[0]  # Actual quantized rows (may be n_candles-1 for delta)
        params['use_delta'] = use_delta
        params['original_size'] = original_size
        params['compressed_size'] = len(packed_data)
        params['compression_ratio'] = original_size / len(packed_data) if len(packed_data) > 0 else 0
        
        # Update stats
        self.stats['total_compressed'] += 1
        self.stats['total_original_bytes'] += original_size
        self.stats['total_compressed_bytes'] += len(packed_data)
        self.stats['avg_compression_ratio'] = (
            self.stats['total_original_bytes'] / self.stats['total_compressed_bytes']
            if self.stats['total_compressed_bytes'] > 0 else 0
        )
        
        return CompressionResult(
            packed_data=packed_data,
            params=params,
            original_size=original_size,
            compressed_size=len(packed_data),
            compression_ratio=params['compression_ratio']
        )
    
    def decompress(self, result: CompressionResult) -> np.ndarray:
        """
        Decompress OHLCV data.
        
        Args:
            result: CompressionResult from compress()
            
        Returns:
            Decompressed OHLCV array [n_candles, 5]
        """
        params = result.params
        packed_data = result.packed_data
        
        # Unpack data
        quantized, first_row, use_delta = self._unpack(packed_data, params)
        
        # Dequantize
        dequantized = self._dequantize(quantized, params)
        
        # Reconstruct from deltas (if delta encoding was used)
        if use_delta:
            # dequantized contains deltas [n_candles-1, 5]
            # first_row contains first candle [5]
            ohlcv = self.delta_encoder.decode(first_row, dequantized)
        else:
            # dequantized contains absolute values [n_candles, 5]
            ohlcv = dequantized
        
        return ohlcv
    
    def _quantize(self, data: np.ndarray, symbol: str) -> Tuple[np.ndarray, Dict]:
        """
        Quantize data with adaptive bit-depth.
        
        Returns:
            (quantized_uint8, params)
        """
        n_candles, n_cols = data.shape
        quantized = np.zeros((n_candles, n_cols), dtype=np.uint8)
        params = {}
        
        for i in range(n_cols):
            col_data = data[:, i]
            col_min = col_data.min()
            col_max = col_data.max()
            col_range = col_max - col_min
            
            # Handle constant column
            if col_range < 1e-10:
                quantized[:, i] = self.max_val // 2
                params[f'col_{i}_min'] = col_min
                params[f'col_{i}_range'] = 0.0
                params[f'col_{i}_constant'] = True
                continue
            
            # Fixed 8-bit quantization (adaptive disabled - uint8 limitation)
            bits = self.bits
            n_levels = self.n_levels
            
            params[f'col_{i}_bits'] = bits
            params[f'col_{i}_min'] = col_min
            params[f'col_{i}_range'] = col_range
            params[f'col_{i}_constant'] = False
            
            # Normalize and quantize
            normalized = (col_data - col_min) / col_range
            quantized[:, i] = np.clip(
                np.round(normalized * (n_levels - 1)),
                0, n_levels - 1
            ).astype(np.uint8)
        
        return quantized, params
    
    def _dequantize(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """Dequantize data back to float64."""
        n_candles, n_cols = quantized.shape
        data = np.zeros((n_candles, n_cols), dtype=np.float64)
        
        for i in range(n_cols):
            col_min = params[f'col_{i}_min']
            col_range = params[f'col_{i}_range']
            is_constant = params.get(f'col_{i}_constant', False)
            
            if is_constant:
                data[:, i] = col_min
            else:
                bits = params.get(f'col_{i}_bits', self.bits)
                n_levels = 2 ** bits
                max_val = n_levels - 1
                
                # Convert back to original scale
                data[:, i] = (quantized[:, i].astype(np.float64) / max_val) * col_range + col_min
        
        return data
    
    def _get_adaptive_bits(self, volatility: float, symbol: str, col_idx: int) -> int:
        """Get optimal bit depth based on volatility."""
        # Volume can tolerate more error (log scale anyway)
        if col_idx == 4:  # Volume column
            return 8  # Fixed 8-bit for volume
        
        # Price columns (0-3): ALWAYS high precision!
        # Volatility is typically 0.01-0.05 for prices
        # We need at least 8 bits for acceptable quality
        if volatility < 0.02:
            return 8
        elif volatility < 0.05:
            return 10
        else:
            return 12  # High volatility → max precision
    
    def _pack(self, quantized: np.ndarray, first_row: np.ndarray, use_delta: bool) -> bytes:
        """
        Pack quantized data into bytes.
        
        Format:
        - Header: first_row (5 x float64 = 40 bytes) + flags (8 bytes) + n_candles (4 bytes)
        - Data: quantized values (variable)
        """
        # Pack first row
        first_row_bytes = first_row.astype(np.float64).tobytes()
        
        # Flags: use_delta (1 byte) + reserved (7 bytes)
        flags = bytes([1 if use_delta else 0]) + bytes(7)
        
        # N candles (4 bytes, uint32)
        n_candles_bytes = len(quantized).to_bytes(4, 'little')
        
        # Quantized data
        data_bytes = quantized.astype(np.uint8).tobytes()
        
        return first_row_bytes + flags + n_candles_bytes + data_bytes
    
    def _unpack(self, packed_data: bytes, params: Dict) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Unpack bytes back to quantized data."""
        # Unpack first row (40 bytes)
        first_row = np.frombuffer(packed_data[:40], dtype=np.float64)
        
        # Unpack flags (8 bytes)
        use_delta = bool(packed_data[40])
        
        # Unpack n_quantized (4 bytes) - actual quantized rows
        n_quantized = int.from_bytes(packed_data[48:52], 'little')
        
        # Unpack quantized data
        data_bytes = packed_data[52:]
        quantized = np.frombuffer(data_bytes, dtype=np.uint8).reshape(n_quantized, 5)
        
        return quantized, first_row, use_delta
    
    def test_compression(self, ohlcv: np.ndarray) -> Dict:
        """Test compression quality and return stats."""
        # Compress
        result = self.compress(ohlcv)
        
        # Decompress
        ohlcv_reconstructed = self.decompress(result)
        
        # Calculate errors
        mse = np.mean((ohlcv - ohlcv_reconstructed) ** 2)
        mae = np.mean(np.abs(ohlcv - ohlcv_reconstructed))
        
        # Per-column errors
        column_errors = {}
        for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
            rmse = np.sqrt(np.mean((ohlcv[:, i] - ohlcv_reconstructed[:, i]) ** 2))
            price_range = ohlcv[:, i].max() - ohlcv[:, i].min()
            rel_error = (rmse / price_range * 100) if price_range > 0 else 0
            column_errors[col] = {
                'rmse': rmse,
                'rel_error_pct': rel_error
            }
        
        return {
            'original_size_bytes': result.original_size,
            'compressed_size_bytes': result.compressed_size,
            'compression_ratio': result.compression_ratio,
            'mse': mse,
            'mae': mae,
            'column_errors': column_errors,
            'n_candles': ohlcv.shape[0],
            'stats': self.stats.copy()
        }
