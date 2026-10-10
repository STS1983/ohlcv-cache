"""
Core compression engine with anchor-based delta encoding and multi-bit support.

Bug 2 fix: Bits parameter actually works — uint16 used when bits > 8.
Bug 4 fix: Anchor-based delta encoding bounds cumulative error.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CompressionResult:
    """Result of compression operation."""
    packed_data: bytes
    params: Dict
    original_size: int
    compressed_size: int
    compression_ratio: float


class AnchorDeltaEncoder:
    """
    Anchor-based delta encoding for OHLCV data.

    Bug 4 fix: Instead of pure cumulative-sum delta (which accumulates
    quantization errors without bound), we insert periodic anchor points
    (absolute values) every `anchor_interval` candles. This bounds the
    maximum cumulative error to anchor_interval quantization steps.

    Example with anchor_interval=3:
        Original:  [100.0, 100.5, 100.3, 101.0, 101.2, 100.8, ...]
        Anchors:   [100.0,  -,    -,    101.0,  -,    -,    ...]
        Deltas:    [ -,    +0.5, -0.2,  -,    +0.2, -0.4,  ...]

    Error is bounded to at most 3 quantization steps instead of N steps.
    """

    def __init__(self, anchor_interval: int = 32):
        self.anchor_interval = max(2, anchor_interval)

    def encode(self, ohlcv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert OHLCV to anchor-delta representation.

        Args:
            ohlcv: [n_candles, 5] array

        Returns:
            (encoded, anchor_mask)
            - encoded: [n_candles, 5] — absolute values at anchors, deltas elsewhere
            - anchor_mask: [n_candles] bool — True at anchor positions
        """
        n = len(ohlcv)
        if n == 0:
            return np.array([]).reshape(0, 5), np.array([], dtype=bool)

        anchor_mask = np.zeros(n, dtype=bool)
        anchor_mask[0] = True  # First candle is always an anchor
        anchor_mask[self.anchor_interval::self.anchor_interval] = True

        encoded = np.empty_like(ohlcv, dtype=np.float64)

        # At anchors: store absolute values
        encoded[anchor_mask] = ohlcv[anchor_mask]

        # Between anchors: store deltas (difference from previous candle)
        # diff_array[i] = ohlcv[i+1] - ohlcv[i]
        # For non-anchor position j, delta = ohlcv[j] - ohlcv[j-1] = diff_array[j-1]
        diff_array = np.diff(ohlcv, axis=0)  # shape: [n-1, 5]
        non_anchor_idx = np.where(~anchor_mask)[0]  # positions where we need deltas
        # All non-anchor positions are > 0 (since position 0 is always anchor)
        # For position j, use diff_array[j-1]
        encoded[non_anchor_idx] = diff_array[non_anchor_idx - 1]

        return encoded, anchor_mask

    def decode(self, encoded: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct OHLCV from anchor-delta representation.

        Args:
            encoded: [n_candles, 5] — absolute at anchors, deltas elsewhere
            anchor_mask: [n_candles] bool — True at anchor positions

        Returns:
            Reconstructed OHLCV [n_candles, 5]
        """
        n = len(encoded)
        if n == 0:
            return np.array([]).reshape(0, 5)

        reconstructed = np.empty_like(encoded, dtype=np.float64)

        # Find anchor positions
        anchor_idx = np.where(anchor_mask)[0]

        # Process each segment between anchors
        for seg_start, seg_end in zip(anchor_idx, list(anchor_idx[1:]) + [n]):
            # Anchor: absolute value
            reconstructed[seg_start] = encoded[seg_start]

            # Non-anchors: cumulative sum of deltas from anchor
            if seg_end > seg_start + 1:
                segment_deltas = encoded[seg_start + 1:seg_end]
                reconstructed[seg_start + 1:seg_end] = (
                    encoded[seg_start] + np.cumsum(segment_deltas, axis=0)
                )

        return reconstructed


class DeltaEncoder:
    """
    Legacy delta encoder (kept for backward compatibility).
    Simple cumulative-sum delta — has unbounded error accumulation.
    """

    def __init__(self):
        pass

    def encode(self, ohlcv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert OHLCV to delta representation."""
        if len(ohlcv) < 1:
            return np.array([]), np.array([])

        first_row = ohlcv[0].copy()
        deltas = np.diff(ohlcv, axis=0)

        return first_row, deltas

    def decode(self, first_row: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Reconstruct OHLCV from delta representation."""
        if len(first_row) == 0:
            return np.array([])

        reconstructed = np.zeros((len(deltas) + 1, 5))
        reconstructed[0] = first_row
        reconstructed[1:] = first_row + np.cumsum(deltas, axis=0)

        return reconstructed


class OHLCVCompressor:
    """
    Production-ready OHLCV compression.

    Bug 2 fix: bits parameter actually controls quantization depth.
      - bits <= 8:  uses uint8 (256 levels)
      - bits > 8:   uses uint16 (up to 65536 levels)
    Adaptive bits now actually calls _get_adaptive_bits.

    Bug 4 fix: Anchor-based delta encoding with bounded error.
      Instead of pure cumulative-sum delta, periodic anchor points
      reset the cumulative error.
    """

    def __init__(
        self,
        bits: int = 8,
        delta_encoding: bool = True,
        adaptive_bits: bool = False,
        delta_anchor_interval: int = 32,
    ):
        self.bits = bits
        self.delta_encoding = delta_encoding
        self.adaptive_bits = adaptive_bits
        self.delta_anchor_interval = delta_anchor_interval

        # Bug 2 fix: Use appropriate dtype based on bits
        if bits <= 8:
            self.dtype = np.uint8
            self.n_levels = 2 ** bits
        else:
            self.dtype = np.uint16
            self.n_levels = 2 ** bits
        self.max_val = self.n_levels - 1

        # Bug 4 fix: Use anchor-based delta encoder
        self.delta_encoder = AnchorDeltaEncoder(anchor_interval=delta_anchor_interval)
        # Keep legacy encoder for backward compat in decompression

        # Statistics
        self.stats = {
            'total_compressed': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'avg_compression_ratio': 0.0,
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
        if ohlcv.ndim != 2 or ohlcv.shape[1] != 5:
            raise ValueError(f"Expected [n, 5] array, got {ohlcv.shape}")

        n_candles = ohlcv.shape[0]
        if n_candles == 0:
            raise ValueError("Cannot compress empty array")

        original_size = ohlcv.nbytes

        # Convert to float64 for precision
        ohlcv = ohlcv.astype(np.float64)

        # Bug 4 fix: Apply anchor-based delta encoding
        if self.delta_encoding and n_candles > 1:
            encoded, anchor_mask = self.delta_encoder.encode(ohlcv)
            use_delta = True
        else:
            encoded = ohlcv.copy()
            anchor_mask = np.ones(n_candles, dtype=bool)  # All anchors = all absolute
            use_delta = False

        # Quantize
        quantized, params = self._quantize(encoded, symbol)

        # Pack data
        packed_data = self._pack(quantized, anchor_mask, use_delta)

        # Add metadata
        params['n_candles'] = n_candles
        params['n_quantized'] = quantized.shape[0]
        params['use_delta'] = use_delta
        params['delta_anchor_interval'] = self.delta_anchor_interval if use_delta else 0
        params['dtype_is_uint16'] = self.dtype == np.uint16
        params['global_bits'] = self.bits
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
            compression_ratio=params['compression_ratio'],
        )

    def decompress(self, result: CompressionResult) -> np.ndarray:
        """Decompress OHLCV data."""
        params = result.params
        packed_data = result.packed_data

        # Unpack data
        quantized, anchor_mask, use_delta = self._unpack(packed_data, params)

        # Dequantize
        dequantized = self._dequantize(quantized, params)

        # Bug 4 fix: Reconstruct from anchor-delta encoding
        if use_delta:
            ohlcv = self.delta_encoder.decode(dequantized, anchor_mask)
        else:
            ohlcv = dequantized

        return ohlcv

    def _quantize(self, data: np.ndarray, symbol: str) -> Tuple[np.ndarray, Dict]:
        """
        Quantize data with per-column bit depth.

        Bug 2 fix: Actually uses self.bits and supports >8 bits.
        Bug 2 fix: Adaptive bits now actually calls _get_adaptive_bits.
        """
        n_candles, n_cols = data.shape

        # Determine per-column bit depths
        per_col_bits = []
        for i in range(n_cols):
            if self.adaptive_bits:
                # Bug 2 fix: Actually call the adaptive bits method
                col_data = data[:, i]
                col_range = col_data.max() - col_data.min()
                volatility = col_range / (np.abs(col_data).mean() + 1e-10)
                bits_i = self._get_adaptive_bits(volatility, symbol, i)
            else:
                bits_i = self.bits
            per_col_bits.append(bits_i)

        # Determine global dtype (use max bits to decide)
        max_bits = max(per_col_bits)
        if max_bits <= 8:
            dtype = np.uint8
        else:
            dtype = np.uint16

        quantized = np.zeros((n_candles, n_cols), dtype=dtype)
        params = {'_max_bits': max_bits, '_dtype_uint16': dtype == np.uint16}

        for i in range(n_cols):
            col_data = data[:, i]
            col_min = col_data.min()
            col_max = col_data.max()
            col_range = col_max - col_min

            bits_i = per_col_bits[i]
            n_levels_i = 2 ** bits_i
            max_val_i = n_levels_i - 1

            # Handle constant column
            if col_range < 1e-10:
                quantized[:, i] = max_val_i // 2
                params[f'col_{i}_min'] = float(col_min)
                params[f'col_{i}_range'] = 0.0
                params[f'col_{i}_constant'] = True
                params[f'col_{i}_bits'] = bits_i
                continue

            params[f'col_{i}_bits'] = bits_i
            params[f'col_{i}_min'] = float(col_min)
            params[f'col_{i}_range'] = float(col_range)
            params[f'col_{i}_constant'] = False

            # Normalize and quantize
            normalized = (col_data - col_min) / col_range
            quantized[:, i] = np.clip(
                np.round(normalized * max_val_i),
                0, max_val_i,
            ).astype(dtype)

        return quantized, params

    def _dequantize(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """Dequantize data back to float64."""
        n_candles, n_cols = quantized.shape
        data = np.zeros((n_candles, n_cols), dtype=np.float64)

        for i in range(n_cols):
            col_min = params[f'col_{i}_min']
            col_range = params[f'col_{i}_range']
            is_constant = params.get(f'col_{i}_constant', False)
            bits_i = params.get(f'col_{i}_bits', self.bits)

            if is_constant:
                data[:, i] = col_min
            else:
                n_levels_i = 2 ** bits_i
                max_val_i = n_levels_i - 1

                data[:, i] = (
                    quantized[:, i].astype(np.float64) / max_val_i
                ) * col_range + col_min

        return data

    def _get_adaptive_bits(self, volatility: float, symbol: str, col_idx: int) -> int:
        """Get optimal bit depth based on volatility."""
        # Volume can tolerate more error (log scale anyway)
        if col_idx == 4:  # Volume column
            return 8

        # Price columns (0-3): use adaptive based on volatility
        if volatility < 0.02:
            return 8
        elif volatility < 0.05:
            return 10
        else:
            return 12

    def _pack(
        self,
        quantized: np.ndarray,
        anchor_mask: np.ndarray,
        use_delta: bool,
    ) -> bytes:
        """
        Pack quantized data into bytes.

        Format:
        - Header: use_delta (1 byte) + dtype_uint16 (1 byte) + n_candles (4 bytes)
        - Anchor mask: packed as bytes
        - Data: quantized values
        """
        n_candles = len(quantized)
        is_uint16 = quantized.dtype == np.uint16

        # Header
        header = bytes([1 if use_delta else 0, 1 if is_uint16 else 0])
        n_candles_bytes = n_candles.to_bytes(4, 'little')

        # Anchor mask: pack as bit array → bytes
        anchor_bytes = np.packbits(anchor_mask).tobytes()
        anchor_len = len(anchor_bytes)
        anchor_len_bytes = anchor_len.to_bytes(4, 'little')

        # Quantized data
        data_bytes = quantized.tobytes()

        return header + n_candles_bytes + anchor_len_bytes + anchor_bytes + data_bytes

    def _unpack(self, packed_data: bytes, params: Dict) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Unpack bytes back to quantized data + anchor mask."""
        offset = 0

        # Header
        use_delta = bool(packed_data[offset]); offset += 1
        is_uint16 = bool(packed_data[offset]); offset += 1

        # n_candles
        n_candles = int.from_bytes(packed_data[offset:offset + 4], 'little'); offset += 4

        # Anchor mask
        anchor_len = int.from_bytes(packed_data[offset:offset + 4], 'little'); offset += 4
        anchor_bytes = packed_data[offset:offset + anchor_len]; offset += anchor_len
        anchor_mask = np.unpackbits(np.frombuffer(anchor_bytes, dtype=np.uint8))[:n_candles].astype(bool)

        # Quantized data
        dtype = np.uint16 if is_uint16 else np.uint8
        itemsize = dtype().itemsize
        data_size = n_candles * 5 * itemsize
        data_bytes = packed_data[offset:offset + data_size]
        quantized = np.frombuffer(data_bytes, dtype=dtype).reshape(n_candles, 5)

        return quantized, anchor_mask, use_delta

    def test_compression(self, ohlcv: np.ndarray) -> Dict:
        """Test compression quality and return stats."""
        result = self.compress(ohlcv)
        ohlcv_reconstructed = self.decompress(result)

        mse = np.mean((ohlcv - ohlcv_reconstructed) ** 2)
        mae = np.mean(np.abs(ohlcv - ohlcv_reconstructed))

        column_errors = {}
        for i, col in enumerate(['open', 'high', 'low', 'close', 'volume']):
            rmse = np.sqrt(np.mean((ohlcv[:, i] - ohlcv_reconstructed[:, i]) ** 2))
            price_range = ohlcv[:, i].max() - ohlcv[:, i].min()
            rel_error = (rmse / price_range * 100) if price_range > 0 else 0
            column_errors[col] = {
                'rmse': float(rmse),
                'rel_error_pct': float(rel_error),
            }

        return {
            'original_size_bytes': result.original_size,
            'compressed_size_bytes': result.compressed_size,
            'compression_ratio': result.compression_ratio,
            'mse': float(mse),
            'mae': float(mae),
            'column_errors': column_errors,
            'n_candles': ohlcv.shape[0],
            'stats': self.stats.copy(),
        }