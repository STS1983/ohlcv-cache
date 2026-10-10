"""
Configuration system for OHLCV Cache.
Unified config that actually connects to the compressor.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class CompressionConfig:
    """Compression settings — actually used by OHLCVCompressor."""
    default_bits: int = 8
    adaptive_enabled: bool = True
    volatility_thresholds: list = field(default_factory=lambda: [0.01, 0.03])
    bit_depths: list = field(default_factory=lambda: [6, 8, 10])
    delta_encoding: bool = True
    delta_anchor_interval: int = 32  # Reset delta every N candles to bound error

    def get_bit_depth(self, volatility: float) -> int:
        """Get optimal bit depth for given volatility."""
        if not self.adaptive_enabled:
            return self.default_bits

        for i, threshold in enumerate(self.volatility_thresholds):
            if volatility < threshold:
                return self.bit_depths[i]

        return self.bit_depths[-1]  # Highest for most volatile

    def to_dict(self) -> Dict[str, Any]:
        return {
            'default_bits': self.default_bits,
            'adaptive_enabled': self.adaptive_enabled,
            'volatility_thresholds': self.volatility_thresholds,
            'bit_depths': self.bit_depths,
            'delta_encoding': self.delta_encoding,
            'delta_anchor_interval': self.delta_anchor_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionConfig':
        return cls(
            default_bits=data.get('default_bits', 8),
            adaptive_enabled=data.get('adaptive_enabled', True),
            volatility_thresholds=data.get('volatility_thresholds', [0.01, 0.03]),
            bit_depths=data.get('bit_depths', [6, 8, 10]),
            delta_encoding=data.get('delta_encoding', True),
            delta_anchor_interval=data.get('delta_anchor_interval', 32),
        )


@dataclass
class CacheConfig:
    """Unified cache settings — compression config is embedded, not disconnected."""
    max_candles: int = 100000
    persist_to: Optional[Path] = None
    auto_save: bool = True
    save_interval_seconds: int = 300
    max_memory_mb: int = 512

    # Compression settings (Bug 3 fix: actually connected)
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    # Symbol-specific overrides (e.g., {'BTC/USDT': {'bits': 10}})
    symbol_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_symbol_bits(self, symbol: str) -> Optional[int]:
        """Get bit depth override for a symbol, if configured."""
        sym_cfg = self.symbol_configs.get(symbol, {})
        return sym_cfg.get('bits')

    def get_compression_config(self, symbol: Optional[str] = None) -> CompressionConfig:
        """
        Get compression config, optionally merged with symbol-specific overrides.
        Returns a new CompressionConfig with symbol overrides applied.
        """
        if symbol and symbol in self.symbol_configs:
            sym_cfg = self.symbol_configs[symbol]
            # Create a merged copy
            merged = CompressionConfig.from_dict(self.compression.to_dict())
            if 'bits' in sym_cfg:
                merged.default_bits = sym_cfg['bits']
            if 'delta_encoding' in sym_cfg:
                merged.delta_encoding = sym_cfg['delta_encoding']
            if 'delta_anchor_interval' in sym_cfg:
                merged.delta_anchor_interval = sym_cfg['delta_anchor_interval']
            return merged
        return self.compression

    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get raw config dict for specific symbol (backward compat)."""
        return self.symbol_configs.get(symbol, {})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheConfig':
        """Create config from dict."""
        # Handle nested compression config
        comp_data = data.get('compression', {})
        # Also check for flat keys (backward compat with old config format)
        if not comp_data:
            flat_keys = {'default_bits', 'adaptive_enabled', 'volatility_thresholds',
                        'bit_depths', 'delta_encoding', 'delta_anchor_interval'}
            comp_data = {k: data[k] for k in flat_keys if k in data}

        compression = CompressionConfig.from_dict(comp_data) if comp_data else CompressionConfig()

        return cls(
            max_candles=data.get('max_candles', 100000),
            persist_to=Path(data['persist_to']) if data.get('persist_to') else None,
            auto_save=data.get('auto_save', True),
            save_interval_seconds=data.get('save_interval_seconds', 300),
            max_memory_mb=data.get('max_memory_mb', 512),
            compression=compression,
            symbol_configs=data.get('symbol_configs', {}),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'CacheConfig':
        """Load config from YAML file."""
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data.get('cache', {}))
        except ImportError:
            return cls.from_json(path.with_suffix('.json'))

    @classmethod
    def from_json(cls, path: Path) -> 'CacheConfig':
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data.get('cache', {}))

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dict."""
        return {
            'max_candles': self.max_candles,
            'persist_to': str(self.persist_to) if self.persist_to else None,
            'auto_save': self.auto_save,
            'save_interval_seconds': self.save_interval_seconds,
            'max_memory_mb': self.max_memory_mb,
            'compression': self.compression.to_dict(),
            'symbol_configs': self.symbol_configs,
        }

    def save(self, path: Path):
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'cache': self.to_dict()}, f, indent=2)


# Default config — compression is now embedded
DEFAULT_CONFIG = CacheConfig(
    max_candles=100000,
    persist_to=None,
    auto_save=True,
    save_interval_seconds=300,
    compression=CompressionConfig(
        default_bits=8,
        adaptive_enabled=True,
        delta_encoding=True,
        delta_anchor_interval=32,
    ),
    symbol_configs={
        'BTC/USDT': {'bits': 10},  # Higher precision for main pair
        'ETH/USDT': {'bits': 8},
    },
)