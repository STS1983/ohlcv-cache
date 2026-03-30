"""
Configuration system for OHLCV Cache.
No more hardcoded values!
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json


@dataclass
class CompressionConfig:
    """Compression settings."""
    default_bits: int = 8
    adaptive_enabled: bool = True
    volatility_thresholds: list = field(default_factory=lambda: [0.01, 0.03])
    bit_depths: list = field(default_factory=lambda: [6, 8, 10])
    delta_encoding: bool = True
    
    def get_bit_depth(self, volatility: float) -> int:
        """Get optimal bit depth for given volatility."""
        if not self.adaptive_enabled:
            return self.default_bits
        
        for i, threshold in enumerate(self.volatility_thresholds):
            if volatility < threshold:
                return self.bit_depths[i]
        
        return self.bit_depths[-1]  # Highest for most volatile


@dataclass
class CacheConfig:
    """Cache settings."""
    max_candles: int = 100000
    persist_to: Optional[Path] = None
    auto_save: bool = True
    save_interval_seconds: int = 300
    max_memory_mb: int = 512
    
    # Symbol-specific settings
    symbol_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get config for specific symbol."""
        return self.symbol_configs.get(symbol, {})
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheConfig':
        """Create config from dict."""
        return cls(
            max_candles=data.get('max_candles', 100000),
            persist_to=Path(data['persist_to']) if data.get('persist_to') else None,
            auto_save=data.get('auto_save', True),
            save_interval_seconds=data.get('save_interval_seconds', 300),
            max_memory_mb=data.get('max_memory_mb', 512),
            symbol_configs=data.get('symbol_configs', {})
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
            # Fallback to JSON
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
            'symbol_configs': self.symbol_configs
        }
    
    def save(self, path: Path):
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'cache': self.to_dict()}, f, indent=2)


# Default config
DEFAULT_CONFIG = CacheConfig(
    max_candles=100000,
    persist_to=None,
    auto_save=True,
    save_interval_seconds=300,
    symbol_configs={
        'BTC/USDT': {'bits': 10},  # Higher precision for main pair
        'ETH/USDT': {'bits': 8},
    }
)
