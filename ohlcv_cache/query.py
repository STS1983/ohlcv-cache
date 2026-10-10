"""
Query builder for flexible OHLCV data retrieval.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
from datetime import datetime


@dataclass
class QueryBuilder:
    """
    Flexible query builder for OHLCV data.

    Usage:
        query = QueryBuilder(
            symbol='BTC/USDT',
            start=0,
            end=100,
            columns=['close', 'volume']
        )

        candles = cache.query(query)
    """

    symbol: Optional[str] = None
    start: Optional[Union[str, datetime, int]] = None
    end: Optional[Union[str, datetime, int]] = None
    columns: Optional[List[str]] = None

    # Computed fields (set after build)
    start_idx: Optional[int] = field(default=None, init=False)
    end_idx: Optional[int] = field(default=None, init=False)

    def with_symbol(self, symbol: str) -> 'QueryBuilder':
        """Set symbol and return self for chaining."""
        self.symbol = symbol
        return self

    def with_range(
        self,
        start: Optional[Union[str, datetime, int]] = None,
        end: Optional[Union[str, datetime, int]] = None,
    ) -> 'QueryBuilder':
        """Set date range and return self for chaining."""
        self.start = start
        self.end = end
        return self

    def with_columns(self, columns: List[str]) -> 'QueryBuilder':
        """Set columns to retrieve and return self for chaining."""
        self.columns = columns
        return self

    def with_limit(self, limit: int) -> 'QueryBuilder':
        """Set limit on number of candles to return."""
        self.end_idx = limit if self.start_idx is None else self.start_idx + limit
        return self

    def with_offset(self, offset: int) -> 'QueryBuilder':
        """Set offset for pagination."""
        self.start_idx = offset
        return self

    def build(self) -> 'QueryBuilder':
        """Build and validate the query."""
        if self.columns:
            valid_columns = {'open', 'high', 'low', 'close', 'volume', 'o', 'h', 'l', 'c', 'v'}
            for col in self.columns:
                if col.lower() not in valid_columns:
                    raise ValueError(f"Invalid column: {col}")
        return self

    def to_dict(self) -> dict:
        """Convert query to dictionary."""
        return {
            'symbol': self.symbol,
            'start': self._serialize_datetime(self.start) if isinstance(self.start, datetime) else self.start,
            'end': self._serialize_datetime(self.end) if isinstance(self.end, datetime) else self.end,
            'columns': self.columns,
        }

    def _serialize_datetime(self, dt: Optional[Union[str, datetime, int]]) -> Optional[str]:
        """Serialize datetime to ISO string."""
        if isinstance(dt, datetime):
            return dt.isoformat()
        return dt

    @classmethod
    def from_dict(cls, data: dict) -> 'QueryBuilder':
        """Create query from dictionary."""
        return cls(
            symbol=data.get('symbol'),
            start=data.get('start'),
            end=data.get('end'),
            columns=data.get('columns'),
        )

    def __str__(self) -> str:
        """Human-readable query string."""
        parts = []
        if self.symbol:
            parts.append(f"symbol={self.symbol}")
        if self.start is not None:
            parts.append(f"start={self.start}")
        if self.end is not None:
            parts.append(f"end={self.end}")
        if self.columns:
            parts.append(f"columns={','.join(self.columns)}")
        return f"Query({', '.join(parts)})" if parts else "Query(all)"


def last_n_candles(symbol: str, n: int, columns: Optional[List[str]] = None) -> QueryBuilder:
    """Query last N candles for a symbol."""
    return QueryBuilder(
        symbol=symbol,
        columns=columns,
    ).with_limit(n)


def candles_between(
    symbol: str,
    start: Union[str, datetime],
    end: Union[str, datetime],
    columns: Optional[List[str]] = None,
) -> QueryBuilder:
    """Query candles between two dates."""
    return QueryBuilder(
        symbol=symbol,
        start=start,
        end=end,
        columns=columns,
    )


def latest_candles(symbols: List[str], n: int = 1) -> QueryBuilder:
    """Query latest N candles for multiple symbols."""
    return QueryBuilder(
        start=0,
        end=n,
    )