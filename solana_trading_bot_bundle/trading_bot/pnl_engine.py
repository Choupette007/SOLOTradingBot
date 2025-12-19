"""
P&L Engine Module for SOLO Trading Bot

This module provides modularized P&L computation logic including:
- Realized and unrealized P&L calculations
- Performance metrics (win rate, profit factor, expectancy)
- Risk metrics (exposure, drawdown)
- Trade statistics and aggregations
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("TradingBot.PnL")

# Constants
PROFIT_THRESHOLD = 0.01  # Minimum profit/loss to avoid rounding issues (cents)


@dataclass
class PnLMetrics:
    """Container for P&L metrics"""
    # P&L metrics
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Performance metrics
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Time metrics
    avg_hold_time_seconds: float = 0.0
    
    # Fee metrics
    total_fees_usd: float = 0.0
    total_fees_sol: float = 0.0
    
    # Drawdown metrics
    max_drawdown_pct: float = 0.0
    
    # Risk metrics
    current_exposure_pct: float = 0.0
    open_positions_at_loss: int = 0
    largest_open_drawdown: float = 0.0
    worst_performing_token: Optional[str] = None
    worst_performing_pnl: float = 0.0


def calculate_realized_pnl(closed_trades: List[Dict[str, Any]]) -> float:
    """
    Calculate realized P&L from closed trades.
    
    Args:
        closed_trades: List of trade dicts with 'profit' field
        
    Returns:
        Total realized P&L in USD
    """
    total = 0.0
    for trade in closed_trades:
        profit = trade.get("profit")
        if profit is not None:
            try:
                total += float(profit)
            except (ValueError, TypeError):
                pass
    return total


def calculate_unrealized_pnl(open_positions: List[Dict[str, Any]]) -> float:
    """
    Calculate unrealized P&L from open positions.
    
    Args:
        open_positions: List of position dicts with buy_price, current price, and amount
        
    Returns:
        Total unrealized P&L in USD
    """
    total = 0.0
    for pos in open_positions:
        buy_price = pos.get("buy_price")
        current_price = pos.get("price")
        buy_amount = pos.get("buy_amount")
        
        if buy_price and current_price and buy_amount:
            try:
                buy_price = float(buy_price)
                current_price = float(current_price)
                buy_amount = float(buy_amount)
                
                cost = buy_price * buy_amount
                value = current_price * buy_amount
                unrealized = value - cost
                total += unrealized
            except (ValueError, TypeError):
                pass
    return total


def calculate_win_rate(closed_trades: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    """
    Calculate win rate from closed trades.
    
    Args:
        closed_trades: List of trade dicts with 'profit' field
        
    Returns:
        Tuple of (win_rate_percent, winning_count, losing_count, breakeven_count)
    """
    wins = 0
    losses = 0
    breakeven = 0
    
    for trade in closed_trades:
        profit = trade.get("profit")
        if profit is not None:
            try:
                p = float(profit)
                if p > PROFIT_THRESHOLD:
                    wins += 1
                elif p < -PROFIT_THRESHOLD:
                    losses += 1
                else:
                    breakeven += 1
            except (ValueError, TypeError):
                pass
    
    total = wins + losses + breakeven
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    
    return win_rate, wins, losses, breakeven


def calculate_profit_factor(closed_trades: List[Dict[str, Any]]) -> float:
    """
    Calculate profit factor (total wins / total losses).
    
    Args:
        closed_trades: List of trade dicts with 'profit' field
        
    Returns:
        Profit factor
    """
    gross_profit = 0.0
    gross_loss = 0.0
    
    for trade in closed_trades:
        profit = trade.get("profit")
        if profit is not None:
            try:
                p = float(profit)
                if p > 0:
                    gross_profit += p
                elif p < 0:
                    gross_loss += abs(p)
            except (ValueError, TypeError):
                pass
    
    if gross_loss > 0:
        return gross_profit / gross_loss
    elif gross_profit > 0:
        return float('inf')
    return 0.0


def calculate_expectancy(closed_trades: List[Dict[str, Any]]) -> float:
    """
    Calculate expectancy per trade (average P&L per trade).
    
    Args:
        closed_trades: List of trade dicts with 'profit' field
        
    Returns:
        Expectancy in USD
    """
    total_profit = 0.0
    count = 0
    
    for trade in closed_trades:
        profit = trade.get("profit")
        if profit is not None:
            try:
                total_profit += float(profit)
                count += 1
            except (ValueError, TypeError):
                pass
    
    return total_profit / count if count > 0 else 0.0


def calculate_avg_hold_time(closed_trades: List[Dict[str, Any]]) -> float:
    """
    Calculate average hold time in seconds.
    
    Args:
        closed_trades: List of trade dicts with 'buy_time' and 'sell_time' fields
        
    Returns:
        Average hold time in seconds
    """
    total_duration = 0.0
    count = 0
    
    for trade in closed_trades:
        buy_time = trade.get("buy_time")
        sell_time = trade.get("sell_time")
        
        if buy_time is not None and sell_time is not None:
            try:
                duration = float(sell_time) - float(buy_time)
                if duration > 0:
                    total_duration += duration
                    count += 1
            except (ValueError, TypeError):
                pass
    
    return total_duration / count if count > 0 else 0.0


def calculate_max_drawdown(closed_trades: List[Dict[str, Any]]) -> float:
    """
    Calculate maximum drawdown from closed trades.
    
    Args:
        closed_trades: List of trade dicts with 'profit' and time fields
        
    Returns:
        Maximum drawdown as a percentage
    """
    if not closed_trades:
        return 0.0
    
    # Sort by time
    sorted_trades = sorted(
        closed_trades,
        key=lambda t: t.get("sell_time") or t.get("buy_time") or 0
    )
    
    # Calculate cumulative P&L curve
    cumulative_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    
    for trade in sorted_trades:
        profit = trade.get("profit")
        if profit is not None:
            try:
                cumulative_pnl += float(profit)
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                
                drawdown = peak - cumulative_pnl
                if drawdown > max_dd:
                    max_dd = drawdown
            except (ValueError, TypeError):
                pass
    
    # Convert to percentage (relative to peak if peak > 0)
    if peak > 0:
        return (max_dd / peak) * 100.0
    return 0.0


def calculate_risk_metrics(
    open_positions: List[Dict[str, Any]], 
    wallet_balance: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate risk metrics for open positions.
    
    Args:
        open_positions: List of open position dicts
        wallet_balance: Current wallet balance in USD (optional)
        
    Returns:
        Dict with risk metrics
    """
    total_exposure = 0.0
    positions_at_loss = 0
    largest_drawdown = 0.0
    worst_token = None
    worst_pnl = 0.0
    
    for pos in open_positions:
        buy_price = pos.get("buy_price")
        current_price = pos.get("price")
        buy_amount = pos.get("buy_amount")
        symbol = pos.get("symbol")
        
        if buy_price and current_price and buy_amount:
            try:
                buy_price = float(buy_price)
                current_price = float(current_price)
                buy_amount = float(buy_amount)
                
                cost = buy_price * buy_amount
                value = current_price * buy_amount
                unrealized = value - cost
                
                total_exposure += cost
                
                if unrealized < 0:
                    positions_at_loss += 1
                    if unrealized < largest_drawdown:
                        largest_drawdown = unrealized
                    if unrealized < worst_pnl:
                        worst_pnl = unrealized
                        worst_token = symbol
            except (ValueError, TypeError):
                pass
    
    exposure_pct = 0.0
    if wallet_balance and wallet_balance > 0:
        exposure_pct = (total_exposure / wallet_balance) * 100.0
    
    return {
        "current_exposure_pct": exposure_pct,
        "open_positions_at_loss": positions_at_loss,
        "largest_open_drawdown": largest_drawdown,
        "worst_performing_token": worst_token,
        "worst_performing_pnl": worst_pnl,
    }


def calculate_comprehensive_metrics(
    closed_trades: List[Dict[str, Any]],
    open_positions: List[Dict[str, Any]],
    wallet_balance: Optional[float] = None,
) -> PnLMetrics:
    """
    Calculate comprehensive P&L metrics.
    
    Args:
        closed_trades: List of closed trade dicts
        open_positions: List of open position dicts
        wallet_balance: Current wallet balance in USD (optional)
        
    Returns:
        PnLMetrics object with all calculated metrics
    """
    metrics = PnLMetrics()
    
    # P&L calculations
    metrics.realized_pnl = calculate_realized_pnl(closed_trades)
    metrics.unrealized_pnl = calculate_unrealized_pnl(open_positions)
    metrics.total_pnl = metrics.realized_pnl + metrics.unrealized_pnl
    
    # Win rate and trade counts
    win_rate, wins, losses, breakeven = calculate_win_rate(closed_trades)
    metrics.win_rate = win_rate
    metrics.winning_trades = wins
    metrics.losing_trades = losses
    metrics.breakeven_trades = breakeven
    metrics.total_trades = wins + losses + breakeven
    
    # Average win/loss
    win_total = 0.0
    loss_total = 0.0
    
    for t in closed_trades:
        try:
            profit = float(t.get("profit", 0))
            if profit > 0:
                win_total += profit
            elif profit < 0:
                loss_total += abs(profit)
        except (ValueError, TypeError):
            pass
    
    metrics.avg_win = win_total / wins if wins > 0 else 0.0
    metrics.avg_loss = loss_total / losses if losses > 0 else 0.0
    
    # Performance metrics
    metrics.profit_factor = calculate_profit_factor(closed_trades)
    metrics.expectancy = calculate_expectancy(closed_trades)
    metrics.avg_hold_time_seconds = calculate_avg_hold_time(closed_trades)
    
    # Drawdown
    metrics.max_drawdown_pct = calculate_max_drawdown(closed_trades)
    
    # Risk metrics
    risk = calculate_risk_metrics(open_positions, wallet_balance)
    metrics.current_exposure_pct = risk["current_exposure_pct"]
    metrics.open_positions_at_loss = risk["open_positions_at_loss"]
    metrics.largest_open_drawdown = risk["largest_open_drawdown"]
    metrics.worst_performing_token = risk["worst_performing_token"]
    metrics.worst_performing_pnl = risk["worst_performing_pnl"]
    
    # Fee calculation (basic - can be enhanced if fee data is available)
    # For now, set to 0 as fee data might not be in all trade records
    metrics.total_fees_usd = 0.0
    metrics.total_fees_sol = 0.0
    
    return metrics


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m", "1d 5h")
    """
    if seconds <= 0:
        return "0s"
    
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 and days == 0:  # Skip minutes if showing days
        parts.append(f"{minutes}m")
    if secs > 0 and days == 0 and hours == 0:  # Only show seconds for short durations
        parts.append(f"{secs}s")
    
    return " ".join(parts[:2]) if parts else "0s"  # Show max 2 units
