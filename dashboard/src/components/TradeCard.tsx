"use client";

import { useEffect, useState } from 'react';

export interface ActiveTrade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  lots: number;
  entry_price: number;
  current_price: number;
  sl: number;
  tp: number;
  pnl: number;
  opened_at: string;
}

interface TradeCardProps {
  trade: ActiveTrade;
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function calculatePips(symbol: string, entry: number, current: number, side: 'buy' | 'sell'): number {
  const multiplier = symbol.includes('JPY') ? 100 : symbol.includes('XAU') ? 10 : 10000;
  const diff = side === 'buy' ? current - entry : entry - current;
  return Math.round(diff * multiplier * 10) / 10;
}

export default function TradeCard({ trade: raw }: TradeCardProps) {
  // Ensure all numeric fields are valid numbers
  const trade = {
    ...raw,
    entry_price: Number(raw.entry_price) || 0,
    current_price: Number(raw.current_price) || 0,
    sl: Number(raw.sl) || 0,
    tp: Number(raw.tp) || 0,
    pnl: Number(raw.pnl) || 0,
    lots: Number(raw.lots) || 0,
  };

  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const openTime = new Date(trade.opened_at).getTime();
    const update = () => setElapsed(Date.now() - openTime);
    update();
    const interval = setInterval(update, 1000);
    return () => clearInterval(interval);
  }, [trade.opened_at]);

  const isBuy = trade.side === 'buy';
  const pnlColor = trade.pnl >= 0 ? 'text-profit' : 'text-loss';
  const sideColor = isBuy ? 'bg-profit' : 'bg-loss';

  // Progress between SL and TP (0 = at SL, 1 = at TP)
  const range = trade.tp - trade.sl;
  const progressRaw = range !== 0 ? (trade.current_price - trade.sl) / range : 0.5;
  const progress = Math.max(0, Math.min(1, progressRaw));

  const slPips = calculatePips(trade.symbol, trade.entry_price, trade.sl, trade.side);
  const tpPips = calculatePips(trade.symbol, trade.entry_price, trade.tp, trade.side);

  return (
    <div className="bg-card rounded-lg border border-border p-4 space-y-3">
      {/* Header: Symbol, direction, lots */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-foreground">{trade.symbol}</span>
          <span className={`${sideColor} text-white text-[10px] font-bold px-1.5 py-0.5 rounded uppercase`}>
            {trade.side}
          </span>
          <span className="text-xs text-muted-foreground">{trade.lots} lots</span>
        </div>
        <span className="text-xs text-muted-foreground">{formatDuration(elapsed)}</span>
      </div>

      {/* Prices row */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-[10px] text-muted-foreground uppercase">Entry</div>
          <div className="text-xs font-mono text-foreground">{trade.entry_price.toFixed(trade.symbol.includes('XAU') ? 2 : 5)}</div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground uppercase">Current</div>
          <div className="text-xs font-mono text-foreground">{trade.current_price.toFixed(trade.symbol.includes('XAU') ? 2 : 5)}</div>
        </div>
        <div>
          <div className="text-[10px] text-muted-foreground uppercase">P&L</div>
          <div className={`text-xs font-mono font-bold ${pnlColor}`}>
            {trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(2)}
          </div>
        </div>
      </div>

      {/* SL / TP with pips */}
      <div className="flex justify-between text-[10px]">
        <span className="text-loss">
          SL: {trade.sl.toFixed(trade.symbol.includes('XAU') ? 2 : 5)} ({Math.abs(slPips)}p)
        </span>
        <span className="text-profit">
          TP: {trade.tp.toFixed(trade.symbol.includes('XAU') ? 2 : 5)} ({Math.abs(tpPips)}p)
        </span>
      </div>

      {/* Progress bar */}
      <div className="relative h-2 bg-muted rounded-full overflow-hidden">
        {/* SL zone (red) */}
        <div
          className="absolute inset-y-0 left-0 bg-loss/30 rounded-l-full"
          style={{ width: '30%' }}
        />
        {/* TP zone (green) */}
        <div
          className="absolute inset-y-0 right-0 bg-profit/30 rounded-r-full"
          style={{ width: '30%' }}
        />
        {/* Price position indicator */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-accent border-2 border-white shadow"
          style={{ left: `calc(${progress * 100}% - 5px)` }}
        />
      </div>
    </div>
  );
}
