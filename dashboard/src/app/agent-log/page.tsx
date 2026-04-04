'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type LogType = 'signal' | 'trade' | 'alert' | 'veto' | 'error';
type FilterTab = 'all' | 'signals' | 'trades' | 'alerts' | 'vetoes';

interface LogEntry {
  id: string;
  timestamp: string;
  type: LogType;
  symbol?: string;
  message: string;
  details?: string;
  side?: 'buy' | 'sell';
}

const MAX_ENTRIES = 500;

/* ------------------------------------------------------------------ */
/*  Badge colors                                                       */
/* ------------------------------------------------------------------ */

function typeBadge(type: LogType, side?: 'buy' | 'sell') {
  const styles: Record<string, string> = {
    signal: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    trade_buy: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    trade_sell: 'bg-red-500/20 text-red-400 border-red-500/30',
    alert: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    veto: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
    error: 'bg-red-600/20 text-red-500 border-red-600/30',
  };

  const key = type === 'trade' ? `trade_${side || 'buy'}` : type;
  const cls = styles[key] || styles.signal;

  const labels: Record<string, string> = {
    signal: 'Signal',
    trade_buy: 'Buy',
    trade_sell: 'Sell',
    alert: 'Alert',
    veto: 'Veto',
    error: 'Error',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border ${cls}`}
    >
      {labels[key] || type}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  Filter tabs                                                        */
/* ------------------------------------------------------------------ */

const FILTER_TABS: { key: FilterTab; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'signals', label: 'Signals' },
  { key: 'trades', label: 'Trades' },
  { key: 'alerts', label: 'Alerts' },
  { key: 'vetoes', label: 'Vetoes' },
];

function filterMatches(entry: LogEntry, tab: FilterTab): boolean {
  if (tab === 'all') return true;
  if (tab === 'signals') return entry.type === 'signal';
  if (tab === 'trades') return entry.type === 'trade';
  if (tab === 'alerts') return entry.type === 'alert' || entry.type === 'error';
  if (tab === 'vetoes') return entry.type === 'veto';
  return true;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

let entryIdCounter = 0;

function makeEntry(type: LogType, data: any): LogEntry {
  entryIdCounter += 1;
  return {
    id: `${type}-${entryIdCounter}-${Date.now()}`,
    timestamp: data.timestamp || new Date().toISOString(),
    type,
    symbol: data.symbol || data.pair || undefined,
    message: data.message || data.reason || data.action || JSON.stringify(data).slice(0, 120),
    details: data.details || data.explanation || undefined,
    side: data.side || data.direction || undefined,
  };
}

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return iso;
  }
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function AgentLogPage() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<FilterTab>('all');
  const [autoScroll, setAutoScroll] = useState(true);

  const scrollRef = useRef<HTMLDivElement>(null);

  /* WebSocket channels */
  const signals = useWebSocket<any>('signals');
  const trades = useWebSocket<any>('trades');
  const alerts = useWebSocket<any>('alerts');
  const vetoes = useWebSocket<any>('vetoes');

  /* Append helper */
  const append = useCallback((entry: LogEntry) => {
    setEntries((prev) => {
      const next = [...prev, entry];
      if (next.length > MAX_ENTRIES) {
        return next.slice(next.length - MAX_ENTRIES);
      }
      return next;
    });
  }, []);

  /* Process incoming data */
  useEffect(() => {
    if (signals.data) append(makeEntry('signal', signals.data));
  }, [signals.data, append]);

  useEffect(() => {
    if (trades.data) append(makeEntry('trade', trades.data));
  }, [trades.data, append]);

  useEffect(() => {
    if (alerts.data) {
      const type: LogType = alerts.data.severity === 'error' ? 'error' : 'alert';
      append(makeEntry(type, alerts.data));
    }
  }, [alerts.data, append]);

  useEffect(() => {
    if (vetoes.data) append(makeEntry('veto', vetoes.data));
  }, [vetoes.data, append]);

  /* Auto-scroll */
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries, autoScroll]);

  /* Export to JSON */
  function exportJSON() {
    const filtered = entries.filter((e) => filterMatches(e, filter));
    const blob = new Blob([JSON.stringify(filtered, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agent-log-${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /* Connection indicators */
  const wsConnected =
    signals.isConnected || trades.isConnected || alerts.isConnected || vetoes.isConnected;

  const filtered = entries.filter((e) => filterMatches(e, filter));

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-2xl font-bold">Agent Log</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time activity feed from the trading agent
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div
              className={`w-2 h-2 rounded-full ${
                wsConnected ? 'bg-profit animate-pulse' : 'bg-loss'
              }`}
            />
            <span className="text-xs text-muted-foreground">
              {wsConnected ? 'Live' : 'Disconnected'}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            {filtered.length} / {entries.length} entries
          </span>
        </div>
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between flex-shrink-0">
        {/* Filter tabs */}
        <div className="flex items-center gap-1 bg-card border border-border rounded-lg p-1">
          {FILTER_TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setFilter(tab.key)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                filter === tab.key
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
              autoScroll
                ? 'border-accent/50 bg-accent/10 text-accent'
                : 'border-border text-muted-foreground hover:text-foreground'
            }`}
          >
            {autoScroll ? 'Auto-scroll ON' : 'Auto-scroll OFF'}
          </button>
          <button
            onClick={exportJSON}
            disabled={filtered.length === 0}
            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-border text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Export JSON
          </button>
          <button
            onClick={() => setEntries([])}
            disabled={entries.length === 0}
            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-border text-muted-foreground hover:text-foreground hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto bg-card border border-border rounded-xl"
      >
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
            {entries.length === 0
              ? 'Waiting for agent activity...'
              : 'No entries match the current filter.'}
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-card border-b border-border">
              <tr className="text-xs text-muted-foreground uppercase tracking-wider">
                <th className="text-left px-4 py-2.5 w-24">Time</th>
                <th className="text-left px-4 py-2.5 w-20">Type</th>
                <th className="text-left px-4 py-2.5 w-24">Symbol</th>
                <th className="text-left px-4 py-2.5">Message</th>
                <th className="text-left px-4 py-2.5 w-64">Details</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry) => (
                <tr
                  key={entry.id}
                  className="border-b border-border/50 hover:bg-muted/30 transition-colors"
                >
                  <td className="px-4 py-2.5 text-xs text-muted-foreground font-mono whitespace-nowrap">
                    {formatTime(entry.timestamp)}
                  </td>
                  <td className="px-4 py-2.5">
                    {typeBadge(entry.type, entry.side)}
                  </td>
                  <td className="px-4 py-2.5 font-medium text-foreground whitespace-nowrap">
                    {entry.symbol || '-'}
                  </td>
                  <td className="px-4 py-2.5 text-foreground">{entry.message}</td>
                  <td className="px-4 py-2.5 text-xs text-muted-foreground truncate max-w-[16rem]">
                    {entry.details || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
