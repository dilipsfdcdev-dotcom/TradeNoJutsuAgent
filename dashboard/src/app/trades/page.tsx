'use client';

import { useState, useEffect, useMemo, useCallback, Fragment } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  entry_price: number;
  exit_price?: number;
  quantity: number;
  pnl?: number;
  rr_ratio?: number;
  confidence?: number;
  quality?: number;
  status: 'open' | 'closed' | 'pending';
  opened_at: string;
  closed_at?: string;
  ai_reasoning?: string;
  ai_review?: string;
}

type SortKey =
  | 'opened_at'
  | 'symbol'
  | 'side'
  | 'entry_price'
  | 'exit_price'
  | 'pnl'
  | 'rr_ratio'
  | 'confidence'
  | 'quality';

type SortDir = 'asc' | 'desc';

const PAGE_SIZE = 50;

export default function TradesPage() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [symbolFilter, setSymbolFilter] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('');

  // Sorting
  const [sortKey, setSortKey] = useState<SortKey>('opened_at');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  // Pagination
  const [page, setPage] = useState(1);

  // Expanded row
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const fetchTrades = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams();
      if (symbolFilter) params.set('symbol', symbolFilter);
      if (statusFilter) params.set('status', statusFilter);
      const query = params.toString();
      const res = await fetch(
        `${API_BASE}/api/trades${query ? `?${query}` : ''}`,
        { signal: AbortSignal.timeout(10000) }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Trade[] = await res.json();
      setTrades(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch trades');
    } finally {
      setLoading(false);
    }
  }, [symbolFilter, statusFilter]);

  useEffect(() => {
    fetchTrades();
  }, [fetchTrades]);

  // Unique symbols for dropdown
  const symbols = useMemo(() => {
    const set = new Set(trades.map((t) => t.symbol));
    return Array.from(set).sort();
  }, [trades]);

  // Sort
  const sorted = useMemo(() => {
    const arr = [...trades];
    arr.sort((a, b) => {
      let av: any = a[sortKey] ?? '';
      let bv: any = b[sortKey] ?? '';
      if (typeof av === 'string') av = av.toLowerCase();
      if (typeof bv === 'string') bv = bv.toLowerCase();
      if (av < bv) return sortDir === 'asc' ? -1 : 1;
      if (av > bv) return sortDir === 'asc' ? 1 : -1;
      return 0;
    });
    return arr;
  }, [trades, sortKey, sortDir]);

  // Paginate
  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const paginated = useMemo(
    () => sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE),
    [sorted, page]
  );

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [symbolFilter, statusFilter, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }

  function sortIndicator(key: SortKey) {
    if (sortKey !== key) return '';
    return sortDir === 'asc' ? ' ▲' : ' ▼';
  }

  function exportCsv() {
    const headers = [
      'Date',
      'Symbol',
      'Direction',
      'Entry',
      'Exit',
      'Quantity',
      'P&L',
      'R:R',
      'Confidence',
      'Quality',
      'Status',
    ];
    const rows = sorted.map((t) => [
      t.opened_at,
      t.symbol,
      t.side.toUpperCase(),
      t.entry_price,
      t.exit_price ?? '',
      t.quantity,
      t.pnl ?? '',
      t.rr_ratio ?? '',
      t.confidence ?? '',
      t.quality ?? '',
      t.status,
    ]);
    const csv = [headers, ...rows].map((r) => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trades_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function formatPnl(val?: number) {
    if (val === undefined || val === null) return '-';
    const sign = val >= 0 ? '+' : '';
    return `${sign}${val.toFixed(2)}`;
  }

  function formatNum(val?: number, decimals = 2) {
    if (val === undefined || val === null) return '-';
    return val.toFixed(decimals);
  }

  const thClass =
    'px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider cursor-pointer select-none hover:text-foreground transition-colors';

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Trade History</h1>
        <button
          onClick={exportCsv}
          className="px-4 py-2 text-sm font-medium rounded-lg bg-accent text-accent-foreground hover:opacity-90 transition-opacity"
        >
          Export CSV
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-4">
        <div>
          <label className="block text-xs text-muted-foreground mb-1">
            Symbol
          </label>
          <select
            value={symbolFilter}
            onChange={(e) => setSymbolFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-muted border border-border text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
          >
            <option value="">All Symbols</option>
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-xs text-muted-foreground mb-1">
            Status
          </label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-muted border border-border text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent"
          >
            <option value="">All</option>
            <option value="open">Open</option>
            <option value="closed">Closed</option>
          </select>
        </div>
        <div className="ml-auto text-sm text-muted-foreground">
          {sorted.length} trade{sorted.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-loss/10 border border-loss/30 text-loss text-sm">
          {error}
          <button
            onClick={fetchTrades}
            className="ml-3 underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Table */}
      <div className="border border-border rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-muted">
              <tr>
                <th className={thClass} onClick={() => handleSort('opened_at')}>
                  Date{sortIndicator('opened_at')}
                </th>
                <th className={thClass} onClick={() => handleSort('symbol')}>
                  Symbol{sortIndicator('symbol')}
                </th>
                <th className={thClass} onClick={() => handleSort('side')}>
                  Direction{sortIndicator('side')}
                </th>
                <th className={thClass} onClick={() => handleSort('entry_price')}>
                  Entry{sortIndicator('entry_price')}
                </th>
                <th className={thClass} onClick={() => handleSort('exit_price')}>
                  Exit{sortIndicator('exit_price')}
                </th>
                <th className={thClass} onClick={() => handleSort('pnl')}>
                  P&L{sortIndicator('pnl')}
                </th>
                <th className={thClass} onClick={() => handleSort('rr_ratio')}>
                  R:R{sortIndicator('rr_ratio')}
                </th>
                <th className={thClass} onClick={() => handleSort('confidence')}>
                  Confidence{sortIndicator('confidence')}
                </th>
                <th className={thClass} onClick={() => handleSort('quality')}>
                  Quality{sortIndicator('quality')}
                </th>
                <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {loading && (
                <tr>
                  <td colSpan={10} className="px-3 py-12 text-center text-muted-foreground">
                    Loading trades...
                  </td>
                </tr>
              )}
              {!loading && paginated.length === 0 && (
                <tr>
                  <td colSpan={10} className="px-3 py-12 text-center text-muted-foreground">
                    No trades found.
                  </td>
                </tr>
              )}
              {!loading &&
                paginated.map((trade) => {
                  const isWin = trade.pnl !== undefined && trade.pnl > 0;
                  const isLoss = trade.pnl !== undefined && trade.pnl < 0;
                  const isExpanded = expandedId === trade.id;

                  return (
                    <Fragment key={trade.id}>
                      <tr
                        onClick={() =>
                          setExpandedId(isExpanded ? null : trade.id)
                        }
                        className={`cursor-pointer hover:bg-muted/50 transition-colors ${
                          isWin
                            ? 'border-l-2 border-l-profit'
                            : isLoss
                            ? 'border-l-2 border-l-loss'
                            : ''
                        }`}
                      >
                        <td className="px-3 py-2.5 whitespace-nowrap text-muted-foreground">
                          {new Date(trade.opened_at).toLocaleDateString()}{' '}
                          <span className="text-xs">
                            {new Date(trade.opened_at).toLocaleTimeString([], {
                              hour: '2-digit',
                              minute: '2-digit',
                            })}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 font-medium">
                          {trade.symbol}
                        </td>
                        <td className="px-3 py-2.5">
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-xs font-bold uppercase ${
                              trade.side === 'buy'
                                ? 'bg-profit/20 text-profit'
                                : 'bg-loss/20 text-loss'
                            }`}
                          >
                            {trade.side}
                          </span>
                        </td>
                        <td className="px-3 py-2.5 font-mono">
                          {formatNum(trade.entry_price, 5)}
                        </td>
                        <td className="px-3 py-2.5 font-mono">
                          {formatNum(trade.exit_price, 5)}
                        </td>
                        <td
                          className={`px-3 py-2.5 font-mono font-semibold ${
                            isWin
                              ? 'text-profit'
                              : isLoss
                              ? 'text-loss'
                              : 'text-muted-foreground'
                          }`}
                        >
                          {formatPnl(trade.pnl)}
                        </td>
                        <td className="px-3 py-2.5 font-mono">
                          {formatNum(trade.rr_ratio)}
                        </td>
                        <td className="px-3 py-2.5">
                          {trade.confidence !== undefined
                            ? `${(trade.confidence * 100).toFixed(0)}%`
                            : '-'}
                        </td>
                        <td className="px-3 py-2.5">
                          {trade.quality !== undefined
                            ? `${trade.quality.toFixed(1)}/10`
                            : '-'}
                        </td>
                        <td className="px-3 py-2.5">
                          <span
                            className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                              trade.status === 'open'
                                ? 'bg-accent/20 text-accent'
                                : trade.status === 'closed'
                                ? 'bg-muted text-muted-foreground'
                                : 'bg-yellow-500/20 text-yellow-500'
                            }`}
                          >
                            {trade.status}
                          </span>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr>
                          <td
                            colSpan={10}
                            className="px-6 py-4 bg-muted/30 border-l-2 border-l-accent"
                          >
                            <div className="space-y-3">
                              {trade.ai_reasoning && (
                                <div>
                                  <h4 className="text-xs font-semibold text-muted-foreground uppercase mb-1">
                                    AI Reasoning
                                  </h4>
                                  <p className="text-sm whitespace-pre-wrap">
                                    {trade.ai_reasoning}
                                  </p>
                                </div>
                              )}
                              {trade.ai_review && (
                                <div>
                                  <h4 className="text-xs font-semibold text-muted-foreground uppercase mb-1">
                                    AI Review
                                  </h4>
                                  <p className="text-sm whitespace-pre-wrap">
                                    {trade.ai_review}
                                  </p>
                                </div>
                              )}
                              {!trade.ai_reasoning && !trade.ai_review && (
                                <p className="text-sm text-muted-foreground">
                                  No AI reasoning or review available for this
                                  trade.
                                </p>
                              )}
                              <div className="flex gap-6 text-xs text-muted-foreground pt-2 border-t border-border">
                                <span>
                                  Opened: {new Date(trade.opened_at).toLocaleString()}
                                </span>
                                {trade.closed_at && (
                                  <span>
                                    Closed: {new Date(trade.closed_at).toLocaleString()}
                                  </span>
                                )}
                                <span>Qty: {trade.quantity}</span>
                                <span>ID: {trade.id}</span>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Page {page} of {totalPages}
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1.5 text-sm rounded-lg border border-border hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              let pageNum: number;
              if (totalPages <= 5) {
                pageNum = i + 1;
              } else if (page <= 3) {
                pageNum = i + 1;
              } else if (page >= totalPages - 2) {
                pageNum = totalPages - 4 + i;
              } else {
                pageNum = page - 2 + i;
              }
              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                    pageNum === page
                      ? 'bg-accent text-accent-foreground border-accent'
                      : 'border-border hover:bg-muted'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1.5 text-sm rounded-lg border border-border hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

