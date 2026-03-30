'use client';

import { useState, useEffect, useCallback, Fragment } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
} from 'recharts';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface BacktestTrade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  entry_price: number;
  exit_price: number;
  pnl: number;
  rr_ratio: number;
  opened_at: string;
  closed_at: string;
}

interface EquityPoint {
  time: string;
  balance: number;
  equity: number;
}

interface BacktestResult {
  id: string;
  symbol: string;
  mode: string;
  initial_balance: number;
  start_date: string;
  end_date: string;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  max_drawdown: number;
  total_pnl: number;
  equity_curve: EquityPoint[];
  trades: BacktestTrade[];
  created_at: string;
}

const SYMBOLS = [
  'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
  'EURGBP', 'EURJPY', 'GBPJPY', 'XAUUSD', 'BTCUSD',
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatCurrency(value: number): string {
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/* ------------------------------------------------------------------ */
/*  Tooltip                                                            */
/* ------------------------------------------------------------------ */

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: string;
}

function ChartTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} className="text-sm font-mono" style={{ color: entry.color }}>
          {entry.dataKey === 'balance' ? 'Balance' : 'Equity'}: {formatCurrency(entry.value)}
        </p>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Page Component                                                     */
/* ------------------------------------------------------------------ */

export default function BacktestPage() {
  // Form state
  const [symbol, setSymbol] = useState(SYMBOLS[0]);
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-12-31');
  const [mode, setMode] = useState<'rules' | 'ai'>('ai');
  const [initialBalance, setInitialBalance] = useState(10000);

  // Run state
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  // Results
  const [results, setResults] = useState<BacktestResult | null>(null);
  const [loadingResults, setLoadingResults] = useState(true);
  const [resultError, setResultError] = useState<string | null>(null);

  // Trade list pagination
  const PAGE_SIZE = 20;
  const [tradePage, setTradePage] = useState(1);

  // Expanded trade row
  const [expandedTradeId, setExpandedTradeId] = useState<string | null>(null);

  /* Fetch latest backtest results on mount */
  const fetchResults = useCallback(async () => {
    setLoadingResults(true);
    setResultError(null);
    try {
      const res = await fetch(`${API_BASE}/api/backtest-results`, {
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: BacktestResult = await res.json();
      setResults(data);
    } catch (err: any) {
      setResultError(err.message || 'Failed to fetch backtest results');
    } finally {
      setLoadingResults(false);
    }
  }, []);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  /* Submit new backtest */
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setRunning(true);
    setRunError(null);
    try {
      const res = await fetch(`${API_BASE}/api/backtest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          start_date: startDate,
          end_date: endDate,
          mode,
          initial_balance: initialBalance,
        }),
        signal: AbortSignal.timeout(120000),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }
      const data: BacktestResult = await res.json();
      setResults(data);
      setTradePage(1);
      setExpandedTradeId(null);
    } catch (err: any) {
      setRunError(err.message || 'Backtest failed');
    } finally {
      setRunning(false);
    }
  }

  /* Derived values for results */
  const trades = results?.trades ?? [];
  const totalTradePages = Math.max(1, Math.ceil(trades.length / PAGE_SIZE));
  const paginatedTrades = trades.slice(
    (tradePage - 1) * PAGE_SIZE,
    tradePage * PAGE_SIZE,
  );

  const chartData = (results?.equity_curve ?? []).map((p) => ({
    ...p,
    formattedTime: formatDate(p.time),
  }));

  const startBalance = results?.initial_balance ?? 0;
  const endEquity = chartData.length > 0 ? chartData[chartData.length - 1].equity : startBalance;
  const isProfit = endEquity >= startBalance;

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  const inputClass =
    'w-full px-3 py-2 rounded-lg bg-muted border border-border text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent';

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Backtest</h1>

      {/* ---- Run Form ---- */}
      <form
        onSubmit={handleSubmit}
        className="bg-card border border-border rounded-xl p-5 space-y-4"
      >
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Run New Backtest
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Symbol */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Symbol</label>
            <select
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              className={inputClass}
            >
              {SYMBOLS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>

          {/* Start Date */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className={inputClass}
            />
          </div>

          {/* End Date */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className={inputClass}
            />
          </div>

          {/* Mode */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Mode</label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as 'rules' | 'ai')}
              className={inputClass}
            >
              <option value="ai">AI</option>
              <option value="rules">Rules</option>
            </select>
          </div>

          {/* Initial Balance */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Initial Balance ($)
            </label>
            <input
              type="number"
              min={100}
              step={100}
              value={initialBalance}
              onChange={(e) => setInitialBalance(Number(e.target.value))}
              className={inputClass}
            />
          </div>
        </div>

        {runError && (
          <div className="p-3 rounded-lg bg-loss/10 border border-loss/30 text-loss text-sm">
            {runError}
          </div>
        )}

        <button
          type="submit"
          disabled={running}
          className="px-5 py-2 text-sm font-medium rounded-lg bg-accent text-accent-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >
          {running ? 'Running Backtest...' : 'Run Backtest'}
        </button>
      </form>

      {/* ---- Results ---- */}
      {loadingResults && (
        <div className="text-center text-muted-foreground py-12">
          Loading latest backtest results...
        </div>
      )}

      {resultError && !results && (
        <div className="p-3 rounded-lg bg-loss/10 border border-loss/30 text-loss text-sm">
          {resultError}
          <button onClick={fetchResults} className="ml-3 underline hover:no-underline">
            Retry
          </button>
        </div>
      )}

      {results && (
        <>
          {/* Summary Metrics */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                Results Summary
              </h2>
              <span className="text-xs text-muted-foreground">
                {results.symbol} &middot; {results.mode.toUpperCase()} &middot;{' '}
                {results.start_date} to {results.end_date}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
              {/* Win Rate */}
              <div className="bg-muted/50 rounded-lg p-4 text-center">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                  Win Rate
                </p>
                <p
                  className={`text-xl font-bold ${
                    results.win_rate >= 50 ? 'text-profit' : 'text-loss'
                  }`}
                >
                  {results.win_rate.toFixed(1)}%
                </p>
              </div>

              {/* Profit Factor */}
              <div className="bg-muted/50 rounded-lg p-4 text-center">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                  Profit Factor
                </p>
                <p
                  className={`text-xl font-bold ${
                    results.profit_factor >= 1 ? 'text-profit' : 'text-loss'
                  }`}
                >
                  {results.profit_factor.toFixed(2)}
                </p>
              </div>

              {/* Max Drawdown */}
              <div className="bg-muted/50 rounded-lg p-4 text-center">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                  Max Drawdown
                </p>
                <p className="text-xl font-bold text-loss">
                  {results.max_drawdown.toFixed(1)}%
                </p>
              </div>

              {/* Total P&L */}
              <div className="bg-muted/50 rounded-lg p-4 text-center">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                  Total P&L
                </p>
                <p
                  className={`text-xl font-bold font-mono ${
                    results.total_pnl >= 0 ? 'text-profit' : 'text-loss'
                  }`}
                >
                  {results.total_pnl >= 0 ? '+' : ''}
                  {formatCurrency(results.total_pnl)}
                </p>
              </div>

              {/* Total Trades */}
              <div className="bg-muted/50 rounded-lg p-4 text-center">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                  Total Trades
                </p>
                <p className="text-xl font-bold text-foreground">
                  {results.total_trades}
                </p>
              </div>
            </div>
          </div>

          {/* Equity Curve */}
          {chartData.length > 0 && (
            <div className="bg-card border border-border rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                  Equity Curve
                </h2>
                <span
                  className={`text-xs font-mono ${
                    isProfit ? 'text-profit' : 'text-loss'
                  }`}
                >
                  {isProfit ? '+' : ''}
                  {formatCurrency(endEquity - startBalance)}
                </span>
              </div>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={chartData}
                    margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
                  >
                    <defs>
                      <linearGradient id="btEquityGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop
                          offset="5%"
                          stopColor={isProfit ? '#22c55e' : '#ef4444'}
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor={isProfit ? '#22c55e' : '#ef4444'}
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                    <XAxis
                      dataKey="formattedTime"
                      tick={{ fill: '#a3a3a3', fontSize: 10 }}
                      tickLine={false}
                      axisLine={{ stroke: '#2a2a4a' }}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      tick={{ fill: '#a3a3a3', fontSize: 10 }}
                      tickLine={false}
                      axisLine={{ stroke: '#2a2a4a' }}
                      tickFormatter={(v: number) => formatCurrency(v)}
                      domain={['dataMin - 100', 'dataMax + 100']}
                      width={80}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <ReferenceLine
                      y={startBalance}
                      stroke="#3b82f6"
                      strokeDasharray="3 3"
                      strokeOpacity={0.5}
                    />
                    <Area
                      type="monotone"
                      dataKey="equity"
                      fill="url(#btEquityGradient)"
                      stroke="none"
                    />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke={isProfit ? '#22c55e' : '#ef4444'}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4, fill: '#fff' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="balance"
                      stroke="#3b82f6"
                      strokeWidth={1}
                      strokeDasharray="4 4"
                      dot={false}
                      activeDot={{ r: 3, fill: '#3b82f6' }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Trade List */}
          {trades.length > 0 && (
            <div className="bg-card border border-border rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
                  Trades ({trades.length})
                </h2>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-muted">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Date
                      </th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Symbol
                      </th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Direction
                      </th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Entry
                      </th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Exit
                      </th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        P&L
                      </th>
                      <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        R:R
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {paginatedTrades.map((trade) => {
                      const isWin = trade.pnl > 0;
                      const isLoss = trade.pnl < 0;
                      const isExpanded = expandedTradeId === trade.id;

                      return (
                        <Fragment key={trade.id}>
                          <tr
                            onClick={() =>
                              setExpandedTradeId(isExpanded ? null : trade.id)
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
                              {new Date(trade.opened_at).toLocaleDateString()}
                            </td>
                            <td className="px-3 py-2.5 font-medium">{trade.symbol}</td>
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
                              {trade.entry_price.toFixed(5)}
                            </td>
                            <td className="px-3 py-2.5 font-mono">
                              {trade.exit_price.toFixed(5)}
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
                              {trade.pnl >= 0 ? '+' : ''}
                              {trade.pnl.toFixed(2)}
                            </td>
                            <td className="px-3 py-2.5 font-mono">
                              {trade.rr_ratio.toFixed(2)}
                            </td>
                          </tr>
                          {isExpanded && (
                            <tr>
                              <td
                                colSpan={7}
                                className="px-6 py-4 bg-muted/30 border-l-2 border-l-accent"
                              >
                                <div className="flex gap-6 text-xs text-muted-foreground">
                                  <span>
                                    Opened: {new Date(trade.opened_at).toLocaleString()}
                                  </span>
                                  <span>
                                    Closed: {new Date(trade.closed_at).toLocaleString()}
                                  </span>
                                  <span>ID: {trade.id}</span>
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

              {/* Pagination */}
              {totalTradePages > 1 && (
                <div className="flex items-center justify-between pt-2">
                  <p className="text-sm text-muted-foreground">
                    Page {tradePage} of {totalTradePages}
                  </p>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setTradePage((p) => Math.max(1, p - 1))}
                      disabled={tradePage === 1}
                      className="px-3 py-1.5 text-sm rounded-lg border border-border hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      Previous
                    </button>
                    {Array.from({ length: Math.min(5, totalTradePages) }, (_, i) => {
                      let pageNum: number;
                      if (totalTradePages <= 5) {
                        pageNum = i + 1;
                      } else if (tradePage <= 3) {
                        pageNum = i + 1;
                      } else if (tradePage >= totalTradePages - 2) {
                        pageNum = totalTradePages - 4 + i;
                      } else {
                        pageNum = tradePage - 2 + i;
                      }
                      return (
                        <button
                          key={pageNum}
                          onClick={() => setTradePage(pageNum)}
                          className={`px-3 py-1.5 text-sm rounded-lg border transition-colors ${
                            pageNum === tradePage
                              ? 'bg-accent text-accent-foreground border-accent'
                              : 'border-border hover:bg-muted'
                          }`}
                        >
                          {pageNum}
                        </button>
                      );
                    })}
                    <button
                      onClick={() =>
                        setTradePage((p) => Math.min(totalTradePages, p + 1))
                      }
                      disabled={tradePage === totalTradePages}
                      className="px-3 py-1.5 text-sm rounded-lg border border-border hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
