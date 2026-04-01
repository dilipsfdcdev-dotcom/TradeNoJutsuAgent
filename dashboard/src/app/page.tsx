"use client";

import { useEffect, useState, useCallback } from 'react';
import LiveChart from '@/components/LiveChart';
import TradeCard, { ActiveTrade } from '@/components/TradeCard';
import EquityCurve from '@/components/EquityCurve';
import MetricsPanel, { MetricsData } from '@/components/MetricsPanel';
import SignalLog from '@/components/SignalLog';
import NewsPanel, { NewsItem } from '@/components/NewsPanel';
import MTFStatusPanel from '@/components/MTFStatusPanel';
import MLScoresPanel from '@/components/MLScoresPanel';
import { useWebSocket } from '@/hooks/useWebSocket';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';

interface AgentStatus {
  status: 'running' | 'paused' | 'stopped';
}

interface PositionsMessage {
  positions?: ActiveTrade[];
}

interface MTFData {
  symbol: string;
  h1_bias: string;
  h1_trend_strength: number;
  h1_structure: string;
  h1_ema_stack: string;
  m15_structure: string;
  m15_poi: { zone_type: string; high: number; low: number } | null;
  m15_active_zones: number;
  m3_momentum: string;
  m3_confirmed: boolean;
  m1_entry_signal: { direction: string; entry_price: number } | null;
  confluence_score: number;
  gates_passed: boolean;
  setup_narrative: string;
}

interface MLScores {
  symbol: string;
  xgb_score: number;
  xgb_pass: boolean;
  xgb_threshold: number;
  xgb_top_features: [string, number][];
  lstm_confidence: number;
  lstm_direction: string;
  lstm_regime: string;
  lstm_pass: boolean;
  xgb_auc_history: number[];
  lstm_acc_history: number[];
  next_retrain: string;
}

export default function Home() {
  const [agentStatus, setAgentStatus] = useState<'running' | 'paused' | 'stopped'>('stopped');
  const [dailyPnl, setDailyPnl] = useState<number>(0);
  const [accountBalance, setAccountBalance] = useState<number>(0);
  const [accountEquity, setAccountEquity] = useState<number>(0);
  const [activeTrades, setActiveTrades] = useState<ActiveTrade[]>([]);
  const [equityData, setEquityData] = useState<{ time: string; balance: number; equity: number }[]>([]);
  const [metrics, setMetrics] = useState<MetricsData>({
    winRate: 0,
    profitFactor: 0,
    totalPnl: 0,
    maxDrawdown: 0,
    sharpeRatio: 0,
    todaysTrades: 0,
    avgRR: 0,
    consecutiveWins: 0,
    consecutiveLosses: 0,
  });
  const [news, setNews] = useState<NewsItem[]>([]);

  const { data: positionsData } = useWebSocket<PositionsMessage>('trades');
  const { data: mtfData } = useWebSocket<MTFData>('mtf_state');
  const { data: mlData } = useWebSocket<MLScores>('ml_scores');

  // Update active trades from WebSocket
  useEffect(() => {
    if (positionsData?.positions) {
      setActiveTrades(positionsData.positions);
    }
  }, [positionsData]);

  // Fetch initial data from REST API
  const fetchDashboardData = useCallback(async () => {
    try {
      const [statusRes, metricsRes, equityRes, newsRes, tradesRes, accountRes] = await Promise.allSettled([
        fetch(`${API_BASE}/api/agent/status`),
        fetch(`${API_BASE}/api/metrics`),
        fetch(`${API_BASE}/api/equity?days=30`),
        fetch(`${API_BASE}/api/news?limit=10`),
        fetch(`${API_BASE}/api/trades?status=open`),
        fetch(`${API_BASE}/api/account`),
      ]);

      if (statusRes.status === 'fulfilled' && statusRes.value.ok) {
        const data: AgentStatus = await statusRes.value.json();
        setAgentStatus(data.status);
      }

      if (metricsRes.status === 'fulfilled' && metricsRes.value.ok) {
        const data = await metricsRes.value.json();
        setMetrics({
          winRate: data.win_rate ?? 0,
          profitFactor: data.profit_factor ?? 0,
          totalPnl: data.total_pnl ?? 0,
          maxDrawdown: data.max_drawdown ?? 0,
          sharpeRatio: data.sharpe_ratio ?? 0,
          todaysTrades: data.todays_trades ?? 0,
          avgRR: data.avg_rr ?? 0,
          consecutiveWins: data.consecutive_wins ?? 0,
          consecutiveLosses: data.consecutive_losses ?? 0,
        });
        setDailyPnl(data.daily_pnl ?? 0);
      }

      if (equityRes.status === 'fulfilled' && equityRes.value.ok) {
        const data = await equityRes.value.json();
        if (Array.isArray(data) && data.length > 0) {
          setEquityData(
            data.map((p: any) => ({
              time: p.date || p.timestamp || '',
              balance: p.ending_balance ?? p.balance ?? p.equity ?? 0,
              equity: p.ending_balance ?? p.equity ?? 0,
            }))
          );
        }
      }

      if (newsRes.status === 'fulfilled' && newsRes.value.ok) {
        const data = await newsRes.value.json();
        setNews(data);
      }

      if (tradesRes.status === 'fulfilled' && tradesRes.value.ok) {
        const data = await tradesRes.value.json();
        setActiveTrades(data);
      }

      if (accountRes.status === 'fulfilled' && accountRes.value.ok) {
        const data = await accountRes.value.json();
        if (data.connected) {
          setAccountBalance(data.balance ?? 0);
          setAccountEquity(data.equity ?? 0);
        }
      }
    } catch {
      // Silently handle fetch errors -- individual panels show empty states
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000);
    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  // Toggle agent status
  const handleToggleAgent = async () => {
    const action = agentStatus === 'running' ? 'pause' : 'resume';
    try {
      const res = await fetch(`${API_BASE}/api/agent/${action}`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setAgentStatus(data.status ?? (action === 'pause' ? 'paused' : 'running'));
      }
    } catch {
      // ignore
    }
  };

  // Build trade markers for chart
  const tradeMarkers = activeTrades.map((t: ActiveTrade) => ({
    entry: Number(t.entry_price) || 0,
    sl: Number(t.sl) || 0,
    tp: Number(t.tp) || 0,
    side: t.side,
  }));

  const statusColor =
    agentStatus === 'running'
      ? 'bg-profit'
      : agentStatus === 'paused'
      ? 'bg-yellow-500'
      : 'bg-loss';

  const statusLabel =
    agentStatus === 'running'
      ? 'Running'
      : agentStatus === 'paused'
      ? 'Paused'
      : 'Stopped';

  const pnlColor = dailyPnl >= 0 ? 'text-profit' : 'text-loss';

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Header bar: Agent status + daily P&L */}
      <div className="flex items-center justify-between bg-card rounded-xl border border-border px-5 py-3">
        <div className="flex items-center gap-3">
          <button
            onClick={handleToggleAgent}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted hover:bg-muted/80 transition-colors"
          >
            <div className={`w-2.5 h-2.5 rounded-full ${statusColor} animate-pulse`} />
            <span className="text-sm font-medium text-foreground">{statusLabel}</span>
          </button>
          <span className="text-xs text-muted-foreground">
            Agent Status
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-[10px] uppercase text-muted-foreground">Balance</div>
            <div className="text-sm font-mono font-bold text-foreground">
              ${(Number(accountBalance) || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] uppercase text-muted-foreground">Equity</div>
            <div className="text-sm font-mono font-bold text-foreground">
              ${(Number(accountEquity) || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] uppercase text-muted-foreground">Today&apos;s P&amp;L</div>
            <div className={`text-sm font-mono font-bold ${pnlColor}`}>
              {dailyPnl >= 0 ? '+' : ''}${Math.abs(dailyPnl).toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] uppercase text-muted-foreground">Open Positions</div>
            <div className="text-sm font-mono font-bold text-foreground">
              {activeTrades.length}
            </div>
          </div>
        </div>
      </div>

      {/* Row 1: Live Chart + Active Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4" style={{ minHeight: '380px' }}>
        <div className="lg:col-span-2 min-h-[340px]">
          <LiveChart symbol="XAUUSD" activeTrades={tradeMarkers} />
        </div>
        <div className="flex flex-col gap-3 overflow-y-auto max-h-[420px]">
          <h3 className="text-sm font-medium text-foreground px-1">Active Positions</h3>
          {activeTrades.length === 0 ? (
            <div className="bg-card rounded-xl border border-border p-6 flex items-center justify-center flex-1">
              <p className="text-sm text-muted-foreground">No open positions</p>
            </div>
          ) : (
            activeTrades.map((trade) => (
              <TradeCard key={trade.id} trade={trade} />
            ))
          )}
        </div>
      </div>

      {/* Row 2: Equity Curve + Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4" style={{ minHeight: '280px' }}>
        <div className="min-h-[260px]">
          <EquityCurve data={equityData} />
        </div>
        <div className="min-h-[260px]">
          <MetricsPanel metrics={metrics} />
        </div>
      </div>

      {/* Row 3: MTF Status Panel */}
      <div style={{ minHeight: '280px' }}>
        <MTFStatusPanel data={mtfData} symbol="XAUUSD" />
      </div>

      {/* Row 4: ML Scores Panel */}
      <div style={{ minHeight: '280px' }}>
        <MLScoresPanel data={mlData} symbol="XAUUSD" />
      </div>

      {/* Row 5: AI Signal Log */}
      <div style={{ minHeight: '240px', maxHeight: '320px' }}>
        <SignalLog />
      </div>

      {/* Row 6: News Sentiment */}
      <div style={{ minHeight: '240px', maxHeight: '360px' }}>
        <NewsPanel news={news} />
      </div>
    </div>
  );
}
