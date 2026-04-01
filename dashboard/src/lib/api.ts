const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;

  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const errorBody = await res.text().catch(() => 'Unknown error');
    throw new Error(`API Error ${res.status}: ${errorBody}`);
  }

  return res.json();
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  entry_price: number;
  exit_price?: number;
  quantity: number;
  pnl?: number;
  status: 'open' | 'closed' | 'pending';
  opened_at: string;
  closed_at?: string;
}

export interface Metrics {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  daily_pnl: number;
  open_positions: number;
  account_balance: number;
}

export interface EquityPoint {
  timestamp: string;
  equity: number;
}

export interface BacktestResult {
  id: string;
  strategy: string;
  symbol: string;
  start_date: string;
  end_date: string;
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  win_rate: number;
  status: 'completed' | 'running' | 'failed';
}

export async function fetchTrades(params?: {
  symbol?: string;
  status?: string;
  limit?: number;
}): Promise<Trade[]> {
  const searchParams = new URLSearchParams();
  if (params?.symbol) searchParams.set('symbol', params.symbol);
  if (params?.status) searchParams.set('status', params.status);
  if (params?.limit) searchParams.set('limit', params.limit.toString());

  const query = searchParams.toString();
  return request<Trade[]>(`/api/trades${query ? `?${query}` : ''}`);
}

export async function fetchMetrics(): Promise<Metrics> {
  return request<Metrics>('/api/metrics');
}

export async function fetchEquityCurve(days?: number): Promise<EquityPoint[]> {
  const query = days ? `?days=${days}` : '';
  return request<EquityPoint[]>(`/api/equity${query}`);
}

export async function fetchBacktestResults(): Promise<BacktestResult[]> {
  return request<BacktestResult[]>('/api/backtest/results');
}

export async function updateSettings(
  settings: Record<string, any>
): Promise<{ success: boolean }> {
  return request<{ success: boolean }>('/api/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}

export async function toggleAgent(
  action: 'pause' | 'resume'
): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/agent/${action}`, {
    method: 'POST',
  });
}
