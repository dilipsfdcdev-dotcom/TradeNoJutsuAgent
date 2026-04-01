'use client';

import { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface MT5Config {
  login: number | '';
  password: string;
  server: string;
  terminal_path: string;
}

interface MT5AccountInfo {
  balance: number;
  equity: number;
  broker: string;
  account_type: string;
}

interface ClaudeConfig {
  api_key: string;
  model: string;
}

interface DatabaseConfig {
  postgres_url: string;
  redis_url: string;
}

interface NewsConfig {
  api_key: string;
}

interface RiskConfig {
  max_risk_per_trade: number;
  max_daily_loss: number;
  max_open_trades: number;
  max_drawdown: number;
  min_rr_ratio: number;
  spread_filter_multiplier: number;
}

interface SymbolConfig {
  symbol: string;
  enabled: boolean;
}

interface TimeframeConfig {
  timeframe: string;
  enabled: boolean;
}

interface SetupSettings {
  mt5: MT5Config;
  claude: ClaudeConfig;
  database: DatabaseConfig;
  news: NewsConfig;
  risk: RiskConfig;
  symbols: SymbolConfig[];
  timeframes: TimeframeConfig[];
  agent_status: 'running' | 'paused' | 'stopped';
}

interface ConnectionStatus {
  mt5: boolean;
  claude: boolean;
  postgres: boolean;
  redis: boolean;
  news: boolean;
}

interface Toast {
  id: number;
  type: 'success' | 'error';
  message: string;
}

const DEFAULT_SETTINGS: SetupSettings = {
  mt5: { login: '', password: '', server: '', terminal_path: '' },
  claude: { api_key: '', model: 'claude-sonnet-4-20250514' },
  database: { postgres_url: '', redis_url: '' },
  news: { api_key: '' },
  risk: {
    max_risk_per_trade: 1.0,
    max_daily_loss: 3.0,
    max_open_trades: 3,
    max_drawdown: 10.0,
    min_rr_ratio: 1.5,
    spread_filter_multiplier: 1.5,
  },
  symbols: [
    { symbol: 'XAUUSD', enabled: true },
    { symbol: 'BTCUSD', enabled: true },
    { symbol: 'XAGUSD', enabled: true },
  ],
  timeframes: [
    { timeframe: 'M1', enabled: true },
    { timeframe: 'M3', enabled: true },
  ],
  agent_status: 'stopped',
};

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SetupPage() {
  const [settings, setSettings] = useState<SetupSettings>(DEFAULT_SETTINGS);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [toastCounter, setToastCounter] = useState(0);

  const [connStatus, setConnStatus] = useState<ConnectionStatus>({
    mt5: false,
    claude: false,
    postgres: false,
    redis: false,
    news: false,
  });

  const [mt5AccountInfo, setMt5AccountInfo] = useState<MT5AccountInfo | null>(null);
  const [claudeModelName, setClaudeModelName] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [showNewsKey, setShowNewsKey] = useState(false);

  // Testing states
  const [testingMT5, setTestingMT5] = useState(false);
  const [testingClaude, setTestingClaude] = useState(false);
  const [testingDB, setTestingDB] = useState(false);
  const [testingNews, setTestingNews] = useState(false);
  const [togglingAgent, setTogglingAgent] = useState(false);

  /* Toast helpers */
  const addToast = useCallback((type: 'success' | 'error', message: string) => {
    setToastCounter((prev) => {
      const id = prev + 1;
      setToasts((t) => [...t, { id, type, message }]);
      setTimeout(() => {
        setToasts((t) => t.filter((toast) => toast.id !== id));
      }, 4000);
      return id;
    });
  }, []);

  /* Fetch settings on mount */
  const fetchSettings = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/settings`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSettings((prev) => ({
        ...prev,
        ...data,
        mt5: { ...prev.mt5, ...data.mt5 },
        claude: { ...prev.claude, ...data.claude },
        database: { ...prev.database, ...data.database },
        news: { ...prev.news, ...data.news },
        risk: { ...prev.risk, ...data.risk },
        symbols: data.symbols?.length ? data.symbols : prev.symbols,
        timeframes: data.timeframes?.length ? data.timeframes : prev.timeframes,
      }));
      if (data.connection_health) {
        setConnStatus((prev) => ({
          ...prev,
          mt5: data.connection_health.mt5 || false,
          postgres: data.connection_health.database || false,
          redis: data.connection_health.redis || false,
        }));
      }
    } catch {
      // Use defaults if settings endpoint is not available
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  /* Test MT5 Connection */
  async function testMT5() {
    setTestingMT5(true);
    try {
      const res = await fetch(`${API_BASE}/api/test-mt5`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings.mt5),
        signal: AbortSignal.timeout(15000),
      });
      const data = await res.json();
      if (res.ok && data.connected) {
        setConnStatus((s) => ({ ...s, mt5: true }));
        setMt5AccountInfo(data.account_info || null);
        addToast('success', 'MT5 connection successful');
      } else {
        setConnStatus((s) => ({ ...s, mt5: false }));
        setMt5AccountInfo(null);
        addToast('error', data.error || 'MT5 connection failed');
      }
    } catch (err: any) {
      setConnStatus((s) => ({ ...s, mt5: false }));
      setMt5AccountInfo(null);
      addToast('error', err.message || 'MT5 connection test failed');
    } finally {
      setTestingMT5(false);
    }
  }

  /* Test Claude API */
  async function testClaude() {
    setTestingClaude(true);
    try {
      const res = await fetch(`${API_BASE}/api/test-claude`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings.claude),
        signal: AbortSignal.timeout(15000),
      });
      const data = await res.json();
      if (res.ok && data.connected) {
        setConnStatus((s) => ({ ...s, claude: true }));
        setClaudeModelName(data.model || settings.claude.model);
        addToast('success', 'Claude API connection successful');
      } else {
        setConnStatus((s) => ({ ...s, claude: false }));
        setClaudeModelName('');
        addToast('error', data.error || 'Claude API connection failed');
      }
    } catch (err: any) {
      setConnStatus((s) => ({ ...s, claude: false }));
      setClaudeModelName('');
      addToast('error', err.message || 'Claude API test failed');
    } finally {
      setTestingClaude(false);
    }
  }

  /* Test Database Connections */
  async function testDB() {
    setTestingDB(true);
    try {
      const res = await fetch(`${API_BASE}/api/test-db`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings.database),
        signal: AbortSignal.timeout(15000),
      });
      const data = await res.json();
      setConnStatus((s) => ({
        ...s,
        postgres: data.postgres || false,
        redis: data.redis || false,
      }));
      if (data.postgres && data.redis) {
        addToast('success', 'All database connections successful');
      } else {
        const failed = [];
        if (!data.postgres) failed.push('PostgreSQL');
        if (!data.redis) failed.push('Redis');
        addToast('error', `Connection failed: ${failed.join(', ')}`);
      }
    } catch (err: any) {
      setConnStatus((s) => ({ ...s, postgres: false, redis: false }));
      addToast('error', err.message || 'Database connection test failed');
    } finally {
      setTestingDB(false);
    }
  }

  /* Test News API */
  async function testNews() {
    setTestingNews(true);
    try {
      const res = await fetch(`${API_BASE}/api/test-news`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: settings.news.api_key }),
        signal: AbortSignal.timeout(15000),
      });
      const data = await res.json();
      if (res.ok && data.connected) {
        setConnStatus((s) => ({ ...s, news: true }));
        addToast('success', 'News API connection successful');
      } else {
        setConnStatus((s) => ({ ...s, news: false }));
        addToast('error', data.error || 'News API connection failed');
      }
    } catch (err: any) {
      setConnStatus((s) => ({ ...s, news: false }));
      addToast('error', err.message || 'News API test failed');
    } finally {
      setTestingNews(false);
    }
  }

  /* Save All Settings */
  async function handleSaveAll() {
    setSaving(true);
    try {
      const res = await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }
      addToast('success', 'All settings saved successfully');
    } catch (err: any) {
      addToast('error', err.message || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  }

  /* Toggle Agent */
  async function handleToggleAgent() {
    const action = settings.agent_status === 'running' ? 'stop' : 'start';
    setTogglingAgent(true);
    try {
      const res = await fetch(`${API_BASE}/api/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSettings((s) => ({
        ...s,
        agent_status: action === 'start' ? 'running' : 'stopped',
      }));
      addToast('success', action === 'start' ? 'Agent started' : 'Agent stopped');
    } catch (err: any) {
      addToast('error', err.message || `Failed to ${action} agent`);
    } finally {
      setTogglingAgent(false);
    }
  }

  /* Field updaters */
  function updateMT5<K extends keyof MT5Config>(key: K, value: MT5Config[K]) {
    setSettings((s) => ({ ...s, mt5: { ...s.mt5, [key]: value } }));
  }

  function updateClaude<K extends keyof ClaudeConfig>(key: K, value: ClaudeConfig[K]) {
    setSettings((s) => ({ ...s, claude: { ...s.claude, [key]: value } }));
  }

  function updateDatabase<K extends keyof DatabaseConfig>(key: K, value: DatabaseConfig[K]) {
    setSettings((s) => ({ ...s, database: { ...s.database, [key]: value } }));
  }

  function updateRisk<K extends keyof RiskConfig>(key: K, value: RiskConfig[K]) {
    setSettings((s) => ({ ...s, risk: { ...s.risk, [key]: value } }));
  }

  function toggleSymbol(index: number) {
    setSettings((s) => {
      const symbols = [...s.symbols];
      symbols[index] = { ...symbols[index], enabled: !symbols[index].enabled };
      return { ...s, symbols };
    });
  }

  function toggleTimeframe(index: number) {
    setSettings((s) => {
      const timeframes = [...s.timeframes];
      timeframes[index] = { ...timeframes[index], enabled: !timeframes[index].enabled };
      return { ...s, timeframes };
    });
  }

  /* ---------------------------------------------------------------- */
  /*  Render helpers                                                   */
  /* ---------------------------------------------------------------- */

  const inputClass =
    'w-full px-3 py-2 rounded-lg bg-muted border border-border text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent';

  const sectionClass = 'bg-card border border-border rounded-xl p-5 space-y-4';

  const sectionTitleClass =
    'text-sm font-semibold text-muted-foreground uppercase tracking-wider';

  const btnPrimary =
    'px-5 py-2.5 text-sm font-medium rounded-lg bg-accent text-accent-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity';

  const btnOutline =
    'px-4 py-2 text-sm font-medium rounded-lg border border-border text-foreground hover:bg-muted disabled:opacity-50 disabled:cursor-not-allowed transition-colors';

  function StatusDot({ connected, label }: { connected: boolean; label?: string }) {
    return (
      <div className="flex items-center gap-2">
        <div
          className={`w-2.5 h-2.5 rounded-full ${
            connected ? 'bg-profit' : 'bg-loss'
          }`}
        />
        {label && (
          <span className={`text-xs font-medium ${connected ? 'text-profit' : 'text-loss'}`}>
            {label}
          </span>
        )}
      </div>
    );
  }

  if (loading) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Setup</h1>
        <div className="text-center text-muted-foreground py-12">
          Loading configuration...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Toast notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`px-4 py-3 rounded-lg text-sm font-medium shadow-lg border animate-in slide-in-from-right ${
              toast.type === 'success'
                ? 'bg-profit/10 border-profit/30 text-profit'
                : 'bg-loss/10 border-loss/30 text-loss'
            }`}
          >
            {toast.message}
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Setup</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure connections, credentials, and trading parameters
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              settings.agent_status === 'running'
                ? 'bg-profit'
                : settings.agent_status === 'paused'
                ? 'bg-yellow-500'
                : 'bg-loss'
            }`}
          />
          <span className="text-sm text-muted-foreground capitalize">
            Agent: {settings.agent_status}
          </span>
        </div>
      </div>

      {/* ---- Section 1: MT5 Connection ---- */}
      <div className={sectionClass}>
        <div className="flex items-center justify-between">
          <h2 className={sectionTitleClass}>MT5 Connection</h2>
          <StatusDot
            connected={connStatus.mt5}
            label={connStatus.mt5 ? 'Connected' : 'Disconnected'}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Login</label>
            <input
              type="number"
              placeholder="MT5 account number"
              value={settings.mt5.login}
              onChange={(e) =>
                updateMT5('login', e.target.value ? Number(e.target.value) : '')
              }
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Password</label>
            <input
              type="password"
              placeholder="MT5 password"
              value={settings.mt5.password}
              onChange={(e) => updateMT5('password', e.target.value)}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Server</label>
            <input
              type="text"
              placeholder="e.g. MetaQuotes-Demo"
              value={settings.mt5.server}
              onChange={(e) => updateMT5('server', e.target.value)}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Terminal Path</label>
            <input
              type="text"
              placeholder="Path to MT5 terminal"
              value={settings.mt5.terminal_path}
              onChange={(e) => updateMT5('terminal_path', e.target.value)}
              className={inputClass}
            />
          </div>
        </div>

        <button onClick={testMT5} disabled={testingMT5} className={btnOutline}>
          {testingMT5 ? 'Testing...' : 'Test Connection'}
        </button>

        {/* Account info when connected */}
        {connStatus.mt5 && mt5AccountInfo && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="px-3 py-2 rounded-lg bg-muted/50 border border-border">
              <p className="text-xs text-muted-foreground">Balance</p>
              <p className="text-sm font-medium">${(Number(mt5AccountInfo.balance) || 0).toLocaleString()}</p>
            </div>
            <div className="px-3 py-2 rounded-lg bg-muted/50 border border-border">
              <p className="text-xs text-muted-foreground">Equity</p>
              <p className="text-sm font-medium">${(Number(mt5AccountInfo.equity) || 0).toLocaleString()}</p>
            </div>
            <div className="px-3 py-2 rounded-lg bg-muted/50 border border-border">
              <p className="text-xs text-muted-foreground">Broker</p>
              <p className="text-sm font-medium">{mt5AccountInfo.broker}</p>
            </div>
            <div className="px-3 py-2 rounded-lg bg-muted/50 border border-border">
              <p className="text-xs text-muted-foreground">Account Type</p>
              <p className="text-sm font-medium">{mt5AccountInfo.account_type}</p>
            </div>
          </div>
        )}
      </div>

      {/* ---- Section 2: Claude API ---- */}
      <div className={sectionClass}>
        <div className="flex items-center justify-between">
          <h2 className={sectionTitleClass}>Claude API</h2>
          <StatusDot
            connected={connStatus.claude}
            label={
              connStatus.claude
                ? `Connected - ${claudeModelName || settings.claude.model}`
                : 'Disconnected'
            }
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">API Key</label>
            <div className="relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                placeholder="sk-ant-..."
                value={settings.claude.api_key}
                onChange={(e) => updateClaude('api_key', e.target.value)}
                className={inputClass}
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1"
              >
                {showApiKey ? 'Hide' : 'Show'}
              </button>
            </div>
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Model</label>
            <select
              value={settings.claude.model}
              onChange={(e) => updateClaude('model', e.target.value)}
              className={inputClass}
            >
              <option value="claude-sonnet-4-20250514">claude-sonnet-4-20250514</option>
              <option value="claude-haiku-4-5-20251001">claude-haiku-4-5-20251001</option>
            </select>
          </div>
        </div>

        <button onClick={testClaude} disabled={testingClaude} className={btnOutline}>
          {testingClaude ? 'Testing...' : 'Test API'}
        </button>
      </div>

      {/* ---- Section 3: Database ---- */}
      <div className={sectionClass}>
        <div className="flex items-center justify-between">
          <h2 className={sectionTitleClass}>Database</h2>
          <div className="flex items-center gap-4">
            <StatusDot
              connected={connStatus.postgres}
              label={connStatus.postgres ? 'PostgreSQL' : 'PostgreSQL'}
            />
            <StatusDot
              connected={connStatus.redis}
              label={connStatus.redis ? 'Redis' : 'Redis'}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">PostgreSQL URL</label>
            <input
              type="text"
              placeholder="postgresql://user:pass@localhost:5432/tradejutsu"
              value={settings.database.postgres_url}
              onChange={(e) => updateDatabase('postgres_url', e.target.value)}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Redis URL</label>
            <input
              type="text"
              placeholder="redis://localhost:6379/0"
              value={settings.database.redis_url}
              onChange={(e) => updateDatabase('redis_url', e.target.value)}
              className={inputClass}
            />
          </div>
        </div>

        <button onClick={testDB} disabled={testingDB} className={btnOutline}>
          {testingDB ? 'Testing...' : 'Test Connections'}
        </button>
      </div>

      {/* ---- Section 4: News API ---- */}
      <div className={sectionClass}>
        <div className="flex items-center justify-between">
          <h2 className={sectionTitleClass}>News API</h2>
          <StatusDot
            connected={connStatus.news}
            label={connStatus.news ? 'Connected' : 'Disconnected'}
          />
        </div>

        <div>
          <label className="block text-xs text-muted-foreground mb-1">API Key</label>
          <div className="relative">
            <input
              type={showNewsKey ? 'text' : 'password'}
              placeholder="News API key"
              value={settings.news.api_key}
              onChange={(e) =>
                setSettings((s) => ({ ...s, news: { ...s.news, api_key: e.target.value } }))
              }
              className={inputClass}
            />
            <button
              type="button"
              onClick={() => setShowNewsKey(!showNewsKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground hover:text-foreground transition-colors px-2 py-1"
            >
              {showNewsKey ? 'Hide' : 'Show'}
            </button>
          </div>
        </div>

        <button onClick={testNews} disabled={testingNews} className={btnOutline}>
          {testingNews ? 'Testing...' : 'Test'}
        </button>
      </div>

      {/* ---- Section 5: Risk Configuration ---- */}
      <div className={sectionClass}>
        <h2 className={sectionTitleClass}>Risk Configuration</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Max Risk Per Trade (%)
            </label>
            <input
              type="number"
              min={0.1}
              max={10}
              step={0.1}
              value={settings.risk.max_risk_per_trade}
              onChange={(e) => updateRisk('max_risk_per_trade', Number(e.target.value))}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Max Daily Loss (%)
            </label>
            <input
              type="number"
              min={0.1}
              max={20}
              step={0.1}
              value={settings.risk.max_daily_loss}
              onChange={(e) => updateRisk('max_daily_loss', Number(e.target.value))}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Max Open Trades</label>
            <input
              type="number"
              min={1}
              max={20}
              step={1}
              value={settings.risk.max_open_trades}
              onChange={(e) => updateRisk('max_open_trades', Number(e.target.value))}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Max Drawdown (%)
            </label>
            <input
              type="number"
              min={1}
              max={50}
              step={0.5}
              value={settings.risk.max_drawdown}
              onChange={(e) => updateRisk('max_drawdown', Number(e.target.value))}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">Min R:R Ratio</label>
            <input
              type="number"
              min={0.5}
              max={10}
              step={0.1}
              value={settings.risk.min_rr_ratio}
              onChange={(e) => updateRisk('min_rr_ratio', Number(e.target.value))}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Spread Filter Multiplier
            </label>
            <input
              type="number"
              min={0.5}
              max={5}
              step={0.1}
              value={settings.risk.spread_filter_multiplier}
              onChange={(e) => updateRisk('spread_filter_multiplier', Number(e.target.value))}
              className={inputClass}
            />
          </div>
        </div>
      </div>

      {/* ---- Section 6: Trading Symbols & Timeframes ---- */}
      <div className={sectionClass}>
        <h2 className={sectionTitleClass}>Trading Symbols</h2>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {settings.symbols.map((sym, i) => (
            <button
              key={sym.symbol}
              onClick={() => toggleSymbol(i)}
              className={`flex items-center justify-between px-4 py-3 rounded-lg border text-sm font-medium transition-colors ${
                sym.enabled
                  ? 'border-profit/50 bg-profit/10 text-profit'
                  : 'border-border bg-muted/50 text-muted-foreground'
              }`}
            >
              <span>{sym.symbol}</span>
              <div
                className={`relative w-9 h-5 rounded-full transition-colors ${
                  sym.enabled ? 'bg-profit' : 'bg-muted-foreground/30'
                }`}
              >
                <div
                  className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                    sym.enabled ? 'translate-x-4' : 'translate-x-0.5'
                  }`}
                />
              </div>
            </button>
          ))}
        </div>

        <h2 className={`${sectionTitleClass} pt-2`}>Timeframes</h2>

        <div className="flex gap-3">
          {settings.timeframes.map((tf, i) => (
            <button
              key={tf.timeframe}
              onClick={() => toggleTimeframe(i)}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border text-sm font-medium transition-colors ${
                tf.enabled
                  ? 'border-accent/50 bg-accent/10 text-accent'
                  : 'border-border bg-muted/50 text-muted-foreground'
              }`}
            >
              <div
                className={`w-3 h-3 rounded border-2 flex items-center justify-center ${
                  tf.enabled ? 'border-accent bg-accent' : 'border-muted-foreground'
                }`}
              >
                {tf.enabled && (
                  <svg
                    className="w-2 h-2 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={4}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>
              <span>{tf.timeframe}</span>
            </button>
          ))}
        </div>
      </div>

      {/* ---- Bottom Actions ---- */}
      <div className="bg-card border border-border rounded-xl p-5">
        <div className="flex flex-wrap items-center gap-3">
          <button onClick={handleSaveAll} disabled={saving} className={btnPrimary}>
            {saving ? 'Saving...' : 'Save All Settings'}
          </button>

          <button
            onClick={handleToggleAgent}
            disabled={togglingAgent}
            className={`px-5 py-2.5 text-sm font-medium rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              settings.agent_status === 'running'
                ? 'border-loss/50 text-loss hover:bg-loss/10'
                : 'border-profit/50 text-profit hover:bg-profit/10'
            }`}
          >
            {togglingAgent
              ? 'Processing...'
              : settings.agent_status === 'running'
              ? 'Stop Agent'
              : 'Start Agent'}
          </button>
        </div>
      </div>
    </div>
  );
}
