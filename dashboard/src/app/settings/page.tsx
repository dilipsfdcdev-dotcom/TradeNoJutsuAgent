'use client';

import { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8765';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface RiskSettings {
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

interface ConnectionHealth {
  mt5: boolean;
  redis: boolean;
  database: boolean;
}

interface Settings {
  risk: RiskSettings;
  symbols: SymbolConfig[];
  connection_health: ConnectionHealth;
  agent_status: 'running' | 'paused' | 'stopped';
}

const DEFAULT_SETTINGS: Settings = {
  risk: {
    max_risk_per_trade: 1.0,
    max_daily_loss: 3.0,
    max_open_trades: 3,
    max_drawdown: 10.0,
    min_rr_ratio: 1.5,
    spread_filter_multiplier: 1.5,
  },
  symbols: [],
  connection_health: { mt5: false, redis: false, database: false },
  agent_status: 'stopped',
};

/* ------------------------------------------------------------------ */
/*  Page Component                                                     */
/* ------------------------------------------------------------------ */

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Kill switch confirmation
  const [showKillConfirm, setShowKillConfirm] = useState(false);
  const [killLoading, setKillLoading] = useState(false);
  const [pauseLoading, setPauseLoading] = useState(false);

  /* Fetch settings */
  const fetchSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/settings`, {
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Settings = await res.json();
      setSettings(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  /* Save settings */
  async function handleSave() {
    setSaving(true);
    setSaveSuccess(false);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          risk: settings.risk,
          symbols: settings.symbols,
        }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(body || `HTTP ${res.status}`);
      }
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err: any) {
      setError(err.message || 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  }

  /* Kill switch */
  async function handleKill() {
    setKillLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'stop' }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSettings((s) => ({ ...s, agent_status: 'stopped' }));
      setShowKillConfirm(false);
    } catch (err: any) {
      setError(err.message || 'Failed to stop agent');
    } finally {
      setKillLoading(false);
    }
  }

  /* Pause / Resume */
  async function handleTogglePause() {
    const nextAction = settings.agent_status === 'paused' ? 'resume' : 'pause';
    setPauseLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: nextAction }),
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSettings((s) => ({
        ...s,
        agent_status: nextAction === 'pause' ? 'paused' : 'running',
      }));
    } catch (err: any) {
      setError(err.message || `Failed to ${nextAction} agent`);
    } finally {
      setPauseLoading(false);
    }
  }

  /* Risk field updater */
  function updateRisk<K extends keyof RiskSettings>(key: K, value: RiskSettings[K]) {
    setSettings((s) => ({
      ...s,
      risk: { ...s.risk, [key]: value },
    }));
  }

  /* Symbol toggle */
  function toggleSymbol(index: number) {
    setSettings((s) => {
      const symbols = [...s.symbols];
      symbols[index] = { ...symbols[index], enabled: !symbols[index].enabled };
      return { ...s, symbols };
    });
  }

  /* ---------------------------------------------------------------- */
  /*  Render                                                           */
  /* ---------------------------------------------------------------- */

  const inputClass =
    'w-full px-3 py-2 rounded-lg bg-muted border border-border text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-accent';

  if (loading) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Settings</h1>
        <div className="text-center text-muted-foreground py-12">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Settings</h1>
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
            {settings.agent_status}
          </span>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 rounded-lg bg-loss/10 border border-loss/30 text-loss text-sm">
          {error}
          <button onClick={() => setError(null)} className="ml-3 underline hover:no-underline">
            Dismiss
          </button>
        </div>
      )}

      {/* Save success */}
      {saveSuccess && (
        <div className="p-3 rounded-lg bg-profit/10 border border-profit/30 text-profit text-sm">
          Settings saved successfully.
        </div>
      )}

      {/* ---- Risk Parameters ---- */}
      <div className="bg-card border border-border rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Risk Parameters
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Max Risk Per Trade */}
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

          {/* Max Daily Loss */}
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

          {/* Max Open Trades */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Max Open Trades
            </label>
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

          {/* Max Drawdown */}
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

          {/* Min R:R Ratio */}
          <div>
            <label className="block text-xs text-muted-foreground mb-1">
              Min R:R Ratio
            </label>
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

          {/* Spread Filter Multiplier */}
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
              onChange={(e) =>
                updateRisk('spread_filter_multiplier', Number(e.target.value))
              }
              className={inputClass}
            />
          </div>
        </div>
      </div>

      {/* ---- Symbol Toggles ---- */}
      {settings.symbols.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-5 space-y-4">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
            Symbols
          </h2>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
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
                {/* Toggle indicator */}
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
        </div>
      )}

      {/* ---- Connection Health ---- */}
      <div className="bg-card border border-border rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Connection Health
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {(
            [
              { key: 'mt5' as const, label: 'MetaTrader 5' },
              { key: 'redis' as const, label: 'Redis' },
              { key: 'database' as const, label: 'Database' },
            ] as const
          ).map(({ key, label }) => {
            const connected = settings.connection_health[key];
            return (
              <div
                key={key}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${
                  connected
                    ? 'border-profit/50 bg-profit/5'
                    : 'border-loss/50 bg-loss/5'
                }`}
              >
                <div
                  className={`w-3 h-3 rounded-full ${
                    connected ? 'bg-profit' : 'bg-loss'
                  }`}
                />
                <div>
                  <p className="text-sm font-medium text-foreground">{label}</p>
                  <p
                    className={`text-xs ${
                      connected ? 'text-profit' : 'text-loss'
                    }`}
                  >
                    {connected ? 'Connected' : 'Disconnected'}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ---- Agent Controls ---- */}
      <div className="bg-card border border-border rounded-xl p-5 space-y-4">
        <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
          Agent Controls
        </h2>

        <div className="flex flex-wrap items-center gap-3">
          {/* Save */}
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-5 py-2.5 text-sm font-medium rounded-lg bg-accent text-accent-foreground hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
          >
            {saving ? 'Saving...' : 'Save Settings'}
          </button>

          {/* Pause / Resume */}
          {settings.agent_status !== 'stopped' && (
            <button
              onClick={handleTogglePause}
              disabled={pauseLoading}
              className={`px-5 py-2.5 text-sm font-medium rounded-lg border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                settings.agent_status === 'paused'
                  ? 'border-profit/50 text-profit hover:bg-profit/10'
                  : 'border-yellow-500/50 text-yellow-500 hover:bg-yellow-500/10'
              }`}
            >
              {pauseLoading
                ? 'Processing...'
                : settings.agent_status === 'paused'
                ? 'Resume Agent'
                : 'Pause Agent'}
            </button>
          )}

          {/* Kill Switch */}
          {!showKillConfirm ? (
            <button
              onClick={() => setShowKillConfirm(true)}
              disabled={settings.agent_status === 'stopped'}
              className="px-5 py-2.5 text-sm font-medium rounded-lg bg-loss text-white hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
            >
              Kill Switch
            </button>
          ) : (
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg border border-loss/50 bg-loss/10">
              <span className="text-sm text-loss font-medium">
                Stop all trading immediately?
              </span>
              <button
                onClick={handleKill}
                disabled={killLoading}
                className="px-3 py-1.5 text-xs font-bold rounded bg-loss text-white hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                {killLoading ? 'Stopping...' : 'Confirm Stop'}
              </button>
              <button
                onClick={() => setShowKillConfirm(false)}
                className="px-3 py-1.5 text-xs font-medium rounded border border-border hover:bg-muted transition-colors"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
