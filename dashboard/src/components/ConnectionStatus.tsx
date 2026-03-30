'use client';

import { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

interface ServiceStatus {
  api: boolean;
  websocket: boolean;
  mt5: boolean;
}

function StatusDot({ connected, label }: { connected: boolean; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={`w-2 h-2 rounded-full ${
          connected ? 'bg-profit animate-pulse' : 'bg-loss'
        }`}
        title={`${label}: ${connected ? 'Connected' : 'Disconnected'}`}
      />
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

export default function ConnectionStatus() {
  const [status, setStatus] = useState<ServiceStatus>({
    api: false,
    websocket: false,
    mt5: false,
  });

  const checkApi = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }, []);

  const checkWebSocket = useCallback((): Promise<boolean> => {
    return new Promise((resolve) => {
      try {
        const ws = new WebSocket(`${WS_BASE}/ws/ping`);
        const timeout = setTimeout(() => {
          ws.close();
          resolve(false);
        }, 5000);

        ws.onopen = () => {
          clearTimeout(timeout);
          ws.close();
          resolve(true);
        };

        ws.onerror = () => {
          clearTimeout(timeout);
          resolve(false);
        };
      } catch {
        resolve(false);
      }
    });
  }, []);

  const checkMT5 = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/mt5/status`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) return false;
      const data = await res.json();
      return data.connected === true;
    } catch {
      return false;
    }
  }, []);

  useEffect(() => {
    let mounted = true;

    const checkAll = async () => {
      const [apiOk, wsOk, mt5Ok] = await Promise.all([
        checkApi(),
        checkWebSocket(),
        checkMT5(),
      ]);

      if (mounted) {
        setStatus({ api: apiOk, websocket: wsOk, mt5: mt5Ok });
      }
    };

    checkAll();
    const interval = setInterval(checkAll, 10000);

    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, [checkApi, checkWebSocket, checkMT5]);

  return (
    <div className="flex items-center gap-4">
      <StatusDot connected={status.api} label="API" />
      <StatusDot connected={status.websocket} label="WS" />
      <StatusDot connected={status.mt5} label="MT5" />
    </div>
  );
}
