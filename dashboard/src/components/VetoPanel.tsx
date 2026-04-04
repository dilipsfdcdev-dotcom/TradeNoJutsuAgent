"use client";

import { useWebSocket } from "@/hooks/useWebSocket";

interface VetoData {
  symbol: string;
  blocked: boolean;
  reason: string;
  risk_modifier: number;
  expires_at: string | null;
  source: string;
}

interface VetoPanelProps {
  symbols?: string[];
}

export default function VetoPanel({ symbols = ["XAUUSD", "BTCUSD", "XAGUSD"] }: VetoPanelProps) {
  const { data: vetoData } = useWebSocket<Record<string, VetoData>>("vetoes");

  const getVetoState = (symbol: string): VetoData | null => {
    if (!vetoData) return null;
    return vetoData[symbol] || null;
  };

  const getStatusColor = (veto: VetoData | null) => {
    if (!veto) return "text-green-400";
    if (veto.blocked) return "text-red-400";
    if (veto.risk_modifier < 1.0) return "text-yellow-400";
    return "text-green-400";
  };

  const getStatusLabel = (veto: VetoData | null) => {
    if (!veto) return "ALLOW";
    if (veto.blocked) return "BLOCKED";
    if (veto.risk_modifier < 1.0) return `REDUCE ${Math.round(veto.risk_modifier * 100)}%`;
    return "ALLOW";
  };

  const getTimeRemaining = (expiresAt: string | null) => {
    if (!expiresAt) return null;
    const diff = new Date(expiresAt).getTime() - Date.now();
    if (diff <= 0) return "Expired";
    const mins = Math.ceil(diff / 60000);
    return `${mins}m remaining`;
  };

  const globalVeto = vetoData?.GLOBAL || null;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">Claude Veto Status</h3>
        <span className="text-xs text-muted-foreground">Updates every 30s</span>
      </div>

      {/* Global veto banner */}
      {globalVeto && globalVeto.blocked && (
        <div className="mb-3 rounded-md bg-red-500/10 border border-red-500/30 p-2">
          <div className="flex items-center gap-2">
            <span className="text-red-400 text-xs font-bold">GLOBAL BLOCK</span>
            <span className="text-red-300 text-xs">{globalVeto.reason}</span>
          </div>
          {globalVeto.expires_at && (
            <span className="text-red-400/60 text-xs">{getTimeRemaining(globalVeto.expires_at)}</span>
          )}
        </div>
      )}

      {/* Per-symbol status */}
      <div className="space-y-2">
        {symbols.map((symbol) => {
          const veto = getVetoState(symbol);
          const statusColor = getStatusColor(veto);
          const statusLabel = getStatusLabel(veto);

          return (
            <div
              key={symbol}
              className="flex items-center justify-between py-1.5 px-2 rounded bg-background/50"
            >
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono font-semibold text-foreground w-16">
                  {symbol}
                </span>
                <span className={`text-xs font-bold ${statusColor}`}>
                  {statusLabel}
                </span>
              </div>

              <div className="flex items-center gap-2">
                {veto?.reason && (
                  <span className="text-xs text-muted-foreground max-w-[200px] truncate">
                    {veto.reason}
                  </span>
                )}
                {veto?.expires_at && (
                  <span className="text-xs text-muted-foreground/60">
                    {getTimeRemaining(veto.expires_at)}
                  </span>
                )}
                {/* Status dot */}
                <div
                  className={`w-2 h-2 rounded-full ${
                    veto?.blocked
                      ? "bg-red-400"
                      : veto && veto.risk_modifier < 1.0
                      ? "bg-yellow-400"
                      : "bg-green-400"
                  }`}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Source info */}
      <div className="mt-3 pt-2 border-t border-border/50">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Source: {globalVeto?.source || "veto_scanner"}</span>
          <span>
            Cost: ~$7-14/day
          </span>
        </div>
      </div>
    </div>
  );
}
