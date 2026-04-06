"use client";

import { useWebSocket } from '@/hooks/useWebSocket';

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

interface MTFStatusPanelProps {
  data: MTFData | null;
  symbol: string;
}

function biasColor(bias: string): string {
  const b = bias.toLowerCase();
  if (b === 'bullish' || b === 'bull' || b === 'long') return 'text-green-400';
  if (b === 'bearish' || b === 'bear' || b === 'short') return 'text-red-400';
  return 'text-gray-400';
}

function biasBg(bias: string): string {
  const b = bias.toLowerCase();
  if (b === 'bullish' || b === 'bull' || b === 'long') return 'border-green-500/30 bg-green-500/5';
  if (b === 'bearish' || b === 'bear' || b === 'short') return 'border-red-500/30 bg-red-500/5';
  return 'border-gray-500/30 bg-gray-500/5';
}

function biasArrow(bias: string): string {
  const b = bias.toLowerCase();
  if (b === 'bullish' || b === 'bull' || b === 'long') return '↑';
  if (b === 'bearish' || b === 'bear' || b === 'short') return '↓';
  return '↔';
}

function confluenceBarColor(score: number): string {
  if (score >= 75) return 'bg-green-500';
  if (score >= 50) return 'bg-yellow-500';
  if (score >= 25) return 'bg-orange-500';
  return 'bg-red-500';
}

function TFCard({ label, icon, status, statusColor }: {
  label: string;
  icon: string;
  status: string;
  statusColor: string;
}) {
  return (
    <div className="flex flex-col items-center gap-1 bg-muted/50 rounded-lg p-3 border border-border">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className={`text-xl ${statusColor}`}>{icon}</span>
      <span className={`text-xs font-bold font-mono ${statusColor}`}>{status}</span>
    </div>
  );
}

export default function MTFStatusPanel({ data, symbol }: MTFStatusPanelProps) {
  const { data: wsData } = useWebSocket<MTFData>('mtf_state');
  const mtf = data ?? wsData;

  // Derive 15M status
  const m15Status = mtf?.m15_poi
    ? `OB ${mtf.m15_poi.zone_type}`
    : mtf?.m15_structure ?? '---';
  const m15Icon = mtf?.m15_poi ? 'OB' : '—';
  const m15Color = mtf?.m15_poi ? 'text-yellow-400' : 'text-gray-400';

  // Derive 3M status
  const m3Status = mtf?.m3_confirmed ? 'CONF' : mtf?.m3_momentum ?? '---';
  const m3Icon = mtf?.m3_confirmed ? '✓' : '—';
  const m3Color = mtf?.m3_confirmed ? 'text-green-400' : 'text-gray-400';

  // Derive 1M status
  const m1Status = mtf?.m1_entry_signal ? mtf.m1_entry_signal.direction.toUpperCase() : 'WAIT';
  const m1Icon = mtf?.m1_entry_signal ? (mtf.m1_entry_signal.direction === 'long' ? '↑' : '↓') : '⏳';
  const m1Color = mtf?.m1_entry_signal
    ? (mtf.m1_entry_signal.direction === 'long' ? 'text-green-400' : 'text-red-400')
    : 'text-gray-400';

  const confluenceScore = mtf?.confluence_score ?? 0;

  return (
    <div className="bg-card rounded-xl border border-border p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">
          MTF Status
          <span className="ml-2 text-xs text-muted-foreground font-mono">{symbol}</span>
        </h3>
        {mtf && (
          <span className={`inline-flex items-center gap-1 text-xs font-bold px-2 py-0.5 rounded-full ${
            mtf.gates_passed
              ? 'bg-green-500/10 text-green-400 border border-green-500/30'
              : 'bg-red-500/10 text-red-400 border border-red-500/30'
          }`}>
            {mtf.gates_passed ? '✓' : '✗'} Gates
          </span>
        )}
      </div>

      {!mtf ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">Waiting for MTF data...</p>
        </div>
      ) : (
        <div className="flex flex-col gap-3 flex-1">
          {/* 4-column TF grid */}
          <div className="grid grid-cols-4 gap-2">
            <TFCard
              label="1H"
              icon={biasArrow(mtf.h1_bias)}
              status={mtf.h1_bias.toUpperCase().slice(0, 4)}
              statusColor={biasColor(mtf.h1_bias)}
            />
            <TFCard
              label="15M"
              icon={m15Icon}
              status={m15Status.slice(0, 6)}
              statusColor={m15Color}
            />
            <TFCard
              label="3M"
              icon={m3Icon}
              status={m3Status.slice(0, 4)}
              statusColor={m3Color}
            />
            <TFCard
              label="1M"
              icon={m1Icon}
              status={m1Status.slice(0, 4)}
              statusColor={m1Color}
            />
          </div>

          {/* 1H detail row */}
          <div className={`flex items-center gap-3 text-xs rounded-lg border px-3 py-2 ${biasBg(mtf.h1_bias)}`}>
            <span className="text-muted-foreground">Structure:</span>
            <span className="font-mono text-foreground">{mtf.h1_structure}</span>
            <span className="text-muted-foreground ml-auto">EMA:</span>
            <span className="font-mono text-foreground">{mtf.h1_ema_stack}</span>
            <span className="text-muted-foreground">Str:</span>
            <span className="font-mono text-foreground">{(Number(mtf.h1_trend_strength) || 0).toFixed(0)}%</span>
          </div>

          {/* Confluence score bar */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted-foreground">
              <span>Confluence</span>
              <span className="font-mono text-foreground text-xs font-bold">{confluenceScore}/100</span>
            </div>
            <div className="w-full h-2.5 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${confluenceBarColor(confluenceScore)}`}
                style={{ width: `${Math.min(100, Math.max(0, confluenceScore))}%` }}
              />
            </div>
          </div>

          {/* 15M zones + 1M entry info */}
          <div className="flex items-center gap-4 text-xs">
            <span className="text-muted-foreground">
              15M zones: <span className="font-mono text-foreground font-bold">{mtf.m15_active_zones}</span>
            </span>
            {mtf.m1_entry_signal && (
              <span className="text-muted-foreground">
                Entry: <span className={`font-mono font-bold ${m1Color}`}>
                  {(Number(mtf.m1_entry_signal.entry_price) || 0).toFixed(2)}
                </span>
              </span>
            )}
          </div>

          {/* Narrative */}
          <div className="mt-auto pt-2 border-t border-border">
            <p className="text-xs text-muted-foreground leading-relaxed italic">
              {mtf.setup_narrative || 'No active narrative.'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
