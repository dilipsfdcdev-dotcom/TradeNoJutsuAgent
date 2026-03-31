"use client";

import { useWebSocket } from '@/hooks/useWebSocket';

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

interface MLScoresPanelProps {
  data: MLScores | null;
  symbol: string;
}

function regimeBadge(regime: string) {
  const r = regime.toLowerCase();
  if (r === 'trending') return { color: 'bg-blue-500/10 text-blue-400 border-blue-500/30', label: 'TRENDING' };
  if (r === 'ranging') return { color: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30', label: 'RANGING' };
  if (r === 'volatile') return { color: 'bg-red-500/10 text-red-400 border-red-500/30', label: 'VOLATILE' };
  return { color: 'bg-gray-500/10 text-gray-400 border-gray-500/30', label: regime.toUpperCase() };
}

function ScoreBar({ label, score, threshold, pass }: {
  label: string;
  score: number;
  threshold: number;
  pass: boolean;
}) {
  const pct = Math.min(100, Math.max(0, score * 100));
  const threshPct = Math.min(100, Math.max(0, threshold * 100));

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-foreground">{label}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold text-foreground">{score.toFixed(2)}</span>
          <span className={`inline-flex items-center gap-0.5 text-[10px] font-bold px-1.5 py-0.5 rounded-full border ${
            pass
              ? 'bg-green-500/10 text-green-400 border-green-500/30'
              : 'bg-red-500/10 text-red-400 border-red-500/30'
          }`}>
            {pass ? '✓' : '✗'} {pass ? 'PASS' : 'FAIL'}
          </span>
        </div>
      </div>
      <div className="relative w-full h-2.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${pass ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${pct}%` }}
        />
        {/* Threshold marker */}
        <div
          className="absolute top-0 h-full w-0.5 bg-white/60"
          style={{ left: `${threshPct}%` }}
          title={`Threshold: ${threshold.toFixed(2)}`}
        />
      </div>
    </div>
  );
}

function TrendIndicator({ history, label }: { history: number[]; label: string }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 text-xs">
      <span className="text-muted-foreground">{label}:</span>
      <div className="flex items-center gap-1 font-mono">
        {history.map((val, i) => {
          const prev = i > 0 ? history[i - 1] : val;
          const trendColor = val > prev ? 'text-green-400' : val < prev ? 'text-red-400' : 'text-gray-400';
          const arrow = i > 0 ? (val > prev ? '↑' : val < prev ? '↓' : '→') : '';
          return (
            <span key={i} className="flex items-center gap-0.5">
              {i > 0 && <span className={`text-[10px] ${trendColor}`}>{arrow}</span>}
              <span className={`font-bold ${trendColor}`}>{val.toFixed(2)}</span>
            </span>
          );
        })}
      </div>
    </div>
  );
}

export default function MLScoresPanel({ data, symbol }: MLScoresPanelProps) {
  const { data: wsData } = useWebSocket<MLScores>('ml_scores');
  const ml = data ?? wsData;

  const regime = ml ? regimeBadge(ml.lstm_regime) : null;

  return (
    <div className="bg-card rounded-xl border border-border p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">
          ML Scores
          <span className="ml-2 text-xs text-muted-foreground font-mono">{symbol}</span>
        </h3>
        {regime && (
          <span className={`inline-flex items-center text-[10px] font-bold px-2 py-0.5 rounded-full border ${regime.color}`}>
            {regime.label}
          </span>
        )}
      </div>

      {!ml ? (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">Waiting for ML scores...</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4 flex-1">
          {/* Score bars */}
          <div className="flex flex-col gap-3">
            <ScoreBar
              label="XGBoost"
              score={ml.xgb_score}
              threshold={ml.xgb_threshold}
              pass={ml.xgb_pass}
            />
            <ScoreBar
              label="LSTM"
              score={ml.lstm_confidence}
              threshold={0.6}
              pass={ml.lstm_pass}
            />
          </div>

          {/* LSTM direction */}
          <div className="flex items-center gap-3 text-xs">
            <span className="text-muted-foreground">LSTM Direction:</span>
            <span className={`font-mono font-bold ${
              ml.lstm_direction === 'long' ? 'text-green-400' : ml.lstm_direction === 'short' ? 'text-red-400' : 'text-gray-400'
            }`}>
              {ml.lstm_direction.toUpperCase()}
            </span>
          </div>

          {/* Top features */}
          {ml.xgb_top_features && ml.xgb_top_features.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Top Features</span>
              <div className="flex flex-wrap gap-1.5">
                {ml.xgb_top_features.slice(0, 5).map(([name, importance]) => (
                  <span
                    key={name}
                    className="inline-flex items-center gap-1 text-[10px] font-mono bg-muted/50 border border-border rounded px-1.5 py-0.5"
                  >
                    <span className="text-foreground">{name}</span>
                    <span className="text-muted-foreground">({(importance * 100).toFixed(0)}%)</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Model performance trends */}
          <div className="flex flex-col gap-1.5 pt-2 border-t border-border">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">Model Performance</span>
            <TrendIndicator history={ml.xgb_auc_history} label="XGB AUC" />
            <TrendIndicator history={ml.lstm_acc_history} label="LSTM Acc" />
          </div>

          {/* Next retrain */}
          <div className="mt-auto pt-2 border-t border-border flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Next retrain:</span>
            <span className="font-mono text-foreground">{ml.next_retrain}</span>
          </div>
        </div>
      )}
    </div>
  );
}
