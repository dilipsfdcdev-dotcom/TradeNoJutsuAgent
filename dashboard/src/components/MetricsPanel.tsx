export interface MetricsData {
  winRate: number;
  profitFactor: number;
  totalPnl: number;
  maxDrawdown: number;
  sharpeRatio: number;
  todaysTrades: number;
  avgRR: number;
  consecutiveWins: number;
  consecutiveLosses: number;
}

interface MetricsPanelProps {
  metrics: MetricsData;
}

function CircularProgress({ value, size = 48, strokeWidth = 4 }: { value: number; size?: number; strokeWidth?: number }) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;
  const color = value >= 60 ? '#22c55e' : value >= 40 ? '#eab308' : '#ef4444';

  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke="#262626"
        strokeWidth={strokeWidth}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
      />
    </svg>
  );
}

interface MetricCardProps {
  label: string;
  children: React.ReactNode;
}

function MetricCard({ label, children }: MetricCardProps) {
  return (
    <div className="bg-muted/50 rounded-lg p-3 flex flex-col items-center justify-center gap-1">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

export default function MetricsPanel({ metrics: raw }: MetricsPanelProps) {
  // Ensure every field is a finite number so .toFixed / arithmetic never throws.
  const metrics: MetricsData = {
    winRate: Number(raw.winRate) || 0,
    profitFactor: Number(raw.profitFactor) || 0,
    totalPnl: Number(raw.totalPnl) || 0,
    maxDrawdown: Number(raw.maxDrawdown) || 0,
    sharpeRatio: Number(raw.sharpeRatio) || 0,
    todaysTrades: Number(raw.todaysTrades) || 0,
    avgRR: Number(raw.avgRR) || 0,
    consecutiveWins: Number(raw.consecutiveWins) || 0,
    consecutiveLosses: Number(raw.consecutiveLosses) || 0,
  };
  const pnlColor = metrics.totalPnl >= 0 ? 'text-profit' : 'text-loss';

  return (
    <div className="bg-card rounded-xl border border-border p-4 h-full flex flex-col">
      <h3 className="text-sm font-medium text-foreground mb-3">Performance Metrics</h3>

      <div className="grid grid-cols-2 gap-2 flex-1">
        {/* Win Rate with circular progress */}
        <MetricCard label="Win Rate">
          <div className="relative flex items-center justify-center">
            <CircularProgress value={metrics.winRate} size={52} />
            <span className="absolute text-xs font-bold text-foreground">
              {metrics.winRate.toFixed(0)}%
            </span>
          </div>
        </MetricCard>

        {/* Profit Factor */}
        <MetricCard label="Profit Factor">
          <span className={`text-lg font-bold ${metrics.profitFactor >= 1 ? 'text-profit' : 'text-loss'}`}>
            {metrics.profitFactor.toFixed(2)}
          </span>
        </MetricCard>

        {/* Total P&L */}
        <MetricCard label="Total P&L">
          <span className={`text-lg font-bold font-mono ${pnlColor}`}>
            {metrics.totalPnl >= 0 ? '+' : ''}${Math.abs(metrics.totalPnl).toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        </MetricCard>

        {/* Max Drawdown */}
        <MetricCard label="Max Drawdown">
          <span className="text-lg font-bold text-loss">
            {metrics.maxDrawdown.toFixed(1)}%
          </span>
        </MetricCard>

        {/* Sharpe Ratio */}
        <MetricCard label="Sharpe Ratio">
          <span className={`text-lg font-bold ${metrics.sharpeRatio >= 1 ? 'text-profit' : metrics.sharpeRatio >= 0 ? 'text-foreground' : 'text-loss'}`}>
            {metrics.sharpeRatio.toFixed(2)}
          </span>
        </MetricCard>

        {/* Today's Trades */}
        <MetricCard label="Today's Trades">
          <span className="text-lg font-bold text-foreground">
            {metrics.todaysTrades}
          </span>
        </MetricCard>

        {/* Avg R:R */}
        <MetricCard label="Avg R:R">
          <span className={`text-lg font-bold ${metrics.avgRR >= 1.5 ? 'text-profit' : 'text-foreground'}`}>
            1:{metrics.avgRR.toFixed(1)}
          </span>
        </MetricCard>

        {/* Consecutive Wins/Losses */}
        <MetricCard label="Streak">
          <div className="flex items-center gap-2">
            <span className="text-xs text-profit font-bold">W{metrics.consecutiveWins}</span>
            <span className="text-muted-foreground">/</span>
            <span className="text-xs text-loss font-bold">L{metrics.consecutiveLosses}</span>
          </div>
        </MetricCard>
      </div>
    </div>
  );
}
