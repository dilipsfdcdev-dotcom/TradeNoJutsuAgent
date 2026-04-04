"use client";

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

interface EquityPoint {
  time: string;
  balance: number;
  equity: number;
}

interface EquityCurveProps {
  data: EquityPoint[];
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatCurrency(value: number): string {
  return `$${value.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number; dataKey: string; color: string }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
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

export default function EquityCurve({ data }: EquityCurveProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-card rounded-xl border border-border p-4 h-full flex items-center justify-center">
        <p className="text-sm text-muted-foreground">No equity data available</p>
      </div>
    );
  }

  // Compute starting balance for reference and determine color segments
  const startBalance = data[0]?.balance ?? 0;

  // Prepare chart data with formatted dates
  const chartData = data.map((point) => ({
    ...point,
    formattedTime: formatDate(point.time),
  }));

  // Determine overall trend color
  const endEquity = data[data.length - 1]?.equity ?? 0;
  const isProfit = endEquity >= startBalance;

  return (
    <div className="bg-card rounded-xl border border-border p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-foreground">Equity Curve</h3>
        <span className={`text-xs font-mono ${isProfit ? 'text-profit' : 'text-loss'}`}>
          {isProfit ? '+' : ''}{formatCurrency(endEquity - startBalance)}
        </span>
      </div>

      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={isProfit ? '#22c55e' : '#ef4444'} stopOpacity={0.3} />
                <stop offset="95%" stopColor={isProfit ? '#22c55e' : '#ef4444'} stopOpacity={0} />
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
              tickFormatter={formatCurrency}
              domain={['dataMin - 100', 'dataMax + 100']}
              width={70}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={startBalance} stroke="#3b82f6" strokeDasharray="3 3" strokeOpacity={0.5} />
            <Area
              type="monotone"
              dataKey="equity"
              fill="url(#equityGradient)"
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
  );
}
