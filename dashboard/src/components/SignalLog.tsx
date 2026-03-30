"use client";

import { useEffect, useRef, useState } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface SignalEntry {
  id: string;
  timestamp: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'WAIT';
  confidence: number;
  reasoning: string;
}

interface SignalMessage {
  signal?: SignalEntry;
  signals?: SignalEntry[];
}

const MAX_ENTRIES = 100;

const SYMBOL_COLORS: Record<string, string> = {
  XAUUSD: 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30',
  BTCUSD: 'bg-orange-600/20 text-orange-400 border-orange-600/30',
  XAGUSD: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
};

const ACTION_STYLES: Record<string, string> = {
  BUY: 'text-profit font-bold',
  SELL: 'text-loss font-bold',
  WAIT: 'text-muted-foreground font-medium',
};

function formatTime(timestamp: string): string {
  const d = new Date(timestamp);
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value * 100));
  const color =
    pct >= 70 ? 'bg-profit' : pct >= 40 ? 'bg-yellow-500' : 'bg-loss';

  return (
    <div className="flex items-center gap-1.5 min-w-[80px]">
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] text-muted-foreground w-8 text-right font-mono">
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export default function SignalLog() {
  const [entries, setEntries] = useState<SignalEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  const { data: signalData } = useWebSocket<SignalMessage>('signals');

  // Process incoming signal data
  useEffect(() => {
    if (!signalData) return;

    setEntries((prev) => {
      let updated = [...prev];

      if (signalData.signals && signalData.signals.length > 0) {
        // Batch of signals (initial load)
        updated = signalData.signals;
      } else if (signalData.signal) {
        // Single new signal
        updated = [...prev, signalData.signal];
      }

      // Trim to max entries
      if (updated.length > MAX_ENTRIES) {
        updated = updated.slice(updated.length - MAX_ENTRIES);
      }

      return updated;
    });
  }, [signalData]);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  // Detect if user scrolled away from bottom
  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 40;
  };

  return (
    <div className="bg-card rounded-xl border border-border flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-foreground">AI Signal Log</h3>
        <span className="text-[10px] text-muted-foreground">
          {entries.length} entries
        </span>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto min-h-0"
      >
        {entries.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-sm text-muted-foreground">
              Waiting for signals...
            </p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {entries.map((entry, idx) => (
              <div
                key={entry.id || idx}
                className="px-4 py-2.5 hover:bg-muted/30 transition-colors"
              >
                <div className="flex items-center gap-2 mb-1">
                  {/* Timestamp */}
                  <span className="text-[10px] font-mono text-muted-foreground shrink-0">
                    {formatTime(entry.timestamp)}
                  </span>

                  {/* Symbol badge */}
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded border shrink-0 ${
                      SYMBOL_COLORS[entry.symbol] ||
                      'bg-muted text-foreground border-border'
                    }`}
                  >
                    {entry.symbol}
                  </span>

                  {/* Action */}
                  <span
                    className={`text-xs shrink-0 ${
                      ACTION_STYLES[entry.action] || 'text-foreground'
                    }`}
                  >
                    {entry.action}
                  </span>

                  {/* Confidence bar */}
                  <ConfidenceBar value={entry.confidence} />
                </div>

                {/* Reasoning snippet */}
                <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2 pl-0">
                  {entry.reasoning}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
