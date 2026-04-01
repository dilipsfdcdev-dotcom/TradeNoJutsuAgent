"use client";

import { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, ColorType, PriceLineOptions, LineStyle } from 'lightweight-charts';
import { useWebSocket } from '@/hooks/useWebSocket';

interface CandleMessage {
  symbol: string;
  candles?: CandlestickData[];
  candle?: CandlestickData;
}

interface ActiveTradeMarker {
  entry: number;
  sl: number;
  tp: number;
  side: 'buy' | 'sell';
}

interface LiveChartProps {
  symbol: string;
  activeTrades?: ActiveTradeMarker[];
}

const SYMBOLS = ['XAUUSD', 'BTCUSD', 'XAGUSD'];

export default function LiveChart({ symbol: initialSymbol, activeTrades = [] }: LiveChartProps) {
  const [selectedSymbol, setSelectedSymbol] = useState(initialSymbol);
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const candlesRef = useRef<Map<number, CandlestickData>>(new Map());

  const { data: priceData } = useWebSocket<CandleMessage>('prices');

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const container = chartContainerRef.current;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1a2e' },
        textColor: '#e0e0e0',
      },
      grid: {
        vertLines: { color: '#2a2a4a' },
        horzLines: { color: '#2a2a4a' },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: '#2a2a4a',
      },
      timeScale: {
        borderColor: '#2a2a4a',
        timeVisible: true,
        secondsVisible: false,
      },
      width: container.clientWidth,
      height: container.clientHeight,
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    });

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Clear candles when symbol changes
  useEffect(() => {
    candlesRef.current.clear();
    if (seriesRef.current) {
      seriesRef.current.setData([]);
    }
  }, [selectedSymbol]);

  // Update chart data from WebSocket
  useEffect(() => {
    if (!priceData || !seriesRef.current) return;
    if (priceData.symbol !== selectedSymbol) return;

    // Coerce OHLC values to numbers (they may arrive as strings from JSON)
    const toCandle = (c: any): CandlestickData => ({
      time: Number(c.time) as CandlestickData['time'],
      open: Number(c.open),
      high: Number(c.high),
      low: Number(c.low),
      close: Number(c.close),
    });

    // Initial batch of candles
    if (priceData.candles && priceData.candles.length > 0) {
      priceData.candles.forEach((c) => {
        const candle = toCandle(c);
        candlesRef.current.set(candle.time as number, candle);
      });
      const sorted = Array.from(candlesRef.current.values()).sort(
        (a, b) => (a.time as number) - (b.time as number)
      );
      seriesRef.current.setData(sorted);
      chartRef.current?.timeScale().scrollToRealTime();
    }

    // Single candle update (real-time tick)
    if (priceData.candle) {
      const candle = toCandle(priceData.candle);
      candlesRef.current.set(candle.time as number, candle);
      seriesRef.current.update(candle);
    }
  }, [priceData, selectedSymbol]);

  // Draw trade markers (entry, SL, TP lines)
  useEffect(() => {
    if (!seriesRef.current) return;

    // Remove existing price lines
    const series = seriesRef.current;
    // lightweight-charts doesn't have a removeAllPriceLines, so we recreate by tracking
    // We'll use createPriceLine and store references

    const lines: ReturnType<typeof series.createPriceLine>[] = [];

    activeTrades.forEach((trade) => {
      const entry = Number(trade.entry) || 0;
      const sl = Number(trade.sl) || 0;
      const tp = Number(trade.tp) || 0;

      if (!entry) return; // skip trades with no valid entry price

      const entryLine = series.createPriceLine({
        price: entry,
        color: '#3b82f6',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        axisLabelVisible: true,
        title: 'Entry',
      } as PriceLineOptions);
      lines.push(entryLine);

      const slLine = series.createPriceLine({
        price: sl,
        color: '#ef4444',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'SL',
      } as PriceLineOptions);
      lines.push(slLine);

      const tpLine = series.createPriceLine({
        price: tp,
        color: '#22c55e',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: 'TP',
      } as PriceLineOptions);
      lines.push(tpLine);
    });

    return () => {
      lines.forEach((line) => {
        try {
          series.removePriceLine(line);
        } catch {
          // Series may have been removed
        }
      });
    };
  }, [activeTrades]);

  return (
    <div className="flex flex-col h-full bg-card rounded-xl border border-border overflow-hidden">
      {/* Symbol tabs */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-border bg-[#1a1a2e]">
        {SYMBOLS.map((sym) => (
          <button
            key={sym}
            onClick={() => setSelectedSymbol(sym)}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              selectedSymbol === sym
                ? 'bg-accent text-accent-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted'
            }`}
          >
            {sym}
          </button>
        ))}
        <span className="ml-auto text-xs text-muted-foreground">{selectedSymbol}</span>
      </div>

      {/* Chart container */}
      <div ref={chartContainerRef} className="flex-1 min-h-0" />
    </div>
  );
}
