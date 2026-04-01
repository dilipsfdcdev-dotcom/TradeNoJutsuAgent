"use client";

export interface NewsItem {
  id: string;
  headline: string;
  source: string;
  sentiment: number; // -1 to +1
  timestamp: string;
}

interface NewsPanelProps {
  news: NewsItem[];
}

function relativeTime(timestamp: string): string {
  const now = Date.now();
  const then = new Date(timestamp).getTime();
  const diffMs = now - then;

  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return `${seconds}s ago`;

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;

  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function sentimentColor(score: number): string {
  // Gradient from red (-1) through yellow (0) to green (+1)
  if (score >= 0.5) return 'text-profit';
  if (score >= 0.2) return 'text-green-400';
  if (score >= -0.2) return 'text-yellow-400';
  if (score >= -0.5) return 'text-orange-400';
  return 'text-loss';
}

function sentimentBg(score: number): string {
  if (score >= 0.5) return 'bg-profit/20';
  if (score >= 0.2) return 'bg-green-400/20';
  if (score >= -0.2) return 'bg-yellow-400/20';
  if (score >= -0.5) return 'bg-orange-400/20';
  return 'bg-loss/20';
}

const SOURCE_COLORS: Record<string, string> = {
  Reuters: 'bg-blue-600/20 text-blue-400',
  Bloomberg: 'bg-purple-600/20 text-purple-400',
  ForexFactory: 'bg-emerald-600/20 text-emerald-400',
  Investing: 'bg-orange-600/20 text-orange-400',
};

export default function NewsPanel({ news }: NewsPanelProps) {
  if (!news || news.length === 0) {
    return (
      <div className="bg-card rounded-xl border border-border p-4 h-full flex items-center justify-center">
        <p className="text-sm text-muted-foreground">No news available</p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl border border-border flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-foreground">News Sentiment</h3>
        <span className="text-[10px] text-muted-foreground">Latest {news.length}</span>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 divide-y divide-border">
        {news.slice(0, 10).map((item, idx) => (
          <div
            key={item.id || idx}
            className="px-4 py-3 hover:bg-muted/30 transition-colors"
          >
            <div className="flex items-start justify-between gap-3">
              {/* Left: headline and meta */}
              <div className="flex-1 min-w-0">
                <p className="text-xs text-foreground leading-relaxed line-clamp-2 mb-1.5">
                  {item.headline}
                </p>
                <div className="flex items-center gap-2">
                  {/* Source badge */}
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded ${
                      SOURCE_COLORS[item.source] || 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {item.source}
                  </span>
                  {/* Relative time */}
                  <span className="text-[10px] text-muted-foreground">
                    {relativeTime(item.timestamp)}
                  </span>
                </div>
              </div>

              {/* Right: sentiment score */}
              <div
                className={`shrink-0 flex items-center justify-center w-12 h-8 rounded ${sentimentBg(
                  item.sentiment ?? 0
                )}`}
              >
                <span
                  className={`text-xs font-mono font-bold ${sentimentColor(
                    item.sentiment ?? 0
                  )}`}
                >
                  {(item.sentiment ?? 0) > 0 ? '+' : ''}
                  {(Number(item.sentiment) || 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
