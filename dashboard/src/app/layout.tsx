import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import Link from 'next/link';
import './globals.css';
import ConnectionStatus from '@/components/ConnectionStatus';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'TradeNoJutsu Dashboard',
  description: 'AI Trading Agent Dashboard',
};

const navItems = [
  { href: '/', label: 'Dashboard', icon: '📊' },
  { href: '/trades', label: 'Trades', icon: '📈' },
  { href: '/backtest', label: 'Backtest', icon: '🔬' },
  { href: '/settings', label: 'Settings', icon: '⚙️' },
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-background text-foreground`}>
        <div className="flex h-screen">
          {/* Sidebar */}
          <aside className="w-64 border-r border-border bg-card flex flex-col">
            <div className="p-6 border-b border-border">
              <h1 className="text-xl font-bold text-accent">TradeNoJutsu</h1>
              <p className="text-sm text-muted-foreground mt-1">AI Trading Agent</p>
            </div>
            <nav className="flex-1 p-4 space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                >
                  <span>{item.icon}</span>
                  <span>{item.label}</span>
                </Link>
              ))}
            </nav>
            <div className="p-4 border-t border-border">
              <p className="text-xs text-muted-foreground">v0.1.0</p>
            </div>
          </aside>

          {/* Main content */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header */}
            <header className="h-14 border-b border-border bg-card flex items-center justify-between px-6">
              <h2 className="text-sm font-medium text-muted-foreground">
                Trading Dashboard
              </h2>
              <ConnectionStatus />
            </header>

            {/* Page content */}
            <main className="flex-1 overflow-auto p-6">
              {children}
            </main>
          </div>
        </div>
      </body>
    </html>
  );
}
