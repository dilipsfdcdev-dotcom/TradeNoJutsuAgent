-- XAUUSD Agent v2.0 Database Schema
-- Run against PostgreSQL 16+

CREATE TABLE IF NOT EXISTS signals (
    id                SERIAL PRIMARY KEY,
    timestamp         TIMESTAMPTZ NOT NULL,
    symbol            VARCHAR(10) DEFAULT 'XAUUSD',
    action            VARCHAR(4) NOT NULL,
    buy_score         FLOAT,
    sell_score        FLOAT,
    entry_threshold   FLOAT,
    ppo_action        INTEGER,
    ppo_confidence    FLOAT,
    lgbm_probability  FLOAT,
    llm_action        VARCHAR(10),
    llm_reason        TEXT,
    htf_score         FLOAT,
    htf_direction     VARCHAR(20),
    d1_score          FLOAT,
    h4_score          FLOAT,
    h1_score          FLOAT,
    m15_at_support    BOOLEAN,
    regime            VARCHAR(30),
    is_counter_trend  BOOLEAN DEFAULT FALSE,
    news_blackout     BOOLEAN DEFAULT FALSE,
    skip_reason       VARCHAR(100),
    was_executed      BOOLEAN DEFAULT FALSE,
    rsi_m3            FLOAT,
    rsi_m5            FLOAT,
    macd_m3           FLOAT,
    spread_pips       FLOAT,
    atr_m3            FLOAT,
    atr_m5            FLOAT
);

CREATE TABLE IF NOT EXISTS trades (
    id                SERIAL PRIMARY KEY,
    ticket            BIGINT UNIQUE NOT NULL,
    signal_id         INTEGER REFERENCES signals(id),
    open_time         TIMESTAMPTZ NOT NULL,
    close_time        TIMESTAMPTZ,
    direction         VARCHAR(4) NOT NULL,
    lot_size          FLOAT NOT NULL,
    open_price        FLOAT NOT NULL,
    close_price       FLOAT,
    sl_price          FLOAT NOT NULL,
    tp_price          FLOAT NOT NULL,
    initial_sl_pips   FLOAT,
    initial_risk_usd  FLOAT,
    profit_usd        FLOAT,
    profit_pips       FLOAT,
    profit_R          FLOAT,
    status            VARCHAR(10) DEFAULT 'OPEN',
    close_reason      VARCHAR(50),
    ratchet_triggered BOOLEAN DEFAULT FALSE,
    ratchet_max_R     FLOAT,
    profit_locked_usd FLOAT,
    is_counter_trend  BOOLEAN DEFAULT FALSE,
    htf_bias_score    FLOAT,
    regime            VARCHAR(30),
    balance_at_entry  FLOAT,
    feature_snapshot  JSONB
);

CREATE TABLE IF NOT EXISTS ratchet_events (
    id            SERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ DEFAULT NOW(),
    ticket        BIGINT NOT NULL,
    old_sl        FLOAT,
    new_sl        FLOAT,
    current_price FLOAT,
    profit_R      FLOAT,
    ratchet_level VARCHAR(20),
    profit_locked_usd FLOAT
);

CREATE TABLE IF NOT EXISTS learning_log (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW(),
    trigger_type    VARCHAR(50),
    param_name      VARCHAR(60),
    old_value       FLOAT,
    new_value       FLOAT,
    reason          TEXT,
    trades_analysed INTEGER,
    win_rate        FLOAT
);

CREATE TABLE IF NOT EXISTS regime_log (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ DEFAULT NOW(),
    regime      VARCHAR(30),
    htf_score   FLOAT,
    d1_score    FLOAT,
    h4_score    FLOAT,
    h1_score    FLOAT,
    atr_m5      FLOAT,
    atr_ratio   FLOAT
);

CREATE TABLE IF NOT EXISTS bot_commands (
    id           SERIAL PRIMARY KEY,
    timestamp    TIMESTAMPTZ DEFAULT NOW(),
    command      VARCHAR(30),
    value        VARCHAR(50),
    processed    BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS bot_state (
    key          VARCHAR(50) PRIMARY KEY,
    value        TEXT,
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Default state
INSERT INTO bot_state (key, value) VALUES ('status', 'RUNNING') ON CONFLICT (key) DO NOTHING;
INSERT INTO bot_state (key, value) VALUES ('risk_pct', '1.0') ON CONFLICT (key) DO NOTHING;
INSERT INTO bot_state (key, value) VALUES ('daily_pnl', '0') ON CONFLICT (key) DO NOTHING;
INSERT INTO bot_state (key, value) VALUES ('daily_trades', '0') ON CONFLICT (key) DO NOTHING;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_action ON signals(action);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trades(ticket);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time DESC);
CREATE INDEX IF NOT EXISTS idx_ratchet_ticket ON ratchet_events(ticket);
CREATE INDEX IF NOT EXISTS idx_ratchet_timestamp ON ratchet_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_learning_timestamp ON learning_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_regime_timestamp ON regime_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_commands_processed ON bot_commands(processed);
