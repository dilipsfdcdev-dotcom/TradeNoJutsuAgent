"""Brain 2: LSTM pattern confidence scorer."""

import numpy as np
import structlog
import torch
import torch.nn as nn
from pathlib import Path

logger = structlog.get_logger()


class MultiTFLSTM(nn.Module):
    """Lightweight LSTM with 3 input branches and multi-head output."""

    def __init__(self):
        super().__init__()
        # Branch A: 1M candles (30 steps x 5 features)
        self.lstm_1m = nn.LSTM(input_size=5, hidden_size=64, batch_first=True)
        self.drop_1m = nn.Dropout(0.2)

        # Branch B: 3M candles (20 steps x 5 features)
        self.lstm_3m = nn.LSTM(input_size=5, hidden_size=32, batch_first=True)
        self.drop_3m = nn.Dropout(0.2)

        # Branch C: 15M candles (10 steps x 5 features)
        self.lstm_15m = nn.LSTM(input_size=5, hidden_size=16, batch_first=True)
        self.drop_15m = nn.Dropout(0.2)

        # Merge: 64 + 32 + 16 = 112
        self.fc1 = nn.Linear(112, 64)
        self.drop_merge = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)

        # Output heads
        self.direction_head = nn.Linear(32, 3)   # buy/sell/wait (softmax)
        self.confidence_head = nn.Linear(32, 1)  # 0-1 (sigmoid)
        self.regime_head = nn.Linear(32, 3)      # trending/ranging/volatile (softmax)

    def forward(self, seq_1m, seq_3m, seq_15m):
        _, (h_1m, _) = self.lstm_1m(seq_1m)
        h_1m = self.drop_1m(h_1m.squeeze(0))

        _, (h_3m, _) = self.lstm_3m(seq_3m)
        h_3m = self.drop_3m(h_3m.squeeze(0))

        _, (h_15m, _) = self.lstm_15m(seq_15m)
        h_15m = self.drop_15m(h_15m.squeeze(0))

        merged = torch.cat([h_1m, h_3m, h_15m], dim=1)
        x = torch.relu(self.fc1(merged))
        x = self.drop_merge(x)
        x = torch.relu(self.fc2(x))

        direction = torch.softmax(self.direction_head(x), dim=1)
        confidence = torch.sigmoid(self.confidence_head(x))
        regime = torch.softmax(self.regime_head(x), dim=1)

        return direction, confidence, regime


class LSTMConfidence:
    DIRECTION_MAP = {0: "buy", 1: "sell", 2: "wait"}
    REGIME_MAP = {0: "trending", 1: "ranging", 2: "volatile"}

    def __init__(self, symbol: str, model_path: str | None = None):
        self.symbol = symbol
        self.device = torch.device("cpu")
        self.model = MultiTFLSTM().to(self.device)
        self.confidence_threshold = 0.60
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, sequences: dict) -> dict:
        """
        Input: lstm_sequences dict with seq_1m (30,5), seq_3m (20,5), seq_15m (10,5)
        Output: {"direction": "buy", "confidence": 0.74, "regime": "trending", "pass": True}
        ~5ms on CPU.
        """
        self.model.eval()
        with torch.no_grad():
            seq_1m = torch.FloatTensor(sequences["seq_1m"]).unsqueeze(0).to(self.device)
            seq_3m = torch.FloatTensor(sequences["seq_3m"]).unsqueeze(0).to(self.device)
            seq_15m = torch.FloatTensor(sequences["seq_15m"]).unsqueeze(0).to(self.device)

            direction, confidence, regime = self.model(seq_1m, seq_3m, seq_15m)

            dir_idx = direction.argmax(dim=1).item()
            conf_val = confidence.item()
            reg_idx = regime.argmax(dim=1).item()

        return {
            "direction": self.DIRECTION_MAP[dir_idx],
            "confidence": round(conf_val, 4),
            "regime": self.REGIME_MAP[reg_idx],
            "pass": conf_val >= self.confidence_threshold,
        }

    def retrain(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        epochs: int = 100,
        patience: int = 10,
        lr: float = 0.001,
        batch_size: int = 64,
    ) -> dict:
        """
        Retrain with early stopping and learning rate scheduling.

        Each dataset sample must yield a tuple of 6 tensors:
            (seq_1m, seq_3m, seq_15m, dir_label, conf_label, regime_label)
        where:
            seq_1m:      (30, 5) float
            seq_3m:      (20, 5) float
            seq_15m:     (10, 5) float
            dir_label:   int in {0, 1, 2}
            conf_label:  float in [0, 1]
            regime_label: int in {0, 1, 2}

        Returns: dict with training metrics.
        """
        # ---- DataLoaders ------------------------------------------------
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        )

        # ---- Loss functions ---------------------------------------------
        direction_loss_fn = nn.CrossEntropyLoss()
        confidence_loss_fn = nn.MSELoss()
        regime_loss_fn = nn.CrossEntropyLoss()

        # ---- Optimizer and scheduler ------------------------------------
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
        )

        # ---- Noise augmentation helper ----------------------------------
        noise_std = 0.01

        def add_noise(tensor: torch.Tensor) -> torch.Tensor:
            """Add small Gaussian noise for data augmentation during training."""
            return tensor + torch.randn_like(tensor) * noise_std

        # ---- Tracking ---------------------------------------------------
        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dir_acc": [],
            "val_regime_acc": [],
            "val_conf_mae": [],
            "lr": [],
        }

        # ---- Training loop ----------------------------------------------
        for epoch in range(1, epochs + 1):
            # ---------- Train phase ----------
            self.model.train()
            train_loss_sum = 0.0
            train_samples = 0

            for batch in train_loader:
                seq_1m, seq_3m, seq_15m, dir_label, conf_label, regime_label = batch

                # Move to device
                seq_1m = seq_1m.to(self.device)
                seq_3m = seq_3m.to(self.device)
                seq_15m = seq_15m.to(self.device)
                dir_label = dir_label.long().to(self.device)
                conf_label = conf_label.float().to(self.device)
                regime_label = regime_label.long().to(self.device)

                # Gaussian noise augmentation on input sequences
                seq_1m_aug = add_noise(seq_1m)
                seq_3m_aug = add_noise(seq_3m)
                seq_15m_aug = add_noise(seq_15m)

                # Forward pass
                dir_pred, conf_pred, regime_pred = self.model(
                    seq_1m_aug, seq_3m_aug, seq_15m_aug,
                )

                # Compute combined loss
                loss_dir = direction_loss_fn(dir_pred, dir_label)
                loss_conf = confidence_loss_fn(conf_pred.squeeze(1), conf_label)
                loss_regime = regime_loss_fn(regime_pred, regime_label)
                loss = loss_dir + loss_conf + loss_regime

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size_actual = seq_1m.size(0)
                train_loss_sum += loss.item() * batch_size_actual
                train_samples += batch_size_actual

            avg_train_loss = train_loss_sum / max(train_samples, 1)

            # ---------- Validation phase ----------
            self.model.eval()
            val_loss_sum = 0.0
            val_samples = 0
            val_dir_correct = 0
            val_regime_correct = 0
            val_conf_abs_err = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    seq_1m, seq_3m, seq_15m, dir_label, conf_label, regime_label = batch

                    seq_1m = seq_1m.to(self.device)
                    seq_3m = seq_3m.to(self.device)
                    seq_15m = seq_15m.to(self.device)
                    dir_label = dir_label.long().to(self.device)
                    conf_label = conf_label.float().to(self.device)
                    regime_label = regime_label.long().to(self.device)

                    dir_pred, conf_pred, regime_pred = self.model(
                        seq_1m, seq_3m, seq_15m,
                    )

                    loss_dir = direction_loss_fn(dir_pred, dir_label)
                    loss_conf = confidence_loss_fn(conf_pred.squeeze(1), conf_label)
                    loss_regime = regime_loss_fn(regime_pred, regime_label)
                    loss = loss_dir + loss_conf + loss_regime

                    batch_size_actual = seq_1m.size(0)
                    val_loss_sum += loss.item() * batch_size_actual
                    val_samples += batch_size_actual

                    # Accuracy metrics
                    val_dir_correct += (dir_pred.argmax(dim=1) == dir_label).sum().item()
                    val_regime_correct += (regime_pred.argmax(dim=1) == regime_label).sum().item()
                    val_conf_abs_err += (conf_pred.squeeze(1) - conf_label).abs().sum().item()

            avg_val_loss = val_loss_sum / max(val_samples, 1)
            val_dir_acc = val_dir_correct / max(val_samples, 1)
            val_regime_acc = val_regime_correct / max(val_samples, 1)
            val_conf_mae = val_conf_abs_err / max(val_samples, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(round(avg_train_loss, 6))
            history["val_loss"].append(round(avg_val_loss, 6))
            history["val_dir_acc"].append(round(val_dir_acc, 4))
            history["val_regime_acc"].append(round(val_regime_acc, 4))
            history["val_conf_mae"].append(round(val_conf_mae, 4))
            history["lr"].append(current_lr)

            # Step the scheduler
            scheduler.step(avg_val_loss)

            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "lstm_train_epoch",
                    symbol=self.symbol,
                    epoch=epoch,
                    train_loss=round(avg_train_loss, 6),
                    val_loss=round(avg_val_loss, 6),
                    val_dir_acc=round(val_dir_acc, 4),
                    val_conf_mae=round(val_conf_mae, 4),
                    lr=current_lr,
                )

            # ---------- Early stopping / best model tracking ----------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state_dict = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "lstm_early_stop",
                        symbol=self.symbol,
                        epoch=epoch,
                        best_val_loss=round(best_val_loss, 6),
                    )
                    break

        # ---- Restore best model -----------------------------------------
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        self.model.eval()

        result = {
            "epochs_run": len(history["train_loss"]),
            "best_val_loss": round(best_val_loss, 6),
            "final_val_dir_acc": history["val_dir_acc"][-1],
            "final_val_regime_acc": history["val_regime_acc"][-1],
            "final_val_conf_mae": history["val_conf_mae"][-1],
            "final_lr": history["lr"][-1],
            "history": history,
        }
        logger.info("lstm_retrain_complete", symbol=self.symbol, **{
            k: v for k, v in result.items() if k != "history"
        })
        return result

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
