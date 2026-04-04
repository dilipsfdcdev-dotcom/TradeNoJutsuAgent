"""Automated ML training pipeline."""
import structlog
from pathlib import Path
from datetime import datetime

logger = structlog.get_logger()
MODELS_DIR = Path(__file__).parent / "models"


class TrainingPipeline:
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)

    def bootstrap_initial_models(self, symbols: list[str]) -> dict:
        """Train initial models from historical/synthetic data."""
        from agent.ml.xgboost_filter import XGBoostFilter
        from agent.ml.lstm_model import LSTMConfidence
        from agent.ml.data_prep import prepare_training_data, prepare_lstm_dataset

        results = {}
        for symbol in symbols:
            logger.info("bootstrap_start", symbol=symbol)
            xgb_data = prepare_training_data(symbol)

            if xgb_data is None or len(xgb_data["features"]) < 50:
                logger.warning("insufficient_bootstrap_data", symbol=symbol)
                results[symbol] = {"status": "skipped", "reason": "insufficient data"}
                continue

            # Train XGBoost
            xgb = XGBoostFilter(symbol)
            xgb_result = xgb.retrain(xgb_data["features"], xgb_data["labels"])
            xgb_path = str(self.models_dir / f"xgb_{symbol.lower()}.json")
            xgb.save_model(xgb_path)

            # Train LSTM
            lstm_data = prepare_lstm_dataset(symbol)
            lstm_result = {"status": "skipped"}
            if lstm_data is not None and len(lstm_data) >= 50:
                import torch
                lstm = LSTMConfidence(symbol)
                split = int(len(lstm_data) * 0.8)
                train_ds = torch.utils.data.Subset(lstm_data, range(split))
                val_ds = torch.utils.data.Subset(lstm_data, range(split, len(lstm_data)))
                lstm_result = lstm.retrain(train_ds, val_ds, epochs=30)
                lstm.save_model(str(self.models_dir / f"lstm_{symbol.lower()}.pt"))

            results[symbol] = {"xgb": xgb_result, "lstm": lstm_result}
            logger.info("bootstrap_complete", symbol=symbol)
        return results

    async def run_xgb_retrain(self, symbol: str, session=None) -> dict:
        """Weekly XGBoost retrain."""
        from agent.ml.data_prep import prepare_training_data_from_db
        from agent.ml.xgboost_filter import XGBoostFilter

        data = await prepare_training_data_from_db(session, symbol, weeks=4) if session else None
        if data is None or len(data.get("features", [])) < 50:
            return {"error": "insufficient_data", "deployed": False}

        model_path = self.models_dir / f"xgb_{symbol.lower()}.json"
        xgb = XGBoostFilter(symbol, str(model_path) if model_path.exists() else None)
        result = xgb.retrain(data["features"], data["labels"])
        if result.get("deployed"):
            xgb.save_model(str(model_path))
        logger.info("xgb_retrain_complete", symbol=symbol, **result)
        return result

    async def run_lstm_retrain(self, symbol: str, session=None) -> dict:
        """Monthly LSTM retrain."""
        from agent.ml.data_prep import prepare_lstm_dataset_from_db
        from agent.ml.lstm_model import LSTMConfidence
        import torch

        dataset = await prepare_lstm_dataset_from_db(session, symbol, months=3) if session else None
        if dataset is None or len(dataset) < 100:
            return {"error": "insufficient_data", "deployed": False}

        split = int(len(dataset) * 0.9)
        train_ds = torch.utils.data.Subset(dataset, range(split))
        val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

        model_path = self.models_dir / f"lstm_{symbol.lower()}.pt"
        lstm = LSTMConfidence(symbol, str(model_path) if model_path.exists() else None)
        result = lstm.retrain(train_ds, val_ds, epochs=100, patience=10)
        if result.get("deployed"):
            lstm.save_model(str(model_path))
        logger.info("lstm_retrain_complete", symbol=symbol)
        return result


training_pipeline = TrainingPipeline()
