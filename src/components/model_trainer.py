import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model, save_json

import mlflow
import mlflow.sklearn


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_report_file_path = os.path.join("artifacts", "model_report.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Training Started")

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.01, max_iter=10000),
                "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
                "RandomForestRegressor": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                ),
                "GradientBoost": GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    random_state=42,
                ),
                "XGBRegressor": XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=3,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.05,
                    depth=3,
                    verbose=False,
                    random_seed=42,
                ),
                "SVR": SVR(C=10, kernel="rbf"),
                "KNNRegressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
            }
            params = {
                "RandomForestRegressor": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [5, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
                "LinearRegression": {},
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
                "Ridge": {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
                "ElasticNet": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                },
                "XGBRegressor": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                },
                "GradientBoost": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                },
                "SVR": {
                    "C": [0.1, 1, 10, 100],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"],
                },
                "CatBoost": {
                    "iterations": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8],
                },
                "KNNRegressor": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2],
                },
            }

            model_metrics, trained_models = evaluate_model(
                X_train, X_test, y_train, y_test, models, params
            )
            save_json(self.model_trainer_config.model_report_file_path, model_metrics)
            mlflow.log_artifact(self.model_trainer_config.model_report_file_path)
            best_model_name = max(
                model_metrics, key=lambda x: model_metrics[x]["r2_test"]
            )
            best_model_score = model_metrics[best_model_name]["r2_test"]

            if best_model_score < 0.7:
                raise CustomException("No best model found", sys)
            logging.info(
                f"Best model found: {best_model_name} score: {best_model_score}"
            )
            best_model = trained_models[best_model_name]
            mlflow.sklearn.log_model(best_model, name="best_model")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            return {
                "best_model_name": best_model_name,
                "best_model_score": float(best_model_score),
            }
        except Exception as e:
            raise CustomException(e, sys)
