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


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_report_file_path = os.path.join("artifacts", "model_report.json")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self, X_train, X_test, y_train, y_test
    ):
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

            model_metrics: dict = evaluate_model(
                X_train, X_test, y_train, y_test, models
            )
            save_json(self.model_trainer_config.model_report_file_path, model_metrics)
            best_model_name = max(
                model_metrics, key=lambda x: model_metrics[x]["r2_test"]
            )
            best_model_score = model_metrics[best_model_name]["r2_test"]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)
            logging.info(
                f"Best model found: {best_model_name} score: {best_model_score}"
            )
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            return best_model_name, best_model_score
        except Exception as e:
            raise CustomException(e, sys)
