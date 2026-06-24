import os
import pickle
import sys
import json

from src.exception import CustomException
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Student Performance Prediction")


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(obj, f, indent=4)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, X_test, y_train, y_test, models, params):

    try:
        report = {}
        trained_models = {}
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                parameters = params[name]

                if parameters:
                    rc = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=parameters,
                        cv=3,
                        n_iter=10,
                        scoring="r2",
                        random_state=42,
                    )
                    rc.fit(X_train, y_train)
                    best_model = rc.best_estimator_
                else:
                    best_model = model
                    best_model.fit(X_train, y_train)

                trained_models[name] = best_model

                # model.fit(X_train, y_train)
                y_test_pred = best_model.predict(X_test)
                y_train_pred = best_model.predict(X_train)

                mse_train = mean_squared_error(y_train, y_train_pred)
                rmse_train = root_mean_squared_error(y_train, y_train_pred)
                mae_train = mean_absolute_error(y_train, y_train_pred)
                r2_train = r2_score(y_train, y_train_pred)

                mse_test = mean_squared_error(y_test, y_test_pred)
                rmse_test = root_mean_squared_error(y_test, y_test_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                r2_test = r2_score(y_test, y_test_pred)
                mlflow.log_params(best_model.get_params())

                mlflow.log_metrics(
                    {
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                        "rmse_train": rmse_train,
                        "rmse_test": rmse_test,
                        "mae_train": mae_train,
                        "mae_test": mae_test,
                        "r2_train": r2_train,
                        "r2_test": r2_test,
                    }
                )
                if parameters:
                    mlflow.log_params(rc.best_params_)
                else:
                    mlflow.log_params(best_model.get_params())

                report[name] = {
                    "mse_train": float(mse_train),
                    "mse_test": float(mse_test),
                    "rmse_train": float(rmse_train),
                    "rmse_test": float(rmse_test),
                    "mae_train": float(mae_train),
                    "mae_test": float(mae_test),
                    "r2_train": float(r2_train),
                    "r2_test": float(r2_test),
                }
        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
