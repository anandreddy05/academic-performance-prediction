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


def evaluate_model(X_train, X_test, y_train, y_test, models):

    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)

            mse_train = mean_squared_error(y_train, y_train_pred)
            rmse_train = root_mean_squared_error(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

            mse_test = mean_squared_error(y_test, y_test_pred)
            rmse_test = root_mean_squared_error(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)

            report[name] = {
                "mse_train": mse_train,
                "mse_test": mse_test,
                "rmse_train": rmse_train,
                "rmse_test": rmse_test,
                "mae_trian": mae_train,
                "mae_test": mae_test,
                "r2_train": r2_train,
                "r2_test": r2_test,
            }
        return report

    except Exception as e:
        raise CustomException(e, sys)
