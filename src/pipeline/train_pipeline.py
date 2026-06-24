from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


def run_pipeline():
    try:
        logging.info("Training Pipeline Started")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _ = (
            data_transformation.initiate_data_transformation(train_path, test_path)
        )
        model_trainer = ModelTrainer()

        results = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        logging.info("Training Pipeline Completed")
        return results

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print(run_pipeline())
