import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_features = ["reading_score", "writing_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            logging.info("Numerical features Scaled")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            logging.info("Categorical features encoded")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features),
                ]
            )
            logging.info("preprocessor pipeline created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data for preprocessing")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = "math_score"
            train_df_X = train_df.drop(columns=[target_column_name])
            train_df_y = train_df[target_column_name]

            test_df_X = test_df.drop(columns=[target_column_name])
            test_df_y = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test datasets")
            X_train = preprocessor_obj.fit_transform(train_df_X)
            X_test = preprocessor_obj.transform(test_df_X)

            logging.info("Saving the preprocessor object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )

            return (
                X_train,
                X_test,
                train_df_y,
                test_df_y,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
