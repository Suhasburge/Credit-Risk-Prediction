import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_preprocessor(self, df):

        numerical_columns = df.select_dtypes(exclude="object").columns
        categorical_columns = df.select_dtypes(include="object").columns

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_columns),
            ("cat", cat_pipeline, categorical_columns)
        ])

        return preprocessor


    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # ---------------- REMOVE ID COLUMN ----------------
            if "Loan_ID" in train_df.columns:
                train_df = train_df.drop(columns=["Loan_ID"])
                test_df = test_df.drop(columns=["Loan_ID"])

            # ---------------- FIX TARGET COLUMN ----------------
            target_column = "Loan_Status"

            # Convert Y/N → 1/0
            train_df[target_column] = train_df[target_column].map({"Y": 1, "N": 0})
            test_df[target_column] = test_df[target_column].map({"Y": 1, "N": 0})

            # ---------------- SPLIT FEATURES & TARGET ----------------
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column].to_numpy().reshape(-1, 1)

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column].to_numpy().reshape(-1, 1)

            # ---------------- PREPROCESS ----------------
            preprocessor = self.get_preprocessor(X_train)

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            # ---------------- SAVE ARTIFACTS ----------------
            import pickle
            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            np.save("artifacts/train_array.npy", train_arr)
            np.save("artifacts/test_array.npy", test_arr)

            logging.info("Data transformation completed")

            return train_arr, test_arr, self.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)