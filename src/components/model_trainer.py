import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing input data")

            # SPLIT FEATURES & TARGET
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Train Shape: {X_train.shape}, {y_train.shape}")
            logging.info(f"Test Shape: {X_test.shape}, {y_test.shape}")

            # TRAIN
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # EVALUATE
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy: {accuracy}")

            # MLFLOW (NO start_run here)
            mlflow.log_param("model", "LogisticRegression")
            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="credit-risk-model"
            )

            logging.info("Model logged to MLflow successfully")

            # LOCAL BACKUP
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            import pickle
            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            logging.info("Model saved locally as backup")

        except Exception as e:
            raise CustomException(e, sys)