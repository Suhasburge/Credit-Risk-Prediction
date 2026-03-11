import sys
import mlflow
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def run_pipeline(self):
        try:
            logging.info("Starting Training Pipeline...")

            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_registry_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("credit-risk-training")

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name="credit-risk-training-run"):

                # 1) Ingestion
                ingestion = DataIngestion()
                train_path, test_path = ingestion.initiate_data_ingestion()

                # 2) Transformation (THIS RETURNS NUMPY — KEEP IT)
                transform = DataTransformation()
                train_arr, test_arr, preprocessor_path = transform.initiate_data_transformation(
                    train_path, test_path
                )

                # 3) Training (PASS NUMPY ONLY)
                trainer = ModelTrainer()
                trainer.initiate_model_trainer(train_arr, test_arr)

                logging.info("Training completed successfully")

        except Exception as e:
            raise CustomException(e, sys)