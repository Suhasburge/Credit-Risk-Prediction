import sys
import os
import pandas as pd
import pickle

from src.exception import CustomException
from src.logger import logging
from src.utils.model_loader import load_production_model


class PredictionPipeline:
    def __init__(self):
        try:
            logging.info("Initializing prediction pipeline")

            # Load preprocessor once at startup
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            if not os.path.exists(preprocessor_path):
                raise Exception("Preprocessing file not found!")

            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)

            logging.info("Preprocessor loaded successfully")

            # Load MLflow production model once at startup
            self.model = load_production_model()

            logging.info("Production model loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction")

            # Transform input data
            data_scaled = self.preprocessor.transform(features)

            # Predict
            preds = self.model.predict(data_scaled)

            logging.info("Prediction completed")

            return preds

        except Exception as e:
            raise CustomException(e, sys)