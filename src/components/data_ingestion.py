import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.config.configuration import ConfigurationManager


class DataIngestion:
    def __init__(self):

        # LOAD CONFIG HERE
        config = ConfigurationManager()

        ingestion_config = config.get_ingestion_config()

        self.raw_data_path = ingestion_config["raw_data_path"]
        self.train_data_path = ingestion_config["train_data_path"]
        self.test_data_path = ingestion_config["test_data_path"]

        self.data_source = config.get_data_source()  


    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            df = pd.read_csv(self.data_source)

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(self.raw_data_path, index=False)
            logging.info("Raw data stored")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logging.info("Train Test split completed")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)