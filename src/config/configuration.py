import yaml
import os

class ConfigurationManager:
    def __init__(self, config_path="configs/paths.yml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get_data_source(self):
        return self.config["data_source"]

    def get_ingestion_config(self):
        return self.config["data_ingestion"]