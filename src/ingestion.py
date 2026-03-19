import pandas as pd
import yaml
from sklearn.datasets import load_diabetes
from logger import get_logger

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def data_ingestion():
    config = load_config()
    logger = get_logger(config["logging"]["log_file"])

    logger.info("Starting data ingestion")

    data = load_diabetes(as_frame=True)
    df = data.frame

    df.to_csv(config["data"]["raw_path"], index=False)
    df.to_csv(config["data"]["processed_path"], index=False)

    logger.info("Data ingestion completed successfully")

if __name__ == "__main__":
    data_ingestion()