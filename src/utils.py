# Utility functions
# Common helper functions used across the project.
import logging
import sys
import os
import requests
import zipfile
import config
import pickle

def get_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def download_data_and_return_folder_path():
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    zip_path = os.path.join(config.DATA_DIR, "data.zip")
    
    if not os.path.exists(zip_path):
        logger.info("Downloading zip file")
        response = requests.get(config.DATA_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(config.RAW_DATA_DIR)
    
    return os.path.join(config.RAW_DATA_DIR, "bullflagdetector", "Q5C0UO")

def load_processed_data():
    data_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data