# Data preprocessing script
# This script handles data loading, cleaning, and transformation.

import os
import requests
import zipfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import pickle
import config
from utils import get_logger, download_data_and_return_folder_path

logger = get_logger()

def load_labels(data_path):
    label_file = os.path.join(data_path, "labels.json")
    
    with open(label_file, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    
    label_name_mapping = {
        'Bullish Normal': 'bull_flag_normal',
        'Bullish Wedge': 'bull_flag_wedge',
        'Bullish Pennant': 'bull_flag_pennant',
        'Bearish Normal': 'bear_flag_normal',
        'Bearish Wedge': 'bear_flag_wedge',
        'Bearish Pennant': 'bear_flag_pennant'
    }
    
    segments_by_file = {}

    for task in label_data:
        file_upload = task.get('file_upload', '')
        csv_filename = file_upload.split('-', 1)[-1] if '-' in file_upload else file_upload
        file_id = os.path.splitext(csv_filename)[0]
        annotations = task.get('annotations', [])

        for annotation in annotations:
            results = annotation.get('result', [])
            for result in results:
                if result.get('type') != 'timeserieslabels':
                    continue

                value = result.get('value', {})
                start_str = value.get('start')
                end_str = value.get('end')
                timeseries_labels = value.get('timeserieslabels', [])

                for label in timeseries_labels:
                    segments_by_file.setdefault(file_id, []).append({
                        'csv_filename': csv_filename,
                        'start': start_str,
                        'end': end_str,
                        'label_key': label_name_mapping[label],
                    })

    return segments_by_file

def load_csv_files(data_path):
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    csv_file_dict = {}
    for csv_file in csv_files:
        file_path = os.path.join(data_path, csv_file)
        df = pd.read_csv(file_path)
        file_id = os.path.splitext(csv_file)[0]
        csv_file_dict[file_id] = df
    logger.info(f"Loaded {len(csv_files)} CSV files")
    return csv_file_dict

def normalize_timeseries(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def _datetime_str_to_epoch_ms(dt_str):
    ts = pd.Timestamp(dt_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return int(ts.value // 1_000_000)

def _adjust_interval_to_df_range(start_ms, end_ms, ts_min, ts_max):
    hour_ms = 3_600_000
    for offset_hours in [0, 1, -1, 2, -2, 3, -3]:
        s = start_ms - offset_hours * hour_ms
        e = end_ms - offset_hours * hour_ms
        if e < ts_min or s > ts_max:
            continue
        return s, e
    return start_ms, end_ms

def prepare_dataset(csv_file_dict, labels_data):
    X_list = []
    y_list = []
    file_ids = []
    
    label_mapping = {
        'bull_flag_normal': 0,
        'bull_flag_wedge': 1,
        'bull_flag_pennant': 2,
        'bear_flag_normal': 3,
        'bear_flag_wedge': 4,
        'bear_flag_pennant': 5
    }
    
    for csv_id, df in csv_file_dict.items():
        if csv_id not in labels_data:
            continue
        price_column = None
        for col in ['close', 'Close', 'price', 'Price', 'value', 'Value']:
            if col in df.columns:
                price_column = col
                break

        if price_column is None:
            continue

        ts_series = df['timestamp'].astype(np.int64)
        ts_min = int(ts_series.min())
        ts_max = int(ts_series.max())

        segments = labels_data[csv_id]
        for seg_idx, seg in enumerate(segments):
            label_key = seg.get('label_key')
            label = label_mapping[label_key]

            start_ms = _datetime_str_to_epoch_ms(seg['start'])
            end_ms = _datetime_str_to_epoch_ms(seg['end'])
            start_ms, end_ms = _adjust_interval_to_df_range(start_ms, end_ms, ts_min, ts_max)

            mask = (ts_series >= start_ms) & (ts_series <= end_ms)
            df_seg = df.loc[mask]
            if df_seg.shape[0] < 2:
                continue

            prices = df_seg[price_column].astype(np.float32).values
            normalized_prices = normalize_timeseries(prices)

            X_list.append(normalized_prices)
            y_list.append(label)
            file_ids.append(f"{csv_id}_seg{seg_idx}")

    logger.info(f"Samples: {len(X_list)}, distribution: {np.bincount(np.array(y_list), minlength=len(label_mapping)).tolist()}")
    return X_list, y_list, file_ids, label_mapping

def pad_sequences(X_list):
    max_length = max(len(x) for x in X_list)
    X_padded = []
    for x in X_list:
        if len(x) < max_length:
            padded = np.pad(x, (0, max_length - len(x)), mode='edge')
        else:
            padded = x[:max_length]
        X_padded.append(padded)
    
    return np.array(X_padded), max_length

def split_dataset(X, y, file_ids):
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, file_ids, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return {
        'X_train': X_train,
        'y_train': y_train,
        'ids_train': ids_train,
        'X_val': X_val,
        'y_val': y_val,
        'ids_val': ids_val,
        'X_test': X_test,
        'y_test': y_test,
        'ids_test': ids_test
    }

def save_processed_data(data_splits):
    logger.info("Saving processed data")
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    save_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_splits, f)
    logger.info(f"Data saved to {save_path}")

def preprocess():
    logger.info("Data preprocessing started")
    data_path = download_data_and_return_folder_path()
    labels_data = load_labels(data_path)
    csv_file_dict = load_csv_files(data_path)

    X_list, y_list, file_ids, label_mapping = prepare_dataset(csv_file_dict, labels_data)
    X_padded, length_of_sequences = pad_sequences(X_list)
    y = np.array(y_list)
    logger.info(f"Dataset shape: {X_padded.shape}, length_of_sequences: {length_of_sequences}")
    data_splits = split_dataset(X_padded, y, file_ids)
    data_splits['label_mapping'] = label_mapping
    data_splits['seq_length'] = length_of_sequences
    save_processed_data(data_splits)
    logger.info("processing done")

if __name__ == "__main__":
    preprocess()
