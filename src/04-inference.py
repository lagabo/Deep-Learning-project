# Inference script
# This script runs the model on new, unseen data.

import os
import pickle
import numpy as np
import torch
import config
from utils import get_logger
from models import get_lstm_model, get_cnn_model
from utils import load_processed_data

logger = get_logger()

def predict_sample(model, sample, label_mapping, device):
    model.eval()
    
    sample_tensor = torch.FloatTensor(sample).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(sample_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0, predicted_class].item()
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = reverse_mapping[predicted_class]
    
    return predicted_label, probability

def predict():
    logger.info("Inference started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_processed_data()
    label_mapping = data['label_mapping']
    lstm_model = get_lstm_model().to(device)
    cnn_model = get_cnn_model().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH_LSTM):
        lstm_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_LSTM))
    if os.path.exists(config.MODEL_SAVE_PATH_CNN):
        cnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_CNN))
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    for i in range(5):
        sample = data['X_test'][i]
        true_label = reverse_mapping[data['y_test'][i]]
        lstm_prediction, lstm_probability = predict_sample(lstm_model, sample, label_mapping, device)
        cnn_prediction, cnn_probability = predict_sample(cnn_model, sample, label_mapping, device)
        logger.info(f"Sample {i+1}: real={true_label}, lstm={lstm_prediction}({lstm_probability:.2f}), cnn={cnn_prediction}({cnn_probability:.2f})")
    logger.info("Inference complete")

if __name__ == "__main__":
    predict()
