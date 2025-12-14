# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import config
from utils import get_logger
from models import get_lstm_model, get_cnn_model
from utils import load_processed_data

logger = get_logger()

def create_test_data_loader(data):
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.LongTensor(data['y_test'])
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return test_loader, data['y_test']

def evaluate_model(model, test_loader, y_true, label_mapping, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_predictions)
    class_labels = list(range(config.NUM_CLASSES))
    f1_macro = f1_score(all_labels, all_predictions, average='macro', labels=class_labels, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions, labels=class_labels)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
    }

def evaluate():
    logger.info("Evaluation started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_processed_data()
    test_loader, y_true = create_test_data_loader(data)
    label_mapping = data['label_mapping']
    logger.info(f"test_samples={len(data['X_test'])}, seq_len={data['seq_length']}")
    
    lstm_model = get_lstm_model().to(device)
    lstm_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_LSTM))
    logger.info("Evaluation LSTM baseline modell")
    lstm_results = evaluate_model(lstm_model, test_loader, y_true, label_mapping, device)
    logger.info("Evaluation done for LSTM modell")
    
    cnn_model = get_cnn_model().to(device)
    cnn_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_CNN))
    logger.info("Evaluation CNN modell")
    cnn_results = evaluate_model(cnn_model, test_loader, y_true, label_mapping, device)
    logger.info("Evaluation done for CNN modell")
    
    logger.info(f"LSTM: acc={lstm_results['accuracy']*100:.2f}%, f1={lstm_results['f1_macro']:.4f}")
    logger.info(f"Confusion Matrix for LSTM:\n{lstm_results['confusion_matrix']}")
    logger.info(f"CNN: acc={cnn_results['accuracy']*100:.2f}%, f1={cnn_results['f1_macro']:.4f}")
    logger.info(f"Confusion Matrix for CNN:\n{cnn_results['confusion_matrix']}")
    logger.info("Evaluation complete")

if __name__ == "__main__":
    evaluate()
