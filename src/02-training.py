# Model training script
# This script defines the model architecture and runs the training loop.

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import config
from utils import get_logger
from models import get_lstm_model, get_cnn_model
from utils import load_processed_data

logger = get_logger()

def create_data_loaders(data):
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.LongTensor(data['y_test'])
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, save_path, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {num_params}")
    logger.info(f"{model}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, loss_function, device)
        
        logger.info(f"Epoch {epoch}/{config.EPOCHS} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}%, val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    return model

def train():
    logger.info("Training started")
    logger.info(f"epochs={config.EPOCHS}, batch_size={config.BATCH_SIZE}, learning_rate={config.LEARNING_RATE}")
    logger.info(f"lstm_hidden_size={config.LSTM_HIDDEN_SIZE}, cnn_kernel_size={config.CNN_KERNEL_SIZE}, num_classes={config.NUM_CLASSES}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"device: {device}")
    
    data = load_processed_data()
    train_loader, val_loader, test_loader = create_data_loaders(data)
    logger.info(f"train={len(data['X_train'])}, val={len(data['X_val'])}, test={len(data['X_test'])}, seq_length={data['seq_length']}")
    
    lstm_model = get_lstm_model().to(device)
    logger.info("Training LSTM baseline modell")
    lstm_model = train_model(lstm_model, train_loader, val_loader, config.MODEL_SAVE_PATH_LSTM, device)
    cnn_model = get_cnn_model().to(device)
    logger.info("Training CNN modell")
    cnn_model = train_model(cnn_model, train_loader, val_loader, config.MODEL_SAVE_PATH_CNN, device)
    logger.info("Training complete")

if __name__ == "__main__":
    train()
