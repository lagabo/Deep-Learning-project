DATA_DIR = "/app/data"
RAW_DATA_DIR = "/app/data/raw"
PROCESSED_DATA_DIR = "/app/data/processed"
MODEL_SAVE_PATH_LSTM = "/app/model_lstm.pth"
MODEL_SAVE_PATH_CNN = "/app/model_cnn.pth"
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAlEFc87da4SLpRVTCs81KwATOAjf5GzI-IxEED_nGrjh0?e=eGgGec&download=1"

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LSTM_HIDDEN_SIZE = 2
CNN_KERNEL_SIZE = 3
CNN_CHANNELS = 16

NUM_CLASSES = 6
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15