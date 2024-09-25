# import dependencies
import matplotlib.pyplot as plt
import os
import requests
import numpy as np
import tensorflow as tf
import zipfile



from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from pandas import read_csv
from scikeras.wrappers import KerasClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

print("####### LOCALIZATION")

# Class for downloads
def download_data():
    url="https://github.com/JJAlmagro/subcellular_localization/raw/refs/heads/master/notebook%20tutorial/data/reduced_train.npz"
    datasetFolderPath = "dataset/"
    train_file = "reduced_train.npz"
    FilePath = os.path.join(datasetFolderPath, train_file)

    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    # Download the zip file if it does not exist
    if not os.path.exists(FilePath):
        download_file(url, FilePath)
    return FilePath
    
def download_val_data():
    url = "https://github.com/JJAlmagro/subcellular_localization/raw/refs/heads/master/notebook%20tutorial/data/reduced_val.npz"
    datasetFolderPath = "dataset/"
    val_file = "reduced_val.npz"
    FilePath = os.path.join(datasetFolderPath, val_file)

    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    # Download the file if it does not exist
    if not os.path.exists(FilePath):
        download_file(url, FilePath)
    return FilePath

download_data()
download_val_data()

# Training data treatment
file_data_train = "dataset/reduced_train.npz"

train = np.load(file_data_train)
X_train = train["X_train"]
Y_train = train["y_train"]
mask_train = train["mask_train"]

# Validation data treatment
file_data_val = "dataset/reduced_val.npz"

val = np.load(file_data_val)
X_val = val["X_val"]
Y_val = val["y_val"]
mask_val = val["mask_val"]

# Print these to visualize.
print("Shape of training dataset X")
print(X_train.shape)
print("Shape of training dataset Y")
print(Y_train.shape)
print("Shape of training dataset Mask")
print(mask_train.shape)

print("Shape of validation dataset X")
print(X_val.shape)
print("Shape of validation dataset Y")
print(Y_val.shape)
print("Shape of validation dataset Mask")
print(mask_val.shape)

classes = preprocessing.LabelEncoder()
classes.fit(Y_train)
classes_Y_train = classes.transform(Y_train)
classes_Y_val = classes.transform(Y_val)

# Encodage format onehot
onehot_Y_train = to_categorical(classes_Y_train)
onehot_Y_val = to_categorical(classes_Y_val)

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# LSTM
model = Sequential()
model.add(LSTM(64, activation="tanh", input_shape=(X_train.shape[1:3])))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dense(onehot_Y_train.shape[1], activation="softmax"))  # softmax car multi-classes

# Compile
model.compile(optimizer=Adam(learning_rate=0.0002), loss="categorical_crossentropy", metrics=["accuracy", "Precision", "Recall"])

# Run
history = model.fit(X_train, onehot_Y_train, epochs=30, batch_size=32, validation_data=(X_val, onehot_Y_val), callbacks=[early_stopping], verbose=1)

# Calculate metrics
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, onehot_Y_val, verbose=1)

# Results in Accuracy, Precision and Recall
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true_classes = np.argmax(onehot_Y_val, axis=1)

# Calculate precision and recall
precision = precision_score(y_val_true_classes, y_val_pred_classes, average='weighted')
recall = recall_score(y_val_true_classes, y_val_pred_classes, average='weighted')

print(f"Sklearn Precision (Weighted): {precision:.4f}")
print(f"Sklearn Recall (Weighted): {recall:.4f}")

# Plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('MLP accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# CNN equivalent
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(onehot_Y_train.shape[1], activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.0002), loss="categorical_crossentropy", metrics=["accuracy", "Precision","Recall"])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, onehot_Y_train, epochs=30, batch_size=32, validation_data=(X_val, onehot_Y_val), callbacks=[early_stopping], verbose=1)

# Evaluate on validation set
val_loss, val_accuracy, val_precision, val_recall = model.evaluate(X_val, onehot_Y_val, verbose=1)

# Print the metrics
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")

# Calculate Precision and Recall using sklearn for more detailed analysis
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true_classes = np.argmax(onehot_Y_val, axis=1)

# Calculate precision and recall using sklearn
precision = precision_score(y_val_true_classes, y_val_pred_classes, average='weighted')
recall = recall_score(y_val_true_classes, y_val_pred_classes, average='weighted')

print(f"Sklearn Precision (Weighted): {precision:.4f}")
print(f"Sklearn Recall (Weighted): {recall:.4f}")

# Plots
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()