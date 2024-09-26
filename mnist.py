
# import dependencies
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import confusion_matrix



train_set = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")

test_set = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")

# function to visualize images
def visual(set, labels):
    """
    image (numpy array) : MNIST image with dims (28, 28)
    label (int) : Label associated with the image
    """

    for image, label in zip(set, labels):

        # remove a dimension for grayscale: (28, 28, 1) => (28, 28)
        image = image.squeeze(2)

        # plot the image
        plt.imshow(image, cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

# show the first two images of each dataset(train/test)
#visual(train_set[0:2], train_labels[0:2])
visual(test_set[0:2], test_labels[0:2])

class models():
    """ Class defining the ML models.
    """

    def __init__(self):
        pass

    def mlp():
        """ First, multilayer model without using convolution.
        """
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(10, activation="softmax"))  # 10 classes
        

        # compile
        model.compile(optimizer=RMSprop(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model

    def cnn():
        """ Second, 2D convolution model using cnn.
        """
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation="softmax"))  # 10 classes
        

        # compile
        model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

# Implemet Early Stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

# Train using MLP
mlp_model = models.mlp()
mlp_history = mlp_model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=30, batch_size=128, callbacks=[early_stopping])

# Train using CNN
cnn_model = models.cnn()
cnn_history = cnn_model.fit(train_set, train_labels, validation_data=(test_set, test_labels), epochs=30, batch_size=128, callbacks=[early_stopping])

# Validation
mlp_loss, mlp_accuracy = mlp_model.evaluate(test_set, test_labels)
cnn_loss, cnn_accuracy = cnn_model.evaluate(test_set, test_labels)

print(f"MLP - Loss: {mlp_loss:.4f}, Accuracy: {mlp_accuracy:.4f}")
print(f"CNN - Loss: {cnn_loss:.4f}, Accuracy: {cnn_accuracy:.4f}")

# Heatmap
mlp_cont_pred = mlp_model.predict(test_set)
cnn_cont_pred = cnn_model.predict(test_set)

mlp_predicted = np.argmax(mlp_cont_pred, axis=1)
cnn_predicted = np.argmax(cnn_cont_pred, axis=1)

sns.heatmap(confusion_matrix(mlp_predicted, test_labels, normalize="true"), annot=True, fmt='.2f', cmap='Blues')
plt.title("MLP Confusion Matrix")
plt.show()
sns.heatmap(confusion_matrix(cnn_predicted, test_labels, normalize="true"), annot=True, fmt='.2f', cmap='Blues')
plt.title("CNN Confusion Matrix")
plt.show()

# Plots
plt.plot(mlp_history.history['accuracy'])
plt.plot(mlp_history.history['val_accuracy'])
plt.title('MLP accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('CNN accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Summary of results

""" The MLP model offered good results, its accuracy measuring above a random guesser algorithm.
However, its performance was superseded by the CNN model. This result is to be expected, as the
Tensorial image data the CNN uses is superior to the vectorial data of the MLP, as it can better
interpret spatial relations between pixels, rendering better predictions.
"""

