import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os

# Paths to MNIST dataset files
dataset_dir = 'E:\\project01\\dataset'
train_images_path = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')

# Function to load MNIST images
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number {magic_number} for images file.")
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

# Function to load MNIST labels
def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        if magic_number != 2049:
            raise ValueError(f"Invalid magic number {magic_number} for labels file.")
        num_labels = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
    return labels

# Load training images and labels
x_train = load_mnist_images(train_images_path)
y_train = load_mnist_labels(train_labels_path)

# Load test images and labels
x_test = load_mnist_images(test_images_path)
y_test = load_mnist_labels(test_labels_path)

# Normalize images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=25, validation_split=0.2)

# Save the model
model.save('mnist_model.h5')

# Load the model
loaded_model = load_model('mnist_model.h5')

# Evaluate the model
y_prob = loaded_model.predict(x_test)
y_pred = y_prob.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')

plt.show()

# Predict a single image
sample_index = 4
plt.imshow(x_test[sample_index], cmap='gray')
plt.title(f'Predicted: {loaded_model.predict(x_test[sample_index].reshape(1, 28, 28)).argmax(axis=1)[0]}')
plt.show()
