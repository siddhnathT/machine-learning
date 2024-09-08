MNIST Handwritten Digit Classification


This project involves training a neural network model to classify handwritten digits from the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The project includes code for training a model, evaluating its performance, and making predictions on new data.

Prerequisites
Ensure you have the following installed:

Python 3.x
TensorFlow
NumPy
Matplotlib
scikit-learn (for evaluating accuracy)
You can install the required packages using pip:

bash
Copy code
pip install tensorflow numpy matplotlib scikit-learn
File Structure
makefile
Copy code
E:\project01\
│
├── dataset\
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── predict.py
├── train_model.py
└── mnist_model.h5
Usage
Training the Model
Prepare the Dataset: Make sure you have the MNIST dataset files (train-images.idx3-ubyte, train-labels.idx1-ubyte, t10k-images.idx3-ubyte, t10k-labels.idx1-ubyte) in the dataset directory.

Run the Training Script: Execute the train_model.py script to train the model and save it as mnist_model.h5.

bash
Copy code
python train_model.py
Review the Training Process: The script will display the training and validation loss/accuracy graphs and save the trained model.

Making Predictions
Run the Prediction Script: Execute the predict.py script to make predictions on test images or draw a digit to be predicted.

bash
Copy code
python predict.py
Interaction:

Drawing Prediction: A graphical window will open where you can draw a digit. Click and drag to draw, then press the "Predict" button to see the classification result.
Sample Prediction: The script will also display a sample image from the test set with the model's predicted digit.
Code Details
Loading MNIST Data: Functions load_mnist_images and load_mnist_labels read the MNIST dataset files and return the image and label arrays.

Model Definition: A Sequential model is defined with:

Flatten layer to reshape the input.
Dense layers with ReLU activation.
Dense output layer with softmax activation for classification.
Training: The model is trained for 25 epochs with validation split. Training history is used to plot loss and accuracy.

Prediction: The trained model is used to predict digits from the test set or from user input (drawing).

Troubleshooting
OneDNN Custom Operations Warning: You may see warnings about oneDNN operations. These can be ignored unless you need precise numerical results.
RectangleSelector Error: Ensure you are using compatible versions of matplotlib and TensorFlow. The drawing code uses matplotlib features for capturing user input.
