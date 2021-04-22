import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# load example dataset of keras containing 60k images of clothing classes on gray 28x28 pixel images
# split dataset into training and testing datasets and split between images and the labels/clothing classes
fashion_mist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mist.load_data()

# as the labels go from 0 to 9, define a list that corresponds to the respective numeric value
class_names = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# plot an example and print the corresponding clothing class
plt.figure()
plt.imshow(train_images[6])
plt.colorbar()
plt.grid(False)
plt.show()
print(class_names[train_labels[6]])

# preprocess image data to make it values between 0 and 1
train_images = train_images / 255.
test_images = test_images / 255.

# create the neural network
model = keras.Sequential([ # a simple type of neural network model
    keras.layers.Flatten(input_shape=(28, 28)),     # input layer
    # "Flatten" method takes the 2D 28x28 picture and flattens it to 1D array of 784, representing the input nodes
    keras.layers.Dense(128, activation="relu"),      # hidden layer
    # 128 nodes in the hidden layer and activation function
    keras.layers.Dense(10, activation="softmax")    # output layer
    # 10 nodes, one for each class, and activation function to make sum of all 10 node values equal 1
])

# compile the neural network
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# train the model
model.fit(train_images, train_labels, epochs=2)

# verify model with test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

# predict all pictures by the model
predictions = model.predict(test_images)