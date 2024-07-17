import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
from prepare_images import load_data
import matplotlib.pyplot as plt

print("We're using TF", tf.__version__)
import keras.datasets.mnist
import keras.models as M
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Reshape
import keras.backend as K
print("We are using Keras", keras.__version__)


# train_ds = tf.keras.utils.image_dataset_from_directory(
#   "TrainDataset",
#   validation_split=0.8,
#   subset="training",
#   label_mode='int',
#   seed=123)
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   "TrainDataset",
#   validation_split=0.2,
#   subset="validation",
#   label_mode='int',
#   seed=123)

# (x_train, x_test), (y_train, y_test) = keras.datasets.mnist.load_data()
(x_test, y_test), (x_train, y_train) = load_data("TrainDataset")
print(x_train)
print(y_train)
print(x_test)
print(y_test)
y_train_cat = keras.utils.to_categorical(y_train, 8)
y_test_cat = keras.utils.to_categorical(y_test, 8)
print(y_train_cat, y_test_cat)

K.clear_session()
model = M.Sequential([
    Reshape((400, 300, 1), input_shape=(400, 300)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Dense(32, activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Dense(64, activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Dense(128, activation='relu'),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Dense(256, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')
])

print(model.summary())

model.compile(  
 loss='categorical_crossentropy', #sparse_categorical_crossentropy
 optimizer='adam',
 metrics=['accuracy'] # выводим процент правильных ответов
)


result = model.fit(
 x_train,
 y_train_cat,
 epochs=6,
 validation_split=0.4,
 batch_size=32,
)
model.evaluate(x_test, y_test_cat)

plt.plot(result.history['accuracy'])
plt.show()

model.save("model")
model.save_weights("model.h5")