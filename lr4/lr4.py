import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from PIL import Image
from numpy import asarray


def build(optim):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_adam2():  # измененный оптимизатор adam изменил скорость обучения
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    myAdam = Adam(learning_rate=0.1)
    model.compile(optimizer=myAdam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def image_to_data(path, num):
    img = Image.open(path)
    numpydata = asarray(img)[:, :, 0]
    res = np.array([np.uint8(num), np.uint8(num)])
    test_label = to_categorical(res, num_classes=10)
    test_image = np.array([numpydata, numpydata]) / 255.0
    return test_image, test_label


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

rain_images = train_images / 255.0
test_images = test_images / 255.0
rain_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

test_images, test_labels = image_to_data('test.tif', 3)

plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()
print(train_labels[0])
model = build('adam')
model.fit(rain_images, rain_labels, epochs=5, batch_size=128)  # заменил все на rain 
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


model = build('sgd')
model.fit(rain_images, rain_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

model = build('RMSprop')
model.fit(rain_images, rain_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

model = build_adam2()
model.fit(rain_images, rain_labels, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
