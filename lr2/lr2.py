import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)  # загрузка входных и выходных данных

# input_dim - количество данных которые передаются, у нас их 60
# kernel_initializer - изначально расставляет веса
model0 = Sequential()
model0.add(Dense(60, input_dim=60, kernel_initializer='random_normal', activation='relu'))
model0.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
# эта функция потерь для бинарной классификации либо 0 либо 1
model0.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dat0 = model0.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)

model1 = Sequential()
model1.add(Dense(30, input_dim=60, kernel_initializer='random_normal', activation='relu'))
model1.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dat1 = model1.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)

model2 = Sequential()
model2.add(Dense(60, input_dim=60, kernel_initializer='random_normal', activation='relu'))
model2.add(Dense(15, kernel_initializer='random_normal', activation='relu'))
model2.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dat2 = model2.fit(X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1)

plt.plot(dat0.history['accuracy'])
plt.plot(dat1.history['accuracy'])
plt.plot(dat2.history['accuracy'])
plt.title('Models accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['accuracy1', 'accuracy2', 'accuracy3'], loc='best')
plt.show()

plt.plot(dat0.history['loss'])
plt.plot(dat1.history['loss'])
plt.plot(dat2.history['loss'])
plt.title('Models loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['loss1', 'loss2', 'loss3'], loc='best')
plt.show()
