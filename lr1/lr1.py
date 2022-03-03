import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)  # входные от 0 до 4
Y = dataset[:, 4]  # выходные

encoder = LabelEncoder()
encoder.fit(Y)  # загружаем выходные данные в энкодер
encoded_Y = encoder.transform(Y)  # считаем количество вариантов
dummy_y = to_categorical(encoded_Y)  # разбиваем

model = Sequential()  # есть один входной и один выходной тензор
model.add(Dense(4, activation='relu'))  # обычные нейронные слои с разной
model.add(Dense(3, activation='softmax'))  # функцией активации

model.compile(  # создание модели
    optimizer='adam',  # оптимизатор меняет веса исходя из функции потерь
    loss='categorical_crossentropy',  # функция потерь используется для поиска ошибок в процессе обучения. (разностью между оцененным и истинным значениями)
    metrics=['accuracy']  # точность модели
)
"""
    batch_size: Количество образцов для изменения весов.
    (Если значение маленькое, потребуется меньше памяти и времени, модель запрашивает меньше выборок, но оценка изменения весов будет менее точной)

    epochs: Количество эпох для обучения модели. Эпоха — это итерация по всем предоставленным данным x и y.

    validation_split: процент данных для проверки
"""

dat0 = model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)
model1 = Sequential()
model1.add(Dense(4, activation='relu'))
model1.add(Dense(3, activation='softmax'))
model1.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
dat1 = model1.fit(X, dummy_y, epochs=75, batch_size=3, validation_split=0.1)  # уменьшил изменение весов
model2 = Sequential()
model2.add(Dense(4, activation='relu'))
model2.add(Dense(3, activation='softmax'))
model2.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
dat2 = model2.fit(X, dummy_y, epochs=25, batch_size=10, validation_split=0.1)  # уменьшил количество эпох
model3 = Sequential()
model3.add(Dense(4, activation='relu'))
model3.add(Dense(20, activation='relu'))
model3.add(Dense(3, activation='softmax'))
model3.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
dat3 = model3.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)  # добавил еще один слой нейронов
plt.plot(dat0.history['accuracy'], color='blue')
plt.plot(dat1.history['accuracy'], color='red')
plt.plot(dat2.history['accuracy'], color='green')
plt.plot(dat3.history['accuracy'], color='orange')
plt.show()
