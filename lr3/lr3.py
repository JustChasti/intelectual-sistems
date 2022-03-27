import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)  # размерность входного обучающего массива
print(test_data.shape)  # размерность тестового массива
print(test_targets)  # выходные данные для тестов

mean = train_data.mean(axis=0)  # среднее значение
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
"""
нормализация данных

Исходные значения признаков могут изменяться в очень большом диапазоне и отличаться друг от друга на несколько порядков.
Данные сильно различаются между собой по абсолютным величинам.
Работа аналитических моделей машинного обучения с такими показателями окажется некорректной:
дисбаланс между значениями признаков может вызвать неустойчивость работы модели, ухудшить результаты обучения и замедлить процесс моделирования.

"""


def build_model():
    model = Sequential()  # модель где каждый слой имеет 1 входной и выходной тензор
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) # создание модели
    return model


num_epochs = 40
k = 5  # делим данные на блоки, так как их мало
num_val_samples = len(train_data) // k
all_scores = []
all_scores_1 = []
data_mse = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]  # тут просто берутся диапазоны данных для обучения модели
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    data0 = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)  # проводим оценку модели
    all_scores.append(val_mae)  # записываем точность
    all_scores_1.append(val_mse)

    dat0_mse = data0.history['loss']
    data_mse.append(dat0_mse)  # записываем ошибку в список

mae0 = np.mean(all_scores)
print(mae0)  # нормализуем значения точности
# plt.plot(all_scores, color='blue')
# plt.show()
mse0 = np.mean(all_scores_1)
print(mse0)  # нормализуем значения ошибки

mse_res = [] # вывожу средний график ошибки
for i in range(len(data_mse[0])):
    mse_res.append((data_mse[0][i] + data_mse[1][i] + data_mse[2][i])/3)
plt.plot(mse_res, color='red')
plt.show()
"""
Вопрос 1:
Классификация - это предсказание метки или категории.

Алгоритм классификации классифицирует требуемый набор данных в одну из двух или более меток
Регрессия - это поиск оптимальной функции для определения данных о непрерывных реальных значениях и прогнозирования этой величины.

То есть, если классификация относит объект к какому-то либо классу, то регрессия выдает оптимальную функцию, для прогнозирования значений

например классификация:
будет ли завтра (солнце дождь или снег)

регрессия:
прогнозирование цены акций ( в зависимости от каких-то внешних данных)

"""
