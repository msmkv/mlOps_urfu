import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,chi2,RFE
from sklearn.ensemble import RandomForestClassifier

# Функция для нормализации данных
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

# Функция для создания данных
def create_data(size, function='sin', noise_level=0.1, anomaly=False, anomaly_intensity=3):
    x = np.linspace(0, 3*np.pi, size)
    if function == 'sin':
        data = np.sin(x)
    elif function == 'cos':
        data = np.cos(x)
    else:
        raise ValueError("Unsupported function type.")
    
    # Добавляем шум
    data += np.random.normal(0, noise_level, size)
    
    # Добавляем аномалии
    if anomaly:
        anomaly_points = np.random.choice(size, size // 10, replace=False)
        data[anomaly_points] += np.random.normal(anomaly_intensity, 0.5, size // 10)
    
    return data

# Создаем различные наборы данных
datasets = {
    'train_sin': create_data(1000, 'sin'),
    'train_cos': create_data(1000, 'cos', noise_level=0.2),
    'train_sin_anomaly': create_data(1000, 'sin', anomaly=True),
    'test_sin': create_data(300, 'sin'),
    'test_cos': create_data(300, 'cos', noise_level=0.2),
    'test_sin_anomaly': create_data(300, 'sin', anomaly=True, anomaly_intensity=5)
}

# Применяем преобразования и сохраняем данные
for name, data in datasets.items():
    # Выводим статистику до нормализации
    print(f'До нормализации: {name} - Среднее: {np.mean(data)}, Стандартное отклонение: {np.std(data)}')
    
    # Нормализуем данные
    normalized_data = normalize_data(data)
    
    # Выводим статистику после нормализации
    print(f'После нормализации: {name} - Среднее: {np.mean(normalized_data)}, Стандартное отклонение: {np.std(normalized_data)}')
    
    # Визуализируем данные
    plt.figure()
    plt.plot(data, label='Original Data')
    plt.plot(normalized_data, label='Normalized Data')
    plt.legend()
    plt.title(name)
    plt.show()
    
    # Сохраняем нормализованные данные
    dir_name = 'train' if 'train' in name else 'test'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    pd.DataFrame(normalized_data).to_csv(f'{dir_name}/{name}.csv', index=False)