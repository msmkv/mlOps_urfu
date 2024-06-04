from sklearn.linear_model import LinearRegression
import pandas as pd

# Загружаем предобработанные данные
train_data = pd.read_csv('train/scaled_train_data.csv')

# Создаем и обучаем модель
model = LinearRegression()
model.fit(train_data, range(len(train_data)))

# Сохраняем модель
import joblib
joblib.dump(model, 'model.pkl')