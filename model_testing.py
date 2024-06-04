import pandas as pd
import joblib

# Загружаем модель и тестовые данные
model = joblib.load('model.pkl')
test_data = pd.read_csv('test/test_data.csv')

# Проверяем модель
predictions = model.predict(test_data)
pd.DataFrame(predictions).to_csv('test/predictions.csv', index=False)