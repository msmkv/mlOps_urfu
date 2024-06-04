from sklearn.preprocessing import StandardScaler
import pandas as pd

# Загружаем данные
train_data = pd.read_csv('train/train_data.csv')

# Предобработка данных
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data)

# Сохраняем предобработанные данные
pd.DataFrame(scaled_train_data).to_csv('train/scaled_train_data.csv', index=False)