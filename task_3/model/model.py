from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pickle

# Dataset loading
iris = load_iris()
X = iris.data  # type: ignore
y = iris.target  # type: ignore

# Dividing the dataset into training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

# Normalisation of data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Saving the model and scaler
dump(model, 'model.joblib')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Saving target_names to a file
with open('target_names.pkl', 'wb') as f:
    pickle.dump(iris.target_names, f)  # type: ignore
