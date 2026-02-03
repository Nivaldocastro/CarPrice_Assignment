import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Carregamento do dataset
df = pd.read_csv('CarPrice_dataset_ajustado.csv')
print(df.head())


# 1. Regressão linear simples (feature mais correlacionada)
X_plot = df[['enginesize']]
y_plot = df['price']

model_simple = LinearRegression()
model_simple.fit(X_plot, y_plot)

y_line = model_simple.predict(X_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X_plot, y_plot, alpha=0.6)
plt.plot(X_plot, y_line, color='red')
plt.xlabel("Engine Size")
plt.ylabel("Price")
plt.title("Regressão Linear Simples (Engine Size vs Price)")
plt.show()

# Regressão Linear Simples
X = df[['enginesize']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Regressão Linear Simples:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

