import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([2, 4, 6, 8]).reshape(-1, 1)
y = np.array([3, 7, 5, 10])
model = LinearRegression()
model.fit(x, y)
m = model.coef_[0]
b = model.intercept_
print("Linear Regression Equation:")
print(f"y = {m:.2f}x + {b:.2f}")
y_pred = model.predict(x)
print("\nPredicted values:", y_pred)
new_x = np.array([[5]])
new_y = model.predict(new_x)
print(f"\nPrediction for x = 5: y = {new_y[0]:.2f}")
