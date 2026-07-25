import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("auto-mpg.csv")
data = data.dropna()

# Features and target
X = data[['displacement']]
y = data['mpg']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.02, random_state=42
)

# Linear Regression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

print("-- Linear Regression --")
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

# Polynomial Regression

print("\n-- Polynomial Regression --")

for degree in [2, 3, 4, 5]:
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nDegree:", degree)
    print("MSE:", mse)
    print("R2 Score:", r2)


# Plot Results

plt.figure(figsize=(10, 6))

# Original data
plt.scatter(X, y, color="black", s=20, label="Data Points")

# Create smooth range for plotting
X_range = np.linspace(
    X['displacement'].min(),
    X['displacement'].max(),
    300
).reshape(-1, 1)

# Linear Regression Line
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear = linear_model.predict(X_range)

plt.plot(X_range, y_linear, linewidth=2, label="Linear")

# Polynomial Curves
for degree in range(1, 15):
    poly = PolynomialFeatures(degree=degree)

    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_range_poly = poly.transform(X_range)
    y_range = model.predict(X_range_poly)

    plt.plot(X_range, y_range, linewidth=2, label=f"Degree {degree}")

# Labels and title
plt.xlabel("Engine Displacement")
plt.ylabel("Miles Per Gallon (MPG)")
plt.title("Linear vs Polynomial Regression on Auto MPG Dataset")
plt.legend()
plt.grid(True)

plt.show()