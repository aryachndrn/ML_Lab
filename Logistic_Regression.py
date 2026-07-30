import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mle_model = LogisticRegression(
    penalty=None,
    max_iter=1000
)

mle_model.fit(X_train, y_train)

y_pred_mle = mle_model.predict(X_test)

map_l2 = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000
)

map_l2.fit(X_train, y_train)

y_pred_l2 = map_l2.predict(X_test)

map_l1 = LogisticRegression(
    penalty='l1',
    C=1.0,
    solver='liblinear',
    max_iter=1000
)

map_l1.fit(X_train, y_train)

y_pred_l1 = map_l1.predict(X_test)

def evaluate(name, y_true, y_pred):

    print("\n", "="*45)
    print(name)
    print("="*45)

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

evaluate("MLE Logistic Regression", y_test, y_pred_mle)

evaluate("MAP (L2 Regularization)", y_test, y_pred_l2)

evaluate("MAP (L1 Regularization)", y_test, y_pred_l1)

coef_df = pd.DataFrame({
    "Feature": data.feature_names,
    "MLE": mle_model.coef_[0],
    "MAP_L2": map_l2.coef_[0],
    "MAP_L1": map_l1.coef_[0]
})

print("\nCoefficient Comparison")
print(coef_df)

print("\nNumber of Non-zero Coefficients")

print("MLE    :", np.sum(mle_model.coef_[0] != 0))
print("MAP L2 :", np.sum(map_l2.coef_[0] != 0))
print("MAP L1 :", np.sum(map_l1.coef_[0] != 0))