import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load Dataset
df = pd.read_csv("students.csv")


# 2. Features & Target
X = df[["hours_studied", "attendance"]]
y = df["pass"]


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Pipelines
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline_dt = Pipeline([
    ("model", DecisionTreeClassifier())
])


# 5. Cross Validation
cv_lr = cross_val_score(pipeline_lr, X_train, y_train, cv=5)
cv_dt = cross_val_score(pipeline_dt, X_train, y_train, cv=5)

print("Logistic Regression CV Accuracy:", cv_lr.mean())
print("Decision Tree CV Accuracy:", cv_dt.mean())


# 6. Train Final Model
pipeline_lr.fit(X_train, y_train)


# 7. Predictions
y_pred = pipeline_lr.predict(X_test)


# 8. Evaluation
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 9. Hyperparameter Tuning
param_grid = {
    "model__C": [0.1, 1, 10]
}

grid = GridSearchCV(pipeline_lr, param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)


# 10. Feature importance / interpretation
print("Model Coefficients:", pipeline_lr.named_steps["model"].coef_)

# 11. Custom prediction
sample = pd.DataFrame([[7, 80]], columns=["hours_studied", "attendance"])
prediction = pipeline_lr.predict(sample)
print("Prediction for sample(hours_studied=7,attendance=80):", prediction)