Student Pass-Fail Detection (Machine Learning Project)

Diagram:


This project predicts whether a student will pass or fail based on study hours and attendance using machine learning models.

1. Import Libraries
We use:
- pandas: data handling
- sklearn: model building, evaluation, and tuning

2. Load Dataset
df = pd.read_csv("students.csv")

Dataset contains:
- hours_studied
- attendance
- pass (target variable)

3. Define Features & Target
X = df[["hours_studied", "attendance"]]
y = df["pass"]

X (features): study hours and attendance
y (label): pass or fail

4. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Split:
- 80% training data
- 20% testing data

5. Build Pipelines
Logistic Regression Pipeline:
- StandardScaler
- LogisticRegression

Decision Tree Pipeline:
- DecisionTreeClassifier

6. Cross Validation (Model Comparison)
cross_val_score(...)

Compare:
- Logistic Regression accuracy
- Decision Tree accuracy

7. Train Final Model
pipeline_lr.fit(X_train, y_train)

8. Predictions
y_pred = pipeline_lr.predict(X_test)

9. Evaluation
- Accuracy
- Confusion Matrix
- Classification Report

10. Hyperparameter Tuning
GridSearchCV(pipeline_lr, param_grid)

C values: [0.1, 1, 10]

11. Model Interpretation
pipeline_lr.named_steps["model"].coef_

12. Custom Prediction
sample = [[7, 80]]
pipeline_lr.predict(sample)

Summary:
Full ML pipeline from preprocessing to prediction.

Tech Stack:
Python, Pandas, Scikit-learn
