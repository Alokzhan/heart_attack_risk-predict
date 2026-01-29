import pandas as pd

df = pd.read_csv("heart_attack_prediction_dataset.csv")
print(df.head())
print(df.info())

from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

# Features & target
X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC()
}


from sklearn.metrics import accuracy_score

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)


from sklearn.metrics import classification_report, confusion_matrix

final_preds = best_model.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, final_preds))
print(confusion_matrix(y_test, final_preds))
print(classification_report(y_test, final_preds))


import joblib

joblib.dump(best_model, "heart_attack_advanced_model.pkl")
joblib.dump(scaler, "scaler.pkl")
