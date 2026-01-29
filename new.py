# 1️⃣ Imports
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 2️⃣ Load dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# 3️⃣ Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# 4️⃣ Features & Target
X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]

# 🔥 SAVE FEATURE NAMES (VERY IMPORTANT)
feature_names = X.columns.tolist()

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6️⃣ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train_scaled, y_train)

# 8️⃣ Accuracy
preds = xgb_model.predict(X_test_scaled)
print("✅ XGBoost Accuracy:", accuracy_score(y_test, preds))

# 9️⃣ SHAP Explainability (USE DATAFRAME FOR FEATURE NAMES)
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_scaled)

shap.summary_plot(
    shap_values.values,
    X_test,
    feature_names=feature_names
)
plt.show()

# 🔟 Save model artifacts (PRODUCTION READY)
joblib.dump(xgb_model, "xgb_heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_names, "features.pkl")

print("✅ Model, scaler & feature names saved successfully")
