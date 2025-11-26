# train_crop_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

RANDOM_STATE = 42

# 1. Load data
data = pd.read_csv("Crop_recommendation.csv")

# 2. Features and target
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'

X = data[FEATURES]
y = data[TARGET]

# 3. Train-test split (stratified to balance crop classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# 4. ML pipeline: Scaling + Random Forest
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE))
])

# 5. Hyperparameter tuning with GridSearchCV
param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# 6. Evaluation
y_pred = best_model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Feature importance (from RandomForest inside pipeline)
rf_model = best_model.named_steps["model"]
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nFeature Importances:\n", feature_importance)

# 8. Save model
joblib.dump(best_model, "crop_recommender.pkl")
print("\n✅ Model saved as crop_recommender.pkl")
