import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("dataset/placement_data.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

# --------------------------------------------------
# Drop Unnecessary Columns
# --------------------------------------------------

df = df.drop(columns=[
    "student_id",
    "salary_lpa",
    "specialization"
])

# --------------------------------------------------
# Features and Target
# --------------------------------------------------

X = df.drop("placed", axis=1)
y = df["placed"]

# --------------------------------------------------
# Identify Categorical and Numerical Columns
# --------------------------------------------------

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

best_pipeline = None
best_accuracy = 0
best_model_name = ""

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_pipeline = pipeline
        best_model_name = model_name

# --------------------------------------------------
# Save Best Pipeline
# --------------------------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(best_pipeline, "model/placement_model.pkl")

print("\n" + "=" * 60)
print("Best Model Saved Successfully")
print("=" * 60)
print("Model :", best_model_name)
print(f"Accuracy : {best_accuracy:.4f}")
print("Saved as : model/placement_model.pkl")