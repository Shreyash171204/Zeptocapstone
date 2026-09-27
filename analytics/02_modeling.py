import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "titanic.csv")
CHART_DIR = os.path.join(BASE, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# LOAD THE CLEANED DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("MODEL DATA")
print("=" * 70)

print("Shape:", df.shape)
print("\nClass balance:")
print(df["survived"].value_counts())
print("\nClass proportions:")
print(df["survived"].value_counts(normalize=True))


# ============================================================
# CLASSIFICATION
# ============================================================

target = "survived"

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df[target]


# Required stratified split BEFORE preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)

print(
    "\nStratification preserves the class distribution between "
    "training and test data, which is important because the "
    "survival classes are imbalanced."
)


numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]


# Fit preprocessing ONLY inside training pipeline
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# THREE CLASSIFIERS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


results = []
fitted_models = {}


def evaluate_model(name, pipeline):
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = pipeline.decision_function(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUC": auc
        }
    )

    fitted_models[name] = pipeline

    return probabilities


roc_data = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    probabilities = evaluate_model(
        name,
        pipeline
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_data[name] = (
        fpr,
        tpr,
        roc_auc_score(y_test, probabilities)
    )


# ============================================================
# ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={auc:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "10_roc_curves.png")
)

plt.close()


# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

tree_pipeline = fitted_models["Decision Tree"]

tree_model = tree_pipeline.named_steps["model"]
tree_preprocessor = tree_pipeline.named_steps["preprocessor"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(22, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=False,
    max_depth=4,
    fontsize=7
)

plt.title("Decision Tree")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "11_decision_tree.png"),
    dpi=150
)

plt.close()


# ============================================================
# CLASSIFICATION COMPARISON TABLE
# ============================================================

classification_results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("CLASSIFICATION COMPARISON")
print("=" * 70)

print(
    classification_results.to_string(
        index=False
    )
)

classification_results.to_csv(
    os.path.join(
        BASE,
        "classification_results.csv"
    ),
    index=False
)


# ============================================================
# IMBALANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("IMBALANCE HANDLING")
print("=" * 70)


def imbalance_evaluation(name, pipeline):

    pipeline.fit(
        X_train,
        y_train
    )

    pred = pipeline.predict(X_test)

    return {
        "Strategy": name,
        "Precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            pred,
            zero_division=0
        )
    }


imbalance_results = []

# A. Baseline
baseline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

imbalance_results.append(
    imbalance_evaluation(
        "Baseline",
        baseline
    )
)


# B. class_weight balanced
balanced_preprocessor = preprocessor

balanced = Pipeline(
    steps=[
        ("preprocessor", balanced_preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

imbalance_results.append(
    imbalance_evaluation(
        "class_weight='balanced'",
        balanced
    )
)


# C. SMOTE on training fold only
smote_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

imbalance_results.append(
    imbalance_evaluation(
        "SMOTE",
        smote_pipeline
    )
)


imbalance_df = pd.DataFrame(
    imbalance_results
)

print(
    imbalance_df.to_string(
        index=False
    )
)

imbalance_df.to_csv(
    os.path.join(
        BASE,
        "imbalance_results.csv"
    ),
    index=False
)


best_imbalance = imbalance_df.loc[
    imbalance_df["F1"].idxmax(),
    "Strategy"
]

print(
    f"\nHighest F1 in this comparison: "
    f"{best_imbalance}"
)


# ============================================================
# RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                random_state=42
            )
        )
    ]
)

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],
    "model__max_depth": [
        4,
        6,
        None
    ],
    "model__max_features": [
        "sqrt",
        "log2"
    ]
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(
    X_train,
    y_train
)

print("Best parameters:")
print(grid.best_params_)

best_rf = grid.best_estimator_

print(
    f"OOB score: "
    f"{best_rf.named_steps['model'].oob_score_:.4f}"
)


# ============================================================
# REGRESSION — PREDICT FARE
# ============================================================

print("\n" + "=" * 70)
print("FARE REGRESSION")
print("=" * 70)

regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]


Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


reg_numeric = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical = [
    "sex",
    "embarked"
]


reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            reg_numeric
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            ),
            reg_categorical
        )
    ]
)


regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    Xr_train,
    yr_train
)

fare_predictions = regression_pipeline.predict(
    Xr_test
)

mae = mean_absolute_error(
    yr_test,
    fare_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        yr_test,
        fare_predictions
    )
)

r2 = r2_score(
    yr_test,
    fare_predictions
)

n = len(yr_test)

p = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

adjusted_r2 = (
    1
    - (1 - r2)
    * (n - 1)
    / (n - p - 1)
)

print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R2         : {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")


# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = yr_test - fare_predictions

plt.figure(figsize=(8, 6))

plt.scatter(
    fare_predictions,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "12_regression_residuals.png"
    )
)

plt.close()


# Simple heteroscedasticity observation
residual_spread = pd.Series(
    residuals
).abs().corr(
    pd.Series(fare_predictions)
)

if abs(residual_spread) > 0.30:
    hetero_statement = (
        "The residual spread shows evidence of "
        "possible heteroscedasticity."
    )
else:
    hetero_statement = (
        "The residual spread does not show strong "
        "evidence of heteroscedasticity."
    )

print("\nHeteroscedasticity:")
print(hetero_statement)


# ============================================================
# SAVE COMPLETE BEST PIPELINE
# ============================================================

# Select Random Forest grid-search pipeline as final model.
full_pipeline = best_rf

model_path = os.path.join(
    BASE,
    "best_pipeline.joblib"
)

joblib.dump(
    full_pipeline,
    model_path
)

print(
    f"\nSaved complete pipeline to: "
    f"{model_path}"
)


# ============================================================
# RELOAD AND TEST RAW INPUT
# ============================================================

loaded_pipeline = joblib.load(
    model_path
)

sample_raw = X_test.iloc[[0]]

prediction = loaded_pipeline.predict(
    sample_raw
)

print("\nReload test:")
print("Raw input:")
print(sample_raw)

print("Prediction:", prediction)


# ============================================================
# FINAL SUMMARY
# ============================================================

best_classifier_row = classification_results.loc[
    classification_results["F1"].idxmax()
]

print("\n" + "=" * 70)
print("FINAL MODEL SUMMARY")
print("=" * 70)

print(
    classification_results.to_string(
        index=False
    )
)

print("\nRegression:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Adjusted R2: {adjusted_r2:.4f}")

print(
    f"\nClassifier with highest F1 in this evaluation: "
    f"{best_classifier_row['Model']}"
)

print(
    f"F1: {best_classifier_row['F1']:.4f}"
)

print(
    "\nThe classifier comparison uses accuracy, precision, "
    "recall, F1 and AUC. The regression metrics are reported "
    "separately because classification and regression metrics "
    "are different quantities."
)

print("\nMODEL PIPELINE COMPLETE")