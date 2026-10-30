import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ========== LOAD ==========
df = pd.read_csv("student_data.csv")
print("Shape:", df.shape)
print("Class balance:\n", df["result"].value_counts())

X = df.drop("result", axis=1)
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== MODELS ==========
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        eval_metric="logloss", random_state=42),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced"))
    ]),
}

# ========== CROSS-VALIDATION ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n=== 5-Fold Cross-Validation ===")
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_scores[name] = scores.mean()
    print(f"{name:25s}  {scores.mean()*100:.2f}%  (+/- {scores.std()*100:.2f})")

# ========== TRAIN ON FULL TRAINING SET ==========
print("\n=== Test Set Results ===")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pred, digits=3))

# ========== PICK BEST MODEL ==========
best_name = max(cv_scores, key=cv_scores.get)
best_model = models[best_name]
print(f"\n⭐ Best model by CV: {best_name}  ({cv_scores[best_name]*100:.2f}%)")

# ========== PLOT 1: Model Comparison ==========
plt.figure(figsize=(8, 4))
names = list(cv_scores.keys())
vals  = [cv_scores[n]*100 for n in names]
bars = plt.barh(names, vals, color="teal")
plt.xlabel("CV Accuracy (%)")
plt.title("Model Comparison (5-Fold CV)")
for bar, v in zip(bars, vals):
    plt.text(v - 3, bar.get_y() + bar.get_height()/2, f"{v:.2f}%",
             va="center", color="white", fontweight="bold")
plt.xlim(0, 100)
plt.tight_layout(); plt.savefig("model_comparison.png"); plt.show()

# ========== PLOT 2: Confusion Matrix (best model) ==========
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.show()

# ========== PLOT 3: ROC Curves ==========
plt.figure(figsize=(7, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
plt.plot([0,1], [0,1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curves"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("roc_curves.png"); plt.show()

# ========== PLOT 4: Feature Importance ==========
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
elif hasattr(best_model, "named_steps"):
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        imp = pd.Series(clf.coef_[0], index=X.columns).sort_values()
    else:
        imp = None
else:
    imp = None

if imp is not None:
    plt.figure(figsize=(7, 5))
    imp.plot(kind="barh", color="teal")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout(); plt.savefig("feature_importance.png"); plt.show()

# ========== SAVE ==========
joblib.dump(best_model, "student_model.pkl")
joblib.dump(best_name, "model_name.pkl")
print(f"\nSaved: student_model.pkl ({best_name})")