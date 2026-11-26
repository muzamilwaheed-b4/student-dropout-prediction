"""
Student Dropout & Academic Success Prediction
Training Script - Lab 07, 08, 10, 11, 12 concepts applied
Dataset: UCI "Predict Students' Dropout and Academic Success"
         Download from: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
         Place as: data/data.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os

# ─────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

print("=" * 60)
print("  Student Dropout Prediction - Training Script")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv("data/data.csv", sep=";")
df.columns = df.columns.str.strip()
print(f"    Shape: {df.shape}")
print(f"    Target distribution:\n{df['Target'].value_counts()}")

# ─────────────────────────────────────────────
# 2. Preprocessing (Lab 07 + Lab 08)
# ─────────────────────────────────────────────
print("\n[2] Preprocessing...")

# 2a. Missing values check
missing = df.isnull().sum()
print(f"    Missing values: {missing.sum()}")
df.fillna(df.median(numeric_only=True), inplace=True)

# 2b. Duplicates
before = len(df)
df = df.drop_duplicates()
print(f"    Duplicates removed: {before - len(df)}")

# 2c. Outlier capping with IQR (Lab 08)
def cap_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

grade_cols = ["Admission grade", "Previous qualification (grade)",
              "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]
for col in grade_cols:
    if col in df.columns:
        df[col] = cap_iqr(df[col])

print("    Outlier capping done (IQR method).")

# 2d. Encode Target (3-class)
target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["Target"] = df["Target"].map(target_map)
print(f"    Target encoded: 0=Dropout, 1=Enrolled, 2=Graduate")

# ─────────────────────────────────────────────
# 3. EDA Plots (Lab 04 style)
# ─────────────────────────────────────────────
print("\n[3] Generating EDA plots...")

label_names = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
colors = ["#e74c3c", "#3498db", "#2ecc71"]

# Plot 1: Target distribution
fig, ax = plt.subplots(figsize=(7, 4))
counts = df["Target"].value_counts().sort_index()
bars = ax.bar([label_names[i] for i in counts.index], counts.values, color=colors)
ax.set_title("Student Outcome Distribution", fontsize=14, fontweight='bold')
ax.set_ylabel("Count")
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("static/plots/target_distribution.png", dpi=100)
plt.close()

# Plot 2: 2nd Sem Approved Units vs Target (Boxplot)
if "Curricular units 2nd sem (approved)" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 5))
    data_by_target = [df[df["Target"] == i]["Curricular units 2nd sem (approved)"].dropna()
                      for i in [0, 1, 2]]
    bp = ax.boxplot(data_by_target, labels=["Dropout", "Enrolled", "Graduate"],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("2nd Semester Approved Units by Outcome", fontsize=13, fontweight='bold')
    ax.set_ylabel("Approved Units")
    plt.tight_layout()
    plt.savefig("static/plots/approved_units_boxplot.png", dpi=100)
    plt.close()

# Plot 3: Tuition fees vs Target (Stacked bar)
if "Tuition fees up to date" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 4))
    cross = pd.crosstab(df["Tuition fees up to date"], df["Target"])
    cross.index = ["Not Up-to-Date", "Up-to-Date"]
    cross.columns = ["Dropout", "Enrolled", "Graduate"]
    cross.plot(kind='bar', stacked=True, ax=ax, color=colors, alpha=0.85)
    ax.set_title("Tuition Fees Status vs Student Outcome", fontsize=13, fontweight='bold')
    ax.set_xlabel("Tuition Fees Status")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig("static/plots/tuition_fees_outcome.png", dpi=100)
    plt.close()

# Plot 4: Correlation heatmap (top numeric features)
numeric_df = df.select_dtypes(include=[np.number])
top_feats = numeric_df.corr()["Target"].abs().sort_values(ascending=False).head(11).index.tolist()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_df[top_feats].corr(), annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap (Top Features)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("static/plots/correlation_heatmap.png", dpi=100)
plt.close()

print("    Plots saved to static/plots/")

# ─────────────────────────────────────────────
# 4. K-Means Clustering (Lab 11)
# ─────────────────────────────────────────────
print("\n[4] K-Means Clustering (at-risk groups)...")

cluster_features = []
for col in ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]:
    if col in df.columns:
        cluster_features.append(col)

if len(cluster_features) >= 2:
    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(df[cluster_features].fillna(0))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_km)

    cluster_dropout = df.groupby("cluster")["Target"].apply(
        lambda x: (x == 0).mean()
    ).sort_values(ascending=False)
    print(f"    Cluster dropout rates:\n{cluster_dropout.round(3)}")

    # Map: highest dropout cluster = "High Risk"
    risk_map = {}
    sorted_clusters = cluster_dropout.index.tolist()
    risk_labels_list = ["High Risk", "Medium Risk", "Low Risk"]
    for i, c in enumerate(sorted_clusters):
        risk_map[c] = risk_labels_list[i]
    df["risk_group"] = df["cluster"].map(risk_map)

    # Plot K-Means clusters
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_colors = {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#2ecc71"}
    for label, grp in df.groupby("risk_group"):
        ax.scatter(grp[cluster_features[0]], grp[cluster_features[1]],
                   c=cluster_colors.get(label, "gray"), label=label, alpha=0.5, s=20)
    ax.set_xlabel(cluster_features[0])
    ax.set_ylabel(cluster_features[1])
    ax.set_title("K-Means Risk Clusters", fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/plots/kmeans_clusters.png", dpi=100)
    plt.close()
    print("    K-Means done. Cluster plot saved.")

# ─────────────────────────────────────────────
# 5. Feature Selection & Train/Test Split
# ─────────────────────────────────────────────
print("\n[5] Feature selection and splitting...")

# Top 10 features for final app (based on domain knowledge + importance)
FEATURES = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Age at enrollment",
    "Admission grade",
    "Gender"
]
# Keep only features that exist in dataset
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"    Features used: {FEATURES}")

X = df[FEATURES].copy()
y = df["Target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 6. Models with & without SMOTE (Lab 10, 12)
# ─────────────────────────────────────────────
print("\n[6] Training 4 models WITH SMOTE...")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
print(f"    After SMOTE - Train size: {X_train_sm.shape[0]}")

models = {
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
}

results_with = {}
results_without = {}
trained_models = {}

for name, model in models.items():
    # With SMOTE
    model.fit(X_train_sm, y_train_sm)
    pred = model.predict(X_test_sc)
    f1   = f1_score(y_test, pred, average="macro")
    acc  = accuracy_score(y_test, pred)
    results_with[name] = {"accuracy": acc, "macro_f1": f1, "pred": pred}
    trained_models[name] = model
    print(f"    {name:25s} | Acc: {acc:.3f} | Macro F1: {f1:.3f}")

print("\n[6b] Training WITHOUT SMOTE (for comparison)...")
for name, _ in models.items():
    m2 = type(models[name])(**models[name].get_params())
    m2.fit(X_train_sc, y_train)
    pred2 = m2.predict(X_test_sc)
    f1_2  = f1_score(y_test, pred2, average="macro")
    acc2  = accuracy_score(y_test, pred2)
    results_without[name] = {"accuracy": acc2, "macro_f1": f1_2}
    print(f"    {name:25s} | Acc: {acc2:.3f} | Macro F1: {f1_2:.3f}")

# ─────────────────────────────────────────────
# 7. Best Model Selection
# ─────────────────────────────────────────────
print("\n[7] Best model selection...")
best_name = max(results_with, key=lambda k: results_with[k]["macro_f1"])
best_model = trained_models[best_name]
print(f"    Best model: {best_name}")
print(f"\n    Classification Report ({best_name}):")
print(classification_report(y_test, results_with[best_name]["pred"],
                             target_names=["Dropout", "Enrolled", "Graduate"]))

# ─────────────────────────────────────────────
# 8. Cross-Validation (5-fold)
# ─────────────────────────────────────────────
print("[8] Cross-validation (5-fold)...")
cv_scores = cross_val_score(best_model, X_train_sm, y_train_sm,
                             cv=5, scoring="f1_macro")
print(f"    CV Macro F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─────────────────────────────────────────────
# 9. Feature Importance Plot
# ─────────────────────────────────────────────
print("\n[9] Feature importance plot...")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
                   color=plt.cm.RdYlGn(feat_df["Importance"] / feat_df["Importance"].max()))
    ax.set_title(f"Feature Importance - {best_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("static/plots/feature_importance.png", dpi=100)
    plt.close()
    print("    Feature importance chart saved.")

# ─────────────────────────────────────────────
# 10. SMOTE vs No-SMOTE Comparison Plot
# ─────────────────────────────────────────────
print("\n[10] SMOTE comparison plot...")
model_names = list(results_with.keys())
f1_with    = [results_with[n]["macro_f1"] for n in model_names]
f1_without = [results_without[n]["macro_f1"] for n in model_names]

x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, f1_without, width, label="Without SMOTE", color="#3498db", alpha=0.8)
b2 = ax.bar(x + width/2, f1_with,    width, label="With SMOTE",    color="#e74c3c", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=10)
ax.set_ylabel("Macro F1 Score")
ax.set_title("SMOTE vs No-SMOTE: Macro F1 Comparison", fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1)
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("static/plots/smote_comparison.png", dpi=100)
plt.close()

# ─────────────────────────────────────────────
# 11. Confusion Matrix (Best Model)
# ─────────────────────────────────────────────
cm = confusion_matrix(y_test, results_with[best_name]["pred"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Dropout", "Enrolled", "Graduate"],
            yticklabels=["Dropout", "Enrolled", "Graduate"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png", dpi=100)
plt.close()

# ─────────────────────────────────────────────
# 12. Save Model + Metadata
# ─────────────────────────────────────────────
print("\n[12] Saving model...")
joblib.dump({
    "model":        best_model,
    "scaler":       scaler,
    "features":     FEATURES,
    "model_name":   best_name,
    "results_with": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                     for k, v in results_with.items()},
    "results_without": results_without,
    "cv_mean":      float(cv_scores.mean()),
    "cv_std":       float(cv_scores.std()),
}, "model/model_bundle.pkl")

print(f"    Model saved: model/model_bundle.pkl")
print("\nTraining complete! Run: python app.py")
print("=" * 60)
