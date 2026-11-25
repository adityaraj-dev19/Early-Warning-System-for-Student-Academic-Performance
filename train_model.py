import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load raw dataset
df = pd.read_csv("student_performance.csv")
print("=== 1. Raw Dataset Info ===")
print(f"Initial Shape: {df.shape}")
print("Missing values before cleaning:\n", df.isnull().sum())

# 2. Data preprocessing (remove duplicates and impute missing values)
df = df.drop_duplicates().copy()
df["Study_Hours_Per_Week"] = df["Study_Hours_Per_Week"].fillna(df["Study_Hours_Per_Week"].median())
df["Previous_SGPA"] = df["Previous_SGPA"].fillna(df["Previous_SGPA"].median())
print(f"\nCleaned Shape after removing duplicates and filling missing values: {df.shape}")

# 3. Exploratory Data Analysis (EDA) and visualization
print("\n=== 2. Exploratory Data Analysis (EDA) ===")
print(df.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Subplot 1: Class distribution (Safe vs. At Risk)
risk_counts = df["At_Risk"].value_counts().sort_index()
axes[0].bar(["Safe (0)", "At Risk (1)"], risk_counts.values, color=["#2E8B57", "#D9534F"])
axes[0].set_title("Student Risk Distribution")
axes[0].set_ylabel("Number of Students")
for i, v in enumerate(risk_counts.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")

# Subplot 2: Attendance vs. Internal Marks scatter plot
safe_df = df[df["At_Risk"] == 0]
risk_df = df[df["At_Risk"] == 1]
axes[1].scatter(safe_df["Attendance_Percentage"], safe_df["Internal_Marks"], color="#2E8B57", label="Safe", alpha=0.7)
axes[1].scatter(risk_df["Attendance_Percentage"], risk_df["Internal_Marks"], color="#D9534F", label="At Risk", alpha=0.7)
axes[1].set_title("Attendance vs Internal Marks")
axes[1].set_xlabel("Attendance (%)")
axes[1].set_ylabel("Internal Marks (out of 30)")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_plot.png", dpi=150)
plt.close()

# 4. Feature scaling and train-test split (80% train, 20% test)
X = df.drop("At_Risk", axis=1)
y = df["At_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train and evaluate three Machine Learning algorithms
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results_list = []
best_model = None
best_f1 = 0.0
best_model_name = ""

print("\n=== 3. Model Comparison & Evaluation Metrics ===")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100

    results_list.append({
        "Model": name,
        "Accuracy (%)": round(acc, 2),
        "Precision (%)": round(prec, 2),
        "Recall (%)": round(rec, 2),
        "F1-Score (%)": round(f1, 2)
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results_list)
results_df.to_csv("results.csv", index=False)
print(results_df.to_string(index=False))
print(f"\nBest Performing Model: {best_model_name} (F1-Score: {best_f1:.2f}%)")

# 6. Save model performance comparison plot
results_df.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]].plot(
    kind="bar", figsize=(9, 4.8), rot=0
)
plt.title("Performance Comparison of Three ML Algorithms")
plt.ylabel("Score (%)")
plt.ylim(0, 115)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150)
plt.close()

# 7. Save best model and feature scaler
model_bundle = {
    "model": best_model,
    "scaler": scaler,
    "best_model_name": best_model_name,
    "features": list(X.columns)
}
with open("early_warning_model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\nSaved: eda_plot.png, comparison_plot.png, results.csv, and early_warning_model.pkl")