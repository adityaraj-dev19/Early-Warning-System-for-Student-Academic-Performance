import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
num_students = 300

attendance = np.random.randint(45, 100, size=num_students)
internal_marks = np.random.randint(8, 30, size=num_students)
assignment_score = np.random.randint(5, 20, size=num_students)
study_hours = np.random.randint(2, 25, size=num_students).astype(float)
previous_sgpa = np.round(np.random.uniform(4.2, 9.6, size=num_students), 2)
backlogs = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.65, 0.20, 0.10, 0.05])

# Determine target label: At_Risk (1 = At Risk, 0 = Safe)
risk_score = (
    (attendance < 70).astype(int) * 2
    + (internal_marks < 15).astype(int) * 2
    + (previous_sgpa < 6.0).astype(int) * 1
    + (backlogs >= 1).astype(int) * 2
    + (study_hours < 7).astype(int) * 1
)
at_risk = (risk_score >= 3).astype(int)

df = pd.DataFrame({
    "Attendance_Percentage": attendance,
    "Internal_Marks": internal_marks,
    "Assignment_Score": assignment_score,
    "Study_Hours_Per_Week": study_hours,
    "Previous_SGPA": previous_sgpa,
    "Backlogs": backlogs,
    "At_Risk": at_risk
})

# Introduce a few missing values and duplicate rows to demonstrate data preprocessing
df.loc[[5, 19, 44, 102, 210], "Study_Hours_Per_Week"] = np.nan
df.loc[[12, 67, 155], "Previous_SGPA"] = np.nan
df = pd.concat([df, df.iloc[:3]], ignore_index=True)

df.to_csv("student_performance.csv", index=False)
print("Dataset created successfully: student_performance.csv")
print(f"Total rows (including duplicates and missing values): {len(df)}")