# 🎓 Early Warning System for Student Academic Performance

A supervised Machine Learning mini-project and interactive **Streamlit** web prototype designed to identify academically at-risk students early in the semester so educators can provide timely mentoring and academic support.

---

## 1. Problem Statement
In many academic institutions, students fall behind due to irregular class attendance, low internal assessment scores, insufficient study hours, or unresolved backlogs. Often, these warning signs go unnoticed until final semester results are published, leaving little time for corrective action. This project addresses that gap by predicting academic risk early using historical and ongoing student performance indicators.

---

## 2. Project Objectives
- **Dataset Preparation:** Generate and structure a clean student academic dataset incorporating key behavioral and performance indicators.
- **Data Preprocessing & EDA:** Handle missing values and duplicate records, normalize numerical features, and visualize patterns through Exploratory Data Analysis (EDA).
- **Multi-Model Benchmarking:** Implement and compare three classification algorithms—**Logistic Regression**, **Decision Tree**, and **Random Forest**.
- **Comprehensive Model Evaluation:** Evaluate all models using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Interactive Prototype:** Deploy the best-performing model in a user-friendly **Streamlit** web application for real-time academic risk prediction and actionable recommendations.

---

## 3. Project Structure

```text
student_early_warning/
├── create_dataset.py        # Generates raw synthetic student dataset (student_performance.csv)
├── train_model.py           # Performs preprocessing, EDA, model training, evaluation & plotting
├── app.py                   # Interactive Streamlit web prototype for live predictions
├── requirements.txt         # Required Python libraries
├── README.md                # Project documentation
├── student_performance.csv  # Generated dataset (created after running create_dataset.py)
├── eda_plot.png             # Generated EDA visualization (created after running train_model.py)
├── comparison_plot.png      # Generated model comparison chart (created after running train_model.py)
├── results.csv              # Saved evaluation metrics table (created after running train_model.py)
└── early_warning_model.pkl  # Saved best model & scaler bundle (created after running train_model.py)
```

---

## 4. Dataset Description
The dataset contains **300 student records** with **6 input features** and **1 binary target variable**:

| Column Name | Description | Range / Values |
| :--- | :--- | :--- |
| `Attendance_Percentage` | Student's class attendance percentage | `45% – 100%` |
| `Internal_Marks` | Marks scored in internal assessments | `0 – 30` |
| `Assignment_Score` | Marks obtained in coursework assignments | `0 – 20` |
| `Study_Hours_Per_Week` | Average self-study hours per week | `2 – 25 hours` |
| `Previous_SGPA` | Semester Grade Point Average from the previous term | `4.2 – 9.6` |
| `Backlogs` | Number of active uncleared subjects | `0, 1, 2, 3` |
| **`At_Risk` (Target)** | Academic risk status (`0` = Safe, `1` = At Risk) | `0` or `1` |

---

## 5. Methodology & Workflow

1. **Data Preprocessing:**
   - Removed duplicate student entries using `drop_duplicates()`.
   - Imputed missing values (`NaN`) in `Study_Hours_Per_Week` and `Previous_SGPA` using column medians.
   - Standardized feature scales using `StandardScaler` after an **80/20 stratified train-test split**.

2. **Exploratory Data Analysis (EDA):**
   - Analyzed class distribution between `Safe (0)` and `At Risk (1)` students.
   - Plotted relationships between `Attendance_Percentage` and `Internal_Marks` (`eda_plot.png`).

3. **Model Implementation & Evaluation:**
   - Trained three supervised classifiers:
     - **Logistic Regression** (linear baseline)
     - **Decision Tree Classifier** (`max_depth=5`)
     - **Random Forest Classifier** (`n_estimators=100`)
   - Evaluated each model on the test set using **Accuracy**, **Precision**, **Recall**, and **F1-Score**, saving results to `results.csv` and `comparison_plot.png`.

---

## 6. Key Experimental Observations
- **Tree-Based Superiority:** Ensemble and tree-based models (**Random Forest** and **Decision Tree**) outperform linear models because they effectively capture non-linear combinations of risk factors (such as low attendance combined with active backlogs).
- **Primary Risk Drivers:** `Attendance_Percentage`, `Internal_Marks`, and `Backlogs` contribute most strongly to identifying at-risk students.
- **Actionable Early Intervention:** Flagging high-risk students mid-semester enables faculty advisors to schedule remedial classes, mentoring, and structured study plans before final exams.

---

## 7. Installation & How to Run

### Prerequisites
- Python 3.8 or higher installed on your system.

### Step-by-Step Execution

1. **Navigate to the project directory:**
   ```bash
   cd student_early_warning
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset:**
   ```bash
   python create_dataset.py
   ```

4. **Run preprocessing, EDA, and model training:**
   ```bash
   python train_model.py
   ```

5. **Launch the Streamlit web prototype:**
   ```bash
   streamlit run app.py
   ```

---

## 8. Prototype Features
- **Live Risk Prediction Tab:** Adjust sliders and inputs for student metrics to view instant risk classification (`Safe` vs. `At Risk`), risk probability percentage, and personalized academic recommendations.
- **EDA Tab:** Inspect dataset statistics, sample rows, and distribution charts directly inside the web app.
- **Model Comparison Tab:** Review the side-by-side evaluation table (`Accuracy`, `Precision`, `Recall`, `F1-Score`) and performance bar chart.

---

## 9. Future Enhancements
- Integrate real institutional ERP/LMS student records for continuous semester-wide tracking.
- Add SHAP or feature-importance explanations for individual student predictions.
- Enable bulk CSV upload in the Streamlit app so faculty can evaluate an entire class at once.

## 9.Author

Name: Aditya Raj

Course: B.Tech in Computer Science & Engineering (AI & ML)

College: Ghani Khan Choudhury Institute of Engineering and Technology
