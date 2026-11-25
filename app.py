import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Early Warning System for Student Academic Performance",
    layout="wide"
)

# Hide default Streamlit deploy button and menu
st.markdown("""
<style>
    .stDeployButton {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load trained model and scaler
with open("early_warning_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
scaler = bundle["scaler"]
best_model_name = bundle["best_model_name"]

df_preview = pd.read_csv("student_performance.csv")
results_df = pd.read_csv("results.csv")
best_f1 = results_df["F1-Score (%)"].max()

# Header
st.title("Early Warning System for Student Academic Performance")
st.caption(f"Model in Use: {best_model_name}")

# Key metrics overview
m1, m2, m3, m4 = st.columns(4)
m1.metric("Selected Algorithm", best_model_name)
m2.metric("Best F1-Score", f"{best_f1:.1f}%")
m3.metric("Total Student Records", len(df_preview))
m4.metric("Input Features", 6)

st.write("")

# Project Overview
with st.container(border=True):
    st.subheader("Project Overview")
    st.write(
        "This application uses supervised Machine Learning to evaluate a student's ongoing semester "
        "records—attendance, internal exam marks, assignment scores, weekly study hours, previous SGPA, "
        "and active backlogs—to predict whether the student is academically Safe or At Risk."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Purpose**")
        st.markdown(
            "- Identify struggling students before final semester exams.\n"
            "- Prevent low internal performance from turning into semester backlogs.\n"
            "- Help faculty prioritize mentoring and extra academic support."
        )
    with col_b:
        st.markdown("**How It Works**")
        st.markdown(
            "1. Enter the student's current academic details in the prediction form.\n"
            "2. Input features are scaled and evaluated by the trained classifier.\n"
            "3. The system outputs the risk status, probability, and suggested actions."
        )

st.write("")

# Main Tabs
tab1, tab2, tab3 = st.tabs([
    "Risk Prediction",
    "Dataset and EDA",
    "Model Comparison"
])

# Tab 1: Live Prediction
with tab1:
    st.subheader("Student Academic Details")
    st.write("Adjust the values below to check a student's academic risk status.")

    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            attendance = st.slider("Attendance Percentage (%)", 40, 100, 76)
            internal_marks = st.slider("Internal Marks (out of 30)", 0, 30, 17)
            assignment_score = st.slider("Assignment Score (out of 20)", 0, 20, 14)

        with col2:
            study_hours = st.slider("Study Hours Per Week", 1, 30, 9)
            previous_sgpa = st.number_input(
                "Previous Semester SGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.1,
                step=0.1
            )
            backlogs = st.number_input(
                "Active Backlogs",
                min_value=0,
                value=0,
                step=1
            )

        predict_clicked = st.button("Predict Academic Status", type="primary")

    if predict_clicked:
        input_df = pd.DataFrame([{
            "Attendance_Percentage": attendance,
            "Internal_Marks": internal_marks,
            "Assignment_Score": assignment_score,
            "Study_Hours_Per_Week": study_hours,
            "Previous_SGPA": previous_sgpa,
            "Backlogs": backlogs
        }])

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] * 100

        st.divider()
        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Status", "At Risk" if prediction == 1 else "Safe")
        r2.metric("Risk Probability", f"{prob:.1f}%")
        r3.metric("Attendance Status", "Below 75%" if attendance < 75 else "Satisfactory")

        if prediction == 1:
            st.error(f"Early Warning: Student is At Risk (Risk Probability: {prob:.1f}%)")
            st.markdown("**Recommended Actions:**")
            if attendance < 75:
                st.write(f"- Increase class attendance from {attendance}% to at least 75%.")
            if internal_marks < 15:
                st.write(f"- Arrange remedial classes or mentoring to improve internal marks ({internal_marks}/30).")
            if backlogs > 0:
                st.write(f"- Create a study schedule to clear {backlogs} active backlog(s).")
            if study_hours < 7:
                st.write(f"- Increase weekly self-study hours from {study_hours} to at least 8-10 hours.")
            if previous_sgpa < 6.0:
                st.write(f"- Provide one-on-one academic guidance to strengthen core subjects (Previous SGPA: {previous_sgpa}).")
        else:
            st.success(f"Safe: Student Performance is Satisfactory (Risk Probability: {prob:.1f}%)")
            st.info("The student's current academic record is steady. Maintain regular attendance and study habits.")

# Tab 2: Dataset & EDA
with tab2:
    st.subheader("Dataset Summary and Exploratory Data Analysis")
    st.write(f"Dataset size: {df_preview.shape[0]} rows and {df_preview.shape[1]} columns.")
    st.image("eda_plot.png", caption="Class Distribution and Attendance vs Internal Marks", use_container_width=True)
    with st.expander("View Sample Dataset Rows"):
        st.dataframe(df_preview.head(15), use_container_width=True, hide_index=True)

# Tab 3: Model Comparison
with tab3:
    st.subheader("Comparison of Machine Learning Algorithms")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.image("comparison_plot.png", caption="Accuracy, Precision, Recall, and F1-Score Comparison", use_container_width=True)

    st.markdown("**Key Observations**")
    st.markdown(
        "- **Tree-Based Models:** Random Forest and Decision Tree achieve higher F1-Scores by capturing combined conditions such as low attendance alongside active backlogs.\n"
        "- **Important Features:** Attendance Percentage, Internal Marks, and Active Backlogs have the strongest impact on academic risk.\n"
        "- **Practical Use:** Evaluating students mid-semester allows faculty to step in before final exams."
    )

st.divider()
st.caption("Developed by Aditya Raj | B.Tech CSE (AI & ML), GKCIET")