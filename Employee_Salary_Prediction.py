import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered",
)

st.title("💼 Employee Salary Predictor")
st.write(
    "Predict **Annual Salary (LPA)** using the trained Gradient Boosting model "
    "(`Employee_Salary_Prediction_Model.pkl`) from `Project2.ipynb`."
)


@st.cache_resource
def load_model():
    return joblib.load("Employee_Salary_Prediction_Model.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Could not find `Employee_Salary_Prediction_Model.pkl`. "
        "Make sure `app.py` is in the same folder as your saved model."
    )
    st.stop()

# Exact mappings used in Project2.ipynb
GENDER_MAP = {"Male": 0, "Female": 1}
EDUCATION_MAP = {"Master": 0, "Bachelor": 1, "Diploma": 2, "PhD": 3}
DEPARTMENT_MAP = {
    "Operations": 0,
    "Marketing": 1,
    "Sales": 2,
    "IT": 3,
    "HR": 4,
}
JOB_LEVEL_MAP = {
    "Junior": 0,
    "Lead": 1,
    "Mid": 2,
    "Senior": 3,
    "Manager": 4,
}
REMOTE_MAP = {"Yes": 0, "No": 1}
CITY_MAP = {
    "Chennai": 0,
    "Delhi": 1,
    "Hyderabad": 2,
    "Mumbai": 3,
}

with st.form("salary_form"):
    st.subheader("Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=14, max_value=90, value=28, step=1)
        gender_label = st.selectbox("Gender", options=list(GENDER_MAP.keys()), index=0)
        education_label = st.selectbox(
            "Education", options=list(EDUCATION_MAP.keys()), index=0
        )
        experience_years = st.number_input(
            "Experience Years", min_value=0.0, max_value=45.0, value=6.0, step=0.5
        )
        department_label = st.selectbox(
            "Department", options=list(DEPARTMENT_MAP.keys()), index=0
        )
        job_level_label = st.selectbox(
            "Job Level", options=list(JOB_LEVEL_MAP.keys()), index=0
        )
        performance_rating = st.slider(
            "Performance Rating", min_value=1, max_value=5, value=1, step=1
        )

    with col2:
        certifications = st.number_input(
            "Certifications", min_value=0, max_value=20, value=4, step=1
        )
        overtime_hours = st.number_input(
            "Overtime Hours", min_value=0, max_value=100, value=15, step=1
        )
        remote_label = st.selectbox(
            "Remote Work", options=list(REMOTE_MAP.keys()), index=0
        )
        city_label = st.selectbox("City", options=list(CITY_MAP.keys()), index=0)
        company_tenure = st.number_input(
            "Company Tenure (Years)", min_value=0, max_value=30, value=6, step=1
        )
        projects_completed = st.number_input(
            "Projects Completed", min_value=0, max_value=100, value=5, step=1
        )
        skill_score = st.number_input(
            "Skill Score", min_value=0.0, max_value=100.0, value=97.0, step=1.0
        )

    submitted = st.form_submit_button("Predict Salary", use_container_width=True)

if submitted:
    # Exact feature order expected by X in Project2.ipynb
    input_df = pd.DataFrame(
        {
            "Age": [int(age)],
            "Gender": [GENDER_MAP[gender_label]],
            "Education": [EDUCATION_MAP[education_label]],
            "Experience_Years": [float(experience_years)],
            "Department": [DEPARTMENT_MAP[department_label]],
            "Job_Level": [JOB_LEVEL_MAP[job_level_label]],
            "Performance_Rating": [int(performance_rating)],
            "Certifications": [int(certifications)],
            "Overtime_Hours": [int(overtime_hours)],
            "Remote_Work": [REMOTE_MAP[remote_label]],
            "City": [CITY_MAP[city_label]],
            "Company_Tenure": [int(company_tenure)],
            "Projects_Completed": [int(projects_completed)],
            "Skill_Score": [float(skill_score)],
        },
        index=[0],
    )

    prediction = float(model.predict(input_df)[0])

    st.divider()
    st.metric(label="Predicted Annual Salary", value=f"{prediction:.2f} LPA")

    with st.expander("View encoded input features sent to model"):
        st.dataframe(input_df, use_container_width=True)