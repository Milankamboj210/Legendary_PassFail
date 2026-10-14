
import streamlit as st
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Legendary PassFail",
    page_icon="🎓",
    layout="centered"
)


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(
        MODEL_DIR / "model.keras"
    )
    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )
    threshold = float(
        joblib.load(MODEL_DIR / "threshold.pkl")
    )
    return model, preprocessor, threshold


model, preprocessor, threshold = load_artifacts()

st.title("🎓 Legendary PassFail")
st.subheader("Student Pass/Fail Prediction")

st.write(
    "Estimate a student's likelihood of passing using "
    "their age, sex, weekly study time, and absences."
)

st.info(
    "This model uses four features and does not use "
    "previous-period grades."
)

with st.form("student_form"):
    sex = st.selectbox(
        "Sex",
        options=["F", "M"]
    )

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=22,
        value=17,
        step=1
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours"
        }[x]
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        max_value=75,
        value=0,
        step=1
    )

    submitted = st.form_submit_button("Predict Result")

if submitted:
    student_data = pd.DataFrame([{
        "sex": sex,
        "age": age,
        "studytime": studytime,
        "absences": absences
    }])

    try:
        processed_data = preprocessor.transform(student_data)

        probability = float(
            model.predict(processed_data, verbose=0)[0][0]
        )

        if probability >= threshold:
            st.success("Prediction: PASS 🎉")
        else:
            st.error("Prediction: FAIL")

        st.metric(
            "Estimated Probability of Passing",
            f"{probability:.2%}"
        )

        st.caption(
            f"Classification threshold: {threshold:.2f}"
        )

        st.caption(
            "This is an experimental model estimate, "
            "not a guarantee of the student's final result."
        )

    except Exception:
        st.error(
            "Prediction failed. Please check that the "
            "model and preprocessing files are compatible."
        )
        st.exception(Exception("See the deployment logs for details."))
