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

    return model, preprocessor


model, preprocessor = load_artifacts()


st.title("🎓 Legendary PassFail")
st.subheader("Student Pass/Fail Prediction")

st.write(
    "Enter the student's details and academic marks "
    "to predict whether they are likely to pass or fail."
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
        "absences": absences,
    }])

    st.write("DEBUG: Form submitted successfully!")

    try:
        processed_data = preprocessor.transform(student_data)

        st.write("DEBUG: Preprocessing successful")
        st.write("Processed input shape:", processed_data.shape)

        prediction_probability = float(
            model.predict(processed_data, verbose=0)[0][0]
        )

        if prediction_probability >= 0.5:
            st.success("Prediction: PASS 🎉")
        else:
            st.error("Prediction: FAIL")

        st.metric(
            "Estimated Probability of Passing",
            f"{prediction_probability:.2%}"
        )

        st.caption(
            "This is a model estimate, not a guarantee "
            "of the student's final result."
        )

    except Exception as error:
        st.error("Prediction failed. Here are the error details:")
        st.exception(error)

