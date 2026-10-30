import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load("student_model.pkl")
    model_name = joblib.load("model_name.pkl")
    return model, model_name


try:
    model, model_name = load_model()

except FileNotFoundError:
    st.error(
        "Model files not found. "
        "Please run train_model.py first."
    )
    st.stop()


# =========================================================
# FEATURE COLUMNS
# =========================================================

FEATURE_COLUMNS = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignments",
    "sleep_hours",
    "extracurricular",
    "family_income",
    "internet_access",
    "part_time_job",
    "tuition",
    "health_score",
    "motivation"
]


# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("🎓 STUDENT PERFORMANCE")

st.caption(
    f"Model: **{model_name}** | "
    "Trained on 3000 students with 12 features"
)


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Prediction",
        "📊 Batch Prediction",
        "📈 Model Info"
    ]
)


# =========================================================
# TAB 1 - SINGLE PREDICTION
# =========================================================

with tab1:

    st.subheader("Enter Student Details")

    c1, c2, c3 = st.columns(3)

    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------

    with c1:

        study_hours = st.slider(
            "Study hours/day",
            min_value=0,
            max_value=12,
            value=6
        )

        attendance = st.slider(
            "Attendance (%)",
            min_value=40,
            max_value=100,
            value=80
        )

        previous_marks = st.slider(
            "Previous marks",
            min_value=0,
            max_value=100,
            value=60
        )

        assignments = st.slider(
            "Assignments",
            min_value=0,
            max_value=10,
            value=7
        )

    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------

    with c2:

        sleep_hours = st.slider(
            "Sleep hours",
            min_value=3,
            max_value=10,
            value=7
        )

        health_score = st.slider(
            "Health (1-10)",
            min_value=1,
            max_value=10,
            value=7
        )

        motivation = st.slider(
            "Motivation (1-10)",
            min_value=1,
            max_value=10,
            value=7
        )

        family_income = st.selectbox(
            "Family income",
            [0, 1, 2],
            format_func=lambda x: [
                "Low",
                "Medium",
                "High"
            ][x]
        )

    # -----------------------------------------------------
    # COLUMN 3
    # -----------------------------------------------------

    with c3:

        extracurricular = st.selectbox(
            "Extracurricular",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        internet_access = st.selectbox(
            "Internet at home",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        part_time_job = st.selectbox(
            "Part-time job",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

        tuition = st.selectbox(
            "Extra tuition",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Predict",
        use_container_width=True,
        type="primary"
    ):

        student_data = pd.DataFrame(
            [
                {
                    "study_hours": study_hours,
                    "attendance": attendance,
                    "previous_marks": previous_marks,
                    "assignments": assignments,
                    "sleep_hours": sleep_hours,
                    "extracurricular": extracurricular,
                    "family_income": family_income,
                    "internet_access": internet_access,
                    "part_time_job": part_time_job,
                    "tuition": tuition,
                    "health_score": health_score,
                    "motivation": motivation
                }
            ]
        )

        try:

            prediction = model.predict(
                student_data
            )[0]

            probability = (
                model.predict_proba(
                    student_data
                )[0][1] * 100
            )

            result_col, probability_col = st.columns(
                [1, 2]
            )

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------

            with result_col:

                if prediction == 1:

                    st.success(
                        f"### ✅ PASS\n"
                        f"**{probability:.2f}% confidence**"
                    )

                else:

                    st.error(
                        f"### ❌ FAIL\n"
                        f"**{100 - probability:.2f}% confidence**"
                    )

            # -------------------------------------------------
            # PROBABILITY
            # -------------------------------------------------

            with probability_col:

                st.metric(
                    "Pass Probability",
                    f"{probability:.2f}%"
                )

                st.progress(
                    int(probability)
                )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError
        ) as error:

            st.error(
                f"Prediction error: {error}"
            )


# =========================================================
# TAB 2 - BATCH PREDICTION
# =========================================================

with tab2:

    st.subheader(
        "📊 Upload CSV to Predict Multiple Students"
    )

    st.write(
        "Upload a CSV file containing the same "
        "12 feature columns used by the trained model."
    )

    # -----------------------------------------------------
    # REQUIRED COLUMNS
    # -----------------------------------------------------

    with st.expander("📋 View Required CSV Columns"):

        st.write(
            "Your CSV must contain exactly these feature "
            "columns:"
        )

        st.code(
            "\n".join(FEATURE_COLUMNS)
        )

    # -----------------------------------------------------
    # CSV TEMPLATE
    # -----------------------------------------------------

    st.subheader("📥 CSV Template")

    template_data = pd.DataFrame(
        [
            {
                "study_hours": 6,
                "attendance": 80,
                "previous_marks": 60,
                "assignments": 7,
                "sleep_hours": 7,
                "extracurricular": 1,
                "family_income": 1,
                "internet_access": 1,
                "part_time_job": 0,
                "tuition": 1,
                "health_score": 7,
                "motivation": 7
            }
        ]
    )

    template_csv = template_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_csv,
        file_name="student_prediction_template.csv",
        mime="text/csv"
    )

    st.divider()

    # -----------------------------------------------------
    # UPLOAD FILE
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(
                uploaded_file
            )

            # ---------------------------------------------
            # CHECK REQUIRED COLUMNS
            # ---------------------------------------------

            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in data.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

                st.warning(
                    "Please download the CSV template "
                    "and use the same column names."
                )

                st.write(
                    "Columns found in uploaded file:"
                )

                st.code(
                    ", ".join(data.columns.tolist())
                )

            else:

                # -----------------------------------------
                # SELECT MODEL FEATURES
                # -----------------------------------------

                input_data = data[
                    FEATURE_COLUMNS
                ]

                # -----------------------------------------
                # PREDICTIONS
                # -----------------------------------------

                predictions = model.predict(
                    input_data
                )

                probabilities = (
                    model.predict_proba(
                        input_data
                    )[:, 1]
                )

                # -----------------------------------------
                # ADD RESULTS
                # -----------------------------------------

                data["prediction"] = np.where(
                    predictions == 1,
                    "PASS",
                    "FAIL"
                )

                data["pass_probability_%"] = (
                    probabilities * 100
                ).round(2)

                # -----------------------------------------
                # SUCCESS
                # -----------------------------------------

                st.success(
                    f"Successfully predicted "
                    f"{len(data)} students."
                )

                # -----------------------------------------
                # SUMMARY
                # -----------------------------------------

                pass_count = int(
                    (predictions == 1).sum()
                )

                fail_count = int(
                    (predictions == 0).sum()
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Students",
                        len(data)
                    )

                with col2:

                    st.metric(
                        "Predicted PASS",
                        pass_count
                    )

                with col3:

                    st.metric(
                        "Predicted FAIL",
                        fail_count
                    )

                # -----------------------------------------
                # DISPLAY RESULTS
                # -----------------------------------------

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    data,
                    use_container_width=True
                )

                # -----------------------------------------
                # HISTOGRAM
                # -----------------------------------------

                fig = px.histogram(
                    data,
                    x="pass_probability_%",
                    color="prediction",
                    title="Distribution of Pass Probabilities",
                    labels={
                        "pass_probability_%":
                            "Pass Probability (%)",
                        "prediction":
                            "Prediction"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # -----------------------------------------
                # DOWNLOAD RESULTS
                # -----------------------------------------

                result_csv = data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=result_csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except pd.errors.EmptyDataError:

            st.error(
                "The uploaded CSV file is empty."
            )

        except pd.errors.ParserError as error:

            st.error(
                f"Could not read the CSV file: {error}"
            )

        except (
            ValueError,
            KeyError,
            TypeError,
            AttributeError
        ) as error:

            st.error(
                f"Prediction error: {error}"
            )


# =========================================================
# TAB 3 - MODEL INFORMATION
# =========================================================

with tab3:

    st.subheader(
        "📈 Model Comparison & Evaluation"
    )

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.info(
        f"Best Model: **{model_name}**"
    )

    st.write(
        "The following charts were generated during "
        "model training."
    )

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    try:

        st.image(
            "model_comparison.png",
            caption="5-Fold Cross-Validation Accuracy",
            use_container_width=True
        )

    except FileNotFoundError:

        st.warning(
            "model_comparison.png not found. "
            "Run train_model.py first."
        )

    # -----------------------------------------------------
    # ROC CURVES
    # -----------------------------------------------------

    try:

        st.image(
            "roc_curves.png",
            caption="ROC Curves",
            use_container_width=True
        )

    except FileNotFoundError:

        st.warning(
            "roc_curves.png not found. "
            "Run train_model.py first."
        )

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    try:

        st.image(
            "confusion_matrix.png",
            caption="Confusion Matrix",
            use_container_width=True
        )

    except FileNotFoundError:

        st.warning(
            "confusion_matrix.png not found. "
            "Run train_model.py first."
        )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    try:

        st.image(
            "feature_importance.png",
            caption="Feature Importance",
            use_container_width=True
        )

    except FileNotFoundError:

        st.warning(
            "feature_importance.png not found. "
            "Run train_model.py first."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎓 Student Pass/Fail Prediction System | "
    "Machine Learning Project"
)