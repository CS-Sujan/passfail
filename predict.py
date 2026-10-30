import joblib
import pandas as pd

model  = joblib.load("student_model.pkl")

def predict_pass(study_hours, attendance, previous_marks,
                 assignments, sleep_hours, extracurricular):
    data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignments": assignments,
        "sleep_hours": sleep_hours,
        "extracurricular": extracurricular
    }])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    return ("PASS" if pred == 1 else "FAIL"), round(prob * 100, 2)

# Example student
result, confidence = predict_pass(
    study_hours=7,
    attendance=85,
    previous_marks=72,
    assignments=9,
    sleep_hours=7,
    extracurricular=1
)
print(f"Prediction: {result}  (Pass probability: {confidence}%)")