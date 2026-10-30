import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

# --- 12 features ---
study_hours     = np.random.randint(0, 13, n)          # 0-12
attendance      = np.random.randint(40, 101, n)        # 40-100
previous_marks  = np.random.randint(20, 101, n)        # 20-100
assignments     = np.random.randint(0, 11, n)          # 0-10
sleep_hours     = np.random.randint(3, 10, n)          # 3-9
extracurricular = np.random.randint(0, 2, n)           # 0/1
family_income   = np.random.choice([0,1,2], n)         # low/mid/high
internet_access = np.random.randint(0, 2, n)
part_time_job   = np.random.randint(0, 2, n)
tuition         = np.random.randint(0, 2, n)
health_score    = np.random.randint(1, 11, n)          # 1-10
motivation      = np.random.randint(1, 11, n)          # 1-10

score = (
    study_hours     * 3.5 +
    (attendance - 40) * 0.35 +
    previous_marks  * 0.45 +
    assignments     * 2.5 +
    sleep_hours     * 1.2 +
    extracurricular * -1.5 +
    family_income   * 2.0 +
    internet_access * 3.0 +
    part_time_job   * -2.5 +
    tuition         * 3.5 +
    health_score    * 1.5 +
    motivation      * 2.5 +
    np.random.normal(0, 6, n)
)

# Adaptive threshold
threshold = np.percentile(score, 35)
result = (score >= threshold).astype(int)

df = pd.DataFrame({
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
    "motivation": motivation,
    "result": result
})

df.to_csv("student_data.csv", index=False)
print("Dataset created:", df.shape)
print(df["result"].value_counts())