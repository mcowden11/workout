import math
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Data-Driven Workout & Heart Rate Planner",
    layout="centered"
)

st.title("🏋️ Data-Driven Workout & Heart Rate Planner")
st.write(
    "This dashboard uses the Final Project CSV data to recommend a workout type and "
    "session duration based on your profile and selected goal."
)

# ---------- DATA LOADER ----------

@st.cache_data
def load_data(csv_path: str):
    """
    Load the final project dataset and compute helper columns.
    """
    df = pd.read_csv(csv_path)

    # Compute BMI if needed
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2)

    # BMI category
    def bmi_category(bmi):
        if pd.isna(bmi):
            return "Unknown"
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"

    df["BMI_cat"] = df["BMI"].apply(bmi_category)

    # Strength volume proxy if Sets/Reps exist
    if "Sets" in df.columns and "Reps" in df.columns:
        df["Volume"] = df["Sets"].fillna(0) * df["Reps"].fillna(0)
    else:
        df["Volume"] = 0

    # Aggregate per Workout_Type
    stats = (
        df.groupby("Workout_Type", dropna=True)
          .agg(
              avg_calories=("Calories_Burned", "mean"),
              avg_duration_hours=("Session_Duration (hours)", "mean"),
              avg_bpm=("Avg_BPM", "mean"),
              avg_volume=("Volume", "mean")
          )
          .reset_index()
          .fillna(0)
    )

    return df, stats


# 👉 EXACT filename you showed me
DATA_CSV = "Final_Project_Data.csv"

try:
    df, workout_stats = load_data(DATA_CSV)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ---------- HELPER FUNCTIONS ----------

def calculate_bmi(weight_kg: float, height_m: float) -> float:
    if height_m <= 0:
        return float("nan")
    return weight_kg / (height_m ** 2)


def bmi_category_simple(bmi: float) -> str:
    if math.isnan(bmi):
        return "Unknown"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def estimate_max_hr(age: int) -> int:
    return 220 - age


def pick_best_workout_for_goal(goal_key: str, stats_df: pd.DataFrame):
    """
    Choose the 'best' workout type for a given goal using the dataset.
    """
    df_w = stats_df.copy()

    max_cal = df_w["avg_calories"].max() or 1
    max_dur = df_w["avg_duration_hours"].max() or 1
    max_vol = df_w["avg_volume"].max() or 1
    max_bpm = df_w["avg_bpm"].max() or 1

    if goal_key == "burn_calories":
        best_idx = df_w["avg_calories"].idxmax()

    elif goal_key == "build_muscle":
        if (df_w["avg_volume"] > 0).any():
            best_idx = df_w["avg_volume"].idxmax()
        else:
            df_w["score"] = (
                (df_w["avg_calories"] / max_cal) * 0.5 +
                (df_w["avg_duration_hours"] / max_dur) * 0.5
            )
            best_idx = df_w["score"].idxmax()

    elif goal_key == "endurance":
        df_w["endurance_score"] = (
            (df_w["avg_duration_hours"] / max_dur) * 0.6 +
            (df_w["avg_bpm"] / max_bpm) * 0.4
        )
        best_idx = df_w["endurance_score"].idxmax()

    elif goal_key == "balanced":
        df_w["norm_cal"] = df_w["avg_calories"] / max_cal
        df_w["norm_dur"] = df_w["avg_duration_hours"] / max_dur
        df_w["norm_vol"] = df_w["avg_volume"] / max_vol
        target = 0.6
        df_w["balanced_score"] = (
            (df_w["norm_cal"] - target).abs() +
            (df_w["norm_dur"] - target).abs() +
            (df_w["norm_vol"] - target).abs()
        )
        best_idx = df_w["balanced_score"].idxmin()

    else:
        best_idx = df_w["avg_calories"].idxmax()

    return df_w.loc[best_idx]


def adjust_duration_for_bmi(base_duration_hours: float, bmi_cat: str) -> float:
    if bmi_cat == "Underweight":
        return base_duration_hours * 0.9
    elif bmi_cat in ("Overweight", "Obese"):
        return base_duration_hours * 1.1
    else:
        return base_duration_hours


# ---------- USER INPUTS ----------

st.subheader("👤 Your Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 15, 80, 25)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])

with col2:
    weight = st.slider("Weight (kg)", 35.0, 200.0, 70.0, 0.5)
    height = st.slider("Height (m)", 1.30, 2.20, 1.70, 0.01)

goal_label = st.selectbox(
    "Fitness Goal",
    [
        "Burn max calories / Lose weight",
        "Build muscle",
        "Improve endurance",
        "Balanced routine"
    ]
)

goal_map = {
    "Burn max calories / Lose weight": "burn_calories",
    "Build muscle": "build_muscle",
    "Improve endurance": "endurance",
    "Balanced routine": "balanced"
}

goal_key = goal_map[goal_label]

# ---------- CALCULATE USER METRICS ----------

user_bmi = calculate_bmi(weight, height)
user_bmi_cat = bmi_category_simple(user_bmi)
user_max_hr = estimate_max_hr(age)

st.markdown("---")
st.subheader("📊 Your Metrics")
st.write(f"**BMI:** {user_bmi:.1f} ({user_bmi_cat})")
st.write(f"**Estimated Max Heart Rate:** {user_max_hr} bpm")


# ---------- DATA-DRIVEN RECOMMENDATION ----------

st.markdown("---")
st.subheader("🤖 Data-Driven Workout Recommendation")

best = pick_best_workout_for_goal(goal_key, workout_stats)

if best is None:
    st.warning("No workout data found.")
else:

    workout_type = best["Workout_Type"]
    base_duration_h = best["avg_duration_hours"]
    avg_bpm_from_data = best["avg_bpm"]
    avg_cal_from_data = best["avg_calories"]

    adjusted_duration_h = adjust_duration_for_bmi(base_duration_h, user_bmi_cat)
    adjusted_duration_min = round(adjusted_duration_h * 60)

    # HR zone
    if avg_bpm_from_data > 0:
        low_hr = round(avg_bpm_from_data * 0.9)
        high_hr = round(avg_bpm_from_data * 1.1)
        avg_hr_display = f"{avg_bpm_from_data:.0f}"
        hr_source = "Based on average BPM from dataset."
    else:
        low_hr = round(user_max_hr * 0.6)
        high_hr = round(user_max_hr * 0.75)
        avg_hr_display = f"{(low_hr + high_hr)//2}"
        hr_source = "Based on % of your estimated max HR."

    colA, colB = st.columns(2)

    with colA:
        st.write(f"**Recommended Workout Type (from data):** `{workout_type}`")
        st.write(f"**Suggested Session Duration:** ~**{adjusted_duration_min} minutes**")
        st.write(f"**Expected Calories Burned:** ~**{avg_cal_from_data:.0f} kcal**")

    with colB:
        st.write("**Target Heart Rate Zone:**")
        st.write(f"- Avg Training HR: **{avg_hr_display} bpm**")
        st.write(f"- Zone: **{low_hr}–{high_hr} bpm**")
        st.caption(hr_source)

    st.markdown("---")
    st.write(
        "This recommendation is based on the **average performance** of each workout type "
        "in the dataset. Your selected goal chooses the optimization criteria, and your BMI "
        "category adjusts the session duration."
    )

st.caption("Educational use only — not medical advice.")
