import math
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------

st.set_page_config(
    page_title="Data-Driven Workout",
    layout="centered"
) #this sets the browser tab title an the page layout

st.title("🏋️ Data-Driven Workout Planner") #creates a big page title
st.write(
    "This dashboard uses data to recommend a workout type and "
    "session duration based on your profile and selected goal."
) #creates a little description of what the website is

# ---------- DATA LOADER ----------

@st.cache_data #this prevents the program from rerunning if the same csv is called. #We used a fucntion to do this because every time a person uses the dashboard in streamlit, it reruns the script so this prevents the dataset from being called over and over again unnecessarily. 
def load_data(csv_path: str):
    """
    Load the final project dataset and compute helper columns.
    """
    df = pd.read_csv(csv_path) #reads our csv into a dataset called df

    # Compute BMI if needed
    if "BMI" not in df.columns:
        df["BMI"] = df["Weight (kg)"] / (df["Height (m)"] ** 2) #calculates if we don't have BMI in data set- we do so i may delete this

    # BMI category
    def bmi_category(bmi): #categorizes the BMI into four categories
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

    df["BMI_cat"] = df["BMI"].apply(bmi_category) #Adds a row for BMI category

    # Strength volume proxy if Sets/Reps exist
    if "Sets" in df.columns and "Reps" in df.columns:
        df["Volume"] = df["Sets"].fillna(0) * df["Reps"].fillna(0) #calculates volume based on sets and reps
    else:
        df["Volume"] = 0 #i we don't have those columns (we do) it goes to zero

    # Aggregate per Workout_Type
    #basically combines the columns that we need for each workout type 
    stats = (
        df.groupby("Workout_Type", dropna=True) #takes all of the rows and groups by workout type
          .agg(
              avg_calories=("Calories_Burned", "mean"),
              avg_duration_hours=("Session_Duration (hours)", "mean"),
              avg_bpm=("Avg_BPM", "mean"),
              avg_volume=("Volume", "mean")
          ) #calculates a summary statistic for each group, we used mean in this case, that way for each workout type we have average calories, duration, bpm, and volume so we are using the average performance of eahc workout type
          .reset_index()
          .fillna(0)
    )

    return df, stats


# Load in our actual data file to be used in the website
DATA_CSV = "Final_Project_Data.csv"

#this part we def don't know how o do but it tries to load and prepare the data but if something fails it immediately stops.Chat says we may need it for the streamlit cloud...
try:
    df, workout_stats = load_data(DATA_CSV)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ---------- HELPER FUNCTIONS ----------

def calculate_bmi(weight_kg: float, height_m: float) -> float: #calculates teh users BMI based on height and weight
    if height_m <= 0:
        return float("nan")
    return weight_kg / (height_m ** 2)


def bmi_category_simple(bmi: float) -> str: #this is different from the first BMI function as now we are categorizing the user's BMI
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


def estimate_max_hr(age: int) -> int: #estimates heart rate by taking 220-age (used to estimate max heart rate)
    return 220 - age


def pick_best_workout_for_goal(goal_key: str, stats_df: pd.DataFrame): #uses the stats dictionary we created earlier
    """
    Choose the 'best' workout type for a given goal using the dataset.
    """
    df_w = stats_df.copy() #uses a copy so we can edit without chnaging the original

    #finds the max in all columns (or 1 if column is all 0s to avoid division by 1)
    max_cal = df_w["avg_calories"].max() or 1
    max_dur = df_w["avg_duration_hours"].max() or 1
    max_vol = df_w["avg_volume"].max() or 1
    max_bpm = df_w["avg_bpm"].max() or 1

    if goal_key == "burn_calories":
        best_idx = df_w["avg_calories"].idxmax() #for a goal of burning calories choose workout with highest avg calories

    elif goal_key == "build_muscle":
        if (df_w["avg_volume"] > 0).any():
            best_idx = df_w["avg_volume"].idxmax() #for a goal of building muscle find the highest training valume (cacluated earlier)
        else: #if we don't have volumne though we can use a similer score that blends calories and duration-- IDK if we need this either bc we do have volume
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
        best_idx = df_w["endurance_score"].idxmax() #for endurance we look for longer sessions with higher BPM 

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
        best_idx = df_w["balanced_score"].idxmin() #for a balanced routine we normalized each metric to 0-1 and picked the workouts whose trio of metirc was closest to the target of 0.6

    else:
        best_idx = df_w["avg_calories"].idxmax()

    return df_w.loc[best_idx] #just in case we don't have the goal it returns the chosen row as a series #don't fully understand but ig we need it 

#this edits the duration based on BMI to perfect the workout duration
def adjust_duration_for_bmi(base_duration_hours: float, bmi_cat: str) -> float:
    if bmi_cat == "Underweight":
        return base_duration_hours * 0.9 #if underweight only do 90% of the duration
    elif bmi_cat in ("Overweight", "Obese"):
        return base_duration_hours * 1.1 #if obese or overweight do 110%
    else:
        return base_duration_hours #if normal do normal time


# ---------- USER INPUTS ----------

st.subheader("👤 Your Profile")

col1, col2 = st.columns(2) #subheader and a 2 column layout for easy use and input layout 

with col1: #left column user enters age using a slider and gender with a dropdown
    age = st.slider("Age (years)", 15, 80, 25)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])

with col2: #rigth column user enters height and weight with sliders 
    weight = st.slider("Weight (kg)", 35.0, 200.0, 70.0, 0.5)
    height = st.slider("Height (m)", 1.30, 2.20, 1.70, 0.01)

#at the bottom we have the goal selection which is also a dropdown menu
goal_label = st.selectbox(
    "Fitness Goal",
    [
        "Burn max calories / Lose weight",
        "Build muscle",
        "Improve endurance",
        "Balanced routine"
    ]
)
# takes the useer entered goal and equates it with a goal from the data set 
goal_map = {
    "Burn max calories / Lose weight": "burn_calories",
    "Build muscle": "build_muscle",
    "Improve endurance": "endurance",
    "Balanced routine": "balanced"
} 

goal_key = goal_map[goal_label] #now we have the goal key

# ---------- CALCULATE USER METRICS ----------
#calculates all of the user's metrics using the previously defined functions
user_bmi = calculate_bmi(weight, height)
user_bmi_cat = bmi_category_simple(user_bmi)
user_max_hr = estimate_max_hr(age)

#shows the user these calculated metrics
st.markdown("---")
st.subheader("📊 Your Metrics")
st.write(f"**BMI:** {user_bmi:.1f} ({user_bmi_cat})")
st.write(f"**Estimated Max Heart Rate:** {user_max_hr} bpm")


# ---------- DATA-DRIVEN RECOMMENDATION ----------

st.markdown("---")
st.subheader("🤖 Data-Driven Workout Recommendation")

best = pick_best_workout_for_goal(goal_key, workout_stats) #use  our function to choose the best workout based on the goal and stats

if best is None: #this is just in case we don't have the information...
    st.warning("No workout data found.")
else:
#extract the recommended rows fields 
    workout_type = best["Workout_Type"]
    base_duration_h = best["avg_duration_hours"]
    avg_bpm_from_data = best["avg_bpm"]
    avg_cal_from_data = best["avg_calories"]

    adjusted_duration_h = adjust_duration_for_bmi(base_duration_h, user_bmi_cat)
    adjusted_duration_min = round(adjusted_duration_h * 60) #adjust the duration and converted to minutes 

    # Figure out the target heart rate zone
    if avg_bpm_from_data > 0: #if we have the avg bpm for that workout we use +-10% for the zone
        low_hr = round(avg_bpm_from_data * 0.9)
        high_hr = round(avg_bpm_from_data * 1.1)
        avg_hr_display = f"{avg_bpm_from_data:.0f}"
        hr_source = "Based on average BPM from dataset." #saves the source to share with user
    else: #if we don't we use 60-75% of the estimated max Heart rate (common aerobic training zone)
        low_hr = round(user_max_hr * 0.6)
        high_hr = round(user_max_hr * 0.75)
        avg_hr_display = f"{(low_hr + high_hr)//2}"
        hr_source = "Based on % of your estimated max HR." #saves the source

    colA, colB = st.columns(2) #again 2 column display

    with colA: #firts column displays workout type, duration and calories
        st.write(f"**Recommended Workout Type (from data):** `{workout_type}`")
        st.write(f"**Suggested Session Duration:** ~**{adjusted_duration_min} minutes**")
        st.write(f"**Expected Calories Burned:** ~**{avg_cal_from_data:.0f} kcal**")

    with colB: #second shows Heart rate info includng avg training HR and the zone as well as where the info came from
        st.write("**Target Heart Rate Zone:**")
        st.write(f"- Avg Training HR: **{avg_hr_display} bpm**")
        st.write(f"- Zone: **{low_hr}–{high_hr} bpm**")
        st.caption(hr_source)

#we lastly wrote a little note to explain where the workout and session suggestion come from and a little note saying this isn't actual advice!
    st.markdown("---")
    st.write(
        "This recommendation is based on the **average performance** of each workout type "
        "in the dataset. Your selected goal chooses the optimization criteria, and your BMI "
        "category adjusts the session duration."
    )

st.caption("Educational use only — not medical advice.")
