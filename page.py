import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Menstrual Cycle Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load trained model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        with open("best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure 'best_model.pkl', 'scaler.pkl', and 'label_encoders.pkl' are in the same directory.")
        return None, None, None

# Load models
best_model, scaler, label_encoders = load_models()

def calculate_bmi(weight_kg, height_feet, height_inches):
    """Calculate BMI from weight in kg and height in feet/inches"""
    # Convert height to meters
    total_height_inches = (height_feet * 12) + height_inches
    height_meters = total_height_inches * 0.0254
    
    # Calculate BMI
    if height_meters > 0:
        bmi = weight_kg / (height_meters ** 2)
        return round(bmi, 1)
    return 0

def scale_numerical_features(df, scaler):
    df = df.copy()
    numerical_features = ["Age", "BMI", "Stress Level", "Sleep Hours", "Period Length"]
    
    missing_numerical = [f for f in numerical_features if f not in df.columns]
    if missing_numerical:
        return df
    
    numerical_data = df[numerical_features].astype(float)
    scaled_data = scaler.transform(numerical_data)
    
    for i, feature in enumerate(numerical_features):
        df[feature] = scaled_data[:, i]
    
    return df

def generate_reason(row, original_exercise, original_diet):
    """
    Generate reasons using the ORIGINAL string values from user input
    """
    reasons = []
    age = row["Age"]
    if age < 18:
        reasons.append("Teen years often involve hormonal fluctuations as cycles establish")
    elif 18 <= age <= 25:
        reasons.append("Young adulthood typically has stable hormonal patterns")
    elif 25 < age <= 35:
        reasons.append("Prime reproductive years with generally stable hormones")
    elif 35 < age <= 45:
        reasons.append("Perimenopausal phase may cause hormonal variations")
    elif age > 45:
        reasons.append("Menopausal transition involves significant hormonal changes")

    # Stress (using the numerical value)
    if row["Stress Level"] >= 4:
        reasons.append("High stress levels can disrupt hormonal balance")
    elif row["Stress Level"] <= 2:
        reasons.append("Low stress levels promote hormonal regularity")
    
    # Sleep (using the numerical value)
    if row["Sleep Hours"] < 7:
        reasons.append("Insufficient sleep may affect cycle regularity")
    elif row["Sleep Hours"] > 9:
        reasons.append("Excessive sleep could indicate hormonal imbalances")

    # BMI (using the numerical value)
    if row["BMI"] < 18.5:
        reasons.append("Low BMI may cause irregular periods")
    elif row["BMI"] > 25:
        reasons.append("High BMI can cause hormonal changes")

    # Exercise - use ORIGINAL string value from user input
    if original_exercise == "Low":
        reasons.append("Low activity may lead to hormonal fluctuations")
    elif original_exercise == "High":
        reasons.append("High exercise intensity can affect cycle timing")

    # Diet - use ORIGINAL string value from user input
    if original_diet == "High Sugar":
        reasons.append("High sugar diet may impact hormonal balance")
    elif original_diet == "Balanced":
        reasons.append("Balanced diet supports reproductive health")
    elif original_diet == "Vegetarian":
        reasons.append("Vegetarian diet may require careful nutrient balance for hormonal health")
    elif original_diet == "Low Carb":
        reasons.append("Low carbohydrate intake can influence energy levels and hormones")

    if not reasons:
        return "Your health parameters appear stable and supportive of regular cycles."
    else:
        return "Factors that may influence your cycle:\n- " + "\n- ".join(reasons)

def generate_insights(predicted_days, actual_mean=28):
    """
    Generate insights based on predicted cycle length
    """
    insights = []
    diff = predicted_days - actual_mean

    if diff > 7:
        insights.append(f"Predicted cycle ({predicted_days:.1f} days) is longer than average ({actual_mean} days)")
        insights.append("Longer cycles can be influenced by stress, diet, or hormonal changes")
        insights.append("Consider tracking ovulation signs for better cycle understanding")
    elif diff < -7:
        insights.append(f"Predicted cycle ({predicted_days:.1f} days) is shorter than average ({actual_mean} days)")
        insights.append("Shorter cycles may relate to lifestyle factors or natural variation")
        insights.append("Ensure adequate rest and nutrition during shorter cycles")
    else:
        insights.append(f"Predicted cycle ({predicted_days:.1f} days) is within normal range")
        insights.append("Your cycle appears regular based on current health parameters")
        insights.append("Continue maintaining healthy lifestyle habits")

    insights.append("Maintain consistent sleep, nutrition, and stress management for cycle regularity")
    return "\n".join(insights)

def encode_categorical_features(df, label_encoders):
    df = df.copy()
    categorical_columns = ["Exercise Frequency", "Diet", "Symptoms", "BMI_Category"]
    
    for col in categorical_columns:
        if col in df.columns:
            try:
                original_value = df[col].iloc[0]
                if original_value not in label_encoders[col].classes_:
                    default_value = label_encoders[col].classes_[0]
                    df[col] = label_encoders[col].transform([default_value])[0]
                else:
                    df[col] = label_encoders[col].transform([original_value])[0]
            except Exception as e:
                default_value = label_encoders[col].classes_[0]
                df[col] = label_encoders[col].transform([default_value])[0]
    
    return df

def apply_feature_engineering(df):
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['Cycle Start Date']):
        df['Cycle Start Date'] = pd.to_datetime(df['Cycle Start Date'])
    
    df['Cycle Start Month'] = df['Cycle Start Date'].dt.month.astype(int)
    df['Cycle Start DayOfWeek'] = df['Cycle Start Date'].dt.dayofweek.astype(int)
    
    # Age grouping
    age_bins = [0, 20, 30, 40, 50, 100]
    age_labels = [0, 1, 2, 3, 4]
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    df['Age_Group'] = df['Age_Group'].astype(int)
    
    # BMI categorization
    conditions = [
        (df['BMI'] < 18.5),
        (df['BMI'] < 25),
        (df['BMI'] < 30),
        (df['BMI'] >= 30)
    ]
    choices = ["Underweight", "Normal", "Overweight", "Obese"]
    df['BMI_Category'] = np.select(conditions, choices, default="Normal")
    
    # Seasonal features
    df['Cycle Start Month_sin'] = np.sin(2 * np.pi * df['Cycle Start Month'].astype(float) / 12)
    df['Cycle Start Month_cos'] = np.cos(2 * np.pi * df['Cycle Start Month'].astype(float) / 12)
    df['Is_Winter'] = df['Cycle Start Month'].isin([12, 1, 2]).astype(int)
    df['Is_Spring'] = df['Cycle Start Month'].isin([3, 4, 5]).astype(int)
    df['Is_Summer'] = df['Cycle Start Month'].isin([6, 7, 8]).astype(int)
    df['Is_Fall'] = df['Cycle Start Month'].isin([9, 10, 11]).astype(int)
    
    return df

def predict_next_cycle(user_df, model, scaler, label_encoders, original_exercise, original_diet):
    if model is None:
        return None, None, "Model not loaded properly", ""
    
    try:
        # Apply feature engineering first
        user_df_processed = apply_feature_engineering(user_df)
        
        # Encode categorical variables
        user_df_encoded = encode_categorical_features(user_df_processed, label_encoders)
        
        # Scale numerical features
        user_df_scaled = scale_numerical_features(user_df_encoded, scaler)
        
        # Define feature columns
        feature_columns = [
            "Age", "BMI", "Stress Level", "Exercise Frequency",
            "Sleep Hours", "Diet", "Period Length", 
            "Symptoms", "Cycle Start Month", "Cycle Start DayOfWeek",
            "Age_Group", "BMI_Category", "Cycle Start Month_sin", "Cycle Start Month_cos",
            "Cycle_Length_Rolling_Avg", "Is_Winter", "Is_Spring", "Is_Summer", "Is_Fall"
        ]
        
        # Check if all required features are present
        missing_features = [f for f in feature_columns if f not in user_df_scaled.columns]
        if missing_features:
            return None, None, f"Missing features: {missing_features}", ""
        
        # Extract features and ensure they are numeric
        prediction_features = user_df_scaled[feature_columns].astype(float)
        
        # Make prediction
        predicted_cycle_length = model.predict(prediction_features)[0]
        
        cycle_start_date = user_df_processed['Cycle Start Date'].iloc[0]
        predicted_next_date = cycle_start_date + timedelta(days=int(predicted_cycle_length))
        
        # Use the processed row but pass original string values for exercise and diet
        row = user_df_processed.iloc[0]
        reasons = generate_reason(row, original_exercise, original_diet)
        insights = generate_insights(predicted_cycle_length, row["Cycle_Length_Rolling_Avg"])
        
        return predicted_cycle_length, predicted_next_date, reasons, insights
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, None, f"Prediction error: {str(e)}", ""

def create_user_dataframe(age, bmi, stress, exercise, diet, sleep, period_length, symptoms, cycle_start_str, avg_cycle_length):
    cycle_start_date = pd.to_datetime(cycle_start_str)
    
    user_df = pd.DataFrame({
        'Age': [float(age)],
        'BMI': [float(bmi)],
        'Stress Level': [float(stress)],
        'Exercise Frequency': [exercise],  # Keep as original string
        'Sleep Hours': [float(sleep)],
        'Diet': [diet],  # Keep as original string
        'Period Length': [float(period_length)],
        'Symptoms': [symptoms],
        'Cycle Start Date': [cycle_start_date],
        "Cycle_Length_Rolling_Avg": [float(avg_cycle_length)]
    })
    
    return user_df

# Streamlit UI
def main():
    st.title("🌸 Menstrual Cycle Predictor")
    st.markdown("""
    Predict your next menstrual cycle start date based on your health parameters and lifestyle factors.
    This AI-powered tool helps you understand patterns in your menstrual health.
    """)
    
    # Sidebar for input
    st.sidebar.header("📋 Your Health Information")
    
    with st.sidebar.form("user_input_form"):
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=12, max_value=60, value=25, step=1)
        
        # Height and Weight inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            height_feet = st.number_input("Height (Feet)", min_value=4, max_value=7, value=5, step=1)
        with col2:
            height_inches = st.number_input("Height (Inches)", min_value=0, max_value=11, value=5, step=1)
        with col3:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.5)
        
        # Calculate BMI
        bmi = calculate_bmi(weight_kg, height_feet, height_inches)
        
        # Display BMI
        st.text_input("Calculated BMI", value=f"{bmi}", disabled=True)
        
        st.subheader("Lifestyle Factors")
        stress = st.slider("Stress Level (1-5)", min_value=1, max_value=5, value=3, 
                          help="1 = Very Low, 5 = Very High")
        sleep = st.slider("Average Sleep Hours", min_value=4.0, max_value=12.0, value=7.5, step=0.5)
        
        exercise = st.selectbox(
            "Exercise Frequency",
            ["Low", "Moderate", "High"],
            help="Low: <2 times/week, Moderate: 2-4 times/week, High: 5+ times/week"
        )
        
        diet = st.selectbox(
            "Diet Type",
            ["Vegetarian", "Balanced", "Low Carb", "High Sugar"],
            help="Select the option that best describes your typical diet"
        )
        
        st.subheader("Cycle Information")
        period_length = st.slider("Period Length (days)", min_value=1, max_value=10, value=5)
        avg_cycle_length = st.slider("Average Cycle Length (days)", min_value=21, max_value=45, value=28)
        
        symptoms = st.selectbox(
            "Common Symptoms",
            ["None", "Bloating", "Fatigue", "Cramps", "Headache", "Mood Swings"],
            help="Select your most common premenstrual symptom"
        )
        
        cycle_start_str = st.date_input(
            "Last Cycle Start Date",
            value=datetime.now() - timedelta(days=28),
            max_value=datetime.now()
        )
        
        submitted = st.form_submit_button("🔮 Predict Next Cycle")

    # Main content area
    if submitted:
        if best_model is None:
            st.error("Please ensure all model files are available in the same directory.")
            return
            
        with st.spinner("🔮 Analyzing your data and predicting next cycle..."):
            # Store original values for use in generate_reason
            original_exercise = exercise
            original_diet = diet
            
            user_df = create_user_dataframe(
                age, bmi, stress, exercise, diet, sleep, 
                period_length, symptoms, cycle_start_str, avg_cycle_length
            )
            
            predicted_length, next_date, reasons, insights = predict_next_cycle(
                user_df, best_model, scaler, label_encoders, original_exercise, original_diet
            )
            
        if predicted_length is not None:
            st.success("✅ Prediction Complete!")
            
            # Display results in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Cycle Length",
                    value=f"{predicted_length:.1f} days",
                    delta=f"{(predicted_length - avg_cycle_length):+.1f} vs average"
                )
            
            with col2:
                st.metric(
                    label="Next Cycle Start",
                    value=next_date.strftime("%b %d, %Y")
                )
            
            with col3:
                st.metric(
                    label="Confidence Range",
                    value="±7 days"
                )
            
            # Display date range
            st.info(f"**Likely Range:** {(next_date - timedelta(days=7)).strftime('%Y-%m-%d')} to {(next_date + timedelta(days=7)).strftime('%Y-%m-%d')}")
            
            # Health Analysis
            st.subheader("🔍 Health Analysis")
            st.text_area("Analysis", reasons, height=150, key="analysis")
            
            # Insights
            st.subheader("💡 Insights & Recommendations")
            st.text_area("Insights", insights, height=120, key="insights")
            
            # Show what changed
            st.subheader("📊 Input Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Personal Details:**")
                st.write(f"- Age: {age} years")
                st.write(f"- Height: {height_feet}'{height_inches}\"")
                st.write(f"- Weight: {weight_kg} kg")
                st.write(f"- BMI: {bmi}")
                
                st.write("**Lifestyle Factors:**")
                st.write(f"- Exercise: {exercise}")
                st.write(f"- Diet: {diet}")
            
            with col2:
                st.write("**Health Metrics:**")
                st.write(f"- Stress Level: {stress}/5")
                st.write(f"- Sleep: {sleep} hours")
                
                st.write("**Cycle Information:**")
                st.write(f"- Period Length: {period_length} days")
                st.write(f"- Average Cycle: {avg_cycle_length} days")
                st.write(f"- Symptoms: {symptoms}")
                st.write(f"- Last Cycle: {cycle_start_str}")
            
        else:
            st.error(f"❌ Prediction failed: {reasons}")

    # Information section (when no prediction yet)
    else:
        st.info("👈 Please fill out the form in the sidebar and click 'Predict Next Cycle' to get started.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Why Track Your Cycle?")
            st.markdown("""
            - **Health Monitoring**: Identify patterns and changes
            - **Fertility Awareness**: Understand your fertile window
            - **Symptom Management**: Prepare for PMS symptoms
            - **Medical Insights**: Detect potential health issues early
            """)
        
        with col2:
            st.subheader("🔍 How It Works")
            st.markdown("""
            - **AI-Powered**: Machine learning model trained on health data
            - **Multi-Factor**: Considers lifestyle, stress, sleep, and more
            - **Personalized**: Adapts to your unique health profile
            - **Educational**: Provides insights into cycle influences
            """)
        
        st.warning("⚠️ **Disclaimer**: This tool is for educational purposes only. Always consult healthcare professionals for medical advice.")

if __name__ == "__main__":
    main()