import streamlit as st
import pandas as pd
import joblib
import gzip
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# Optional: Set a light background color
def set_custom_style():
    st.markdown("""
        <style>
        body {
            background-color: #f8f9fa;
        }
        h1 {
            color: #dc3545;
        }
        </style>
    """, unsafe_allow_html=True)

set_custom_style()

# Load model
@st.cache_resource
def load_model(path):
    with gzip.open(path, 'rb') as f:
        return joblib.load(f)

from sklearn.ensemble import ExtraTreesClassifier
params = np.load("best_model_et_params.npy", allow_pickle=True).item()
model = ExtraTreesClassifier(**params)
model.fit(X_train_sel, y_train)


# Feature names
feature_names = [
    'unique_session_count_cumulative_y', 
    'user_lifetime_purchase_count', 
    'days_since_previous_purchase', 
    'user_lifetime_event_count', 
    'purchase_value', 
    'cumulative_product_id_count', 
    'purchase_repeat_product%'
]

# Session state navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_predict():
    st.session_state.page = "predict"

# --- HOME PAGE ---
if st.session_state.page == "home":
    # Optional logo (if you have it)
    try:
        st.image("eae_logo.png", width=280)
    except:
        pass

    st.markdown("""
    <div style='text-align: center;'>
        <h1>Customer Churn Prediction</h1>
        <p style='font-size: 1.3em;'>Welcome to the Customer Churn Prediction App.</p>
        <p style='font-size: 1.1em;'>Click below to start predicting whether a customer is likely to churn or not.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button("🚀 Start Predicting", on_click=go_to_predict)

    st.markdown("---")
    st.markdown("🎓 Developed as part of the TFM for **EAE Business School**", unsafe_allow_html=True)

# --- PREDICTION PAGE ---
elif st.session_state.page == "predict":
    st.title("Predict Customer Churn")
    st.markdown("Enter the following customer data:")

    input_data = {}
    for feature in feature_names:
        label = feature.replace('_', ' ').replace('%', ' (%)').title()
        input_data[feature] = st.number_input(label, value=0.0)

    if st.button("🔍 Predict Churn"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction Result:")
        st.write("**✅ No Churn**" if prediction == 0 else "**⚠️ Churn Likely**")
        st.write(f"Churn Probability: `{prediction_proba[1]:.2f}`")

        # Pie chart
        labels = ['No Churn', 'Churn']
        colors = ['#4CAF50', '#F44336']
        fig, ax = plt.subplots()
        ax.pie(prediction_proba, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("🎓 Developed for the **TFM - EAE Business School**", unsafe_allow_html=True)

