import streamlit as st
import pandas as pd
import joblib


# Cargar el modelo
@st.cache_resource
def load_model():
    return joblib.load("best_model_et.pkl")

model = load_model()


feature_names = ['unique_session_count_cumulative_y', 'user_lifetime_purchase_count', 'days_since_previous_purchase', 'user_lifetime_event_count', 'purchase_value', 'cumulative_product_id_count', 'purchase_repeat_product%']


# Título
st.title("🔍 Customer Churn Prediction")
st.markdown("Completa la información del cliente para predecir si hará churn (repeated purchase in 60d).")

# Formulario dinámico basado en tipos de datos
user_input = {}

# Entradas numéricas
user_input['unique_session_count_cumulative_y'] = st.number_input("Sesiones únicas acumuladas", value=5)
user_input['user_lifetime_purchase_count'] = st.number_input("Compras acumuladas", value=3)
user_input['days_since_previous_purchase'] = st.number_input("Días desde la última compra", value=15.0)
user_input['user_lifetime_event_count'] = st.number_input("Proporción media de productos con descuento", min_value=0.0, max_value=1.0, value=0.2)
user_input['purchase_value'] = st.number_input("Proporción de productos repetidos", min_value=0.0, max_value=1.0, value=0.3)
user_input['cumulative_product_id_count'] = st.number_input("Duración de la sesión (segundos)", min_value=0.0, value=300.0)
user_input['purchase_repeat_product%'] = st.number_input("Número de productos", value=2)


# Convertir a DataFrame
input_df = pd.DataFrame([user_input])


# Predicción
if st.button("Predecir"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ El cliente probablemente hará churn. Probabilidad: {prob:.2%}")
    else:
        st.success(f"✅ El cliente probablemente NO hará churn. Probabilidad de churn: {prob:.2%}")
