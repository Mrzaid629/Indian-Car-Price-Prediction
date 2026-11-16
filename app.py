
import streamlit as st
import pandas as pd
import joblib

# Load trained model and application metadata
@st.cache_resource
def load_resources():
    model = joblib.load("indian_car_price_model.pkl")
    metadata = joblib.load("app_metadata.pkl")
    return model, metadata

model, metadata = load_resources()

# Page configuration
st.set_page_config(
    page_title="Indian Used Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Application title
st.title("🚗 Indian Pre-Owned Car Price Prediction")
st.write(
    "Enter your car details below to estimate its market price."
)

st.divider()

# Input form
with st.form("car_prediction_form"):

    st.subheader("Enter Car Details")

    input_data = {}

    categorical_features = metadata["categorical_options"]
    numerical_features = metadata["numerical_ranges"]

    col1, col2 = st.columns(2)

    # Categorical inputs
    for index, feature in enumerate(categorical_features):
        options = categorical_features[feature]

        if not options:
            continue

        with (col1 if index % 2 == 0 else col2):
            input_data[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options
            )

    # Numerical inputs
    for index, feature in enumerate(numerical_features):
        details = numerical_features[feature]

        minimum = details["min"]
        maximum = details["max"]
        median = details["median"]

        with (col1 if index % 2 == 0 else col2):
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(median)
            )

    submitted = st.form_submit_button(
        "Predict Car Price",
        use_container_width=True
    )

# Prediction
if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=metadata["features"]
    )

    try:
        predicted_price = float(model.predict(input_df)[0])

        if predicted_price < 0:
            st.error("The model returned an invalid negative price.")
        else:
            lower_price = predicted_price * 0.95
            upper_price = predicted_price * 1.05

            st.divider()
            st.subheader("Car Price Prediction")

            st.success(
                f"Estimated Car Price: ₹{predicted_price:,.0f}"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Lower Estimate",
                f"₹{lower_price:,.0f}"
            )

            col2.metric(
                "Upper Estimate",
                f"₹{upper_price:,.0f}"
            )

            st.caption(
                "The valuation range is an indicative ±5% "
                "range around the model prediction, not a "
                "statistical confidence interval."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
