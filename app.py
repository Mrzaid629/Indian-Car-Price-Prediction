
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AutoValue | Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: #0b1220;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
.hero {
    background: linear-gradient(120deg, #142b4d, #174b78);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 28px;
    border: 1px solid #31577e;
}
.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #c5d9ee;
    font-size: 16px;
}
.section {
    background: #121e30;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #263b54;
    margin-bottom: 20px;
}
div[data-testid="stForm"] {
    background: #121e30;
    padding: 24px;
    border: 1px solid #263b54;
    border-radius: 16px;
}
div[data-testid="stMetric"] {
    background: #16263b;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #31506e;
}
.stButton button, 
.stFormSubmitButton button {
    background: #2384d8;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: 600;
}
.stButton button:hover,
.stFormSubmitButton button:hover {
    background: #176bb5;
    color: white;
}
label, p, h2, h3 {
    color: #e2e8f0 !important;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = joblib.load("indian_car_price_model.pkl")
    metadata = joblib.load("app_metadata.pkl")
    return model, metadata

model, metadata = load_resources()

# Header
st.markdown("""
<div class="hero">
    <h1>🚗 AutoValue</h1>
    <p>Indian Pre-Owned Car Price Prediction</p>
    <p>Discover the estimated market value of your car using machine learning.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## Enter Your Car Details")
st.write("Provide the vehicle information below to get an estimated price.")

categorical_features = metadata["categorical_options"]
numerical_features = metadata["numerical_ranges"]
input_data = {}

with st.form("car_prediction_form"):

    st.markdown("### 🚘 Vehicle Information")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(categorical_features):
        options = categorical_features[feature]
        if options:
            with (col1 if index % 2 == 0 else col2):
                input_data[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options
                )

    st.markdown("### ⚙️ Technical Specifications")

    col3, col4 = st.columns(2)

    for index, feature in enumerate(numerical_features):
        details = numerical_features[feature]

        with (col3 if index % 2 == 0 else col4):
            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=float(details["min"]),
                max_value=float(details["max"]),
                value=float(details["median"])
            )

    submitted = st.form_submit_button(
        "✨ Predict Car Price",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame(
        [input_data],
        columns=metadata["features"]
    )

    try:
        predicted_price = float(model.predict(input_df)[0])

        if predicted_price < 0:
            st.error("Invalid prediction. Please check your inputs.")
        else:
            lower_price = predicted_price * 0.95
            upper_price = predicted_price * 1.05

            st.markdown("---")
            st.markdown("## 📊 Your Car Valuation")

            st.markdown(
                f"""
                <div class="hero" style="text-align:center;">
                    <p>ESTIMATED MARKET PRICE</p>
                    <h1 style="font-size:44px;">₹{predicted_price:,.0f}</h1>
                    <p>Based on your vehicle details and our trained ML model</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Lower Estimate",
                    f"₹{lower_price:,.0f}"
                )

            with c2:
                st.metric(
                    "Upper Estimate",
                    f"₹{upper_price:,.0f}"
                )

            st.caption(
                "The ±5% range is an indicative valuation range, "
                "not a statistical confidence interval."
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.markdown("---")
st.caption("AutoValue | Machine Learning Based Car Price Prediction")
