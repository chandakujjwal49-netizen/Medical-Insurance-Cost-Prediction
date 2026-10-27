import os
import pickle
import streamlit as st
import pandas as pd

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# ==========================================
# 2. MODEL FILE PATH
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "insurance_regression_model.pkl"
)

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


if not os.path.exists(MODEL_PATH):
    st.error("PKL model file not found!")
    st.write("Expected location:", MODEL_PATH)
    st.stop()

try:
    model_bundle = load_model()

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    numerical_features = model_bundle["numerical_features"]
    categorical_features = model_bundle["categorical_features"]
    category_options = model_bundle["category_options"]
    numeric_defaults = model_bundle["numeric_defaults"]

except Exception as error:
    st.error("Unable to load the PKL model.")
    st.code(str(error))
    st.warning(
        "If the error mentions a missing scikit-learn attribute, "
        "retrain and resave the model using the current Python "
        "environment. Changing app.py alone will not fix "
        "an incompatible pickle file."
    )
    st.stop()

# ==========================================
# 4. PAGE HEADER
# ==========================================
st.title("🏥 Medical Insurance Cost Prediction")

st.write(
    "Enter your details to estimate medical insurance charges "
    "using a trained machine learning model."
)

st.success("Machine learning model loaded successfully!")

st.divider()

# ==========================================
# 5. INPUT FORM
# ==========================================
st.subheader("📋 Enter Your Details")

with st.form("insurance_form"):

    user_input = {}

    col1, col2 = st.columns(2)

    # Numerical features
    for index, feature in enumerate(numerical_features):

        container = col1 if index % 2 == 0 else col2
        default_value = numeric_defaults.get(feature, 0)

        with container:

            if feature.lower() == "age":
                user_input[feature] = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=max(18, min(100, int(default_value))),
                    step=1
                )

            elif feature.lower() == "bmi":
                user_input[feature] = st.number_input(
                    "BMI",
                    min_value=10.0,
                    max_value=100.0,
                    value=max(
                        10.0,
                        min(100.0, float(default_value))
                    ),
                    step=0.1
                )

            elif feature.lower() in ["children", "child"]:
                user_input[feature] = st.number_input(
                    "Number of Children",
                    min_value=0,
                    max_value=20,
                    value=max(0, min(20, int(default_value))),
                    step=1
                )

            else:
                user_input[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=float(default_value)
                )

    # Categorical features
    for feature in categorical_features:

        options = category_options.get(feature, [])

        if not options:
            st.error(
                f"No category options available for {feature}."
            )
            st.stop()

        label = feature.replace("_", " ").title()

        user_input[feature] = st.selectbox(
            label,
            options=options
        )

    submitted = st.form_submit_button(
        "🔍 Predict Insurance Cost",
        use_container_width=True
    )

# ==========================================
# 6. PREDICTION
# ==========================================
if submitted:

    try:
        input_df = pd.DataFrame(
            [user_input],
            columns=feature_columns
        )

        prediction = float(model.predict(input_df)[0])

        st.divider()
        st.subheader("📊 Prediction Result")

        st.metric(
            label="Estimated Annual Insurance Charges",
            value=f"{prediction:,.2f}"
        )

        st.caption(
            "This is a model-generated estimate, not a guaranteed "
            "insurance quote."
        )

        # Display evaluation metrics, if saved
        metrics = model_bundle.get("metrics", {})

        if metrics:
            with st.expander("View Model Performance"):
                for name, value in metrics.items():
                    try:
                        st.write(f"**{name}:** {float(value):.4f}")
                    except (TypeError, ValueError):
                        st.write(f"**{name}:** {value}")

    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))
        st.info(
            "Check that the input feature names and data types "
            "match those used when training the model."
        )

# ==========================================
# 7. FOOTER
# ==========================================
st.divider()

st.caption(
    "Medical Insurance Cost Prediction | Machine Learning Project"
)