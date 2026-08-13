import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FREIGHT_MODEL_PATH = ROOT / "freight_cost_prediction" / "models" / "predict_freight_model.pkl"
FLAG_MODEL_PATH = ROOT / "invoice_flagging" / "models" / "predict_flag_invoice.pkl"
SCALER_PATH = ROOT / "invoice_flagging" / "models" / "scaler.pkl"


def load_model(path: Path):
    with open(path, "rb") as f:
        return joblib.load(f)


def load_scaler(path: Path):
    with open(path, "rb") as f:
        return joblib.load(f)


def predict_freight(dollars: float, quantity: float):
    model = load_model(FREIGHT_MODEL_PATH)
    input_df = pd.DataFrame({"Dollars": [dollars], "Quantity": [quantity]})
    input_df["Predicted_Freight"] = model.predict(input_df).round(2)
    return input_df


def predict_invoice_flag(values: dict):
    model = load_model(FLAG_MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    input_df = pd.DataFrame(values)
    input_scaled = scaler.transform(input_df)
    input_df["Predicted_Flag"] = model.predict(input_scaled)
    return input_df


def main():
    st.set_page_config(
        page_title="Invoice Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5em;
        color: #2196F3;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #666;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">📊 Invoice Intelligence</h1>', unsafe_allow_html=True)
    st.write("🚀 Use trained ML models to predict freight costs and detect suspicious invoices.")

    page = st.sidebar.selectbox(
        "Choose Prediction Type",
        ["Freight Cost Prediction", "Invoice Flag Prediction", "About"]
    )

    if page == "Freight Cost Prediction":
        st.markdown('<h2 class="sub-header">🚚 Freight Cost Prediction</h2>', unsafe_allow_html=True)
        st.write("Enter invoice details to predict freight cost using Linear Regression.")

        col1, col2 = st.columns(2)
        with col1:
            dollars = st.number_input(
                "💰 Invoice Dollars",
                min_value=0.0,
                value=10000.0,
                step=100.0,
                format="%.2f",
                help="Total dollar amount of the invoice"
            )
        with col2:
            quantity = st.number_input(
                "📦 Invoice Quantity",
                min_value=0.0,
                value=100.0,
                step=1.0,
                help="Total quantity of items"
            )

        if st.button("🔮 Predict Freight Cost"):
            with st.spinner("Predicting..."):
                try:
                    prediction = predict_freight(dollars, quantity)
                    freight = prediction.loc[0, 'Predicted_Freight']
                    st.success(f"🎉 Predicted Freight Cost: **${freight:.2f}**")
                    st.dataframe(prediction.style.highlight_max(axis=0))
                except FileNotFoundError as exc:
                    st.error(f"❌ Model file not found: {exc}")
                except Exception as exc:
                    st.error(f"❌ Prediction failed: {exc}")

    elif page == "Invoice Flag Prediction":
        st.markdown('<h2 class="sub-header">🚨 Invoice Flag Prediction</h2>', unsafe_allow_html=True)
        st.write("Enter invoice fields to check if it should be flagged as suspicious.")

        with st.form(key="flag_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                invoice_quantity = st.number_input(
                    "📦 Invoice Quantity",
                    min_value=0.0,
                    value=100.0,
                    step=1.0
                )
                freight = st.number_input(
                    "🚚 Freight",
                    min_value=0.0,
                    value=200.0,
                    step=10.0,
                    format="%.2f"
                )
            with col2:
                invoice_dollars = st.number_input(
                    "💰 Invoice Dollars",
                    min_value=0.0,
                    value=10000.0,
                    step=100.0,
                    format="%.2f"
                )
                total_item_quantity = st.number_input(
                    "📊 Total Item Quantity",
                    min_value=0.0,
                    value=100.0,
                    step=1.0
                )
            with col3:
                total_item_dollars = st.number_input(
                    "💵 Total Item Dollars",
                    min_value=0.0,
                    value=9950.0,
                    step=100.0,
                    format="%.2f"
                )

            submitted = st.form_submit_button("🔍 Check Invoice Flag")

        if submitted:
            values = {
                "invoice_quantity": [invoice_quantity],
                "invoice_dollars": [invoice_dollars],
                "Freight": [freight],
                "total_item_quantity": [total_item_quantity],
                "total_item_dollars": [total_item_dollars],
            }
            with st.spinner("Analyzing..."):
                try:
                    prediction = predict_invoice_flag(values)
                    flag = prediction.loc[0, "Predicted_Flag"]
                    if flag == 1:
                        st.error("🚨 Invoice Flagged as Suspicious!")
                    else:
                        st.success("✅ Invoice is Clean")
                    st.dataframe(prediction.style.applymap(
                        lambda x: 'background-color: red' if x == 1 else 'background-color: green',
                        subset=['Predicted_Flag']
                    ))
                except FileNotFoundError as exc:
                    st.error(f"❌ Model or scaler file not found: {exc}")
                except Exception as exc:
                    st.error(f"❌ Prediction failed: {exc}")

    else:  # About
        st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)
        st.write("""
        This app uses machine learning models trained on invoice data to:
        - Predict freight costs based on invoice dollars and quantity.
        - Flag potentially suspicious invoices based on various features.

        Models:
        - Freight Prediction: Linear Regression
        - Invoice Flagging: Random Forest Classifier

        Built with Streamlit and scikit-learn.
        """)
        st.info("📈 Models trained on historical invoice data for accurate predictions.")

    # st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()