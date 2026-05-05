import streamlit as st
import pandas as pd

from preprocessing import preprocess_data
from models import get_models, train_models
from evaluation import evaluate_models

st.set_page_config(page_title="ML Comparison App", layout="wide")

st.title("📊 Machine Learning Model Comparison")

# Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select target
    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("🚀 Run Models"):
        with st.spinner("Processing..."):

            # Preprocessing
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

            # Models
            models = get_models()
            trained_models = train_models(models, X_train, y_train)

            # Evaluation
            results = evaluate_models(trained_models, X_test, y_test)

            results_df = pd.DataFrame(results)

        st.success("Done!")

        st.subheader("📈 Results")
        st.dataframe(results_df)

        # Best model
        best_model = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

        st.subheader("🏆 Best Model")
        st.write(best_model)

        # Chart
        st.subheader("📊 Accuracy Comparison")
        st.bar_chart(results_df.set_index("Model")["Accuracy"])
