import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from models import get_models, train_models
from evaluation import evaluate_models

st.set_page_config(page_title="ML Comparison App", layout="centered")
st.title("ML Model Comparison")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    target = st.selectbox("Target Column", [c for c in df.columns if "id" not in c.lower() and not c.endswith("_date")])

    if st.button("Run"):
        X_train, X_test, y_train, y_test, task = preprocess_data(df, target)
        st.info(f"**{task.upper()}** — {y_train.nunique()} unique values")

        models = get_models(task)
        trained = train_models(models, X_train, y_train)
        results = pd.DataFrame(evaluate_models(trained, X_test, y_test, task))

        st.dataframe(results)
        metric = "R2" if task == "regression" else "Accuracy"
        best = results.sort_values(metric, ascending=False).iloc[0]
        st.success(f"Best: {best['Model']} ({metric}={best[metric]:.4f})")
        st.bar_chart(results.set_index("Model")[metric])
