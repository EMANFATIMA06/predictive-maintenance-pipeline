import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from pipeline.data_loader import load_and_prepare
from pipeline.model_trainer import train_model
from pipeline.evaluator import evaluate

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

st.title("🔧 Predictive Maintenance Dashboard")
st.write("Upload a manufacturing dataset and configure columns to train a failure-prediction model.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()

    st.subheader("Configure your columns")
    target_col = st.selectbox("Which column is the failure label (0/1)?", all_columns)
    numeric_features = st.multiselect("Select numeric sensor/feature columns", all_columns)
    categorical_features = st.multiselect("Select categorical columns (if any)", all_columns)
    id_cols = st.multiselect("Select ID columns to drop (not useful for prediction)", all_columns)

    threshold = st.slider("Decision threshold (lower = catch more failures, more false alarms)", 0.05, 0.95, 0.3, 0.05)

    if st.button("Train Model"):
        config = {
            "target_col": target_col,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "id_cols": id_cols,
            "drop_cols": []
        }

        X, y = load_and_prepare(df, config)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = train_model(X_train, y_train)
        pred, proba = evaluate(model, X_test, y_test, threshold=threshold)

        st.subheader("Model Performance")
        cm = confusion_matrix(y_test, pred)
        st.write("Confusion Matrix (rows=actual, cols=predicted):")
        st.dataframe(pd.DataFrame(cm, index=["Actual: No Failure", "Actual: Failure"],
                                       columns=["Predicted: No Failure", "Predicted: Failure"]))

        report = classification_report(y_test, pred, output_dict=True)
        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Feature Importance")
        importances = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(importances["feature"], importances["importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("What drives failure predictions")
        st.pyplot(fig)

        st.subheader("Business Insight")
        top_feature = importances.iloc[0]["feature"]
        recall = report["1"]["recall"] if "1" in report else report["1.0"]["recall"]
        st.write(f"The model catches **{recall*100:.0f}%** of actual failures at this threshold. "
                 f"The strongest predictor of failure is **{top_feature}**.")
else:
    st.info("Upload a CSV file to get started. Try the AI4I 2020 dataset from your `data` folder.")