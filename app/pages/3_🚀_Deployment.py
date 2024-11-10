import streamlit as st
import pandas as pd
import json
import pickle
from app.core.system import AutoMLSystem


st.set_page_config(page_title="Deployment", page_icon="🚀")

st.write("# 🚀 Deployment")
st.write("Load a saved pipeline and perform predictions on new data.")

# Initialize AutoMLSystem
automl = AutoMLSystem.get_instance()

# Step 1: Load a Saved Pipeline
st.write("## Step 1: Load a Saved Pipeline")
pipelines = automl.registry.list(type="pipeline")
pipeline_names = [pipeline.name for pipeline in pipelines]

selected_pipeline = st.selectbox("Select Pipeline", pipeline_names)

if selected_pipeline:
    try:
        pipeline_artifact = next(
            p for p in pipelines
            if p.name == selected_pipeline
        )
        pipeline_config = json.loads(pipeline_artifact.data.decode("utf-8"))

        # Display pipeline summary
        st.write("### Pipeline Summary:")
        st.json(pipeline_config)

        st.success(f"Pipeline '{selected_pipeline}' loaded successfully!")

        # Ensure the trained model is loaded
        if (
            "trained_model" in pipeline_config
            and pipeline_config["trained_model"]
        ):
            trained_model = pickle.loads(
                bytes.fromhex(pipeline_config["trained_model"])
            )
            st.info("Trained model is loaded and ready for predictions.")
        else:
            st.error(
                "The pipeline does not contain a trained model. "
                "Please ensure the model was trained and saved correctly."
            )
            st.stop()

        # Step 2: Upload Data for Prediction
        st.write("## Step 2: Upload Data for Prediction")
        uploaded_file = st.file_uploader(
            "Upload a CSV file for prediction",
            type=["csv"]
        )

        if uploaded_file:
            try:
                # Read the uploaded file
                pred_df = pd.read_csv(uploaded_file)

                # Validate features
                expected_features = pipeline_config.get("input_features", [])
                if not set(expected_features).issubset(pred_df.columns):
                    missing_features = (
                        set(expected_features)
                        - set(pred_df.columns)
                    )
                    raise ValueError(
                        f"The features in the uploaded dataset do not match "
                        f"the expected features: {list(missing_features)}"
                    )

                # Step 3: Perform Predictions
                st.write("## Step 3: Perform Predictions")
                predictions = trained_model.predict(pred_df[expected_features])

                # Display prediction results
                st.write("### Prediction Results:")
                pred_df["Predictions"] = predictions
                st.dataframe(pred_df)

                # Optionally download the results
                csv = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"An error occurred while predicting: {e}")
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
else:
    st.warning("No pipeline selected.")
