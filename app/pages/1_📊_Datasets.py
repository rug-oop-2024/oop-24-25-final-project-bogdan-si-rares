import streamlit as st
import pandas as pd
import os
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact

st.set_page_config(page_title="Datasets", page_icon="📊")

# Singleton instance of AutoMLSystem
automl = AutoMLSystem.get_instance()

st.write("# 📊 Datasets")
st.write("Manage datasets by uploading and saving them as artifacts.")

# Ensure the datasets directory exists
os.makedirs("./datasets", exist_ok=True)

# Upload and preview dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Dataset", df.head())

    # Generate unique name and asset path
    # Remove file extension for the name
    name = os.path.splitext(uploaded_file.name)[0]
    asset_path = f"./datasets/{name}.csv"

    # Create an Artifact instance from Dataset
    artifact = Artifact(
        name=name,
        type="dataset",
        data=df.to_csv(index=False).encode(),
        asset_path=asset_path,
        version="1.0.0",
        etadata={"original_name": uploaded_file.name},
    )

    # Convert to Artifact
    if st.button("Convert to Dataset Artifact"):
        # Register the artifact
        automl.registry.register(artifact)
        st.success(
            f"Dataset '{name}' has been added to the artifact registry."
        )

    # Save Dataset Artifact
    if st.button("Save Dataset"):
        automl.registry.register(artifact)
        st.success(
            f"Dataset '{artifact.name}' "
            "has been saved to the artifact registry."
        )

# List available datasets
st.write("## Available Datasets")
datasets = automl.registry.list(type="dataset")
if datasets:
    for dataset in datasets:
        st.write(f"- {dataset.name}")
else:
    st.warning("No datasets available in the registry.")
