import streamlit as st
import pandas as pd
import io
import json
import pickle
import os
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.artifact import Artifact
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score

st.set_page_config(page_title="Modelling", page_icon="⚙")

st.write("# ⚙ Modelling")
st.write("Design a machine learning pipeline to train a model on a dataset.")

# Initialize AutoMLSystem
automl = AutoMLSystem.get_instance()

# Step 1: Select Dataset
st.write("## Step 1: Select a Dataset")
datasets = automl.registry.list(type="dataset")
dataset_names = [dataset.name for dataset in datasets]
selected_dataset = st.selectbox("Select Dataset", dataset_names)

if selected_dataset:
    dataset = next(
        (ds for ds in datasets if ds.name == selected_dataset),
        None
    )

    # Decode dataset data into a DataFrame
    try:
        csv_data = dataset.data.decode("utf-8")
        dataset_df = pd.read_csv(io.StringIO(csv_data))
        st.write("Preview of Dataset:")
        st.dataframe(dataset_df.head())  # Display a preview of the dataset
    except Exception as e:
        st.error(f"Failed to read dataset: {e}")
        dataset_df = None

    if dataset_df is not None:
        # Convert DataFrame back to Dataset
        dataset = Dataset.from_dataframe(
            data=dataset_df,
            name=selected_dataset,
            asset_path=f"./assets/{selected_dataset}.csv"
        )

        # Step 2: Feature Selection
        st.write("## Step 2: Select Features")
        try:
            detected_features = detect_feature_types(dataset)
            numerical_features = [
                f.name for f in detected_features if f.type == "numerical"
            ]
            categorical_features = [
                f.name for f in detected_features if f.type == "categorical"
            ]

            input_features = st.multiselect(
                "Select Input Features",
                numerical_features
            )
            target_feature = st.selectbox(
                "Select Target Feature",
                numerical_features + categorical_features
            )
        except Exception as e:
            st.error(f"Error detecting features: {e}")
            input_features, target_feature = None, None

        if input_features and target_feature:
            st.success(
                f"Input Features: {input_features}, "
                f"Target Feature: {target_feature}"
            )

            # Step 3: Select Model
            st.write("## Step 3: Select a Model")
            task_type = (
                "classification" if target_feature in categorical_features
                else "regression"
            )
            models = {
                "classification": [
                    "Logistic Regression",
                    "Decision Tree",
                    "KNN"
                ],
                "regression": [
                    "Linear Regression",
                    "Decision Tree Regressor",
                    "Random Forest Regressor"
                ]
            }
            selected_model = st.selectbox("Select Model", models[task_type])
            st.success(f"Selected Model: {selected_model}")

            # Step 4: Configure Metrics
            st.write("## Step 4: Select Metrics")
            metrics = {
                "classification": ["Accuracy", "Precision", "Recall"],
                "regression": ["Mean Squared Error"]
            }
            selected_metrics = st.multiselect(
                "Select Metrics",
                metrics[task_type]
            )
            st.success(f"Selected Metrics: {selected_metrics}")

            # Step 5: Split Dataset
            st.write("## Step 5: Split Dataset")
            test_size = st.slider("Test Size (as %)", 10, 50, 20) / 100
            train_df, test_df = train_test_split(
                dataset_df,
                test_size=test_size,
                random_state=42
            )
            st.write(
                f"Train Set: {len(train_df)} rows, "
                f"Test Set: {len(test_df)} rows"
            )

            # Step 6: Train Pipeline
            st.write("## Step 6: Train the Pipeline")
            if st.button("Train"):
                X_train = train_df[input_features]
                y_train = train_df[target_feature]
                X_test = test_df[input_features]
                y_test = test_df[target_feature]

                # Train the model
                model = None
                if selected_model == "Logistic Regression":
                    model = LogisticRegression()
                elif selected_model == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif selected_model == "KNN":
                    model = KNeighborsClassifier()
                elif selected_model == "Linear Regression":
                    model = LinearRegression()
                elif selected_model == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                elif selected_model == "Random Forest Regressor":
                    model = RandomForestRegressor()

                if model:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)

                    # Evaluate the model
                    results = {}
                    if "Accuracy" in selected_metrics:
                        results["Accuracy"] = accuracy_score(
                            y_test,
                            predictions
                        )
                    if "Mean Squared Error" in selected_metrics:
                        results["Mean Squared Error"] = mean_squared_error(
                            y_test,
                            predictions
                        )
                    if "Precision" in selected_metrics:
                        results["Precision"] = precision_score(
                            y_test,
                            predictions,
                            average="weighted"
                        )
                    if "Recall" in selected_metrics:
                        results["Recall"] = recall_score(
                            y_test,
                            predictions,
                            average="weighted"
                        )

                    # Display Results
                    st.write("Pipeline Results:")
                    st.json(results)
                    st.success("Training complete!")

                    # Save pipeline configuration in session state
                    st.session_state.pipeline_config = {
                        "dataset": selected_dataset,
                        "input_features": input_features,
                        "target_feature": target_feature,
                        "model": selected_model,
                        "trained_model": pickle.dumps(model).hex(),
                        "model_params": model.get_params(),
                        "metrics": selected_metrics,
                        "split": {
                            "train_size": len(train_df) / len(dataset_df),
                            "test_size": len(test_df) / len(dataset_df)
                        },
                    }
                    st.success("Pipeline configuration saved successfully!")
                else:
                    st.error(
                        "Failed to train the model. "
                        "Please check your configuration."
                    )

            # Step 7: Save the Pipeline
            st.write("## Step 7: Save the Pipeline")

            # Check if pipeline configuration exists
            if (
                "pipeline_config" not in st.session_state
                or not st.session_state.pipeline_config
            ):
                st.error(
                    "No pipeline configuration found. "
                    "Please complete the previous steps."
                )
                st.stop()

            # Prompt for pipeline name and version
            pipeline_name = st.text_input(
                "Pipeline Name",
                value=f"{st.session_state.pipeline_config['model']}_pipeline"
            )
            pipeline_version = st.text_input("Pipeline Version", value="1.0.0")

            # Save Pipeline Button
            if st.button("Save Pipeline"):
                try:
                    pipeline_config = st.session_state.pipeline_config

                    # Validate the pipeline configuration
                    if (
                        "trained_model" not in pipeline_config
                        or not pipeline_config["trained_model"]
                    ):
                        st.error(
                            "No trained model found. "
                            "Please train the model before saving."
                        )
                        st.stop()

                    # Serialize the pipeline configuration
                    pipeline_data = json.dumps(pipeline_config).encode("utf-8")

                    # Dynamically generate the asset path for the pipeline
                    asset_path = os.path.join(
                        "./pipelines",
                        f"{pipeline_name}_v{pipeline_version}.json"
                    )
                    os.makedirs(os.path.dirname(asset_path), exist_ok=True)
                    # Ensure directory exists

                    # Create artifact
                    pipeline_artifact = Artifact(
                        name=pipeline_name,
                        type="pipeline",
                        data=pipeline_data,
                        asset_path=asset_path,  # Set asset_path correctly
                        metadata={"version": pipeline_version},
                    )

                    # Save artifact in registry
                    automl.registry.register(pipeline_artifact)
                    st.success(
                        f"Pipeline '{pipeline_name}' (v{pipeline_version}) "
                        f"saved successfully at {asset_path}!"
                    )

                except Exception as e:
                    st.error(f"Error saving pipeline: {e}")
