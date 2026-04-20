from kfp import dsl
from kfp import compiler

# --- Component 1: Extract Data ---
@dsl.component(packages_to_install=["pandas", "google-cloud-bigquery", "db-dtypes"])
def extract_data(project_id: str, bq_table: str, dataset_out: dsl.Output[dsl.Dataset]):
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project_id)
    # E.g., 'SELECT * FROM `my_project.my_dataset.my_table`'
    query = f"""SELECT IND, RAIN, IND1, T_MAX, IND_2, T_MIN, T_MIN_G,
    T_MAX_CLEANSED, T_MAX_CLEANSED_FROM_RAW, T_MAX_FLOAT, T_MIN_CLEANSED, WIND FROM `{bq_table}`"""
    
    df = client.query(query).to_dataframe()
    # Save to the pipeline artifact path
    df.to_csv(dataset_out.path, index=False)

# --- Component 2: Train Regression Model ---
@dsl.component(packages_to_install=["pandas", "scikit-learn", "joblib"])
def train_model(
    dataset_in: dsl.Input[dsl.Dataset], 
    model_out: dsl.Output[dsl.Model], 
    metrics_out: dsl.Output[dsl.Metrics]
):
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib
    import os

    df = pd.read_csv(dataset_in.path)
    
    # Assuming the last column is the target variable for regression
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log metrics to Vertex AI
    metrics_out.log_metric("MSE", mse)
    metrics_out.log_metric("R2_Score", r2)

    # Vertex AI requires the model file to be named exactly 'model.joblib' in the GCS directory
    os.makedirs(model_out.path, exist_ok=True)
    joblib.dump(model, os.path.join(model_out.path, "model.joblib"))


# --- Component 3: Deploy to Vertex AI ---
@dsl.component(packages_to_install=["google-cloud-aiplatform"])
def deploy_model(
    model_in: dsl.Input[dsl.Model], 
    project_id: str, 
    region: str
):
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    # Upload model to Vertex AI Model Registry
    # Using pre-built Scikit-Learn container
    model = aiplatform.Model.upload(
        display_name="regression-model",
        artifact_uri=model_in.uri, # Points to the GCS directory containing model.joblib
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
    )

    # Deploy to Endpoint (creates a REST API)
    # Note: endpoints incur 24/7 costs. Adjust machine type as needed.
    endpoint = model.deploy(
        deployed_model_display_name="regression-model-endpoint",
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=1
    )
    print(f"Model deployed to endpoint: {endpoint.resource_name}")


# --- Define Pipeline ---
@dsl.pipeline(
    name="end-to-end-regression-pipeline",
    description="Extracts data, trains a regression model, and deploys it."
)
def regression_pipeline(
    project_id: str, 
    region: str, 
    bq_table: str
):
    # Step 1
    data_task = extract_data(project_id=project_id, bq_table=bq_table)
    
    # Step 2
    train_task = train_model(dataset_in=data_task.outputs["dataset_out"])
    
    # Step 3
    deploy_task = deploy_model(
        model_in=train_task.outputs["model_out"],
        project_id=project_id,
        region=region
    )