import os
from kfp import compiler
from google.cloud import aiplatform
from pipeline import regression_pipeline

# Fetch environment variables injected by Cloud Build
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
BQ_TABLE = os.environ.get("BQ_TABLE")

PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"
COMPILED_PIPELINE_PATH = "pipeline.yaml"

# 1. Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=regression_pipeline,
    package_path=COMPILED_PIPELINE_PATH
)

# 2. Submit to Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name="regression-ci-cd-run",
    template_path=COMPILED_PIPELINE_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "bq_table": BQ_TABLE
    }
)

job.submit()
print("Pipeline submitted successfully!")