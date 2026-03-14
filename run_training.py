import mlflow
from src.pipeline.training_pipeline import TrainingPipeline

# connect to K8s MLflow (via port-forward)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

print("Starting training to Kubernetes MLflow...")

pipeline = TrainingPipeline()
pipeline.run_pipeline()

print("Training completed and model registered!")