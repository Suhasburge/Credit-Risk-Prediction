import os
import pickle
import mlflow
import mlflow.pyfunc

MODEL_NAME = "credit-risk-model"
MODEL_ALIAS = "production"


def running_inside_docker():
    """Detect if app is running inside Docker container"""
    return os.path.exists("/.dockerenv")


def load_production_model():

    # ---------------- DOCKER MODE ----------------
    if running_inside_docker():
        print("Docker environment detected → loading packaged model")

        local_model_path = os.path.join("artifacts", "model.pkl")
        with open(local_model_path, "rb") as f:
            model = pickle.load(f)

        return model

    # ---------------- DEV MODE (MLflow) ----------------
    try:
        print("Local environment detected → loading from MLflow")

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        model = mlflow.pyfunc.load_model(model_uri)
        return model

    except Exception:
        print("MLflow unavailable → fallback to local model")

        local_model_path = os.path.join("artifacts", "model.pkl")
        with open(local_model_path, "rb") as f:
            model = pickle.load(f)

        return model