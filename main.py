import os
from src.pipeline.training_pipeline import TrainingPipeline

MODEL_PATH = "artifacts/model.pkl"

if __name__ == "__main__":

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    else:
        print("Model already exists. Skipping training.")