import os
from huggingface_hub import HfApi, upload_folder

HF_TOKEN = os.environ["HF_TOKEN"]
DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID", "YOUR_HF_USERNAME/superkart-sales-data")
MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "YOUR_HF_USERNAME/superkart-sales-model")
SPACE_REPO_ID = os.environ.get("HF_SPACE_REPO_ID", "YOUR_HF_USERNAME/superkart-sales-app")

api = HfApi(token=HF_TOKEN)
api.create_repo(DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
api.create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)
api.create_repo(SPACE_REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)

upload_folder(repo_id=DATASET_REPO_ID, repo_type="dataset", folder_path="data", token=HF_TOKEN)
upload_folder(repo_id=MODEL_REPO_ID, repo_type="model", folder_path="models", token=HF_TOKEN)
upload_folder(repo_id=SPACE_REPO_ID, repo_type="space", folder_path="app", token=HF_TOKEN)
upload_folder(repo_id=SPACE_REPO_ID, repo_type="space", folder_path=".", allow_patterns=["Dockerfile", "requirements.txt"], token=HF_TOKEN)
