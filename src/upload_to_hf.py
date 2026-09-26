import os
from pathlib import Path
from huggingface_hub import HfApi, upload_file, upload_folder

HF_TOKEN = os.environ["HF_TOKEN"]
DATASET_REPO_ID = os.environ.get("HF_DATASET_REPO_ID", "manjunathans/superkart-sales-data")
MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "manjunathans/superkart-sales-model")
SPACE_REPO_ID = os.environ.get("HF_SPACE_REPO_ID", "manjunathans/superkart-sales-app")

api = HfApi(token=HF_TOKEN)
api.create_repo(DATASET_REPO_ID, repo_type="dataset", exist_ok=True)
api.create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)
api.create_repo(SPACE_REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)

upload_folder(repo_id=DATASET_REPO_ID, repo_type="dataset", folder_path="data", token=HF_TOKEN)
upload_folder(repo_id=MODEL_REPO_ID, repo_type="model", folder_path="models", token=HF_TOKEN)

upload_file(repo_id=SPACE_REPO_ID, repo_type="space", path_or_fileobj="app/app.py", path_in_repo="app.py", token=HF_TOKEN)
upload_file(repo_id=SPACE_REPO_ID, repo_type="space", path_or_fileobj="app/README.md", path_in_repo="README.md", token=HF_TOKEN)
upload_file(repo_id=SPACE_REPO_ID, repo_type="space", path_or_fileobj="requirements.txt", path_in_repo="requirements.txt", token=HF_TOKEN)
