# SuperKart MLOps Sales Forecasting

Automated MLOps project for SuperKart sales forecasting.

## Structure

- `data/`: raw/clean/train/test data artifacts
- `src/train.py`: model training, tuning, evaluation, and model artifact generation
- `src/upload_to_hf.py`: Hugging Face dataset/model/space registration helper
- `app/app.py`: Streamlit prediction frontend
- `Dockerfile`: Hugging Face Spaces Docker deployment
- `.github/workflows/pipeline.yml`: GitHub Actions CI/CD workflow

## Required GitHub Secrets

- `HF_TOKEN`
- `HF_DATASET_REPO_ID`
- `HF_MODEL_REPO_ID`
- `HF_SPACE_REPO_ID`
