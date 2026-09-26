# SuperKart MLOps Sales Forecasting

Automated MLOps project for SuperKart sales forecasting.

## Public links

- GitHub: https://github.com/manju-greatlearning/superkart
- Hugging Face Dataset: https://huggingface.co/datasets/manjunathans/superkart-sales-data
- Hugging Face Model: https://huggingface.co/manjunathans/superkart-sales-model
- Hugging Face Space: https://huggingface.co/spaces/manjunathans/superkart-sales-app

## Structure

- `data/`: raw/clean/train/test data artifacts
- `src/train.py`: model training, tuning, evaluation, and model artifact generation
- `src/upload_to_hf.py`: Hugging Face dataset/model/Gradio Space registration helper
- `app/app.py`: Gradio prediction frontend
- `.github/workflows/pipeline.yml`: GitHub Actions CI/CD workflow

## Required GitHub Secrets

- `HF_TOKEN`
- `HF_DATASET_REPO_ID=manjunathans/superkart-sales-data`
- `HF_MODEL_REPO_ID=manjunathans/superkart-sales-model`
- `HF_SPACE_REPO_ID=manjunathans/superkart-sales-app`
