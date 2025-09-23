# F25-PMLDL-Assignment1

## CI Pipeline (GitHub Actions)
- A workflow runs on manual dispatch to automate:
  - Data processing split into `data/processed/`
  - Model training with fast CI overrides (`EPOCHS=1`, sample caps)
  - Docker build of API and App images
  - Compose up and API health checks (`/health`, `/labels`)
- Workflow file: `.github/workflows/pipeline.yml`

### Trigger manually
- Go to GitHub → Actions → "MLOps Pipeline" → Run workflow.

### Notes
- CI uses small sample caps and 1 epoch for speed. Local training uses defaults.
- Built images are not pushed; compose runs locally in the CI runner.