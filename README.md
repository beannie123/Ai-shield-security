# AIShield — Adversarial AI Security Lab (Enhanced)

An educational FastAPI + PyTorch web application for testing image classifiers against adversarial perturbations.

## Features
- MNIST digit and GTSRB traffic-sign classification.
- FGSM and PGD attack experiments with adjustable epsilon and PGD steps.
- Baseline vs adversarially trained CNNs.
- Original/result image previews and a difference heatmap.
- Same-upload comparison of baseline and defended models.
- Benchmark dashboard based on measured labelled test subsets.
- Experimental security score derived transparently from measured defended-model accuracy.
- Downloadable PDF report for a selected image experiment.
- JSON API endpoints for status, benchmarks, analysis, comparison, score, and PDF report.

## Important limitations
- No trained model weights or fabricated metrics are bundled. You must train the four models first.
- The experimental score is a simple educational composite, not a certified security rating.
- Uploaded images have no verified ground-truth label. The attack uses the model's clean prediction as a pseudo-label, so a changed prediction is not a dataset attack-success rate.
- The difference heatmap visualizes pixel changes; it does not prove that the input is malicious.
- This is a research/college prototype, not certified safety software or a production vehicle-control system.
- Do not load untrusted model files into your server. Custom model uploads are not enabled in this version; the API is structured for adding trusted model adapters safely.

## Setup (Windows)
1. Install Python 3.10 or 3.11.
2. Open this folder in CMD or VS Code terminal.
3. Create and activate a virtual environment:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bat
   pip install -r requirements.txt
   ```
5. Train and evaluate models:
   ```bat
   python train_models.py
   ```
   Quick smoke test (limited training; not meaningful research metrics):
   ```bat
   set TRAIN_LIMIT=2000
   set TEST_LIMIT=500
   set EPOCHS=1
   set ADV_EPOCHS=1
   python train_models.py
   ```
6. Start the website:
   ```bat
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```
7. Open http://127.0.0.1:8000 and keep the terminal running.

## API endpoints
- `GET /api/status`
- `GET /api/benchmark`
- `GET /api/security-score`
- `POST /api/analyze` (multipart: image, task, variant, attack, epsilon, steps)
- `POST /api/compare` (multipart: image, task, attack, epsilon, steps)
- `POST /api/report` (multipart: image, task, variant, attack, epsilon, steps; returns PDF)

## Deployment
A `render.yaml` is included. Train the models and ensure model weights plus benchmark results are available in the deployed environment. Check host memory, disk, request limits, and cold-start behavior. A small/free web instance may not have enough resources for PyTorch. Use HTTPS and restrict access if this is extended beyond a demo.
