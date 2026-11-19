# Predictive Maintenance & RUL Prediction

Academic MLOps prototype using UCI AI4I 2020 data.

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

## Run the application

Terminal 1 — API:
```powershell
uvicorn app:app --reload
```

Terminal 2 — MLflow UI:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` for MLflow and `http://127.0.0.1:8000/docs` for API docs.

Open `frontend/index.html` in your browser.

## Drift demonstration

```powershell
python drift.py
```

## Important scientific limitation

AI4I 2020 provides machine-failure labels but does not provide actual Remaining Useful Life labels. This prototype therefore predicts a **tool-life proxy** defined as `max(250 - Tool wear [min], 0)`. It must not be presented as ground-truth engine RUL. For real RUL evaluation, use NASA C-MAPSS with a cycle-based target.

## Project components

- `train.py`: data acquisition, preprocessing, two failure classifiers, RUL-proxy regression, MLflow tracking and model saving.
- `app.py`: FastAPI inference endpoints.
- `drift.py`: simple standardized mean-shift monitoring demonstration.
- `frontend/`: browser dashboard.
- `models/`: saved model bundle and evaluation metrics.
