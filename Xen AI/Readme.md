# Autolabel Web App (prototype)

## Structure
- backend/ → FastAPI server
- data/labeled/ → seed images grouped by user label
- data/unlabeled/ → bulk uploads, grouped by session (timestamp)
- outputs/ → CSVs and processed results

## Run
```bash
conda activate autolabel3
cd web_app
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
uvicorn backend.main:app --reload --port 8000
python -m backend.evaluate
python -m backend.plot_eval