# Intelligent Groundwater Quality Assessment System

Software-only groundwater quality screening application using CGWB Uttar Pradesh groundwater data, BIS-based screening rules, Random Forest, Gradio and SQLite.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Render

- Build command: `pip install -r requirements.txt`
- Start command: `python app.py`
- Instance: Free

The app downloads the official CGWB Uttar Pradesh CSV on first startup if the model file is not already present, trains the Random Forest, saves the model, and starts the Gradio web application.

Note: Render's free filesystem is not persistent storage. SQLite history may reset after service replacement/redeployment. The public URL remains the same.
