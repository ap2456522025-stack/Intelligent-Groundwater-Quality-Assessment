
import os
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "groundwater_quality_model_real.pkl"
DB_FILE = "groundwater.db"
CSV_FILE = "gwq_chemical_parameter_manual_cgwb_up_1961_2025.csv"

SOURCE_URL = (
    "https://nwdp.nwic.gov.in/dataset/7bb1e7c7-bcc1-48bb-8bcd-32c19473804a/"
    "resource/5bddb48e-136f-49c9-8a35-37c75f981c66/download/"
    "gwq_chemical_parameter_manual_cgwb_up_1961_2025.csv"
)

MODEL_COLUMNS = [
    "Potential of Hydrogen (pH)",
    "Electric Conductivity (μS/cm)",
    "Chloride (mg/L)",
    "Sulphate (mg/L)",
    "Total Hardness (mgCaCO3/L)",
    "Calcium (mg/L)",
    "Magnesium (mg/L)",
    "Sodium (mg/L)",
]

UI_COLUMNS = ["pH", "EC", "Chloride", "Sulphate", "Hardness", "Calcium", "Magnesium", "Sodium"]

PARAM_LIMITS = {
    "pH": {"min": 6.5, "max": 8.5, "perm": None, "unit": ""},
    "EC": {"min": None, "max": None, "perm": None, "unit": "μS/cm"},
    "Chloride": {"min": 0, "max": 250, "perm": 1000, "unit": "mg/L"},
    "Sulphate": {"min": 0, "max": 200, "perm": 400, "unit": "mg/L"},
    "Hardness": {"min": 0, "max": 200, "perm": 600, "unit": "mg/L as CaCO₃"},
    "Calcium": {"min": 0, "max": 75, "perm": 200, "unit": "mg/L"},
    "Magnesium": {"min": 0, "max": 30, "perm": 100, "unit": "mg/L"},
    "Sodium": {"min": None, "max": None, "perm": None, "unit": "mg/L"},
}

def classify_water(row):
    poor = (
        row["Potential of Hydrogen (pH)"] < 6.5
        or row["Potential of Hydrogen (pH)"] > 8.5
        or row["Chloride (mg/L)"] > 1000
        or row["Sulphate (mg/L)"] > 400
        or row["Total Hardness (mgCaCO3/L)"] > 600
        or row["Calcium (mg/L)"] > 200
        or row["Magnesium (mg/L)"] > 100
    )
    if poor:
        return "Poor"

    moderate = (
        row["Chloride (mg/L)"] > 250
        or row["Sulphate (mg/L)"] > 200
        or row["Total Hardness (mgCaCO3/L)"] > 200
        or row["Calcium (mg/L)"] > 75
        or row["Magnesium (mg/L)"] > 30
    )
    return "Moderate" if moderate else "Good"

def load_or_train_model():
    if Path(MODEL_FILE).exists():
        return joblib.load(MODEL_FILE)

    if not Path(CSV_FILE).exists():
        import urllib.request
        print("Downloading CGWB dataset...")
        urllib.request.urlretrieve(SOURCE_URL, CSV_FILE)

    df = pd.read_csv(CSV_FILE)
    for col in MODEL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df_model = df[MODEL_COLUMNS].dropna().copy()
    df_model["Quality"] = df_model.apply(classify_water, axis=1)

    X = df_model[MODEL_COLUMNS]
    y = df_model["Quality"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    return model

model = load_or_train_model()

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ph REAL, ec REAL, chloride REAL, sulphate REAL,
    hardness REAL, calcium REAL, magnesium REAL, sodium REAL,
    quality TEXT, confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def load_history():
    return pd.read_sql_query(
        """SELECT id, ph AS pH, ec AS EC, chloride AS Chloride,
                  sulphate AS Sulphate, hardness AS Hardness,
                  calcium AS Calcium, magnesium AS Magnesium,
                  sodium AS Sodium, quality AS Quality,
                  confidence AS Confidence, created_at AS "Date & Time"
           FROM assessments ORDER BY id DESC""",
        conn,
    )

def dashboard():
    d = load_history()
    if d.empty:
        counts = pd.Series([0, 0, 0], index=["Good", "Moderate", "Poor"])
        avg = "0.00%"
    else:
        counts = d["Quality"].value_counts().reindex(
            ["Good", "Moderate", "Poor"], fill_value=0
        )
        avg = f"{d['Confidence'].mean():.2f}%"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Assessment History — Groundwater Quality")
    ax.set_xlabel("Quality Class")
    ax.set_ylabel("Number of Assessments")
    ax.set_xticklabels(["Good", "Moderate", "Poor"], rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return str(len(d)), str(counts["Good"]), str(counts["Moderate"]), str(counts["Poor"]), avg, fig, d

def parameter_status(name, value):
    p = PARAM_LIMITS[name]
    if p["min"] is None and p["max"] is None:
        return "REFERENCE", "No direct BIS limit is used for this parameter in this project."

    if name == "pH":
        if 6.5 <= value <= 8.5:
            return "GOOD", "Within the selected acceptable range (6.5–8.5)."
        return "POOR", "Outside the selected pH range."

    if value <= p["max"]:
        return "GOOD", f"Within the selected acceptable limit (≤ {p['max']})."
    if p["perm"] is not None and value <= p["perm"]:
        return "MODERATE", f"Above acceptable limit ({p['max']}) but within permissible limit ({p['perm']})."
    return "POOR", f"Above the selected permissible limit ({p['perm']})."

def status_html(name, value):
    status, reason = parameter_status(name, value)
    colors = {"GOOD": "#198754", "MODERATE": "#f28c28", "POOR": "#dc3545", "REFERENCE": "#e0b100"}
    return (
        f'<div style="margin:4px 0 10px;padding:8px 12px;border-radius:8px;'
        f'background:{colors[status]};color:white;font-weight:600;">'
        f'{status} — {reason}</div>'
    )

def parameter_table(values):
    rows = []
    for name, value in zip(UI_COLUMNS, values):
        status, reason = parameter_status(name, value)
        p = PARAM_LIMITS[name]
        acceptable = (
            "No direct BIS limit used" if p["min"] is None
            else ("6.5–8.5" if name == "pH" else f"≤ {p['max']} {p['unit']}")
        )
        permissible = (
            "—" if p["perm"] is None else f"{p['perm']} {p['unit']}"
        )
        rows.append((name, value, p["unit"], acceptable, permissible, status, reason))

    css = {
        "GOOD": ("#d1e7dd", "#0f5132"),
        "MODERATE": ("#ffe5cc", "#8a4b08"),
        "POOR": ("#f8d7da", "#842029"),
        "REFERENCE": ("#fff3cd", "#664d03"),
    }

    html = """
    <div style="overflow-x:auto">
    <table style="width:100%;border-collapse:collapse;font-size:14px">
    <tr style="background:#16324F;color:white">
      <th style="padding:9px">Parameter</th><th>Actual</th><th>Unit</th>
      <th>Acceptable</th><th>Permissible</th><th>Status</th><th>Reason</th>
    </tr>
    """
    for r in rows:
        bg, fg = css[r[5]]
        html += (
            f'<tr style="border-bottom:1px solid #ddd;background:{bg};color:{fg}">'
            f'<td style="padding:8px;font-weight:600">{r[0]}</td>'
            f'<td>{r[1]:.2f}</td><td>{r[2]}</td><td>{r[3]}</td>'
            f'<td>{r[4]}</td><td style="font-weight:700">{r[5]}</td><td>{r[6]}</td></tr>'
        )
    html += "</table></div>"
    return html

def assess_water(pH, EC, Chloride, Sulphate, Hardness, Calcium, Magnesium, Sodium):
    values = [pH, EC, Chloride, Sulphate, Hardness, Calcium, Magnesium, Sodium]
    if any(v is None for v in values):
        return ("Error", "—", "Please enter all 8 parameters.", "", *dashboard_outputs())

    x = pd.DataFrame([values], columns=MODEL_COLUMNS)
    prediction = str(model.predict(x)[0])
    confidence = float(np.max(model.predict_proba(x)[0]) * 100)

    lines = [status_html(n, float(v)) for n, v in zip(UI_COLUMNS, values)]
    table = parameter_table(values)

    attention = [
        n for n, v in zip(UI_COLUMNS, values)
        if parameter_status(n, float(v))[0] in ("MODERATE", "POOR")
    ]
    if attention:
        recommendation = "Parameters needing attention: " + ", ".join(attention) + "."
    else:
        recommendation = "All parameters with an applicable BIS limit are within the selected acceptable limits."

    conn.execute(
        """INSERT INTO assessments
        (ph,ec,chloride,sulphate,hardness,calcium,magnesium,sodium,quality,confidence)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        tuple(map(float, values)) + (prediction, round(confidence, 2)),
    )
    conn.commit()

    stats = dashboard()
    return (
        prediction, f"{confidence:.2f}%", recommendation, table,
        *lines, stats[6], *stats[:5], stats[5]
    )

def dashboard_outputs():
    s = dashboard()
    return (s[6], *s[:5], s[5])

def clear_history():
    conn.execute("DELETE FROM assessments")
    conn.commit()
    s = dashboard()
    return ("", *[""] * 8, s[6], *s[:5], s[5])

def refresh_dashboard():
    s = dashboard()
    return (s[6], *s[:5], s[5])

initial = dashboard()

with gr.Blocks(title="Intelligent Groundwater Quality Assessment System") as demo:
    gr.Markdown(
        "# 💧 Intelligent Groundwater Quality Assessment System\n"
        "### CGWB Uttar Pradesh groundwater data + Random Forest"
    )
    gr.Markdown(
        "Enter the eight groundwater parameters and click **Assess Water Quality**.\n\n"
        "🟢 Good &nbsp;&nbsp; 🟠 Moderate &nbsp;&nbsp; 🔴 Poor &nbsp;&nbsp; 🟡 Reference"
    )

    with gr.Row():
        with gr.Column():
            pH = gr.Number(label="pH", value=7.2)
            pH_status = gr.HTML()
            EC = gr.Number(label="Electrical Conductivity (μS/cm)", value=660)
            EC_status = gr.HTML()
            Chloride = gr.Number(label="Chloride (mg/L)", value=28.36)
            Chloride_status = gr.HTML()
            Sulphate = gr.Number(label="Sulphate (mg/L)", value=15)
            Sulphate_status = gr.HTML()
            Hardness = gr.Number(label="Total Hardness (mgCaCO3/L)", value=230)
            Hardness_status = gr.HTML()
            Calcium = gr.Number(label="Calcium (mg/L)", value=36)
            Calcium_status = gr.HTML()
            Magnesium = gr.Number(label="Magnesium (mg/L)", value=34.07)
            Magnesium_status = gr.HTML()
            Sodium = gr.Number(label="Sodium (mg/L)", value=46)
            Sodium_status = gr.HTML()
            btn = gr.Button("🔍 Assess Water Quality", variant="primary")

        with gr.Column():
            quality = gr.Textbox(label="Overall Water Quality")
            confidence = gr.Textbox(label="Model Confidence")
            recommendation = gr.Textbox(label="Recommendation", lines=4)

    gr.Markdown("## 🔬 Parameter-wise Color-Coded Analysis")
    parameter_html = gr.HTML("Enter the 8 values and click **Assess Water Quality**.")

    gr.Markdown("## 📋 Assessment History")
    history = gr.Dataframe(value=initial[6], interactive=False)

    with gr.Row():
        clear = gr.Button("🗑️ Clear History")
        refresh = gr.Button("🔄 Refresh Dashboard")

    gr.Markdown("## 📊 Live Dashboard")
    with gr.Row():
        total = gr.Textbox(label="Total Assessments", value=initial[0], interactive=False)
        good = gr.Textbox(label="Good", value=initial[1], interactive=False)
        moderate = gr.Textbox(label="Moderate", value=initial[2], interactive=False)
        poor = gr.Textbox(label="Poor", value=initial[3], interactive=False)
        avg = gr.Textbox(label="Average Confidence", value=initial[4], interactive=False)

    chart = gr.Plot(value=initial[5], label="Quality Distribution")

    outputs = [
        quality, confidence, recommendation, parameter_html,
        pH_status, EC_status, Chloride_status, Sulphate_status,
        Hardness_status, Calcium_status, Magnesium_status, Sodium_status,
        history, total, good, moderate, poor, avg, chart
    ]

    btn.click(
        assess_water,
        [pH, EC, Chloride, Sulphate, Hardness, Calcium, Magnesium, Sodium],
        outputs,
    )

    clear.click(
        clear_history,
        outputs=[
            parameter_html, pH_status, EC_status, Chloride_status,
            Sulphate_status, Hardness_status, Calcium_status,
            Magnesium_status, Sodium_status, history, total, good,
            moderate, poor, avg, chart
        ],
    )

    refresh.click(
        refresh_dashboard,
        outputs=[history, total, good, moderate, poor, avg, chart],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)
