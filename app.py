
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
    <tr style="background:#16324F!important;color:white!important">
      <th style="padding:9px;background:#16324F!important;color:#ffffff!important">Parameter</th><th style="background:#16324F!important;color:#ffffff!important">Actual</th><th style="background:#16324F!important;color:#ffffff!important">Unit</th>
      <th style="background:#16324F!important;color:#ffffff!important">Acceptable</th><th style="background:#16324F!important;color:#ffffff!important">Permissible</th><th style="background:#16324F!important;color:#ffffff!important">Status</th><th style="background:#16324F!important;color:#ffffff!important">Reason</th>
    </tr>
    """
    for r in rows:
        bg, fg = css[r[5]]
        html += (
            f'<tr style="border-bottom:1px solid #ddd;background:{bg}!important;color:{fg}!important">'
            f'<td style="padding:8px;font-weight:700;background:{bg}!important;color:{fg}!important">{r[0]}</td>'
            f'<td style="background:{bg}!important;color:{fg}!important">{r[1]:.2f}</td><td style="background:{bg}!important;color:{fg}!important">{r[2]}</td><td style="background:{bg}!important;color:{fg}!important">{r[3]}</td>'
            f'<td style="background:{bg}!important;color:{fg}!important">{r[4]}</td><td style="font-weight:900;background:{bg}!important;color:{fg}!important">{r[5]}</td><td style="background:{bg}!important;color:{fg}!important">{r[6]}</td></tr>'
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

CUSTOM_CSS = """
/* =========================
   3D WATER GATE / INTRO
   ========================= */
#intro-gate{
  position:fixed;
  inset:0;
  z-index:99999;
  display:flex;
  align-items:center;
  justify-content:center;
  overflow:hidden;
  background:
    radial-gradient(circle at 50% 45%, rgba(41,216,255,.15), transparent 22%),
    radial-gradient(circle at 50% 75%, rgba(72,240,209,.08), transparent 32%),
    linear-gradient(145deg,#020b13 0%,#061a29 48%,#020912 100%);
  transition:opacity .9s ease, visibility .9s ease;
}
#intro-gate.gate-open{
  opacity:0;
  visibility:hidden;
  pointer-events:none;
}
#intro-gate .gate-scene{
  position:relative;
  width:min(900px,92vw);
  height:min(620px,82vh);
  perspective:1400px;
  display:flex;
  align-items:center;
  justify-content:center;
}
#intro-gate .water-floor{
  position:absolute;
  width:78%;
  height:28%;
  bottom:7%;
  border-radius:50%;
  background:radial-gradient(ellipse,rgba(41,216,255,.20),rgba(41,216,255,.05) 48%,transparent 72%);
  filter:blur(2px);
  transform:rotateX(68deg);
  box-shadow:0 0 90px rgba(41,216,255,.15);
}
#intro-gate .orb{
  position:absolute;
  width:180px;
  height:180px;
  border-radius:50%;
  background:
    radial-gradient(circle at 32% 25%,rgba(255,255,255,.75) 0 3%,rgba(126,241,255,.38) 12%,transparent 28%),
    radial-gradient(circle at 42% 42%,#168ab7 0,#0a4567 42%,#031927 78%);
  border:1px solid rgba(105,232,255,.48);
  box-shadow:
    inset -28px -28px 55px rgba(0,0,0,.52),
    inset 18px 12px 35px rgba(129,244,255,.16),
    0 0 55px rgba(41,216,255,.28);
  animation:gateOrbFloat 4.5s ease-in-out infinite;
}
#intro-gate .orb:before,
#intro-gate .orb:after{
  content:"";
  position:absolute;
  inset:18%;
  border:1px solid rgba(124,238,255,.18);
  border-radius:48% 52% 55% 45%;
  transform:rotate(28deg);
  animation:gateRing 5s linear infinite;
}
#intro-gate .orb:after{
  inset:9%;
  transform:rotate(-32deg) scaleY(.72);
  animation-duration:7s;
}
#intro-gate .door{
  position:absolute;
  top:10%;
  width:37%;
  height:74%;
  background:
    linear-gradient(90deg,rgba(14,58,82,.98),rgba(8,34,53,.96)),
    linear-gradient(180deg,rgba(255,255,255,.05),transparent);
  border:1px solid rgba(91,224,255,.34);
  box-shadow:inset 0 0 35px rgba(41,216,255,.08),0 20px 60px rgba(0,0,0,.45);
  transform-style:preserve-3d;
  transition:transform 1.35s cubic-bezier(.75,0,.2,1);
}
#intro-gate .door.left{
  left:13%;
  transform-origin:left center;
  transform:rotateY(0deg) translateZ(20px);
  border-radius:26px 7px 7px 26px;
}
#intro-gate .door.right{
  right:13%;
  transform-origin:right center;
  transform:rotateY(0deg) translateZ(20px);
  border-radius:7px 26px 26px 7px;
}
#intro-gate.opening .door.left{
  transform:rotateY(78deg) translateZ(20px);
}
#intro-gate.opening .door.right{
  transform:rotateY(-78deg) translateZ(20px);
}
#intro-gate .door:before{
  content:"";
  position:absolute;
  inset:14px;
  border:1px solid rgba(80,230,255,.16);
  border-radius:inherit;
  background:
    repeating-linear-gradient(90deg,transparent 0 38px,rgba(80,230,255,.035) 39px 40px),
    repeating-linear-gradient(0deg,transparent 0 38px,rgba(80,230,255,.025) 39px 40px);
}
#intro-gate .gate-content{
  position:absolute;
  z-index:5;
  text-align:center;
  width:min(640px,75vw);
  color:#eafcff;
  transform:translateZ(90px);
  pointer-events:none;
}
#intro-gate .gate-kicker{
  display:inline-block;
  padding:8px 14px;
  border-radius:999px;
  border:1px solid rgba(72,240,209,.32);
  background:rgba(72,240,209,.07);
  color:#83ffe7;
  font-size:11px;
  font-weight:800;
  letter-spacing:2.2px;
  text-transform:uppercase;
}
#intro-gate .gate-title{
  margin:18px 0 7px;
  font-size:clamp(28px,5vw,58px);
  line-height:1.02;
  font-weight:900;
  letter-spacing:-1.8px;
  background:linear-gradient(100deg,#fff,#76edff 45%,#62f4d8);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
}
#intro-gate .gate-subtitle{
  color:#9dbccc;
  font-size:14px;
  line-height:1.6;
}
#intro-gate .gate-enter{
  pointer-events:auto;
  margin-top:25px;
  border:0;
  border-radius:15px;
  padding:13px 25px;
  color:#03131d;
  font-weight:900;
  background:linear-gradient(135deg,#39e1ff,#55efcf);
  box-shadow:0 12px 35px rgba(41,216,255,.25),inset 0 1px 0 rgba(255,255,255,.65);
  cursor:pointer;
  transition:transform .25s ease,box-shadow .25s ease;
}
#intro-gate .gate-enter:hover{
  transform:translateY(-3px) scale(1.03);
  box-shadow:0 18px 45px rgba(41,216,255,.34);
}
#intro-gate .particles{
  position:absolute;
  inset:0;
  pointer-events:none;
}
#intro-gate .particle{
  position:absolute;
  width:5px;height:5px;border-radius:50%;
  background:rgba(105,235,255,.65);
  box-shadow:0 0 14px rgba(105,235,255,.55);
  animation:particleRise linear infinite;
}
@keyframes gateOrbFloat{
  0%,100%{transform:translateY(0) rotate(0deg)}
  50%{transform:translateY(-15px) rotate(5deg)}
}
@keyframes gateRing{to{transform:rotate(388deg)}}
@keyframes particleRise{
  from{transform:translateY(40px);opacity:0}
  15%{opacity:.8}
  85%{opacity:.55}
  to{transform:translateY(-420px);opacity:0}
}
@media (max-width:700px){
  #intro-gate .door{width:42%;height:65%}
  #intro-gate .door.left{left:7%}
  #intro-gate .door.right{right:7%}
  #intro-gate .orb{width:130px;height:130px}
}


:root {
 --bg:#06131f; --panel:rgba(10,31,48,.78); --cyan:#29d8ff;
 --blue:#4f8cff; --aqua:#48f0d1; --text:#eaf8ff; --muted:#8eafc3;
}
body,.gradio-container{
 background:
 radial-gradient(circle at 15% 10%,rgba(41,216,255,.16),transparent 28%),
 radial-gradient(circle at 85% 18%,rgba(79,140,255,.14),transparent 30%),
 radial-gradient(circle at 50% 100%,rgba(72,240,209,.10),transparent 32%),
 linear-gradient(135deg,#04101a,#071b2a 45%,#04101a)!important;
 color:var(--text)!important; min-height:100vh;
}
.gradio-container{max-width:1500px!important;margin:auto!important}
#hero{
 position:relative;overflow:hidden;min-height:300px;padding:42px 48px;
 border:1px solid rgba(73,226,255,.24);border-radius:32px;
 background:linear-gradient(135deg,rgba(12,42,64,.92),rgba(5,20,34,.74));
 box-shadow:0 35px 90px rgba(0,0,0,.48),inset 0 1px 0 rgba(255,255,255,.1);
}
#hero:after{
 content:"";position:absolute;width:340px;height:340px;right:-70px;top:-80px;border-radius:50%;
 background:radial-gradient(circle at 35% 30%,rgba(255,255,255,.22) 0 3%,rgba(41,216,255,.18) 14%,rgba(41,216,255,.05) 42%,transparent 70%);
 border:1px solid rgba(80,230,255,.25);
 box-shadow:inset -35px -25px 60px rgba(0,0,0,.28),0 0 80px rgba(41,216,255,.16);
 animation:floatOrb 6s ease-in-out infinite;
}
.hero-kicker{
 display:inline-block;padding:7px 14px;border-radius:999px;
 border:1px solid rgba(72,240,209,.3);background:rgba(72,240,209,.08);
 color:#82ffe7;font-size:12px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
}
.hero-title{
 margin:18px 0 8px;font-size:clamp(32px,5vw,64px);line-height:1.02;font-weight:900;
 letter-spacing:-2px;background:linear-gradient(100deg,#fff,#7ceeff 45%,#62f4d8);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-subtitle{color:#a9c9d9;font-size:17px;max-width:760px;line-height:1.6}
.hero-badges{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}
.hero-badge{
 padding:8px 13px;border-radius:12px;background:rgba(255,255,255,.045);
 border:1px solid rgba(255,255,255,.09);color:#cce8f5;
}
.section-title{margin:30px 0 14px;color:#dff8ff;font-size:23px;font-weight:850}
.glass-card{
 border:1px solid rgba(108,221,255,.17)!important;border-radius:24px!important;
 background:linear-gradient(145deg,rgba(14,45,67,.82),rgba(5,23,37,.72))!important;
 box-shadow:0 24px 55px rgba(0,0,0,.3),inset 0 1px 0 rgba(255,255,255,.075);
 backdrop-filter:blur(18px);transition:.35s;
}
.glass-card:hover{transform:translateY(-5px);border-color:rgba(72,240,209,.32)!important}
.input-card{
 padding:22px;border-radius:24px;border:1px solid rgba(100,220,255,.14);
 background:rgba(7,29,45,.68);box-shadow:inset 0 1px 0 rgba(255,255,255,.05),0 18px 45px rgba(0,0,0,.2);
}
.input-card input{
 background:rgba(2,17,28,.7)!important;color:#effcff!important;
 border:1px solid rgba(97,215,255,.16)!important;border-radius:14px!important;
 box-shadow:inset 0 3px 12px rgba(0,0,0,.24);transition:.25s;
}
.input-card input:focus{
 border-color:rgba(41,216,255,.7)!important;
 box-shadow:0 0 0 3px rgba(41,216,255,.08),0 0 25px rgba(41,216,255,.1);
}
#assess-btn{
 border:0!important;border-radius:16px!important;min-height:58px!important;
 font-size:16px!important;font-weight:850!important;
 background:linear-gradient(135deg,#19c7ed,#427dff 52%,#38e5c3)!important;
 box-shadow:0 15px 35px rgba(37,181,255,.24),inset 0 1px 0 rgba(255,255,255,.28);
 transition:.25s;
}
#assess-btn:hover{transform:translateY(-3px) scale(1.01);box-shadow:0 20px 48px rgba(37,181,255,.34)}
#result-card{
 min-height:245px;padding:26px;border-radius:26px;
 background:radial-gradient(circle at 85% 15%,rgba(41,216,255,.12),transparent 30%),linear-gradient(145deg,rgba(14,47,70,.92),rgba(5,22,36,.86));
 border:1px solid rgba(84,225,255,.18);box-shadow:0 30px 65px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.07);
}
#result-card input,#result-card textarea{
 background:rgba(2,17,28,.62)!important;color:#effcff!important;border-radius:14px!important;
 border-color:rgba(97,215,255,.13)!important;
}
#param-table{border-radius:22px;overflow:hidden;border:1px solid rgba(100,220,255,.13);box-shadow:0 20px 45px rgba(0,0,0,.22)}
#param-table table{background:#071d2b!important;color:#dcecf4!important}
#param-table th{background:linear-gradient(135deg,#0b4864,#123d5a)!important;color:#ffffff!important;padding:13px!important}
#param-table td{padding:11px!important;font-weight:600!important;text-shadow:none!important}
#param-table td:nth-child(6){font-weight:900!important;letter-spacing:.4px}
#param-table td:nth-child(7){font-weight:600!important}

.kpi{
 padding:18px 15px;min-height:95px;border-radius:20px;
 border:1px solid rgba(100,220,255,.14);background:linear-gradient(145deg,rgba(14,45,66,.78),rgba(5,22,36,.78));
 box-shadow:0 18px 38px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.06);
}
.kpi input{color:#dffaff!important;font-weight:800!important;background:transparent!important;border:0!important}
.kpi label{color:#83aabd!important}
.dashboard-plot{border-radius:24px;overflow:hidden;border:1px solid rgba(100,220,255,.13);box-shadow:0 24px 55px rgba(0,0,0,.25)}
#footer{margin-top:30px;padding:20px;text-align:center;color:#6f92a7;font-size:12px;border-top:1px solid rgba(100,220,255,.1)}
@keyframes floatOrb{0%,100%{transform:translate3d(0,0,0) rotate(0)}50%{transform:translate3d(-12px,15px,25px) rotate(5deg)}}
"""

with gr.Blocks(
    title="Intelligent Groundwater Quality Assessment",
    css=CUSTOM_CSS,
    js="""
    function() {
      const openGate = () => {
        const gate = document.getElementById("intro-gate");
        if (!gate || gate.dataset.ready === "1") return;
        gate.dataset.ready = "1";
        const enter = document.getElementById("gate-enter");
        const run = () => {
          gate.classList.add("opening");
          setTimeout(() => gate.classList.add("gate-open"), 1050);
        };
        if (enter) enter.addEventListener("click", run);
        setTimeout(run, 4200);
      };
      setTimeout(openGate, 300);
      setTimeout(openGate, 1000);
    }
    """,
    theme=gr.themes.Base(
        primary_hue="cyan", secondary_hue="blue", neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"]
    )
) as demo:

    gr.HTML("""
    <div id="intro-gate" aria-label="3D entrance">
      <div class="particles">
        <span class="particle" style="left:12%;bottom:8%;animation-duration:5s"></span>
        <span class="particle" style="left:23%;bottom:2%;animation-duration:7s;animation-delay:1s"></span>
        <span class="particle" style="left:36%;bottom:10%;animation-duration:6s;animation-delay:.5s"></span>
        <span class="particle" style="left:61%;bottom:5%;animation-duration:8s;animation-delay:1.5s"></span>
        <span class="particle" style="left:76%;bottom:12%;animation-duration:5.5s;animation-delay:.8s"></span>
        <span class="particle" style="left:88%;bottom:3%;animation-duration:7.5s;animation-delay:2s"></span>
      </div>
      <div class="gate-scene">
        <div class="water-floor"></div>
        <div class="door left"></div>
        <div class="door right"></div>
        <div class="orb"></div>
        <div class="gate-content">
          <div class="gate-kicker">Aqua Intelligence Portal</div>
          <div class="gate-title">Enter the Water<br>Intelligence Lab</div>
          <div class="gate-subtitle">Groundwater quality assessment powered by data, screening criteria and machine learning.</div>
          <button class="gate-enter" id="gate-enter" type="button">ENTER PORTAL&nbsp; →</button>
        </div>
      </div>
    </div>
    """)

    gr.HTML("""
    <section id="hero">
      <div class="hero-kicker">B.Tech CSE • Software-Only ML Platform</div>
      <div class="hero-title">Intelligent Groundwater<br>Quality Assessment</div>
      <div class="hero-subtitle">A modern groundwater screening platform using CGWB Uttar Pradesh data, BIS-based screening criteria and a Random Forest classifier.</div>
      <div class="hero-badges">
        <span class="hero-badge">💧 CGWB Data</span><span class="hero-badge">🌐 8 Chemical Parameters</span>
        <span class="hero-badge">🧠 Random Forest</span><span class="hero-badge">📊 Live Dashboard</span>
        <span class="hero-badge">🗃️ SQLite History</span>
      </div>
    </section>
    """)

    gr.Markdown("## 💧 Water Quality Assessment", elem_classes="section-title")
    with gr.Row():
        with gr.Column(scale=7, elem_classes="glass-card"):
            gr.Markdown("### Enter groundwater parameters")
            gr.Markdown("Provide the measured values below. Each parameter receives an instant screening status.")
            with gr.Row():
                with gr.Column(elem_classes="input-card"):
                    pH=gr.Number(label="pH",value=7.2); pH_status=gr.HTML()
                    EC=gr.Number(label="Electrical Conductivity (μS/cm)",value=660); EC_status=gr.HTML()
                    Chloride=gr.Number(label="Chloride (mg/L)",value=28.36); Chloride_status=gr.HTML()
                    Sulphate=gr.Number(label="Sulphate (mg/L)",value=15); Sulphate_status=gr.HTML()
                with gr.Column(elem_classes="input-card"):
                    Hardness=gr.Number(label="Total Hardness (mgCaCO₃/L)",value=230); Hardness_status=gr.HTML()
                    Calcium=gr.Number(label="Calcium (mg/L)",value=36); Calcium_status=gr.HTML()
                    Magnesium=gr.Number(label="Magnesium (mg/L)",value=34.07); Magnesium_status=gr.HTML()
                    Sodium=gr.Number(label="Sodium (mg/L)",value=46); Sodium_status=gr.HTML()
            btn=gr.Button("💧  ANALYZE WATER QUALITY",variant="primary",elem_id="assess-btn")

        with gr.Column(scale=5,elem_id="result-card"):
            gr.Markdown("### ✨ Assessment Result")
            quality=gr.Textbox(label="Overall Water Quality")
            confidence=gr.Textbox(label="Model Confidence")
            recommendation=gr.Textbox(label="Recommendation",lines=5)

    gr.Markdown("## 🔬 Parameter Intelligence",elem_classes="section-title")
    parameter_html=gr.HTML("Enter the 8 values and click **Analyze Water Quality**.",elem_id="param-table")
    gr.Markdown("## 📋 Assessment History",elem_classes="section-title")
    history=gr.Dataframe(value=initial[6],interactive=False)

    with gr.Row():
        clear=gr.Button("🗑️ Clear History")
        refresh=gr.Button("🔄 Refresh Dashboard")

    gr.Markdown("## 📊 Live Analytics",elem_classes="section-title")
    with gr.Row():
        total=gr.Textbox(label="Total Assessments",value=initial[0],interactive=False,elem_classes="kpi")
        good=gr.Textbox(label="Good",value=initial[1],interactive=False,elem_classes="kpi")
        moderate=gr.Textbox(label="Moderate",value=initial[2],interactive=False,elem_classes="kpi")
        poor=gr.Textbox(label="Poor",value=initial[3],interactive=False,elem_classes="kpi")
        avg=gr.Textbox(label="Average Confidence",value=initial[4],interactive=False,elem_classes="kpi")
    chart=gr.Plot(value=initial[5],label="Quality Distribution",elem_classes="dashboard-plot")

    gr.HTML('<div id="footer">Intelligent Groundwater Quality Assessment • CGWB Uttar Pradesh • Screening application — not laboratory or regulatory certification.</div>')

    outputs=[quality,confidence,recommendation,parameter_html,pH_status,EC_status,Chloride_status,Sulphate_status,Hardness_status,Calcium_status,Magnesium_status,Sodium_status,history,total,good,moderate,poor,avg,chart]
    btn.click(assess_water,[pH,EC,Chloride,Sulphate,Hardness,Calcium,Magnesium,Sodium],outputs)
    clear.click(clear_history,outputs=[parameter_html,pH_status,EC_status,Chloride_status,Sulphate_status,Hardness_status,Calcium_status,Magnesium_status,Sodium_status,history,total,good,moderate,poor,avg,chart])
    refresh.click(refresh_dashboard,outputs=[history,total,good,moderate,poor,avg,chart])




if __name__ == "__main__":
    port=int(os.environ.get("PORT","7860"))
    demo.launch(server_name="0.0.0.0",server_port=port)
