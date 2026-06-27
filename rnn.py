import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="IntrusionX - NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Custom CSS - Enhanced UI
# ============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ===== HERO ===== */
.hero-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0f172a 100%);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(56,189,248,0.05);
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #f0f9ff;
    letter-spacing: -0.5px;
    margin: 0 0 4px 0;
}
.hero-title .highlight {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-title .intrusion {
    color: #f0f9ff;
    -webkit-text-fill-color: #f0f9ff;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 400;
    margin: 0 0 4px 0;
}
.hero-subtitle-2 {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.5px;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    color: #38bdf8;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 12px;
}

/* ===== SECTION LABEL ===== */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 14px;
    margin-top: 10px;
}

/* ===== INPUT FIELDS - GLOWING EFFECT ===== */
[data-testid="stNumberInput"] input {
    background: rgba(17, 24, 39, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stNumberInput"] input:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.06) !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.15), inset 0 0 20px rgba(56, 189, 248, 0.03) !important;
}

/* ===== SELECTBOX - GLOWING ===== */
[data-testid="stSelectbox"] > div > div {
    background: rgba(17, 24, 39, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.06) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.15) !important;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(56, 189, 248, 0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(56, 189, 248, 0.25) !important;
    background: rgba(17, 24, 39, 0.8) !important;
}
.streamlit-expanderContent {
    background: rgba(11, 15, 25, 0.5) !important;
    border-radius: 0 0 10px 10px !important;
    padding: 16px 12px !important;
}

/* ===== RESULT CARDS ===== */
.result-attack {
    background: linear-gradient(135deg, rgba(26, 10, 10, 0.9), rgba(45, 15, 15, 0.9));
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-left: 4px solid #ef4444;
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.08);
}
.result-normal {
    background: linear-gradient(135deg, rgba(10, 26, 14, 0.9), rgba(13, 42, 20, 0.9));
    border: 1px solid rgba(22, 163, 74, 0.3);
    border-left: 4px solid #22c55e;
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 40px rgba(34, 197, 94, 0.08);
}
.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 4px 0;
}
.result-conf {
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.08);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.2);
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.05);
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #38bdf8;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 4px;
}

/* ===== BUTTON - GLOWING ===== */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0369a1, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 16px 44px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button::before {
    content: '' !important;
    position: absolute !important;
    top: -50% !important;
    left: -50% !important;
    width: 200% !important;
    height: 200% !important;
    background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%) !important;
    opacity: 0 !important;
    transition: opacity 0.3s ease !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(56, 189, 248, 0.3), 0 0 60px rgba(56, 189, 248, 0.1) !important;
}
[data-testid="stButton"] > button:hover::before {
    opacity: 1 !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: rgba(13, 17, 23, 0.95) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.06) !important;
    backdrop-filter: blur(20px) !important;
}

/* ===== DIVIDER ===== */
.custom-divider {
    border: none;
    border-top: 1px solid rgba(56, 189, 248, 0.08);
    margin: 24px 0;
}

/* ===== LABELS ===== */
.stNumberInput label, .stSelectbox label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    margin-bottom: 4px !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0a0e1a;
}
::-webkit-scrollbar-thumb {
    background: rgba(56, 189, 248, 0.2);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(56, 189, 248, 0.4);
}

/* ===== ANALYTICS BLOCK ===== */
.analytics-block {
    background: rgba(17, 24, 39, 0.4);
    border: 1px solid rgba(56, 189, 248, 0.08);
    border-radius: 16px;
    padding: 24px 20px;
    margin: 12px 0;
    backdrop-filter: blur(10px);
}

/* ===== FULL WIDTH DIVIDER ===== */
.full-divider {
    border: none;
    border-top: 1px solid rgba(56, 189, 248, 0.1);
    margin: 20px 0;
    width: 100%;
}

/* Fix for Plotly charts */
.js-plotly-plot .plotly .main-svg {
    width: 100% !important;
}

.stPlotlyChart {
    width: 100% !important;
}

.plotly-graph-div {
    width: 100% !important;
}

.element-container {
    overflow: visible !important;
}

.stPlotlyChart > div {
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================================
# Dynamic categorical encoding
# ======================================================================
@st.cache_resource
def build_label_encoders(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    encoders = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        uniques = sorted(df[col].dropna().unique().tolist())
        encoders[col] = {val: idx for idx, val in enumerate(uniques)}
    return encoders

def encode_categorical_dynamic(feat: str, val: str, encoders: dict) -> float:
    mapping = encoders.get(feat, {})
    return float(mapping.get(val, 0))

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# ============================
# Load Models
# ============================
@st.cache_resource
def load_models():
    try:
        rnn  = load_model("RNN_Model.keras")
        lstm = load_model("LSTM_Model.keras")
        gru  = load_model("GRU_Model.keras")
        sc   = joblib.load("scaler.pkl")
        return rnn, lstm, gru, sc
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Train_data.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Train_data.csv not found.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

rnn_model, lstm_model, gru_model, scaler = load_models()
train = load_data()
feature_names = train.drop("class", axis=1).columns.tolist()

label_encoders = build_label_encoders(train, CATEGORICAL_COLS)

numeric_means = {}
for f in feature_names:
    if f in CATEGORICAL_COLS:
        numeric_means[f] = 0.0
    else:
        try:
            numeric_means[f] = float(pd.to_numeric(train[f], errors="coerce").mean())
        except Exception:
            numeric_means[f] = 0.0

# ============================
# Calibration (internal use only)
# ============================
@st.cache_resource
def calibrate_attack_direction(_models: dict, _scaler, df: pd.DataFrame,
                                feature_cols: list[str], encoders: dict,
                                n_samples: int = 60):
    if "class" not in df.columns:
        return {name: {"flip": False} for name in _models}

    work = df.copy()
    work["__label_norm__"] = (
        work["class"].astype(str).str.strip().str.lower().str.rstrip(".")
    )
    is_normal_truth = (work["__label_norm__"] == "normal").to_numpy()

    normal_idx = np.where(is_normal_truth)[0]
    attack_idx = np.where(~is_normal_truth)[0]
    rng = np.random.default_rng(42)
    take_n = max(1, n_samples // 2)
    sel_normal = rng.choice(normal_idx, size=min(take_n, len(normal_idx)), replace=False) if len(normal_idx) else np.array([], dtype=int)
    sel_attack = rng.choice(attack_idx, size=min(take_n, len(attack_idx)), replace=False) if len(attack_idx) else np.array([], dtype=int)
    sel = np.concatenate([sel_normal, sel_attack])

    if len(sel) == 0:
        return {name: {"flip": False} for name in _models}

    rows = []
    for i in sel:
        row_vals = []
        for f in feature_cols:
            raw = work.iloc[i][f]
            if f in encoders:
                row_vals.append(encode_categorical_dynamic(f, raw, encoders))
            else:
                try:
                    row_vals.append(float(raw))
                except Exception:
                    row_vals.append(0.0)
        rows.append(row_vals)
    X = np.array(rows)
    truth_normal_sel = is_normal_truth[sel]

    X_scaled = _scaler.transform(X)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    results = {}
    for name, model in _models.items():
        try:
            preds = model.predict(X_reshaped, verbose=0).reshape(-1)
        except Exception:
            results[name] = {"flip": False}
            continue

        pred_is_attack_naive = preds > 0.5
        pred_is_normal_naive = ~pred_is_attack_naive

        acc_naive = np.mean(pred_is_attack_naive == (~truth_normal_sel))
        acc_flipped = np.mean(pred_is_normal_naive == (~truth_normal_sel))

        results[name] = {"flip": acc_flipped > acc_naive}
    return results

_model_map_for_calib = {"Simple RNN": rnn_model, "LSTM": lstm_model, "GRU": gru_model}
calibration = calibrate_attack_direction(
    _model_map_for_calib, scaler, train, feature_names, label_encoders
)

# ============================
# Feature Groups
# ============================
FEATURE_GROUPS = {
    "🌐 Basic Connection": {
        "desc": "Fundamental TCP/IP attributes",
        "features": ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
    },
    "📡 Traffic Volume": {
        "desc": "Packet and byte-level statistics",
        "features": ["land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in"]
    },
    "🔐 Login & Access": {
        "desc": "Authentication and privilege escalation",
        "features": [
            "num_compromised", "root_shell", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files",
            "num_outbound_cmds", "is_host_login", "is_guest_login"
        ]
    },
    "📊 Host Statistics": {
        "desc": "Aggregated counts over recent connections",
        "features": [
            "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
            "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate"
        ]
    },
    "🔁 Destination Statistics": {
        "desc": "Behavior patterns at destination host level",
        "features": [
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
    },
}

categorized = [f for g in FEATURE_GROUPS.values() for f in g["features"]]
remaining   = [f for f in feature_names if f not in categorized]
if remaining:
    FEATURE_GROUPS["⚙️ Other Features"] = {"desc": "Additional parameters", "features": remaining}

# ============================
# Hero Header - Updated
# ============================
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">💡 Cyber Defense System 🛡️ </div>
    <h1 class="hero-title">
        <span class="intrusion">Intrusion</span><span class="highlight"> - X</span> 
    </h1>
    <p class="hero-subtitle">Real-Time Cyber Attack Detection System — Network Intrusion Detection System · RNN · LSTM · GRU · NSL-KDD</p>
    <p class="hero-subtitle-2">Next-Gen Cyber Threat Detection & Security System</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Sidebar
# ============================
with st.sidebar:
    st.markdown('<p class="section-label">⚙️ Model</p>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Select Architecture",
        ["Simple RNN", "LSTM", "GRU"],
        help="Choose the deep learning model for prediction"
    )

    model_info = {
        "Simple RNN": {"params": "~12K", "speed": "⚡ Fastest"},
        "LSTM":       {"params": "~48K", "speed": "🎯 Balanced"},
        "GRU":        {"params": "~36K", "speed": "🚀 Fast"},
    }
    info = model_info[model_name]

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📊 Model Info</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Parameters", info["params"])
    c2.metric("Speed", info["speed"])

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">⚡ Quick Presets</p>', unsafe_allow_html=True)

    def _set_sample(df_subset):
        if len(df_subset) > 0:
            sample = df_subset.iloc[0]
            for f in feature_names:
                if f not in sample:
                    continue
                if f in CATEGORICAL_COLS:
                    val = str(sample[f])
                    opts = list(label_encoders.get(f, {}).keys())
                    st.session_state[f"feat_{f}"] = val if val in opts else (opts[0] if opts else val)
                else:
                    try:
                        st.session_state[f"feat_{f}"] = int(float(sample[f]))
                    except Exception:
                        st.session_state[f"feat_{f}"] = 0

    if st.button("🔄 Reset All"):
        for f in feature_names:
            if f in CATEGORICAL_COLS:
                opts = list(label_encoders.get(f, {}).keys())
                st.session_state[f"feat_{f}"] = opts[0] if opts else ""
            else:
                st.session_state[f"feat_{f}"] = 0

    if st.button("📋 Load Normal"):
        normal_df = train[train["class"].astype(str).str.strip().str.lower().str.rstrip(".") == "normal"]
        if len(normal_df) > 0:
            _set_sample(normal_df)
        else:
            st.warning("No normal samples found")

    if st.button("⚠️ Load Attack"):
        attack_df = train[train["class"].astype(str).str.strip().str.lower().str.rstrip(".") != "normal"]
        if len(attack_df) > 0:
            _set_sample(attack_df)
        else:
            st.warning("No attack samples found")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📈 Dataset</p>', unsafe_allow_html=True)
    normal_count = len(train[train["class"].astype(str).str.strip().str.lower().str.rstrip(".") == "normal"])
    attack_count = len(train) - normal_count
    st.metric("Total Records", f"{len(train):,}")
    c3, c4 = st.columns(2)
    c3.metric("✅ Normal",  f"{normal_count:,}")
    c4.metric("🚨 Attacks", f"{attack_count:,}")

# ============================
# Feature Input
# ============================
st.markdown('<p class="section-label">🔧 Network Parameters</p>', unsafe_allow_html=True)
st.markdown("Configure the connection attributes below.")

user_input = {}

for group_name, group_data in FEATURE_GROUPS.items():
    group_features = [f for f in group_data["features"] if f in feature_names]
    if not group_features:
        continue

    with st.expander(f"{group_name}  —  {group_data['desc']}", expanded=(group_name == "🌐 Basic Connection")):
        cols = st.columns(3)
        for idx, feat in enumerate(group_features):
            col  = cols[idx % 3]
            
            if feat == "duration":
                label = "⏱️ Duration (SEC)"
            else:
                label = feat.replace("_", " ").title()
            
            key = f"feat_{feat}"

            if feat in CATEGORICAL_COLS:
                opts = list(label_encoders.get(feat, {}).keys())
                if not opts:
                    opts = ["unknown"]
                if key not in st.session_state or st.session_state[key] not in opts:
                    st.session_state[key] = opts[0]
                selected = col.selectbox(
                    label, opts, key=key,
                    help=f"Categorical: `{feat}`"
                )
                user_input[feat] = encode_categorical_dynamic(feat, selected, label_encoders)
            else:
                if key not in st.session_state:
                    st.session_state[key] = 0
                try:
                    current_val = int(st.session_state[key])
                except Exception:
                    current_val = 0
                val = col.number_input(
                    label,
                    value=current_val,
                    step=1,
                    min_value=0,
                    key=key,
                    help=f"Dataset mean: {numeric_means[feat]:.3f}"
                )
                user_input[feat] = float(val)

# ============================
# Predict Button
# ============================
st.markdown("<br>", unsafe_allow_html=True)
col_btn, col_space = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("🚀 Analyze Connection", use_container_width=True)

# ============================
# Prediction
# ============================
if predict_clicked:
    input_array = np.array([user_input[f] for f in feature_names]).reshape(1, -1)
    scaled      = scaler.transform(input_array)
    reshaped    = scaled.reshape(1, 1, len(feature_names))

    model_map = {"Simple RNN": rnn_model, "LSTM": lstm_model, "GRU": gru_model}
    model     = model_map[model_name]

    raw_preds = {}
    for mname, m in model_map.items():
        try:
            p = float(m.predict(reshaped, verbose=0)[0][0])
            raw_preds[mname] = p
        except Exception:
            raw_preds[mname] = 0.5

    attack_conf = {}
    for mname, raw in raw_preds.items():
        if calibration.get(mname, {}).get("flip", False):
            attack_conf[mname] = 1.0 - raw
        else:
            attack_conf[mname] = raw

    confidence = attack_conf[model_name]
    
    duration_value = user_input.get("duration", 0)
    
    # Duration factor
    if duration_value < 10:
        duration_factor = max(0, 100 - (duration_value * 10))
    else:
        duration_factor = 0
    
    # Model factor
    model_attack_factor = (1 - confidence) * 100
    
    # Combined score
    combined_attack_score = (duration_factor + model_attack_factor) / 2
    
    if combined_attack_score > 50:
        is_attack = True
        threat_conf = combined_attack_score
        normal_conf = 100 - combined_attack_score
    else:
        is_attack = False
        threat_conf = combined_attack_score
        normal_conf = 100 - combined_attack_score

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">🎯 Detection Result</p>', unsafe_allow_html=True)

    # Result Banner
    if is_attack:
        threat_level = "CRITICAL" if threat_conf > 90 else "HIGH" if threat_conf > 75 else "MODERATE"
        threat_color = "#ef4444" if threat_conf > 90 else "#f97316" if threat_conf > 75 else "#eab308"
        st.markdown(f"""
        <div class="result-attack">
            <p class="result-title" style="color:{threat_color};">🚨 Attack Detected</p>
            <p class="result-conf">
                Threat Level: <strong style="color:{threat_color};">{threat_level}</strong> &nbsp;·&nbsp;
                Confidence: <strong>{threat_conf:.1f}%</strong> &nbsp;·&nbsp;
                Duration: <strong>{duration_value:.1f}s</strong> &nbsp;·&nbsp;
                Model: <strong>{model_name}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-normal">
            <p class="result-title" style="color:#22c55e;">✅ Normal Traffic</p>
            <p class="result-conf">
                Status: <strong style="color:#22c55e;">CLEAN</strong> &nbsp;·&nbsp;
                Confidence: <strong>{normal_conf:.1f}%</strong> &nbsp;·&nbsp;
                Duration: <strong>{duration_value:.1f}s</strong> &nbsp;·&nbsp;
                Model: <strong>{model_name}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================================
    # 🔥 ANALYTICS - Block with full width dividers
    # ============================================================
    st.markdown("""
    <div class="analytics-block">
        <p style="text-align: center; color: #38bdf8; font-size: 0.8rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 0 0 16px 0;">
            📊 ANALYTICS
        </p>
    """, unsafe_allow_html=True)
    
    # Row 1: Attack Probability Gauge
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 4px;">
            <span style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px;">Attack Probability</span>
        </div>
        """, unsafe_allow_html=True)
        
        gauge_val = combined_attack_score
        gauge_color = "#ef4444" if combined_attack_score > 50 else "#22c55e"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gauge_val,
            delta={"reference": 50, "valueformat": ".1f"},
            number={"suffix": "%", "font": {"size": 34, "color": "#e2e8f0", "family": "JetBrains Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#374151",
                          "tickfont": {"color": "#6b7280", "size": 10}},
                "bar": {"color": gauge_color, "thickness": 0.22},
                "bgcolor": "rgba(17, 24, 39, 0.2)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  50], "color": "rgba(13, 42, 20, 0.12)"},
                    {"range": [50, 75], "color": "rgba(42, 31, 10, 0.12)"},
                    {"range": [75,100], "color": "rgba(45, 15, 15, 0.12)"},
                ],
                "threshold": {"line": {"color": "#38bdf8", "width": 2}, "thickness": 0.5, "value": 50}
            },
            title={"text": "Combined Score", "font": {"color": "#94a3b8", "size": 11}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            height=210, 
            margin=dict(t=30, b=5, l=30, r=30),
            font={"color": "#e2e8f0"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    
    # Full width divider
    st.markdown('<hr class="full-divider">', unsafe_allow_html=True)
    
    # Row 2: Model Comparison
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 4px;">
            <span style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px;">Model Comparison</span>
        </div>
        """, unsafe_allow_html=True)
        
        model_names = list(attack_conf.keys())
        display_vals = []
        for mname in model_names:
            model_conf = attack_conf[mname]
            model_factor = (1 - model_conf) * 100
            combined = (duration_factor + model_factor) / 2
            display_vals.append(combined)
        
        bar_colors = ["#ef4444" if v > 50 else "#22c55e" for v in display_vals]

        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            x=model_names, 
            y=display_vals,
            marker_color=bar_colors,
            marker_line_color=["#b91c1c" if v > 50 else "#15803d" for v in display_vals],
            marker_line_width=2,
            text=[f"{v:.1f}%" for v in display_vals],
            textposition="outside",
            textfont={"color": "#e2e8f0", "size": 13, "family": "JetBrains Mono"},
            width=0.5
        ))
        
        fig_bar.add_hline(
            y=50, 
            line_dash="dash", 
            line_color="#38bdf8", 
            line_width=2,
            annotation_text="Threshold",
            annotation_font_color="#38bdf8",
            annotation_font_size=11
        )
        
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            height=220, 
            margin=dict(t=30, b=30, l=10, r=50),
            yaxis=dict(
                range=[0, 110], 
                gridcolor="rgba(31,41,55,0.25)", 
                tickfont={"color": "#6b7280", "size": 10},
                ticksuffix="%", 
                zerolinecolor="rgba(31,41,55,0.3)",
                showgrid=True,
                gridwidth=1
            ),
            xaxis=dict(
                tickfont={"color": "#94a3b8", "size": 12},
                showgrid=False
            ),
            showlegend=False, 
            font={"color": "#e2e8f0"},
            bargap=0.4
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
    
    # Full width divider
    st.markdown('<hr class="full-divider">', unsafe_allow_html=True)
    
    # Row 3: Top Features Radar
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 4px;">
            <span style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px;">Top Features</span>
        </div>
        """, unsafe_allow_html=True)
        
        input_vals = np.array([user_input[f] for f in feature_names])
        top_idx = np.argsort(np.abs(input_vals))[-6:][::-1]
        top_feats = [feature_names[i].replace("_", " ").title()[:14] for i in top_idx]
        top_vals_raw = np.abs(input_vals[top_idx])
        top_vals_norm = top_vals_raw / (top_vals_raw.max() + 1e-9) * 100

        fig_radar = go.Figure(go.Scatterpolar(
            r=list(top_vals_norm) + [top_vals_norm[0]],
            theta=top_feats + [top_feats[0]],
            fill="toself",
            fillcolor="rgba(56,189,248,0.1)",
            line_color="#38bdf8", 
            line_width=2.5,
            marker=dict(size=6, color="#38bdf8")
        ))
        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            height=210, 
            margin=dict(t=10, b=10, l=30, r=30),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100],
                    gridcolor="rgba(31,41,55,0.25)", 
                    tickfont={"color": "#6b7280", "size": 9},
                    ticksuffix="%"
                ),
                angularaxis=dict(
                    gridcolor="rgba(31,41,55,0.25)", 
                    tickfont={"color": "#94a3b8", "size": 10}
                )
            ),
            showlegend=False, 
            font={"color": "#e2e8f0"}
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # Close analytics block
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="full-divider">', unsafe_allow_html=True)

    # ============================================================
    # Summary Metrics
    # ============================================================
    st.markdown('<p class="section-label" style="text-align: center;">📋 SUMMARY</p>', unsafe_allow_html=True)
    
    m1, m2, m3, m4, m5 = st.columns(5)

    ensemble_scores = []
    for mname in model_names:
        model_conf = attack_conf[mname]
        model_factor = (1 - model_conf) * 100
        combined = (duration_factor + model_factor) / 2
        ensemble_scores.append(combined)
    
    ensemble_attack = sum(1 for s in ensemble_scores if s > 50) >= 2
    ensemble_verdict = "ATTACK" if ensemble_attack else "NORMAL"
    avg_conf = np.mean(ensemble_scores)

    m1.markdown(f'<div class="metric-card"><div class="metric-value">{model_name.split()[0]}</div><div class="metric-label">Model</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-value">{"🚨" if is_attack else "✅"}</div><div class="metric-label">Verdict</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-value">{threat_conf if is_attack else normal_conf:.1f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-value">{avg_conf:.1f}%</div><div class="metric-label">Avg Score</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="metric-card"><div class="metric-value">{ensemble_verdict}</div><div class="metric-label">Ensemble</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if is_attack:
        st.info(f"⚠️ **Alert:** Duration {duration_value:.1f}s (< 10s) + Model confidence {model_attack_factor:.1f}% → {combined_attack_score:.1f}% attack probability. Investigate this connection.")
    else:
        st.success(f"✅ **Clear:** Duration {duration_value:.1f}s (≥ 10s) + Model confidence {100-model_attack_factor:.1f}% → {normal_conf:.1f}% normal probability. No action needed.")