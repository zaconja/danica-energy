import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from energy_sankey import create_energy_sankey
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

from report_generator import PDFReport
from milp_optimizer import MILPDayAheadOptimizer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Danica Energy Optimizer PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK PREMIUM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ===== BASE DARK THEME ===== */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #070D1A !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #070D1A !important;
}
[data-testid="stHeader"] {
    background: rgba(7,13,26,0.95) !important;
    border-bottom: 1px solid rgba(0,188,212,0.1) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0F1E; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00BCD4; }

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #070D1A 100%) !important;
    border-right: 1px solid rgba(0,188,212,0.12) !important;
}
section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
section[data-testid="stSidebar"] .stRadio > div { background: transparent; padding: 0; }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex; flex-direction: column; gap: 5px;
}
section[data-testid="stSidebar"] .stRadio div[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 0;
    transition: all 0.2s;
    cursor: pointer;
    font-weight: 500;
    color: #64748B !important;
    display: flex;
    align-items: center;
}
section[data-testid="stSidebar"] .stRadio div[data-testid="stRadio"] label:hover {
    background: rgba(0,188,212,0.08);
    border-color: rgba(0,188,212,0.2);
    color: #CBD5E1 !important;
    transform: translateX(3px);
}

/* ===== PAGE HEADER ===== */
.main-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #00BCD4 60%, #1565C0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.sub-title {
    color: #475569;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* ===== METRIC CARDS – GLASSMORPHISM DARK ===== */
.metric-card {
    background: linear-gradient(135deg, rgba(13,27,55,0.9) 0%, rgba(15,23,42,0.95) 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: border-color 0.25s, transform 0.2s;
    margin-bottom: 0.8rem;
}
.metric-card:hover {
    border-color: rgba(0,188,212,0.3);
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.9px;
    color: #475569;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #F1F5F9;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}
.delta-positive { color: #4CAF50; font-weight: 600; font-size: 0.8rem; margin-top: 3px; }
.delta-negative { color: #EF5350; font-weight: 600; font-size: 0.8rem; margin-top: 3px; }

/* ===== GLASS CARD ===== */
.card {
    background: rgba(10,18,35,0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}

/* ===== PROGRESS BAR ===== */
.progress-container {
    background: rgba(255,255,255,0.07);
    border-radius: 6px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}
.progress-fill {
    height: 10px;
    border-radius: 6px;
    color: white;
    font-size: 0.7rem;
    line-height: 10px;
    text-align: right;
    padding-right: 6px;
    transition: width 0.4s ease;
}

/* ===== STATUS BADGE ===== */
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(46,125,50,0.15);
    border: 1px solid rgba(76,175,80,0.35);
    color: #4CAF50;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
}
.pulse {
    width: 7px; height: 7px;
    background: #4CAF50;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ===== SIDEBAR INFO ===== */
.sidebar-info {
    background: rgba(0,188,212,0.07);
    border-radius: 10px;
    padding: 10px 12px;
    margin: 10px 0;
    font-size: 0.82rem;
    border-left: 3px solid #00BCD4;
}
.sidebar-info p { margin: 3px 0; color: #94A3B8 !important; }
.sidebar-info strong { color: #CBD5E1 !important; }
.sidebar-footer {
    font-size: 0.75rem;
    color: #334155 !important;
    text-align: center;
    margin-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 10px;
}

/* ===== INPUTS DARK ===== */
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: rgba(10,18,35,0.9) !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: rgba(0,188,212,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,188,212,0.08) !important;
}
.stSelectbox > div > div {
    background: rgba(10,18,35,0.9) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #E2E8F0 !important;
}
.stSlider > div > div { background: #1E3A5F !important; }
.stSlider > div > div > div { background: #00BCD4 !important; }
.stCheckbox label { color: #94A3B8 !important; }

/* ===== BUTTONS DARK ===== */
.stButton > button {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1565C0 0%, #00BCD4 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0,188,212,0.25) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00838F 0%, #00BCD4 100%) !important;
    box-shadow: 0 2px 10px rgba(0,188,212,0.2) !important;
}
.stDownloadButton > button {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0A1220, #0D1A30);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.76rem !important; text-transform: uppercase; letter-spacing: 0.5px !important; }
[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.4rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.77rem !important; }

/* ===== DATA EDITOR ===== */
[data-testid="stDataEditor"] { background: #0A1220 !important; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06) !important; }

/* ===== DATAFRAME ===== */
.dataframe, [data-testid="stDataFrame"] { background: #0A1220 !important; color: #E2E8F0 !important; }

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: rgba(10,18,35,0.8) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    color: #64748B !important;
}
.streamlit-expanderContent {
    background: rgba(7,13,26,0.9) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* ===== TABS ===== */
.stTabs [role="tablist"] { background: rgba(10,18,35,0.8); border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [role="tab"] { color: #475569 !important; border-radius: 6px; font-weight: 500; }
.stTabs [role="tab"][aria-selected="true"] { 
    background: linear-gradient(135deg, #0D47A1, #1565C0) !important; 
    color: white !important; 
}

/* ===== ALERTS ===== */
.stSuccess > div { background: rgba(46,125,50,0.12) !important; border-color: #2E7D32 !important; color: #A5D6A7 !important; }
.stError > div { background: rgba(198,40,40,0.12) !important; border-color: #C62828 !important; color: #EF9A9A !important; }
.stWarning > div { background: rgba(245,124,0,0.12) !important; border-color: #F57C00 !important; color: #FFCC80 !important; }
.stInfo > div { background: rgba(21,101,192,0.12) !important; border-color: #1565C0 !important; color: #90CAF9 !important; }

/* ===== SPINNER ===== */
.stSpinner > div { border-top-color: #00BCD4 !important; }

/* ===== PROGRESS ===== */
.stProgress > div > div > div { background: linear-gradient(90deg, #0D47A1, #00BCD4) !important; }

/* ===== DIVIDER ===== */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1rem 0 !important; }

/* ===== HEADERS ===== */
h1, h2, h3 { color: #E2E8F0 !important; }
.stHeader, [data-testid="stHeader"] h1 { color: #F1F5F9 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY DARK TEMPLATE
# ============================================================
DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,18,35,0.6)',
    font=dict(family='Inter, sans-serif', color='#94A3B8', size=11),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.07)',
               color='#475569', linecolor='rgba(255,255,255,0.08)', showline=True),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.07)',
               color='#475569', linecolor='rgba(255,255,255,0.08)', showline=True),
    legend=dict(bgcolor='rgba(10,18,35,0.85)', bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1, font=dict(color='#CBD5E1', size=11)),
    hoverlabel=dict(bgcolor='#0D1A30', font_size=12, font_family='Inter',
                    bordercolor='rgba(0,188,212,0.3)'),
    margin=dict(l=50, r=25, t=50, b=45),
    colorway=['#00BCD4','#4CAF50','#FF6B35','#1565C0','#AB47BC','#FF7043','#26A69A','#FFB300'],
)

def dark_fig(fig: go.Figure, h: int = 420, title: str = None) -> go.Figure:
    upd = dict(**DARK_LAYOUT, height=h)
    if title:
        upd['title'] = dict(text=title, font=dict(size=16, color='#CBD5E1', family='Inter'), x=0.5)
    fig.update_layout(**upd)
    return fig

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_eur(x):
    if abs(x) >= 1e6: return f"{x/1e6:.1f}M €"
    elif abs(x) >= 1e3: return f"{x/1e3:.0f}k €"
    return f"{x:.0f} €"

def format_co2(x):
    if abs(x) >= 1e3: return f"{x/1e3:.1f}k tCO₂"
    return f"{x:.0f} tCO₂"

def metric_card(label, value, delta=None, delta_color="normal", suffix=""):
    if isinstance(value, (int, float)):
        val_str = f"{value:,.0f}{suffix}" if suffix else f"{value:,.0f}"
    else:
        val_str = str(value)
    delta_html = ""
    if delta is not None:
        cls = "delta-positive" if delta > 0 else "delta-negative"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div style="margin-top:5px;"><span class="{cls}">{sign}{delta:,.0f}</span></div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val_str}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def progress_bar(value, max_value, label="", color="#00BCD4"):
    pct = min(100, value / max_value * 100) if max_value > 0 else 0
    st.markdown(f"""
    <div style="margin-bottom:0.8rem;">
        <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#475569;margin-bottom:4px;">
            <span>{label}</span><span style="color:#94A3B8">{value:,.0f} / {max_value:,.0f}</span>
        </div>
        <div class="progress-container">
            <div class="progress-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if 'portfolio_contracts' not in st.session_state:
    st.session_state.portfolio_contracts = [
        {"Energija": "Tranša 1", "Količina": 5000.0, "Jedinica": "MWh", "Cijena": 59.0, "Status": "Fiksno"},
        {"Energija": "Tranša 2", "Količina": 3000.0, "Jedinica": "MWh", "Cijena": 65.0, "Status": "Fiksno"},
        {"Energija": "Plin",     "Količina": 1_500_000.0, "Jedinica": "m³","Cijena": 35.0, "Status": "Fiksno"},
    ]
if 'portfolio_biomass' not in st.session_state:
    st.session_state.portfolio_biomass = [
        {"Količina": 1000.0, "Cijena": 120.0},
        {"Količina": 1500.0, "Cijena": 115.0},
    ]
if 'co2_total' not in st.session_state:    st.session_state.co2_total = 8400.0
if 'co2_purchased' not in st.session_state: st.session_state.co2_purchased = 6500.0
if 'cropex_spot' not in st.session_state:  st.session_state.cropex_spot = 78.5
if 'ob_now' not in st.session_state:
    st.session_state.ob_now = {
        'fne_power': 1850.0, 'grid_import': 3650.0, 'grid_export': 3550.0,
        'bess_charge': 800.0, 'bess_discharge': 700.0, 'thermal_power': 220.0,
        'co2_rate': 4.02, 'plan_fne': 105000.0, 'plan_bess': 70.0,
        'gas_boiler': 90.0, 'biomass_boiler': 30.0,
        'gas_remaining': 100000.0, 'biomass_remaining': 120000.0,
    }
if 'optimizer_load' not in st.session_state:
    np.random.seed(42)
    h = np.arange(24)
    st.session_state.optimizer_load  = np.clip(80 + 40*np.sin((h-6)*np.pi/12)**2 + np.random.normal(0,5,24), 60, 145)
    st.session_state.optimizer_fne   = np.clip(60*np.sin((h-7)*np.pi/11) + np.random.normal(0,5,24), 0, None)
    st.session_state.optimizer_spot  = np.clip(np.random.normal(75, 10, 24), 40, 130)
    st.session_state.optimizer_eua   = np.clip(np.random.normal(35, 8, 24), 20, 80)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    try:
        st.image("icon.jpg", width=180)
    except:
        st.markdown("<div style='font-size:2rem;text-align:center'>⚡</div>", unsafe_allow_html=True)

    st.markdown("---")
    menu = st.radio(
        "Navigacija",
        ["📊 Pregled portfelja", "⚡ Operativna bilanca",
         "📅 Optimizacija D-1", "💰 Investicijski kalkulator", "🧩 Modularni dizajner"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    now = datetime.now()
    st.markdown(f"""
    <div class="sidebar-info">
        <p>🕐 <strong>{now.strftime('%H:%M')}</strong> · {now.strftime('%d.%m.%Y')}</p>
        <p>⚡ CROPEX: <strong>{st.session_state.cropex_spot:.1f} €/MWh</strong></p>
        <p>💨 CO₂ slobodnih: <strong>{st.session_state.co2_total - st.session_state.co2_purchased:.0f} t</strong></p>
        <p><span class="status-badge"><span class="pulse"></span> Sustav aktivan</span></p>
    </div>
    <div class="sidebar-footer">
        v7.0 · MILP Extreme Dark<br/>
        EKONERG © 2026
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     background:linear-gradient(135deg,#0A1628 0%,#0D2040 50%,#0A1628 100%);
     border:1px solid rgba(0,188,212,0.15);border-radius:14px;
     padding:1.4rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
  <div style="position:absolute;top:-60%;right:-5%;width:280px;height:280px;
       background:radial-gradient(circle,rgba(0,188,212,0.06) 0%,transparent 70%);pointer-events:none;"></div>
  <div>
    <div class="main-title">⚡ Danica Energy Optimizer PRO</div>
    <div class="sub-title">Napredna MILP optimizacija · Analiza investicija · Izvještavanje</div>
  </div>
  <div style="text-align:right;font-size:0.75rem;color:#334155;">
    {now.strftime('%A, %d. %B %Y.')}
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. PREGLED PORTFELJA
# ============================================================
if menu == "📊 Pregled portfelja":
    st.header("📊 Pregled energetskog portfelja")

    with st.expander("➕ Dodaj / Uredi ugovor", expanded=False):
        with st.form("contract_form"):
            cols = st.columns(4)
            with cols[0]: en_type = st.text_input("Energija", "Tranša 3")
            with cols[1]:
                quantity = st.number_input("Količina", min_value=0.0, value=2000.0, step=100.0)
                unit = st.selectbox("Jedinica", ["MWh", "m³", "t"])
            with cols[2]: price = st.number_input("Cijena (€/jed)", min_value=0.0, value=70.0, step=1.0)
            with cols[3]: status = st.selectbox("Status", ["Fiksno", "Indeksirano"])
            if st.form_submit_button("Dodaj ugovor", use_container_width=True):
                st.session_state.portfolio_contracts.append(
                    {"Energija": en_type, "Količina": quantity, "Jedinica": unit,
                     "Cijena": price, "Status": status})
                st.success("✅ Ugovor dodan!")
                st.rerun()

    st.subheader("📋 Ugovorene energije")
    df_contracts = pd.DataFrame(st.session_state.portfolio_contracts)
    edited_df = st.data_editor(
        df_contracts, use_container_width=True, num_rows="dynamic",
        column_config={"Količina": st.column_config.NumberColumn(format="%.0f"),
                       "Cijena": st.column_config.NumberColumn(format="%.2f")}
    )
    if not edited_df.equals(df_contracts):
        st.session_state.portfolio_contracts = edited_df.to_dict('records')
        st.rerun()

    with st.expander("🌱 Biomasa – zalihe", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Postojeće zalihe")
            df_bio = pd.DataFrame(st.session_state.portfolio_biomass)
            if not df_bio.empty: st.dataframe(df_bio, use_container_width=True)
        with col2:
            st.subheader("Dodaj")
            q_bio = st.number_input("Količina (t)", min_value=0.0, value=500.0, step=100.0, key="bio_q")
            p_bio = st.number_input("Cijena (€/t)", min_value=0.0, value=110.0, step=5.0, key="bio_p")
            if st.button("Dodaj", key="add_bio", use_container_width=True):
                st.session_state.portfolio_biomass.append({"Količina": q_bio, "Cijena": p_bio})
                st.rerun()

    with st.expander("💨 CO₂ obveze", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.co2_total = st.number_input(
                "Ukupne emisije (tCO₂)", min_value=0.0, value=st.session_state.co2_total, step=100.0)
        with col2:
            st.session_state.co2_purchased = st.number_input(
                "Otkupljeno (tCO₂)", min_value=0.0, value=st.session_state.co2_purchased, step=100.0)
        remaining = max(0, st.session_state.co2_total - st.session_state.co2_purchased)
        progress_bar(remaining, st.session_state.co2_total, "Preostala obveza", "#EF5350")

    st.subheader("📈 CROPEX Spot cijena")
    st.session_state.cropex_spot = st.number_input(
        "Današnja cijena (€/MWh)", min_value=0.0, value=st.session_state.cropex_spot, step=1.0)

    df_el = edited_df[edited_df['Jedinica'] == 'MWh'] if not edited_df.empty else pd.DataFrame()
    total_mwh  = df_el['Količina'].sum() if not df_el.empty else 0.0
    total_cost = (df_el['Količina'] * df_el['Cijena']).sum() if not df_el.empty else 0.0
    avg_price  = total_cost / total_mwh if total_mwh > 0 else 0.0
    remaining  = max(0, st.session_state.co2_total - st.session_state.co2_purchased)

    cols = st.columns(4)
    with cols[0]: metric_card("Ukupno ugovoreno", total_mwh, suffix=" MWh")
    with cols[1]: metric_card("Prosječna cijena", avg_price, suffix=" €/MWh")
    with cols[2]: metric_card("CO₂ preostalo", remaining, suffix=" tCO₂")
    with cols[3]: metric_card("CROPEX Spot", st.session_state.cropex_spot, suffix=" €/MWh")

    st.markdown("---")
    st.subheader("🔄 What-If analiza (Spot vs Fiksno)")
    col1, col2 = st.columns(2)
    with col1:
        wi_demand = st.number_input("Ukupna potrošnja (MWh)", min_value=0.0, value=10000.0, step=500.0, key="wi_d")
        wi_spot   = st.number_input("Spot cijena (€/MWh)", min_value=0.0, value=st.session_state.cropex_spot, step=1.0, key="wi_s")
    with col2:
        wi_fixed_share = st.slider("Udio fiksnog dijela", 0.0, 1.0, 0.5, 0.01)
        st.caption(f"Spot udio: {1-wi_fixed_share:.1%}")

    fixed_vol       = wi_demand * wi_fixed_share
    scale           = min(1.0, fixed_vol / total_mwh) if total_mwh > 0 else 0.0
    new_fixed_cost  = (df_el['Količina'] * df_el['Cijena'] * scale).sum() if not df_el.empty else 0.0
    new_spot_cost   = (wi_demand - fixed_vol) * wi_spot
    new_total       = new_fixed_cost + new_spot_cost
    savings         = total_cost - new_total

    cols = st.columns(4)
    cols[0].metric("Originalni trošak", format_eur(total_cost))
    cols[1].metric("Novi trošak", format_eur(new_total),
                   delta=f"{savings:,.0f} €" if savings != 0 else None, delta_color="inverse")
    cols[2].metric("Spot izloženost", f"{(wi_demand-fixed_vol)/wi_demand:.1%}" if wi_demand else "0%")
    cols[3].metric("Prosječna cijena", f"{new_total/wi_demand:.2f} €/MWh" if wi_demand else "0")

    if not df_el.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = go.Figure(go.Pie(
                labels=df_el['Energija'], values=df_el['Količina'],
                hole=0.5,
                marker=dict(colors=['#1565C0','#00BCD4','#4CAF50','#FF6B35'],
                            line=dict(color='#070D1A', width=2)),
                textfont=dict(color='#E2E8F0'),
            ))
            dark_fig(fig_pie, 340, "Udjeli u portfelju (MWh)")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            cijene = df_el['Cijena'].tolist()
            colors_b = ['#4CAF50' if c <= st.session_state.cropex_spot else '#EF5350' for c in cijene]
            fig_b = go.Figure(go.Bar(
                x=df_el['Energija'], y=cijene,
                marker_color=colors_b, text=[f"{c:.1f}" for c in cijene],
                textposition='outside', textfont=dict(color='#E2E8F0'),
            ))
            fig_b.add_hline(y=st.session_state.cropex_spot, line_dash="dash",
                            line_color="#FF6B35",
                            annotation_text=f"CROPEX: {st.session_state.cropex_spot:.1f} €/MWh",
                            annotation_font_color="#FF6B35")
            dark_fig(fig_b, 340, "Cijena tranši vs CROPEX spot")
            fig_b.update_layout(yaxis_title="€/MWh")
            st.plotly_chart(fig_b, use_container_width=True)

    # Dinamički Sankey
    st.subheader("🔀 Dinamički prikaz tokova energije")
    with st.expander("🔧 Podesi snage za eksperimentiranje", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            fne_s  = st.slider("FNE (kW)", 0, 500, int(st.session_state.ob_now['fne_power']), 5)
            load_s = st.slider("Potrošnja (kW)", 0, 500, int(st.session_state.ob_now['grid_import']), 5)
        with col2:
            bch_s  = st.slider("Punjenje baterije (kW)", 0.0, 50.0, st.session_state.ob_now['bess_charge'], 2.5)
            bdis_s = st.slider("Pražnjenje baterije (kW)", 0.0, 50.0, st.session_state.ob_now['bess_discharge'], 2.5)
    fig_sankey = create_energy_sankey(fne=fne_s, load=load_s, batt_ch=bch_s, batt_dis=bdis_s, electrolyzer=0)
    st.plotly_chart(fig_sankey, use_container_width=True)

    if st.button("📥 Preuzmi PDF izvještaj (Portfelj)", use_container_width=True):
        pdf = PDFReport("Izvještaj o energetskom portfelju")
        pdf.add_title()
        pdf.add_paragraph("Prikaz ugovorenih energija, What-If analiza spot/fiksno, CO₂ pozicija.")
        pdf.add_metric_cards({"Ukupno MWh": total_mwh, "Avg cijena €/MWh": avg_price,
                               "CO₂ preostalo t": remaining, "CROPEX €/MWh": st.session_state.cropex_spot})
        pdf.add_dataframe(edited_df, "Ugovorene energije")
        pdf_bytes = pdf.save()
        st.download_button("📄 Preuzmi PDF", pdf_bytes,
                           file_name=f"portfelj_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                           mime="application/pdf")

# ============================================================
# 2. OPERATIVNA BILANCA
# ============================================================
elif menu == "⚡ Operativna bilanca":
    st.header("⚡ Operativna energetska bilanca")
    st.markdown("##### Trenutno stanje – uredi i osvježi")

    with st.form("operativa_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.ob_now['fne_power']    = st.number_input("FNE (kW)", 0.0, value=st.session_state.ob_now['fne_power'], step=10.0)
            st.session_state.ob_now['grid_import']  = st.number_input("Iz mreže (kW)", 0.0, value=st.session_state.ob_now['grid_import'], step=10.0)
            st.session_state.ob_now['grid_export']  = st.number_input("U mrežu (kW)", 0.0, value=st.session_state.ob_now['grid_export'], step=10.0)
        with col2:
            st.session_state.ob_now['bess_charge']      = st.number_input("BESS punjenje (kW)", 0.0, value=st.session_state.ob_now['bess_charge'], step=10.0)
            st.session_state.ob_now['bess_discharge']   = st.number_input("BESS pražnjenje (kW)", 0.0, value=st.session_state.ob_now['bess_discharge'], step=10.0)
            st.session_state.ob_now['thermal_power']    = st.number_input("Toplinski sustav (kW)", 0.0, value=st.session_state.ob_now['thermal_power'], step=10.0)
        with col3:
            st.session_state.ob_now['co2_rate']  = st.number_input("CO₂ emisije (tCO₂/h)", 0.0, value=st.session_state.ob_now['co2_rate'], step=0.1, format="%.2f")
            st.session_state.ob_now['plan_fne']  = st.number_input("Plan FNE (kWh)", 0.0, value=st.session_state.ob_now['plan_fne'], step=100.0)
            st.session_state.ob_now['plan_bess'] = st.number_input("Plan BESS pražnjenje (%)", 0.0, value=st.session_state.ob_now['plan_bess'], step=1.0)
        st.form_submit_button("Ažuriraj bilancu", use_container_width=True)

    balance  = (st.session_state.ob_now['fne_power'] + st.session_state.ob_now['bess_discharge']
                - st.session_state.ob_now['bess_charge'] - st.session_state.ob_now['grid_export']
                + st.session_state.ob_now['grid_import'])
    co2_daily = st.session_state.ob_now['co2_rate'] * 24

    col1, col2, col3 = st.columns(3)
    with col1: metric_card("Stanje bilance", balance, suffix=" kW")
    with col2: metric_card("CO₂ emisije danas", co2_daily, suffix=" tCO₂")
    with col3: metric_card("FNE proizvodnja", st.session_state.ob_now['fne_power'], suffix=" kW")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("FNE (dnevno)", f"{st.session_state.ob_now['fne_power']*24:,.0f} kWh",
                  delta=f"{st.session_state.ob_now['fne_power']*24 - st.session_state.ob_now['plan_fne']:,.0f} kWh",
                  delta_color="inverse")
    with col2:
        st.metric("BESS pražnjenje", f"{st.session_state.ob_now['bess_discharge']:.0f} kW",
                  delta=f"{st.session_state.ob_now['bess_discharge'] - st.session_state.ob_now['plan_bess']:.0f} kW",
                  delta_color="inverse")

    df_power = pd.DataFrame({
        "Kategorija": ["FNE","BESS pražnjenje","Iz mreže","U mrežu","BESS punjenje","Toplina"],
        "Snaga (kW)": [
            st.session_state.ob_now['fne_power'],  st.session_state.ob_now['bess_discharge'],
            st.session_state.ob_now['grid_import'], -st.session_state.ob_now['grid_export'],
            -st.session_state.ob_now['bess_charge'], st.session_state.ob_now['thermal_power']
        ],
        "Tip": ["Proizvodnja","Proizvodnja","Proizvodnja","Potrošnja","Potrošnja","Proizvodnja"]
    })

    fig_bar = go.Figure()
    colors_bar = ['#4CAF50' if t=='Proizvodnja' else '#EF5350' for t in df_power['Tip']]
    fig_bar.add_trace(go.Bar(
        x=df_power['Kategorija'], y=df_power['Snaga (kW)'],
        marker_color=colors_bar, text=df_power['Snaga (kW)'].round(0),
        textposition='outside', textfont=dict(color='#E2E8F0'),
    ))
    dark_fig(fig_bar, 400, "Trenutni tokovi energije")
    fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="kW")
    st.plotly_chart(fig_bar, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        df_prod = df_power[df_power['Tip']=='Proizvodnja'].copy()
        df_prod['Snaga (kW)'] = df_prod['Snaga (kW)'].abs()
        fig_pie2 = go.Figure(go.Pie(
            labels=df_prod['Kategorija'], values=df_prod['Snaga (kW)'],
            hole=0.45,
            marker=dict(colors=['#4CAF50','#00BCD4','#1565C0','#FF6B35'],
                        line=dict(color='#070D1A', width=2)),
        ))
        dark_fig(fig_pie2, 320, "Udio u proizvodnji")
        st.plotly_chart(fig_pie2, use_container_width=True)

    with col_b:
        st.subheader("🔥 Toplinska energija & Zalihe")
        st.session_state.ob_now['gas_boiler']    = st.number_input("Plinski kotao (MW)", 0.0, value=st.session_state.ob_now['gas_boiler'], step=5.0)
        st.session_state.ob_now['biomass_boiler']= st.number_input("Kotao na biomasu (MW)", 0.0, value=st.session_state.ob_now['biomass_boiler'], step=5.0)
        st.session_state.ob_now['gas_remaining'] = st.number_input("Preostalo plina (m³)", 0.0, value=st.session_state.ob_now['gas_remaining'], step=1000.0)
        st.session_state.ob_now['biomass_remaining'] = st.number_input("Preostalo biomase (t)", 0.0, value=st.session_state.ob_now['biomass_remaining'], step=1000.0)
        progress_bar(st.session_state.ob_now['gas_remaining'], 200000.0, "Plin", "#FF6B35")
        progress_bar(st.session_state.ob_now['biomass_remaining'], 200000.0, "Biomasa", "#4CAF50")

    st.subheader("📈 Dnevni profil (simulacija)")
    col1, col2 = st.columns(2)
    with col1:
        peak_load  = st.slider("Max potrošnja (MWh/h)", 80.0, 200.0, 150.0)
        peak_fne   = st.slider("Max FNE (MWh/h)", 30.0, 100.0, 70.0)
    with col2:
        load_pattern = st.selectbox("Obrazac potrošnje", ["Industrijski","Uslužni","Stambeni"])
        fne_pattern  = st.selectbox("Obrazac FNE", ["Sunčano","Oblačno","Varijabilno"])

    profile_key = f"dp_{peak_load}_{load_pattern}_{peak_fne}_{fne_pattern}"
    if profile_key not in st.session_state:
        hrs = list(range(24))
        if load_pattern=="Industrijski":
            lc = 80 + 40*np.sin(np.linspace(0,2*np.pi,24)+0.5) + np.random.normal(0,5,24)
        elif load_pattern=="Uslužni":
            lc = 60 + 50*(np.sin(np.linspace(-1.5,1.5,24))**2) + np.random.normal(0,5,24)
        else:
            lc = 50 + 30*np.sin(np.linspace(0,2*np.pi,24)) + np.random.normal(0,3,24)
        lc = np.clip(lc*peak_load/100, 50, peak_load+20)

        if fne_pattern=="Sunčano":
            fc = peak_fne*np.array([0,0,0,0,0,5,30,60,85,95,100,95,85,70,50,30,15,5,0,0,0,0,0,0])/100
        elif fne_pattern=="Oblačno":
            fc = peak_fne*np.array([0,0,0,0,0,2,15,35,50,60,55,45,35,25,20,12,5,1,0,0,0,0,0,0])/100
        else:
            fc = peak_fne*np.array([0,0,0,0,0,5,30,70,90,70,50,30,80,90,60,30,10,5,0,0,0,0,0,0])/100
        fc = np.clip(fc + np.random.normal(0,2,24), 0, peak_fne)
        st.session_state[profile_key] = (hrs, lc, fc)
    else:
        hrs, lc, fc = st.session_state[profile_key]

    df_day = pd.DataFrame({'Sat': hrs, 'Potrošnja (MWh)': lc, 'FNE (MWh)': fc, 'Neto (MWh)': lc-fc})
    fig_day = go.Figure()
    fig_day.add_trace(go.Scatter(x=hrs, y=lc, name='Potrošnja', line=dict(color='#EF5350', width=2.5)))
    fig_day.add_trace(go.Scatter(x=hrs, y=fc, name='FNE', fill='tozeroy',
                                  fillcolor='rgba(76,175,80,0.15)', line=dict(color='#4CAF50', width=2.5)))
    fig_day.add_trace(go.Scatter(x=hrs, y=lc-fc, name='Neto', line=dict(color='#FF6B35', width=2, dash='dot')))
    dark_fig(fig_day, 380, "Simulirani dnevni profil")
    fig_day.update_layout(xaxis_title="Sat", yaxis_title="MWh", hovermode='x unified',
                           legend=dict(**DARK_LAYOUT['legend'], orientation='h', y=1.05, x=0.5, xanchor='center'))
    st.plotly_chart(fig_day, use_container_width=True)

    if 'df_day' in dir() or True:
        fig_hm = go.Figure(data=go.Heatmap(
            z=[df_day['Potrošnja (MWh)'].values, df_day['FNE (MWh)'].values],
            x=[f"{h}h" for h in hrs],
            y=['Potrošnja','FNE'],
            colorscale=[[0,'#0A1628'],[0.5,'#1565C0'],[1,'#00BCD4']],
            showscale=False,
        ))
        dark_fig(fig_hm, 180)
        fig_hm.update_layout(xaxis_title="Sat", margin=dict(l=70, r=20, t=20, b=30))
        st.plotly_chart(fig_hm, use_container_width=True)

    df_heat = pd.DataFrame({'Izvor': ['Plinski kotao','Biomasa'],
                             'Snaga (MW)': [st.session_state.ob_now['gas_boiler'],
                                            st.session_state.ob_now['biomass_boiler']]})

    if st.button("📥 Preuzmi PDF izvještaj (Bilanca)", use_container_width=True):
        pdf = PDFReport("Izvještaj o operativnoj bilanci")
        pdf.add_title()
        pdf.add_metric_cards({"Bilanca kW": balance, "CO₂ danas t": co2_daily,
                               "FNE kW": st.session_state.ob_now['fne_power'],
                               "BESS pražnjenje kW": st.session_state.ob_now['bess_discharge']})
        pdf.add_dataframe(df_heat, "Toplinski izvori")
        fig_pdf = go.Figure(go.Bar(x=df_power['Kategorija'], y=df_power['Snaga (kW)'],
                                    marker_color=['#4CAF50' if t=='Proizvodnja' else '#EF5350'
                                                  for t in df_power['Tip']]))
        pdf.add_plotly_chart(fig_pdf, "Tokovi energije")
        pdf_bytes = pdf.save()
        st.download_button("📄 Preuzmi PDF", pdf_bytes,
                           file_name=f"bilanca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                           mime="application/pdf")

# ============================================================
# 3. OPTIMIZACIJA D-1
# ============================================================
elif menu == "📅 Optimizacija D-1":
    st.header("📅 Optimizirani plan dan-unaprijed – MILP")
    st.markdown("##### Mixed Integer Linear Programming – realistično modeliranje baterije")

    with st.expander("📈 Uredi prognozu (24h)", expanded=False):
        for key, label in [('optimizer_spot','CROPEX Spot (€/MWh)'),
                           ('optimizer_load','Potrošnja (MWh/h)'),
                           ('optimizer_fne','FNE (MWh/h)'),
                           ('optimizer_eua','EUA (€/tCO₂)')]:
            st.markdown(f"**{label}**")
            txt = st.text_area(label, value=",".join([f"{x:.1f}" for x in st.session_state[key]]),
                               key=f"ta_{key}", height=70)
            try:
                arr = np.array([float(x.strip()) for x in txt.split(",")])
                if len(arr) == 24: st.session_state[key] = arr
                else: st.warning(f"{label}: treba 24 vrijednosti")
            except: pass

    col1, col2, col3 = st.columns(3)
    with col1:
        contracted_vol   = st.number_input("Ugovorena količina (MWh)", 0.0, value=100.0, step=10.0)
        contracted_price = st.number_input("Ugovorena cijena (€/MWh)", 0.0, value=60.0, step=5.0)
    with col2:
        batt_cap = st.number_input("Kapacitet baterije (MWh)", 0.0, value=6.0, step=1.0)
        batt_pow = st.number_input("Snaga baterije (MW)",      0.0, value=1.0, step=0.5)
    with col3:
        co2_price = st.number_input("Cijena EUA (€/tCO₂)",       0.0, value=80.0, step=5.0)
        feedin    = st.number_input("Otkupna cijena viška (€/MWh)",0.0, value=50.0, step=5.0)

    st.markdown("### 🧠 Napredno modeliranje baterije (MILP)")
    col_adv1, col_adv2, col_adv3 = st.columns(3)
    with col_adv1: use_milp = st.checkbox("Koristi MILP (preporučeno)", value=True)
    with col_adv2: batt_min_power = st.number_input("Min. snaga (MW)", 0.0, max_value=float(batt_pow), value=0.1, step=0.05)
    with col_adv3: batt_cycle_cost = st.number_input("Trošak degradacije (€/MWh)", 0.0, value=5.0, step=1.0)

    if st.button("🚀 Pokreni MILP optimizaciju", type="primary", use_container_width=True):
        optimizer = MILPDayAheadOptimizer(
            st.session_state.optimizer_load, st.session_state.optimizer_fne,
            st.session_state.optimizer_spot, contracted_vol, contracted_price,
            batt_cap, batt_pow, co2_price=co2_price, feedin_tariff=feedin,
            co2_intensity=0.4, batt_min_power=batt_min_power, batt_cycle_cost=batt_cycle_cost
        )
        with st.spinner("🧠 Rješavanje MILP modela..."):
            res = optimizer.optimize(initial_soc=0.0)

        if res['status'] == 'optimal':
            st.success("✅ MILP optimizacija uspješno završena!")

            col1, col2, col3, col4 = st.columns(4)
            with col1: metric_card("Ukupni trošak", res['total_cost'], suffix=" €")
            with col2: metric_card("CO₂ emisije", res['co2_emissions'], suffix=" tCO₂")
            with col3: metric_card("Korištenje baterije", np.sum(res['batt_dis']), suffix=" MWh")
            with col4: metric_card("Prodaja u mrežu", np.sum(res['grid_sales']), suffix=" MWh")

            # Build result key mapping (handles both old & new optimizer output keys)
            spot_key  = 'spot_buy' if 'spot_buy' in res else 'spot'
            contr_key = 'contract' if 'contract' in res else 'contr'

            df_res = pd.DataFrame({
                'Sat': range(1, 25),
                'Spot (MWh)': res[spot_key],
                'Tranše (MWh)': res[contr_key],
                'Prodaja (MWh)': res['grid_sales'],
                'FNE (MWh)': st.session_state.optimizer_fne,
                'Punjenje (MWh)': res['batt_ch'],
                'Pražnjenje (MWh)': res['batt_dis'],
                'SOC (MWh)': res['soc'],
            })

            # Stacked area – struktura opskrbe
            fig1 = go.Figure()
            for col_name, color, fill in [
                ('FNE (MWh)',      'rgba(76,175,80,0.75)',   'tozeroy'),
                ('Tranše (MWh)',   'rgba(21,101,192,0.8)',   'tonexty'),
                ('Spot (MWh)',     'rgba(255,107,53,0.75)',  'tonexty'),
            ]:
                fig1.add_trace(go.Scatter(
                    x=df_res['Sat'], y=df_res[col_name],
                    mode='lines', line=dict(width=0, color=color),
                    stackgroup='one', groupnorm='percent',
                    name=col_name.replace(' (MWh)',''), fillcolor=color
                ))
            dark_fig(fig1, 420, "Struktura opskrbe po satu (%)")
            fig1.update_layout(yaxis_title="%", xaxis_title="Sat",
                               hovermode='x unified',
                               legend=dict(**DARK_LAYOUT['legend'], orientation='h', y=1.05, x=0.5, xanchor='center'))
            st.plotly_chart(fig1, use_container_width=True)

            # BESS dual-axis
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name='Pražnjenje', x=df_res['Sat'], y=df_res['Pražnjenje (MWh)'],
                                   marker_color='#4CAF50', opacity=0.85))
            fig3.add_trace(go.Bar(name='Punjenje', x=df_res['Sat'], y=-df_res['Punjenje (MWh)'],
                                   marker_color='#EF5350', opacity=0.85))
            fig3.add_trace(go.Scatter(name='SOC', x=df_res['Sat'], y=df_res['SOC (MWh)'],
                                       mode='lines+markers',
                                       line=dict(color='#00BCD4', width=3, dash='dot'),
                                       marker=dict(size=7, symbol='diamond'), yaxis='y2'))
            dark_fig(fig3, 420, "BESS – punjenje/pražnjenje i SOC")
            fig3.update_layout(
                barmode='relative', yaxis_title="MWh",
                yaxis2=dict(title="SOC (MWh)", overlaying='y', side='right',
                            color='#00BCD4', gridcolor='rgba(0,0,0,0)'),
                hovermode='x unified',
                legend=dict(**DARK_LAYOUT['legend'], orientation='h', y=1.05, x=0.5, xanchor='center')
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Spot vs akcija baterije
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=df_res['Sat'], y=df_res['Pražnjenje (MWh)'],
                                   name='BESS pražnjenje', marker_color='rgba(0,188,212,0.6)'))
            fig4.add_trace(go.Bar(x=df_res['Sat'], y=-df_res['Punjenje (MWh)'],
                                   name='BESS punjenje', marker_color='rgba(239,83,80,0.55)'))
            fig4.add_trace(go.Scatter(x=df_res['Sat'], y=st.session_state.optimizer_spot,
                                       name='Spot €/MWh', line=dict(color='#FF6B35', width=2.5),
                                       yaxis='y2'))
            dark_fig(fig4, 320, "Arbitraža – spot cijena vs BESS aktivnost")
            fig4.update_layout(
                barmode='relative',
                yaxis2=dict(title="€/MWh", overlaying='y', side='right',
                            color='#FF6B35', gridcolor='rgba(0,0,0,0)'),
                hovermode='x unified'
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Heatmap korelacija
            fig_hm = go.Figure(data=go.Heatmap(
                z=[st.session_state.optimizer_spot, st.session_state.optimizer_load],
                x=[f"{s}h" for s in df_res['Sat']],
                y=['Spot cijena','Potrošnja'],
                colorscale=[[0,'#0A1628'],[0.5,'#1565C0'],[1,'#00BCD4']],
                showscale=False,
            ))
            dark_fig(fig_hm, 200)
            fig_hm.update_layout(title=dict(text="Korelacija spot cijene i potrošnje",
                                             font=dict(size=14, color='#94A3B8'), x=0.5),
                                  margin=dict(l=80, r=20, t=40, b=30))
            st.plotly_chart(fig_hm, use_container_width=True)

            with st.expander("📋 Detaljna tablica po satima"):
                st.dataframe(df_res.style.format({k: '{:.2f}' for k in df_res.columns if k != 'Sat'}),
                             use_container_width=True, hide_index=True)

            if st.button("📥 Preuzmi PDF izvještaj (Optimizacija)", use_container_width=True):
                pdf = PDFReport("Izvještaj optimizacije D-1 – MILP")
                pdf.add_title()
                pdf.add_paragraph(
                    "MILP s binarnim varijablama za bateriju. Cilj minimizacije: "
                    "nabavni troškovi + CO₂ + degradacija baterije − prihod od prodaje.")
                pdf.add_metric_cards({
                    "Ukupni trošak €": res['total_cost'], "CO₂ t": res['co2_emissions'],
                    "Baterija MWh": float(np.sum(res['batt_dis'])), "Prodaja MWh": float(np.sum(res['grid_sales']))
                })
                pdf.add_dataframe(df_res.round(2), "Rezultati po satima")
                pdf.add_plotly_chart(fig1, "Struktura opskrbe")
                pdf.add_plotly_chart(fig3, "SOC baterije")
                pdf_bytes = pdf.save()
                st.download_button("📄 Preuzmi PDF", pdf_bytes,
                                   file_name=f"milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                   mime="application/pdf")
        else:
            st.error(f"❌ MILP optimizacija nije uspjela: {res.get('message','Nepoznata greška')}")

# ============================================================
# 4. INVESTICIJSKI KALKULATOR
# ============================================================
elif menu == "💰 Investicijski kalkulator":
    st.header("💰 Napredni investicijski kalkulator")
    st.markdown('<div class="sub-title">Interaktivna analiza isplativosti – unesi vlastite parametre</div>',
                unsafe_allow_html=True)

    tech_defaults = {
        'BESS (baterija)':  dict(capex_kw=400,  opex_kw=15,  lifetime=15, co2=0.1,  prod=0.0, desc='Litij-ionski, 2h',         icon='🔋'),
        'FNE (solarna)':    dict(capex_kw=700,  opex_kw=10,  lifetime=25, co2=-0.8, prod=1.2, desc='Fotonaponska elektrana',    icon='☀️'),
        'Elektrokotao':     dict(capex_kw=150,  opex_kw=5,   lifetime=20, co2=-0.4, prod=2.0, desc='Zamjena plinskog kotla',    icon='🔥'),
        'FNE + BESS':       dict(capex_kw=1100, opex_kw=25,  lifetime=20, co2=-1.0, prod=1.2, desc='Integrirani sustav',        icon='⚡'),
        'Vjetroelektrana':  dict(capex_kw=1200, opex_kw=30,  lifetime=25, co2=-0.9, prod=2.5, desc='Vjetroagregat – offshore',  icon='💨'),
    }

    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔧 Odabir tehnologije")
        tech = st.selectbox("Tip postrojenja", list(tech_defaults.keys()),
                             format_func=lambda x: f"{tech_defaults[x]['icon']} {x}")
        st.caption(tech_defaults[tech]['desc'])
        capacity = st.number_input("Instalirani kapacitet (kW)", 1.0, value=1000.0, step=50.0)

        st.markdown("**Financijski parametri**")
        use_custom_capex = st.checkbox("Ručni unos CAPEX")
        if use_custom_capex:
            capex = st.number_input("Ukupni CAPEX (€)", 0.0, value=capacity*tech_defaults[tech]['capex_kw'], step=10000.0)
        else:
            capex = capacity * tech_defaults[tech]['capex_kw']
            st.metric("Preporučeni CAPEX", format_eur(capex))

        use_custom_opex = st.checkbox("Ručni unos OPEX")
        if use_custom_opex:
            opex = st.number_input("Godišnji OPEX (€)", 0.0, value=capacity*tech_defaults[tech]['opex_kw'], step=1000.0)
        else:
            opex = capacity * tech_defaults[tech]['opex_kw']
            st.metric("Preporučeni OPEX", format_eur(opex))

        lifetime = st.number_input("Ekonomski vijek (god)", 1, value=tech_defaults[tech]['lifetime'], step=1)
        discount = st.slider("Diskontna stopa (%)", 0.0, 15.0, 5.0, 0.5) / 100
        inflation = st.slider("Inflacija (%)", 0.0, 5.0, 2.0, 0.1) / 100

        st.markdown("**Energetski parametri**")
        if tech == 'Elektrokotao':
            prod_factor = st.number_input("Potrošnja (MWh/kW/god)", 0.0, value=float(tech_defaults[tech]['prod']), step=0.1)
            gas_price   = st.number_input("Cijena plina (€/MWh)", 0.0, value=45.0, step=5.0)
            elec_price = self_cons = feedin = 0.0
        else:
            prod_factor = st.number_input("Specifična proizvodnja (MWh/kW/god)", 0.0, value=float(tech_defaults[tech]['prod']), step=0.1)
            elec_price  = st.number_input("Cijena el. energije (€/MWh)", 0.0, value=80.0, step=5.0)
            self_cons   = st.slider("Udio vlastite potrošnje", 0.0, 1.0, 0.8, 0.05)
            feedin      = st.number_input("Otkupna cijena viška (€/MWh)", 0.0, value=50.0, step=5.0)
            gas_price   = 0.0
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        annual_prod = capacity * prod_factor
        if tech == 'Elektrokotao':
            annual_savings = annual_prod * gas_price
        else:
            annual_savings = annual_prod * self_cons * elec_price + annual_prod * (1-self_cons) * feedin

        cf = np.zeros(lifetime + 1)
        cf[0] = -capex
        for t in range(1, lifetime+1):
            cf[t] = annual_savings*(1+inflation)**(t-1) - opex*(1+inflation)**(t-1)

        t_vec = np.arange(lifetime+1)
        npv = np.sum(cf / (1+discount)**t_vec)
        try:
            irr = brentq(lambda r: np.sum(cf/(1+r)**t_vec), -0.99, 1.0)
        except:
            irr = None

        cum = np.cumsum(cf)
        payback = float('inf')
        for i in range(1, len(cum)):
            if cum[i] >= 0:
                payback = i - cum[i-1]/(cum[i]-cum[i-1])
                break

        lcoe = (capex + np.sum(opex/(1+discount)**t_vec[1:])) / (annual_prod*lifetime) if annual_prod > 0 else 0.0
        co2_red = abs(tech_defaults[tech]['co2']) * capacity

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Rezultati isplativosti")
        c1, c2, c3 = st.columns(3)
        c1.metric("NPV", format_eur(npv))
        c2.metric("IRR", f"{irr:.1%}" if irr else "n/a")
        c3.metric("Payback", f"{payback:.1f} god")
        c1, c2, c3 = st.columns(3)
        c1.metric("LCOE", f"{lcoe:.1f} €/MWh" if lcoe > 0 else "n/a")
        c2.metric("CO₂ redukcija", format_co2(co2_red))
        c3.metric("God. proizvodnja", f"{annual_prod:,.0f} MWh")
        st.metric("Godišnja ušteda", format_eur(annual_savings))
        st.markdown('</div>', unsafe_allow_html=True)

        years = list(range(lifetime+1))
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=years, y=cf,
                                 marker_color=['#EF5350' if x<0 else '#4CAF50' for x in cf],
                                 opacity=0.85))
        dark_fig(fig_cf, 320, "Godišnji novčani tokovi")
        fig_cf.update_layout(xaxis_title="Godina", yaxis_title="€", xaxis_dtick=1)
        st.plotly_chart(fig_cf, use_container_width=True)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=years, y=np.cumsum(cf), mode='lines+markers',
                                      line=dict(color='#00BCD4', width=3),
                                      marker=dict(size=8, symbol='diamond', color='#00BCD4')))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.2)")
        dark_fig(fig_cum, 280, "Kumulativni novčani tok")
        fig_cum.update_layout(xaxis_title="Godina", yaxis_title="€", xaxis_dtick=1)
        st.plotly_chart(fig_cum, use_container_width=True)

        fig_water = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative","relative","relative","relative","total"],
            x=["Ušteda EE","Smjena plina","Trošak BESS","Prihod od prodaje","Ukupno"],
            y=[-annual_savings*0.7, -annual_savings*0.25, opex, -annual_savings*0.15, 0],
            connector={"line": {"color": "rgba(255,255,255,0.1)", "width": 1}},
            decreasing={"marker": {"color": "#4CAF50"}},
            increasing={"marker": {"color": "#EF5350"}},
            totals={"marker": {"color": "#1565C0"}},
            textfont=dict(color="#E2E8F0"),
        ))
        dark_fig(fig_water, 360, "Struktura godišnje promjene troška")
        fig_water.update_layout(showlegend=False, yaxis_title="€", yaxis_tickformat=',.0f')
        st.plotly_chart(fig_water, use_container_width=True)

        if st.button("🔄 Generiraj radar usporedbu", key="radar_btn", use_container_width=True):
            with st.spinner("Izračunavam usporedbu..."):
                techs = list(tech_defaults.keys())
                rows = []
                for t_name in techs:
                    t = tech_defaults[t_name]
                    cap, c_, o_, p_ = 1000.0, 1000*t['capex_kw'], 1000*t['opex_kw'], 1000*t['prod']
                    s_ = p_*45 if t_name=='Elektrokotao' else p_*0.8*80+p_*0.2*50
                    cf_ = np.zeros(21); cf_[0]=-c_
                    for y in range(1,21): cf_[y]=s_-o_
                    df_ = (1+0.05)**np.arange(21)
                    npv_ = np.sum(cf_/df_)
                    try: irr_ = brentq(lambda r: np.sum(cf_/(1+r)**np.arange(21)), -0.99, 2.0)
                    except: irr_ = 0.0
                    cum_ = np.cumsum(cf_)
                    pb_ = next((i-cum_[i-1]/(cum_[i]-cum_[i-1]) for i in range(1,len(cum_)) if cum_[i]>=0), 999)
                    lc_ = (c_+np.sum(o_/df_[1:]))/(p_*20) if p_>0 else 0
                    rows.append({'Tehnologija': t_name, 'NPV M€': npv_/1e6,
                                 'IRR %': irr_*100, 'Payback god': pb_,
                                 'LCOE €/MWh': lc_, 'CO₂ red t': abs(t['co2'])*cap})

                df_r = pd.DataFrame(rows)
                df_m = df_r.melt(id_vars='Tehnologija', var_name='Parametar', value_name='Val')
                df_m['Norm'] = df_m.groupby('Parametar')['Val'].transform(
                    lambda x: (x-x.min())/(x.max()-x.min()) if x.max()>x.min() else x)

                fig_rad = px.line_polar(df_m, r='Norm', theta='Parametar',
                                        color='Tehnologija', line_close=True,
                                        color_discrete_sequence=['#00BCD4','#4CAF50','#FF6B35','#1565C0','#AB47BC'])
                fig_rad.update_layout(
                    polar=dict(
                        bgcolor='rgba(10,18,35,0.8)',
                        radialaxis=dict(visible=True, range=[0,1], color='#475569',
                                        gridcolor='rgba(255,255,255,0.06)'),
                        angularaxis=dict(color='#94A3B8', gridcolor='rgba(255,255,255,0.06)')
                    ),
                    **{k:v for k,v in DARK_LAYOUT.items() if k not in ['xaxis','yaxis']},
                    height=500,
                    title=dict(text="Usporedba tehnologija (normalizirano)", x=0.5,
                               font=dict(size=16, color='#CBD5E1'))
                )
                st.plotly_chart(fig_rad, use_container_width=True)

        if st.button("📥 Preuzmi PDF izvještaj (Investicija)", use_container_width=True):
            pdf = PDFReport("Izvještaj o isplativosti investicije")
            pdf.add_title()
            pdf.add_metric_cards({"Tehnologija": tech, "Kapacitet kW": capacity,
                                   "CAPEX €": capex, "OPEX €/god": opex})
            pdf.add_metric_cards({"NPV €": npv, "IRR": f"{irr:.1%}" if irr else "n/a",
                                   "Payback god": f"{payback:.1f}",
                                   "CO₂ red t/god": co2_red, "God. ušteda €": annual_savings})
            fig_cf_pdf = go.Figure(go.Bar(x=years, y=cf,
                                           marker_color=['#EF5350' if x<0 else '#4CAF50' for x in cf]))
            fig_cf_pdf.update_layout(title="Godišnji novčani tokovi", xaxis_title="Godina", yaxis_title="€")
            pdf.add_plotly_chart(fig_cf_pdf, "Novčani tok")
            pdf_bytes = pdf.save()
            st.download_button("📄 Preuzmi PDF", pdf_bytes,
                               file_name=f"investicija_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf")

# ============================================================
# 5. MODULARNI DIZAJNER
# ============================================================
elif menu == "🧩 Modularni dizajner":
    import modular_energy_designer as med
    med.show_designer()

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("EKONERG – Institut za energetiku i zaštitu okoliša | 2026")
