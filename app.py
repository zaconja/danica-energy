"""
Danica Energy Optimizer PRO – v6.0
Moderni energetski dashboard s MILP optimizacijom, investicijskim kalkulatorom,
scenarij analizom i PDF izvještavanjem.

Autor: EKONERG – Institut za energetiku i zaštitu okoliša
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# IMPORTI
# ============================================================
try:
    from report_generator import PDFReport
    REPORT_OK = True
except Exception:
    REPORT_OK = False

try:
    from milp_optimizer import MILPDayAheadOptimizer
    MILP_OK = True
except Exception:
    MILP_OK = False

# ============================================================
# KONFIGURACIJA STRANICE
# ============================================================
st.set_page_config(
    page_title="Danica Energy Optimizer PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "**Danica Energy Optimizer PRO v6.0**\nEKONERG – Institut za energetiku i zaštitu okoliša\n© 2026"
    }
)

# ============================================================
# MODERNI DARK THEME CSS
# ============================================================
st.markdown("""
<style>
/* ---- Import Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ---- Base ---- */
html, body, .stApp {
    background-color: #0A0F1E !important;
    color: #E2E8F0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0A0F1E; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #0A0F1E 100%) !important;
    border-right: 1px solid rgba(0, 188, 212, 0.15) !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] .stRadio label { 
    font-size: 0.88rem; 
    padding: 0.4rem 0;
    transition: color 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #00BCD4 !important; }

/* ---- Header ---- */
.main-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #162035 50%, #0D1B2A 100%);
    border: 1px solid rgba(0, 188, 212, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0, 188, 212, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, #00BCD4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-subtitle {
    color: #64748B;
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 400;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(46, 125, 50, 0.15);
    border: 1px solid rgba(46, 125, 50, 0.4);
    color: #4CAF50;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.status-dot {
    width: 7px; height: 7px;
    background: #4CAF50;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.3); }
}

/* ---- KPI Cards ---- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: linear-gradient(135deg, #0D1B2A 0%, #111827 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.3s, transform 0.2s;
}
.kpi-card:hover {
    border-color: rgba(0, 188, 212, 0.35);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #475569;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-delta {
    font-size: 0.75rem;
    margin-top: 0.4rem;
    font-weight: 600;
}
.kpi-delta.up { color: #4CAF50; }
.kpi-delta.down { color: #EF5350; }
.kpi-delta.neutral { color: #64748B; }
.kpi-icon {
    font-size: 1.4rem;
    float: right;
    opacity: 0.6;
}

/* ---- Glass Cards ---- */
.glass-card {
    background: rgba(13, 27, 42, 0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}
.glass-card-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- Section Headers ---- */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(0, 188, 212, 0.15);
}
.section-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #0D47A1, #00BCD4);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F1F5F9;
}

/* ---- Table overrides ---- */
.dataframe {
    background: #0D1B2A !important;
    color: #E2E8F0 !important;
    border-radius: 8px;
    overflow: hidden;
}
.dataframe thead th {
    background: #162035 !important;
    color: #00BCD4 !important;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
.dataframe tbody tr:nth-child(even) { background: rgba(255,255,255,0.03) !important; }
.dataframe tbody tr:hover { background: rgba(0, 188, 212, 0.07) !important; }

/* ---- Inputs ---- */
.stNumberInput input, .stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: #0D1B2A !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: rgba(0, 188, 212, 0.5) !important;
    box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.1) !important;
}
.stSlider > div > div { background: #1E3A5F !important; }
.stSlider > div > div > div { background: #00BCD4 !important; }

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #0D47A1 0%, #1565C0 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1565C0 0%, #00BCD4 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(0, 188, 212, 0.3) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00838F 0%, #00BCD4 100%) !important;
}

/* ---- Download Button ---- */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1B5E20, #2E7D32) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ---- Metrics ---- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0D1B2A, #111827);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: #0D1B2A !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    color: #94A3B8 !important;
}
.streamlit-expanderContent {
    background: #0A1120 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ---- Alerts ---- */
.stSuccess { background: rgba(46, 125, 50, 0.15) !important; border-color: #2E7D32 !important; }
.stError { background: rgba(198, 40, 40, 0.15) !important; border-color: #C62828 !important; }
.stWarning { background: rgba(245, 124, 0, 0.15) !important; border-color: #F57C00 !important; }
.stInfo { background: rgba(21, 101, 192, 0.15) !important; border-color: #1565C0 !important; }

/* ---- Checkbox ---- */
.stCheckbox label { color: #94A3B8 !important; }

/* ---- Tab ---- */
.stTabs [role="tablist"] { background: #0D1B2A; border-radius: 8px; padding: 4px; }
.stTabs [role="tab"] { color: #64748B !important; border-radius: 6px; }
.stTabs [role="tab"][aria-selected="true"] { 
    background: linear-gradient(135deg, #0D47A1, #1565C0) !important; 
    color: white !important; 
}

/* ---- Spinner overlay ---- */
.stSpinner > div { border-top-color: #00BCD4 !important; }

/* ---- Progress ---- */
.stProgress > div > div { background: linear-gradient(90deg, #0D47A1, #00BCD4) !important; }

/* ---- Waterfall bar annotation ---- */
.annotation { color: #E2E8F0 !important; }

/* ---- Separator ---- */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ---- Sidebar version badge ---- */
.version-badge {
    background: rgba(0, 188, 212, 0.1);
    border: 1px solid rgba(0, 188, 212, 0.2);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.8rem;
    color: #00BCD4;
    text-align: center;
    margin-top: 1rem;
}

/* ---- Energy flow badge ---- */
.flow-positive { color: #4CAF50; font-weight: 700; }
.flow-negative { color: #EF5350; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY DARK THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13, 27, 42, 0.5)',
    font=dict(family='Space Grotesk, sans-serif', color='#CBD5E1', size=11),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.08)', color='#64748B'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.08)', color='#64748B'),
    legend=dict(bgcolor='rgba(13,27,42,0.8)', bordercolor='rgba(255,255,255,0.08)', borderwidth=1, font=dict(color='#CBD5E1')),
    margin=dict(l=40, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor='#162035', font_size=12, font_family='Space Grotesk', bordercolor='rgba(0,188,212,0.3)'),
    colorway=['#00BCD4', '#1565C0', '#4CAF50', '#FF6B35', '#AB47BC', '#FF7043', '#26A69A'],
)

COLOR_MAP = {
    'FNE': '#4CAF50',
    'Tranše': '#1565C0',
    'Spot': '#FF6B35',
    'Baterija dis': '#00BCD4',
    'Baterija ch': '#EF5350',
    'Prodaja': '#AB47BC',
    'Plin': '#FF7043',
    'Biomasa': '#8D6E63',
}

# ============================================================
# HELPER FUNKCIJE
# ============================================================
def fmt_eur(x: float, decimals: int = 0) -> str:
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M €"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.1f}k €"
    return f"{x:.{decimals}f} €"

def fmt_mwh(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x/1000:.2f} GWh"
    return f"{x:.1f} MWh"

def fmt_co2(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x/1000:.2f} ktCO₂"
    return f"{x:.0f} tCO₂"

def fmt_pct(x: float) -> str:
    return f"{x:.1f}%"

def kpi_card(label: str, value: str, delta: str = "", delta_dir: str = "neutral", icon: str = ""):
    delta_class = {"up": "up", "down": "down"}.get(delta_dir, "neutral")
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    icon_html = f'<span class="kpi-icon">{icon}</span>' if icon else ""
    st.markdown(f"""
    <div class="kpi-card">
        {icon_html}
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def section_header(icon: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <div class="section-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def make_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ============================================================
# SESSION STATE – INICIJALIZACIJA
# ============================================================
def init_state():
    if 'initialized' in st.session_state:
        return

    np.random.seed(42)
    hours = np.arange(24)

    st.session_state.portfolio_contracts = [
        {"Energija": "Tranša 1", "Količina MWh": 5000.0, "Cijena €/MWh": 59.0, "Status": "Fiksno", "Dobavljač": "HEP"},
        {"Energija": "Tranša 2", "Količina MWh": 3000.0, "Cijena €/MWh": 65.0, "Status": "Fiksno", "Dobavljač": "A1"},
        {"Energija": "Plin", "Količina m³": 1_500_000.0, "Cijena €/m³": 0.035, "Status": "Indeksirano", "Dobavljač": "Geoplin"},
    ]
    st.session_state.biomass_batches = [
        {"Isporuka": "2025-01-15", "Količina t": 1000.0, "Cijena €/t": 120.0, "Vlažnost %": 25.0},
        {"Isporuka": "2025-03-01", "Količina t": 1500.0, "Cijena €/t": 115.0, "Vlažnost %": 22.0},
    ]
    st.session_state.co2_allocated = 8400.0
    st.session_state.co2_used = 6500.0
    st.session_state.cropex_spot = 78.5
    st.session_state.eua_price = 82.0

    # Realistični dnevni profil opterećenja
    base_load = 80 + 40 * np.sin((hours - 6) * np.pi / 12) ** 2
    base_load = np.clip(base_load + np.random.normal(0, 5, 24), 60, 140)
    fne_profile = np.maximum(0, 60 * np.sin((hours - 7) * np.pi / 11) + np.random.normal(0, 5, 24))
    spot_profile = 60 + 25 * np.sin((hours - 8) * np.pi / 12) + np.random.normal(0, 5, 24)
    spot_profile = np.clip(spot_profile, 35, 130)
    eua_profile = np.random.normal(82, 3, 24).clip(65, 100)

    st.session_state.optimizer_load = base_load
    st.session_state.optimizer_fne = fne_profile
    st.session_state.optimizer_spot = spot_profile
    st.session_state.optimizer_eua = eua_profile

    st.session_state.ob_now = {
        'fne_power': 1850.0,
        'grid_import': 3650.0,
        'grid_export': 3550.0,
        'bess_charge': 800.0,
        'bess_discharge': 700.0,
        'thermal_power': 220.0,
        'co2_rate': 4.02,
        'plan_fne': 105000.0,
        'plan_bess': 70.0,
        'gas_boiler': 90.0,
        'biomass_boiler': 30.0,
        'gas_remaining': 100000.0,
        'biomass_remaining': 120000.0,
        'bess_soc_pct': 68.0,
    }

    st.session_state.last_milp_result = None
    st.session_state.initialized = True

init_state()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem;">⚡</div>
        <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9;">Danica Energy</div>
        <div style="font-size:0.75rem; color:#475569;">Optimizer PRO</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    menu = st.radio(
        "Navigacija",
        options=[
            "📊 Portfelj",
            "⚡ Operativna bilanca",
            "📅 Optimizacija D-1",
            "💰 Investicijski kalkulator",
            "🔬 Scenarij analiza",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Live status
    now = datetime.now()
    st.markdown(f"""
    <div style="font-size:0.75rem; color:#475569; text-align:center;">
        🕐 {now.strftime('%H:%M:%S')} | {now.strftime('%d.%m.%Y')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:0.8rem; font-size:0.75rem; text-align:center;">
        <span style="color:#4CAF50;">●</span> CROPEX: <strong style="color:#F1F5F9;">{st.session_state.cropex_spot:.1f} €/MWh</strong><br/>
        <span style="color:#FF6B35;">●</span> EUA: <strong style="color:#F1F5F9;">{st.session_state.eua_price:.1f} €/tCO₂</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="version-badge">
        v6.0 – MILP Enhanced<br/>
        <span style="font-size:0.7rem; opacity:0.7;">EKONERG © 2026</span>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="main-header">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
        <div>
            <h1 class="main-title">⚡ Danica Energy Optimizer PRO</h1>
            <p class="main-subtitle">Napredna MILP optimizacija energetskog portfelja · Analiza investicija · Scenarij planiranje</p>
        </div>
        <div style="display:flex; flex-direction:column; align-items:flex-end; gap:0.5rem;">
            <div class="status-badge"><div class="status-dot"></div> Sustav aktivan</div>
            <div style="font-size:0.75rem; color:#475569;">{datetime.now().strftime('%A, %d. %B %Y. – %H:%M')}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 1. PREGLED PORTFELJA
# ============================================================
if menu == "📊 Portfelj":
    section_header("📊", "Pregled energetskog portfelja")

    # --- TOP KPI-ovi ---
    contracts = st.session_state.portfolio_contracts
    total_ee = sum(c.get('Količina MWh', 0) for c in contracts[:2])
    total_cost_ee = sum(c.get('Količina MWh', 0) * c.get('Cijena €/MWh', 0) for c in contracts[:2])
    gas_m3 = contracts[2].get('Količina m³', 0) if len(contracts) > 2 else 0
    gas_cost = gas_m3 * contracts[2].get('Cijena €/m³', 0) if len(contracts) > 2 else 0
    total_cost = total_cost_ee + gas_cost
    co2_free_pct = (st.session_state.co2_allocated - st.session_state.co2_used) / st.session_state.co2_allocated * 100

    cols = st.columns(5)
    with cols[0]:
        kpi_card("Ukupna EE", fmt_mwh(total_ee), icon="⚡")
    with cols[1]:
        kpi_card("Trošak EE", fmt_eur(total_cost_ee), icon="💶")
    with cols[2]:
        kpi_card("Plin ukupno", f"{gas_m3/1e6:.2f}M m³", icon="🔥")
    with cols[3]:
        kpi_card("Ukupni trošak", fmt_eur(total_cost), icon="💰")
    with cols[4]:
        kpi_card("CO₂ slobodni", fmt_pct(co2_free_pct),
                 delta="Dostupni kvota", delta_dir="up" if co2_free_pct > 20 else "down", icon="🌿")

    st.divider()

    # --- UGOVORI ---
    tab1, tab2, tab3 = st.tabs(["📋 Električna energija", "🔥 Plin & Biomasa", "🌿 CO₂ kvote"])

    with tab1:
        with st.expander("➕ Dodaj novi ugovor", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: en_type = st.text_input("Naziv tranše", "Tranša 3", key="new_en_type")
            with c2: quantity = st.number_input("Količina MWh", min_value=0.0, value=2000.0, step=100.0, key="new_qty")
            with c3: price = st.number_input("Cijena €/MWh", min_value=0.0, value=70.0, step=0.5, key="new_price")
            with c4: dobavljac = st.text_input("Dobavljač", "HEP", key="new_dob")
            with c5:
                status = st.selectbox("Status", ["Fiksno", "Indeksirano"], key="new_status")
                if st.button("Dodaj ✓", key="add_contr"):
                    st.session_state.portfolio_contracts.append({
                        "Energija": en_type, "Količina MWh": quantity,
                        "Cijena €/MWh": price, "Status": status, "Dobavljač": dobavljac
                    })
                    st.success("Ugovor dodan!")
                    st.rerun()

        df_c = pd.DataFrame(st.session_state.portfolio_contracts)
        df_c['Ukupno €'] = df_c.get('Količina MWh', df_c.get('Količina m³', pd.Series([0]*len(df_c)))) * df_c.get('Cijena €/MWh', df_c.get('Cijena €/m³', pd.Series([0]*len(df_c))))
        edited = st.data_editor(
            df_c, use_container_width=True, num_rows="dynamic",
            column_config={
                "Količina MWh": st.column_config.NumberColumn(format="%.0f MWh"),
                "Cijena €/MWh": st.column_config.NumberColumn(format="%.2f €/MWh"),
                "Ukupno €": st.column_config.NumberColumn(format="%.0f €", disabled=True),
            },
            key="contracts_editor"
        )

        # Grafikon strukture portfelja
        if len(df_c) > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                ee_contracts = [c for c in contracts if 'Količina MWh' in c]
                if ee_contracts:
                    labels = [c['Energija'] for c in ee_contracts]
                    values = [c['Količina MWh'] for c in ee_contracts]
                    fig_pie = go.Figure(go.Pie(
                        labels=labels, values=values,
                        hole=0.55,
                        marker=dict(colors=['#1565C0', '#00BCD4', '#4CAF50', '#FF6B35'],
                                    line=dict(color='#0A0F1E', width=2)),
                        textfont=dict(color='#E2E8F0'),
                    ))
                    fig_pie.add_annotation(text="EE<br>MWh", x=0.5, y=0.5, showarrow=False,
                                           font=dict(size=12, color='#94A3B8'))
                    fig_pie.update_layout(title="Struktura EE portfelja", **PLOTLY_LAYOUT,
                                          margin=dict(l=20, r=20, t=40, b=20), height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                # Usporedba cijena tranši vs CROPEX spot
                if ee_contracts:
                    tranše = [c['Energija'] for c in ee_contracts]
                    cijene = [c['Cijena €/MWh'] for c in ee_contracts]
                    colors_bar = ['#4CAF50' if c <= st.session_state.cropex_spot else '#EF5350' for c in cijene]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=tranše, y=cijene,
                        marker_color=colors_bar,
                        name="Ugovorena cijena",
                        text=[f"{c:.1f}" for c in cijene],
                        textposition='outside',
                        textfont=dict(color='#E2E8F0'),
                    ))
                    fig_bar.add_hline(
                        y=st.session_state.cropex_spot, line_dash="dash",
                        line_color="#FF6B35", annotation_text=f"CROPEX: {st.session_state.cropex_spot:.1f} €/MWh",
                        annotation_font_color="#FF6B35",
                    )
                    fig_bar.update_layout(title="Cijena vs CROPEX spot", **PLOTLY_LAYOUT,
                                          yaxis_title="€/MWh", height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.markdown("#### 🌱 Biomasa – isporuke i zalihe")
        df_bio = pd.DataFrame(st.session_state.biomass_batches)
        df_bio['Ukupno €'] = df_bio['Količina t'] * df_bio['Cijena €/t']
        df_bio['Energija MWh'] = df_bio['Količina t'] * (1 - df_bio['Vlažnost %'] / 100) * 3.2  # ~3.2 MWh/t suhe
        edited_bio = st.data_editor(df_bio, use_container_width=True, num_rows="dynamic",
                                     column_config={
                                         "Količina t": st.column_config.NumberColumn(format="%.0f t"),
                                         "Cijena €/t": st.column_config.NumberColumn(format="%.1f €/t"),
                                         "Vlažnost %": st.column_config.NumberColumn(format="%.1f %%"),
                                         "Ukupno €": st.column_config.NumberColumn(format="%.0f €", disabled=True),
                                         "Energija MWh": st.column_config.NumberColumn(format="%.0f MWh", disabled=True),
                                     }, key="bio_editor")

        # Plin troškovi
        st.markdown("#### 🔥 Plin – parametri")
        c1, c2, c3 = st.columns(3)
        with c1: hho_val = st.number_input("Ogrijevna vrijednost (kWh/m³)", 9.5, 11.5, 10.3, 0.1)
        with c2: plinska_eff = st.number_input("Efikasnost kotla (%)", 80.0, 98.0, 92.0, 1.0)
        with c3: gas_price = st.number_input("Cijena plina (€/m³)", 0.01, 0.20, 0.035, 0.001, format="%.3f")

        if gas_m3 > 0:
            energy_plin = gas_m3 * hho_val / 1000 * plinska_eff / 100  # MWh toplinska
            st.info(f"📊 Energetski sadržaj plina (toplinski): **{fmt_mwh(energy_plin)}** | "
                    f"Ukupni trošak: **{fmt_eur(gas_m3 * gas_price)}**")

    with tab3:
        st.markdown("#### 🌿 EU emisijski certifikati (EUA)")
        c1, c2, c3 = st.columns(3)
        with c1: co2_total = st.number_input("Ukupna alokacija (tCO₂)", 0.0, 50000.0,
                                              st.session_state.co2_allocated, 100.0)
        with c2: co2_used = st.number_input("Iskorišteno (tCO₂)", 0.0, co2_total,
                                             st.session_state.co2_used, 50.0)
        with c3: eua_price_inp = st.number_input("Cijena EUA (€/tCO₂)", 0.0, 200.0,
                                                  st.session_state.eua_price, 1.0)

        st.session_state.co2_allocated = co2_total
        st.session_state.co2_used = co2_used
        st.session_state.eua_price = eua_price_inp

        free = co2_total - co2_used
        pct = co2_used / max(co2_total, 1) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Slobodni kvota", fmt_co2(free))
        col2.metric("Iskorištenost", fmt_pct(pct), delta=f"{fmt_co2(free)} preostalo")
        col3.metric("Vrijednost kvote", fmt_eur(free * eua_price_inp))

        # CO2 gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            delta={'reference': 75, 'increasing': {'color': '#EF5350'}, 'decreasing': {'color': '#4CAF50'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#64748B', 'tickfont': {'color': '#64748B'}},
                'bar': {'color': '#EF5350' if pct > 80 else '#F57C00' if pct > 60 else '#4CAF50'},
                'bgcolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(76,175,80,0.1)'},
                    {'range': [60, 80], 'color': 'rgba(245,124,0,0.1)'},
                    {'range': [80, 100], 'color': 'rgba(239,83,80,0.1)'},
                ],
                'threshold': {'line': {'color': '#FF6B35', 'width': 2}, 'thickness': 0.75, 'value': 90},
            },
            number={'suffix': '%', 'font': {'color': '#F1F5F9', 'size': 28}},
            title={'text': "Iskorištenost CO₂ kvote", 'font': {'color': '#94A3B8', 'size': 13}},
        ))
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # PDF izvještaj portfelj
    if REPORT_OK and st.button("📥 Generiraj PDF izvještaj portfelja"):
        pdf = PDFReport("Izvještaj energetskog portfelja", subtitle="Pregled ugovorenih energija i CO₂ pozicije")
        pdf.add_title_page()
        pdf.add_heading("1. Električna energija – ugovorene tranše")
        pdf.add_metric_cards({
            "Ukupna EE": fmt_mwh(total_ee),
            "Trošak EE": fmt_eur(total_cost_ee),
            "CO₂ slobodni": fmt_co2(free),
        })
        pdf.add_dataframe(pd.DataFrame(ee_contracts) if ee_contracts else pd.DataFrame(), "Popis ugovora")
        pdf.add_heading("2. CO₂ pozicija")
        pdf.add_metric_cards({
            "Alokacija": fmt_co2(co2_total),
            "Iskorišteno": fmt_co2(co2_used),
            "Slobodni": fmt_co2(free),
            "Iskorištenost": fmt_pct(pct),
        })
        pdf_bytes = pdf.save()
        st.download_button("📄 Preuzmi PDF", pdf_bytes,
                           file_name=f"portfelj_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                           mime="application/pdf")


# ============================================================
# 2. OPERATIVNA BILANCA
# ============================================================
elif menu == "⚡ Operativna bilanca":
    section_header("⚡", "Operativna energetska bilanca – Real-time pregled")

    ob = st.session_state.ob_now

    # Sliders za real-time ažuriranje
    with st.expander("⚙️ Uredi trenutne vrijednosti", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ob['fne_power'] = st.slider("FNE snaga (kW)", 0.0, 5000.0, ob['fne_power'], 50.0)
            ob['bess_soc_pct'] = st.slider("BESS SOC (%)", 0.0, 100.0, ob.get('bess_soc_pct', 68.0), 1.0)
        with c2:
            ob['grid_import'] = st.slider("Uvoz iz mreže (kW)", 0.0, 10000.0, ob['grid_import'], 50.0)
            ob['grid_export'] = st.slider("Izvoz u mrežu (kW)", 0.0, 10000.0, ob['grid_export'], 50.0)
        with c3:
            ob['bess_charge'] = st.slider("BESS punjenje (kW)", 0.0, 2000.0, ob['bess_charge'], 25.0)
            ob['bess_discharge'] = st.slider("BESS pražnjenje (kW)", 0.0, 2000.0, ob['bess_discharge'], 25.0)
        st.session_state.ob_now = ob

    # KPI kartice
    total_gen = ob['fne_power'] + ob['bess_discharge'] + ob['grid_import']
    total_cons = ob['grid_export'] + ob['bess_charge'] + ob['thermal_power']
    net_balance = total_gen - total_cons
    self_suff = ob['fne_power'] / max(ob['fne_power'] + ob['grid_import'], 1) * 100

    st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
    cols = st.columns(5)
    with cols[0]: kpi_card("FNE snaga", f"{ob['fne_power']:.0f} kW", icon="☀️")
    with cols[1]: kpi_card("BESS SOC", f"{ob.get('bess_soc_pct',68):.0f}%",
                            delta=f"{'Punjenje' if ob['bess_charge']>ob['bess_discharge'] else 'Pražnjenje'}",
                            delta_dir="up" if ob['bess_charge'] > ob['bess_discharge'] else "down", icon="🔋")
    with cols[2]: kpi_card("Mrežni uvoz", f"{ob['grid_import']:.0f} kW", icon="🔌")
    with cols[3]: kpi_card("Samodostatnost", fmt_pct(self_suff),
                            delta_dir="up" if self_suff > 40 else "down", icon="🌿")
    with cols[4]: kpi_card("Neto bilanca", f"{net_balance:+.0f} kW",
                            delta_dir="up" if net_balance > 0 else "down", icon="⚖️")
    st.markdown('</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        # Sankey dijagram energetskih tokova
        source = [0, 1, 2, 0, 1]
        target = [3, 3, 3, 4, 4]
        value = [
            ob['fne_power'] / 10,
            ob['bess_discharge'] / 10,
            ob['grid_import'] / 10,
            ob['grid_export'] / 10,
            ob['bess_charge'] / 10,
        ]
        label = ['☀️ FNE', '🔋 BESS dis.', '🔌 Mreža uvoz', '🏭 Potrošači', '🔋 BESS pun.']
        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color='rgba(0,188,212,0.3)', width=0.5),
                label=label,
                color=['#4CAF50', '#00BCD4', '#1565C0', '#FF6B35', '#AB47BC'],
            ),
            link=dict(
                source=source, target=target, value=value,
                color=['rgba(76,175,80,0.3)', 'rgba(0,188,212,0.3)',
                       'rgba(21,101,192,0.3)', 'rgba(171,71,188,0.3)', 'rgba(21,101,192,0.2)'],
            ),
            arrangement='snap',
        ))
        fig_sankey.update_layout(title="Tokovi energije (Sankey)", **PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_sankey, use_container_width=True)

    with col_right:
        # Toplinska bilanca
        st.markdown("#### 🌡️ Toplinska bilanca")
        heat_sources = {
            '🔥 Plinski kotao': ob['gas_boiler'],
            '🌱 Biomasa': ob['biomass_boiler'],
            '⚡ Elektrokotao': ob['thermal_power'],
        }
        for name, val in heat_sources.items():
            max_val = 300 if 'Plinski' in name else 150 if 'Biomasa' in name else 200
            pct_heat = val / max_val * 100
            color = '#4CAF50' if pct_heat < 70 else '#F57C00' if pct_heat < 90 else '#EF5350'
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#94A3B8; margin-bottom:3px;">
                    <span>{name}</span><span>{val:.0f} kW ({pct_heat:.0f}%)</span>
                </div>
                <div style="background:rgba(255,255,255,0.06); border-radius:4px; height:8px;">
                    <div style="width:{pct_heat}%; background:{color}; height:8px; border-radius:4px; transition:width 0.5s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 💨 CO₂ emisije – tekući sat")
        co2_now = ob['co2_rate']
        co2_target = 4.5
        co2_delta = co2_now - co2_target
        st.metric("CO₂ intenzitet", f"{co2_now:.2f} tCO₂/h",
                  delta=f"{co2_delta:+.2f} vs cilj",
                  delta_color="inverse")

        # Zalihe
        st.divider()
        st.markdown("#### 📦 Zalihe goriva")
        gas_pct = ob['gas_remaining'] / 150000 * 100
        bio_pct = ob['biomass_remaining'] / 200000 * 100
        col_g, col_b = st.columns(2)
        with col_g:
            st.metric("Plin", f"{ob['gas_remaining']/1000:.0f}k m³")
            st.progress(gas_pct / 100)
        with col_b:
            st.metric("Biomasa", f"{ob['biomass_remaining']/1000:.0f}k t")
            st.progress(bio_pct / 100)

    # Planirano vs ostvareno – 24h pregled
    st.divider()
    section_header("📈", "Planirano vs ostvareno – 24h")
    hours = list(range(24))
    np.random.seed(123)
    planned = st.session_state.optimizer_load
    actual = planned + np.random.normal(0, 8, 24)
    actual = np.clip(actual, 50, 160)

    fig_pa = go.Figure()
    fig_pa.add_trace(go.Scatter(
        x=hours, y=planned, name='Planirano', mode='lines',
        line=dict(color='#1565C0', width=2, dash='dash'),
    ))
    fig_pa.add_trace(go.Scatter(
        x=hours, y=actual, name='Ostvareno', mode='lines+markers',
        line=dict(color='#00BCD4', width=2.5),
        marker=dict(size=5, color='#00BCD4'),
    ))
    fig_pa.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=np.concatenate([planned * 1.05, (planned * 0.95)[::-1]]),
        fill='toself', fillcolor='rgba(21,101,192,0.07)',
        line=dict(color='rgba(0,0,0,0)'), name='±5% zona', hoverinfo='skip',
    ))
    fig_pa.update_layout(
        title="Potrošnja – planirano vs ostvareno (MWh/h)", **PLOTLY_LAYOUT,
        xaxis_title="Sat", yaxis_title="MWh/h", height=320,
    )
    st.plotly_chart(fig_pa, use_container_width=True)


# ============================================================
# 3. OPTIMIZACIJA D-1 – MILP
# ============================================================
elif menu == "📅 Optimizacija D-1":
    section_header("📅", "Optimizirani plan dan-unaprijed – MILP v2")

    if not MILP_OK:
        st.error("⚠️ MILP optimizer nije dostupan. Provjeri instalaciju PuLP paketa.")
        st.stop()

    st.info("**MILP v2** modelira: binarne varijable punjenje/pražnjenje baterije, degradaciju, "
            "CO₂ troškove, mrežna ograničenja, min/max SOC i opcionalni završni SOC.", icon="🧠")

    # --- TABS ---
    tab_prog, tab_params, tab_adv = st.tabs(["📈 Prognoza D-1", "⚙️ Parametri", "🔧 Napredno"])

    with tab_prog:
        st.markdown("#### Uredi 24h prognozu")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Spot cijena CROPEX (€/MWh)**")
            spot_txt = st.text_area("Spot (24 vrijednosti, zarez)", 
                                     ",".join([f"{x:.1f}" for x in st.session_state.optimizer_spot]),
                                     height=80, key="spot_in")
            try:
                ns = np.array([float(x.strip()) for x in spot_txt.split(",")])
                if len(ns) == 24: st.session_state.optimizer_spot = ns
                else: st.warning("Treba 24 vrijednosti")
            except: pass

            st.markdown("**Prognoza potrošnje (MWh/h)**")
            load_txt = st.text_area("Potrošnja",
                                     ",".join([f"{x:.1f}" for x in st.session_state.optimizer_load]),
                                     height=80, key="load_in")
            try:
                nl = np.array([float(x.strip()) for x in load_txt.split(",")])
                if len(nl) == 24: st.session_state.optimizer_load = nl
            except: pass

        with col_b:
            st.markdown("**FNE prognoza (MWh/h)**")
            fne_txt = st.text_area("FNE",
                                    ",".join([f"{x:.1f}" for x in st.session_state.optimizer_fne]),
                                    height=80, key="fne_in")
            try:
                nf = np.array([float(x.strip()) for x in fne_txt.split(",")])
                if len(nf) == 24: st.session_state.optimizer_fne = nf
            except: pass

            st.markdown("**EUA cijena (€/tCO₂)**")
            eua_txt = st.text_area("EUA",
                                    ",".join([f"{x:.1f}" for x in st.session_state.optimizer_eua]),
                                    height=80, key="eua_in")
            try:
                ne = np.array([float(x.strip()) for x in eua_txt.split(",")])
                if len(ne) == 24: st.session_state.optimizer_eua = ne
            except: pass

        # Pregled prognoze
        hours = list(range(24))
        fig_prog = go.Figure()
        fig_prog.add_trace(go.Scatter(x=hours, y=st.session_state.optimizer_load,
                                      name='Potrošnja', line=dict(color='#EF5350', width=2)))
        fig_prog.add_trace(go.Scatter(x=hours, y=st.session_state.optimizer_fne,
                                      name='FNE', fill='tozeroy',
                                      fillcolor='rgba(76,175,80,0.15)',
                                      line=dict(color='#4CAF50', width=2)))
        fig_prog.add_trace(go.Bar(x=hours, y=st.session_state.optimizer_spot,
                                   name='Spot €/MWh', opacity=0.5,
                                   marker_color='#FF6B35', yaxis='y2'))
        fig_prog.update_layout(
            title="24h prognoza", **PLOTLY_LAYOUT, height=320,
            yaxis=dict(title="MWh/h", **PLOTLY_LAYOUT['yaxis']),
            yaxis2=dict(title="€/MWh", overlaying='y', side='right',
                        color='#FF6B35', gridcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_prog, use_container_width=True)

    with tab_params:
        c1, c2, c3 = st.columns(3)
        with c1:
            contracted_vol = st.number_input("Ugovorena količina (MWh)", 0.0, 5000.0, 100.0, 10.0, key="contr_vol")
            contracted_price = st.number_input("Ugovorena cijena (€/MWh)", 0.0, 200.0, 60.0, 1.0, key="contr_price")
            feedin = st.number_input("Otkupna cijena viška (€/MWh)", 0.0, 200.0, 50.0, 1.0, key="feedin")
        with c2:
            batt_cap = st.number_input("Kapacitet baterije (MWh)", 0.1, 100.0, 6.0, 0.5, key="batt_cap")
            batt_pow = st.number_input("Max snaga baterije (MW)", 0.1, 50.0, 1.0, 0.5, key="batt_pow")
            batt_eff = st.slider("Efikasnost baterije (%)", 80, 99, 92, key="batt_eff") / 100
        with c3:
            co2_price = st.number_input("Cijena CO₂ EUA (€/tCO₂)", 0.0, 300.0, float(st.session_state.eua_price), 1.0, key="co2_price")
            co2_intensity = st.number_input("CO₂ intenzitet mreže (tCO₂/MWh)", 0.0, 1.0, 0.40, 0.01, key="co2_int")
            initial_soc_pct = st.slider("Početni SOC (%)", 0, 100, 50, key="init_soc") / 100

    with tab_adv:
        c1, c2, c3 = st.columns(3)
        with c1:
            batt_min_power = st.number_input("Min. snaga (MW)", 0.0, float(batt_pow), 0.1, 0.05, key="bmin_pow")
            batt_cycle_cost = st.number_input("Degradacija (€/MWh protoka)", 0.0, 50.0, 5.0, 0.5, key="b_cyc")
        with c2:
            min_soc_pct = st.slider("Min SOC (%)", 0, 30, 10, key="min_soc") / 100
            max_soc_pct = st.slider("Max SOC (%)", 70, 100, 95, key="max_soc") / 100
        with c3:
            final_soc_pct_val = st.slider("Ciljni završni SOC (%)", 0, 100, 50, key="fin_soc") / 100
            use_final_soc = st.checkbox("Aktiviraj završni SOC", value=False, key="use_fin")
            max_cycles = st.number_input("Max ciklusa/dan", 0, 10, 0, key="max_cyc")

        st.markdown("**Peak satni tarifa (drži Ctrl za višestruki odabir)**")
        peak_hours_sel = st.multiselect(
            "Peak sati",
            options=list(range(24)),
            default=[7, 8, 9, 17, 18, 19, 20],
            format_func=lambda h: f"{h:02d}:00",
            key="peak_h",
        )
        peak_mult = st.slider("Peak množač tarife", 1.0, 3.0, 1.5, 0.1, key="peak_mult")

    # --- POKRETANJE OPTIMIZACIJE ---
    st.divider()
    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        run_milp = st.button("🚀 Pokreni MILP", type="primary", use_container_width=True)
    with col_btn2:
        solver_limit = st.number_input("Limit (s)", 10, 300, 60, 10, key="solv_lim")

    if run_milp:
        optimizer = MILPDayAheadOptimizer(
            load=st.session_state.optimizer_load,
            fne=st.session_state.optimizer_fne,
            spot_price=st.session_state.optimizer_spot,
            contracted_volume=contracted_vol,
            contracted_price=contracted_price,
            batt_capacity_mwh=batt_cap,
            batt_power_mw=batt_pow,
            batt_efficiency=batt_eff,
            co2_intensity=co2_intensity,
            co2_price=co2_price,
            feedin_tariff=feedin,
            batt_min_power=batt_min_power,
            batt_cycle_cost=batt_cycle_cost,
            min_soc_pct=min_soc_pct,
            max_soc_pct=max_soc_pct,
            peak_hours=peak_hours_sel,
            peak_multiplier=peak_mult,
            final_soc_pct=final_soc_pct_val if use_final_soc else None,
            max_cycles_per_day=int(max_cycles) if max_cycles > 0 else None,
            solver_time_limit_s=int(solver_limit),
        )

        with st.spinner(f"🧠 Rješavanje MILP modela (max {solver_limit}s)..."):
            res = optimizer.optimize(initial_soc=initial_soc_pct)

        st.session_state.last_milp_result = res

    res = st.session_state.get('last_milp_result')
    if res is not None:
        if res.status == 'optimal':
            st.success(f"✅ Optimalno rješenje pronađeno za {res.solver_time_s:.1f}s")

            # KPI kartice
            cols = st.columns(6)
            kpi_data = [
                ("Ukupni trošak", fmt_eur(res.total_cost)),
                ("Spot trošak", fmt_eur(res.spot_cost)),
                ("CO₂ emisije", fmt_co2(res.co2_emissions_t)),
                ("Samodostatnost", fmt_pct(res.self_sufficiency_pct)),
                ("Peak shaving", f"{res.peak_shaving_mw:.1f} MW"),
                ("Prodaja viška", fmt_mwh(float(np.sum(res.sales_mwh)))),
            ]
            for col, (lbl, val) in zip(cols, kpi_data):
                with col:
                    kpi_card(lbl, val)

            hours = list(range(1, 25))

            # --- GRAFIKONI ---
            col1, col2 = st.columns(2)
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(name='FNE', x=hours, y=st.session_state.optimizer_fne,
                                      marker_color=COLOR_MAP['FNE']))
                fig1.add_trace(go.Bar(name='Tranše', x=hours, y=res.contract_mwh,
                                      marker_color=COLOR_MAP['Tranše']))
                fig1.add_trace(go.Bar(name='Spot', x=hours, y=res.spot_mwh,
                                      marker_color=COLOR_MAP['Spot']))
                fig1.add_trace(go.Bar(name='BESS dis.', x=hours, y=res.batt_discharge_mwh,
                                      marker_color=COLOR_MAP['Baterija dis']))
                fig1.add_trace(go.Scatter(name='Potrošnja', x=hours, y=st.session_state.optimizer_load,
                                          mode='lines+markers', line=dict(color='#FFFFFF', width=2.5, dash='dot')))
                fig1.update_layout(barmode='stack', title="Optimizirani plan opskrbe (MWh/h)",
                                   **PLOTLY_LAYOUT, xaxis_title="Sat", yaxis_title="MWh", height=380)
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=hours, y=res.soc_mwh, name='SOC', fill='tozeroy',
                    fillcolor='rgba(0,188,212,0.1)',
                    line=dict(color='#00BCD4', width=3),
                    marker=dict(size=6, color='#00BCD4', symbol='circle'),
                ))
                fig2.add_hline(y=batt_cap * (min_soc_pct if 'min_soc_pct' in dir() else 0.1),
                               line_dash="dot", line_color="#EF5350", annotation_text="Min SOC")
                fig2.add_hline(y=batt_cap * (max_soc_pct if 'max_soc_pct' in dir() else 0.95),
                               line_dash="dot", line_color="#FF6B35", annotation_text="Max SOC")
                fig2.update_layout(title="Stanje napunjenosti baterije – SOC (MWh)",
                                   **PLOTLY_LAYOUT, xaxis_title="Sat", yaxis_title="MWh", height=380)
                st.plotly_chart(fig2, use_container_width=True)

            # BESS waterfall
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name='Pražnjenje ↑', x=hours, y=res.batt_discharge_mwh,
                                   marker_color='#4CAF50'))
            fig3.add_trace(go.Bar(name='Punjenje ↓', x=hours, y=-res.batt_charge_mwh,
                                   marker_color='#EF5350'))
            fig3.update_layout(barmode='relative', title="BESS aktivnost – punjenje / pražnjenje",
                               **PLOTLY_LAYOUT, xaxis_title="Sat", yaxis_title="MWh", height=300)
            st.plotly_chart(fig3, use_container_width=True)

            # Spot price vs BESS action
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=hours, y=st.session_state.optimizer_spot,
                                      name='Spot cijena', line=dict(color='#FF6B35', width=2), yaxis='y2'))
            fig4.add_trace(go.Bar(x=hours, y=res.batt_discharge_mwh,
                                   name='BESS pražnjenje', marker_color='rgba(0,188,212,0.6)'))
            fig4.add_trace(go.Bar(x=hours, y=-res.batt_charge_mwh,
                                   name='BESS punjenje', marker_color='rgba(239,83,80,0.6)'))
            fig4.update_layout(
                barmode='relative', title="Arbitraža – spot cijena vs BESS",
                **PLOTLY_LAYOUT, height=300,
                yaxis=dict(title="MWh", **PLOTLY_LAYOUT['yaxis']),
                yaxis2=dict(title="€/MWh", overlaying='y', side='right',
                            color='#FF6B35', gridcolor='rgba(0,0,0,0)'),
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Detaljna tablica
            with st.expander("📋 Detaljna tablica rezultata"):
                df_res = pd.DataFrame({
                    'Sat': hours,
                    'FNE (MWh)': st.session_state.optimizer_fne.round(2),
                    'Spot (MWh)': res.spot_mwh.round(2),
                    'Tranše (MWh)': res.contract_mwh.round(2),
                    'BESS pun. (MWh)': res.batt_charge_mwh.round(2),
                    'BESS pra. (MWh)': res.batt_discharge_mwh.round(2),
                    'SOC (MWh)': res.soc_mwh.round(2),
                    'Prodaja (MWh)': res.sales_mwh.round(2),
                    'Spot €/MWh': st.session_state.optimizer_spot.round(2),
                    'CO₂ t': (res.spot_mwh * co2_intensity).round(3),
                })
                st.dataframe(df_res, use_container_width=True, hide_index=True)

                # Download CSV
                csv = df_res.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                st.download_button("⬇️ Preuzmi CSV", csv,
                                   file_name=f"milp_rezultati_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv")

            # PDF IZVJEŠTAJ
            if REPORT_OK:
                if st.button("📥 Generiraj PDF izvještaj optimizacije"):
                    pdf = PDFReport("Izvještaj MILP optimizacije D-1",
                                    subtitle="Mixed Integer Linear Programming – Energetski plan")
                    pdf.add_title_page()
                    pdf.add_heading("1. Metodologija")
                    pdf.add_paragraph(
                        "Mixed Integer Linear Programming (MILP v2) s binarnim varijablama za kontrolu "
                        "baterije, ograničenjima SOC, troškovima degradacije i CO₂, mrežnim ograničenjima "
                        "i varijabilnom CO₂ intensifikacijom po satu."
                    )
                    pdf.add_heading("2. Ključni pokazatelji")
                    pdf.add_metric_cards({
                        "Ukupni trošak": fmt_eur(res.total_cost),
                        "Spot trošak": fmt_eur(res.spot_cost),
                        "Degradacija": fmt_eur(res.battery_degradation_cost),
                        "Prihod prodaje": fmt_eur(res.sales_revenue),
                        "CO₂ emisije": fmt_co2(res.co2_emissions_t),
                        "Samodostatnost": fmt_pct(res.self_sufficiency_pct),
                    })
                    pdf.add_heading("3. Rezultati po satima")
                    pdf.add_dataframe(df_res, "Optimizirani plan (24h)")
                    pdf.add_plotly_chart(fig1, "Struktura opskrbe")
                    pdf.add_plotly_chart(fig2, "SOC baterije")
                    pdf_bytes = pdf.save()
                    st.download_button("📄 Preuzmi PDF", pdf_bytes,
                                       file_name=f"milp_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                       mime="application/pdf")
        else:
            st.error(f"❌ MILP neuspješan: {res.message}")
            st.warning("Provjeri parametre: je li potrošnja veća od FNE + baterija + max. ugovor + spot?")


# ============================================================
# 4. INVESTICIJSKI KALKULATOR
# ============================================================
elif menu == "💰 Investicijski kalkulator":
    section_header("💰", "Napredni investicijski kalkulator")

    TECH = {
        'BESS (baterija 2h)':   dict(capex_kw=400,  opex_kw=15,  life=15, co2=-0.1,  prod=0.0,  desc='Li-ion, 2h isporuka'),
        'FNE (solarna)':        dict(capex_kw=700,  opex_kw=10,  life=25, co2=-0.8,  prod=1.2,  desc='Fotonaponska elektrana'),
        'Elektrokotao':         dict(capex_kw=150,  opex_kw=5,   life=20, co2=-0.4,  prod=0.0,  desc='Zamjena plinskog kotla'),
        'FNE + BESS':           dict(capex_kw=1100, opex_kw=25,  life=20, co2=-1.0,  prod=1.2,  desc='Integrirani sustav'),
        'Vjetroelektrana':      dict(capex_kw=1300, opex_kw=30,  life=25, co2=-1.2,  prod=2.2,  desc='Kopneni vjetar'),
        'Biomasa CHP':          dict(capex_kw=2000, opex_kw=80,  life=20, co2=-0.6,  prod=0.75, desc='Kogeneracija na biomasu'),
        'Toplotna pumpa':       dict(capex_kw=800,  opex_kw=20,  life=20, co2=-0.5,  prod=0.0,  desc='COP=3.5, grijanje/hlađenje'),
    }

    col_l, col_r = st.columns([1, 1.3])

    with col_l:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="glass-card-title">🔧 Odabir tehnologije i financijski parametri</div>', unsafe_allow_html=True)

        tech = st.selectbox("Tip postrojenja", list(TECH.keys()), key="inv_tech")
        t = TECH[tech]
        st.caption(f"ℹ️ {t['desc']}")

        capacity = st.number_input("Instalirani kapacitet (kW)", 1.0, 100000.0, 1000.0, 50.0, key="inv_cap")

        st.divider()
        use_custom_capex = st.checkbox("Ručni unos CAPEX/OPEX", key="inv_custom")
        if use_custom_capex:
            capex = st.number_input("Ukupni CAPEX (€)", 0.0, 1e9, capacity * t['capex_kw'], 10000.0, key="inv_capex")
            opex = st.number_input("Godišnji OPEX (€)", 0.0, 1e8, capacity * t['opex_kw'], 1000.0, key="inv_opex")
        else:
            capex = capacity * t['capex_kw']
            opex = capacity * t['opex_kw']
            st.info(f"CAPEX: **{fmt_eur(capex)}** | OPEX: **{fmt_eur(opex)}/god**")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            discount_rate = st.slider("Diskontna stopa (%)", 1.0, 15.0, 5.0, 0.5, key="inv_disc") / 100
            electricity_price = st.number_input("Cijena EE (€/MWh)", 20.0, 300.0, 80.0, 5.0, key="inv_ee")
        with c2:
            inflation = st.slider("Inflacija (%/god)", 0.0, 8.0, 2.5, 0.5, key="inv_inf") / 100
            grant_pct = st.slider("Subvencija/bespovratna sredstva (%)", 0.0, 60.0, 0.0, 5.0, key="inv_grant") / 100

        lifetime = t['life']
        net_capex = capex * (1 - grant_pct)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- PRORAČUN ---
    annual_prod = capacity * t['prod'] * 8760 / 1000 if t['prod'] > 0 else 0  # MWh/god
    if tech == 'Elektrokotao':
        annual_savings = capacity * 3.5 * 0.8 * 45.0 / 1000 * 8760  # COP * η * plin_ušted
    elif tech == 'Toplotna pumpa':
        annual_savings = capacity * 3.5 * (electricity_price - 20) / 1e3 * 4000  # gruba procjena
    else:
        # Prihod od prodaje = 80% samo-potrošnja + 20% prodaja
        annual_savings = annual_prod * (0.8 * electricity_price + 0.2 * 50.0)

    # Godišnji novčani tok s inflacijom
    cf = np.zeros(lifetime + 1)
    cf[0] = -net_capex
    for y in range(1, lifetime + 1):
        inflation_factor = (1 + inflation) ** y
        cf[y] = annual_savings * inflation_factor - opex * inflation_factor

    disc_factors = (1 + discount_rate) ** np.arange(lifetime + 1)
    npv = float(np.sum(cf / disc_factors))

    # IRR
    try:
        irr = brentq(lambda r: np.sum(cf / (1 + r) ** np.arange(lifetime + 1)), -0.99, 2.0)
    except:
        irr = None

    # Payback (simpli)
    cum_cf = np.cumsum(cf)
    payback = next(
        (i - cum_cf[i-1] / (cum_cf[i] - cum_cf[i-1]) for i in range(1, len(cum_cf)) if cum_cf[i] >= 0),
        float('inf')
    )

    # LCOE
    if annual_prod > 0:
        lcoe = (net_capex + np.sum(opex / disc_factors[1:])) / max(annual_prod * lifetime, 1)
    else:
        lcoe = 0.0

    # CO₂ redukcija
    co2_reduction = abs(t['co2']) * capacity  # t/god

    with col_r:
        # Rezultati
        c1, c2, c3 = st.columns(3)
        c1.metric("NPV", fmt_eur(npv), delta="Pozitivan" if npv > 0 else "Negativan",
                  delta_color="normal" if npv > 0 else "inverse")
        c2.metric("IRR", f"{irr:.1%}" if irr else "N/A")
        c3.metric("Payback", f"{payback:.1f} god" if payback != float('inf') else "N/A")

        c1, c2, c3 = st.columns(3)
        c1.metric("LCOE", f"{lcoe:.1f} €/MWh" if lcoe > 0 else "N/A")
        c2.metric("CO₂ redukcija", fmt_co2(co2_reduction) + "/god")
        c3.metric("God. prihod", fmt_eur(annual_savings))

        st.metric("Subvencija", fmt_eur(capex * grant_pct), delta=f"Neto CAPEX: {fmt_eur(net_capex)}")

        # Novčani tokovi
        years = list(range(lifetime + 1))
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=years, y=cf,
            marker_color=['#EF5350' if v < 0 else '#4CAF50' for v in cf],
            name="Godišnji tok",
        ))
        fig_cf.add_trace(go.Scatter(
            x=years, y=cum_cf, name="Kumulativno", mode='lines+markers',
            line=dict(color='#00BCD4', width=2.5), yaxis='y2',
        ))
        fig_cf.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig_cf.update_layout(
            title="Novčani tok", **PLOTLY_LAYOUT, height=350,
            yaxis=dict(title="€/god", **PLOTLY_LAYOUT['yaxis']),
            yaxis2=dict(title="€ kumulativno", overlaying='y', side='right',
                        color='#00BCD4', gridcolor='rgba(0,0,0,0)'),
        )
        st.plotly_chart(fig_cf, use_container_width=True)

    # --- USPOREDBA TEHNOLOGIJA ---
    st.divider()
    section_header("🕸️", "Usporedba tehnologija")

    if st.button("🔄 Generiraj usporedbu svih tehnologija", key="gen_comp"):
        comp_rows = []
        for t_name, t_data in TECH.items():
            cap = 1000.0
            c_ = cap * t_data['capex_kw']
            o_ = cap * t_data['opex_kw']
            p_ = cap * t_data['prod'] * 8760 / 1000
            s_ = p_ * 80.0 if p_ > 0 else cap * 3.0 * 45
            cf_ = np.zeros(t_data['life'] + 1)
            cf_[0] = -c_
            for y in range(1, t_data['life'] + 1):
                cf_[y] = s_ - o_
            df_ = (1 + 0.05) ** np.arange(t_data['life'] + 1)
            npv_ = float(np.sum(cf_ / df_))
            try:
                irr_ = brentq(lambda r: np.sum(cf_ / (1+r)**np.arange(t_data['life']+1)), -0.99, 2.0)
            except:
                irr_ = 0.0
            cum_ = np.cumsum(cf_)
            pb_ = next((i - cum_[i-1]/(cum_[i]-cum_[i-1]) for i in range(1,len(cum_)) if cum_[i]>=0), 999)
            lcoe_ = (c_ + np.sum(o_/df_[1:])) / max(p_ * t_data['life'], 1) if p_ > 0 else 0
            comp_rows.append({
                'Tehnologija': t_name,
                'NPV (k€)': npv_ / 1000,
                'IRR (%)': irr_ * 100,
                'Payback (god)': pb_,
                'LCOE (€/MWh)': lcoe_,
                'CO₂ red. (t/god)': abs(t_data['co2']) * cap,
            })

        df_comp = pd.DataFrame(comp_rows)

        # Radar chart
        cats = ['NPV (k€)', 'IRR (%)', 'CO₂ red. (t/god)']
        df_norm = df_comp[['Tehnologija'] + cats].copy()
        for c in cats:
            mx = df_norm[c].abs().max()
            if mx > 0:
                df_norm[c] = df_norm[c] / mx * 10

        fig_radar = go.Figure()
        palette = ['#00BCD4', '#4CAF50', '#FF6B35', '#1565C0', '#AB47BC', '#FF7043', '#26A69A']
        for i, row in df_norm.iterrows():
            r_vals = [row[c] for c in cats] + [row[cats[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals, theta=cats + [cats[0]],
                fill='toself', name=row['Tehnologija'],
                line=dict(color=palette[i % len(palette)]),
                fillcolor=palette[i % len(palette)].replace('#', 'rgba(') + ',0.1)',
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(13,27,42,0.8)',
                radialaxis=dict(visible=True, range=[0, 10], color='#475569', gridcolor='rgba(255,255,255,0.06)'),
                angularaxis=dict(color='#94A3B8'),
            ),
            title="Usporedba tehnologija (normalizirano, 1000 kW baza)",
            **PLOTLY_LAYOUT, height=450,
        )
        col_rad, col_tbl = st.columns([1.2, 1])
        with col_rad:
            st.plotly_chart(fig_radar, use_container_width=True)
        with col_tbl:
            st.dataframe(
                df_comp.style.format({
                    'NPV (k€)': '{:.0f}',
                    'IRR (%)': '{:.1f}',
                    'Payback (god)': '{:.1f}',
                    'LCOE (€/MWh)': '{:.1f}',
                    'CO₂ red. (t/god)': '{:.0f}',
                }).background_gradient(subset=['NPV (k€)', 'IRR (%)'], cmap='RdYlGn'),
                use_container_width=True, hide_index=True,
            )

    # PDF
    if REPORT_OK and st.button("📥 PDF izvještaj investicije"):
        pdf = PDFReport(f"Investicijska analiza – {tech}", subtitle=f"{capacity:.0f} kW | {t['desc']}")
        pdf.add_title_page()
        pdf.add_heading("1. Ključni parametri")
        pdf.add_metric_cards({
            "Tehnologija": tech, "Kapacitet": f"{capacity:.0f} kW",
            "CAPEX": fmt_eur(capex), "OPEX/god": fmt_eur(opex),
            "Subvencija": fmt_pct(grant_pct * 100), "Neto CAPEX": fmt_eur(net_capex),
        })
        pdf.add_heading("2. Rezultati isplativosti")
        pdf.add_metric_cards({
            "NPV": fmt_eur(npv), "IRR": f"{irr:.1%}" if irr else "N/A",
            "Payback": f"{payback:.1f} god" if payback != float('inf') else "N/A",
            "LCOE": f"{lcoe:.1f} €/MWh" if lcoe > 0 else "N/A",
            "God. prihod": fmt_eur(annual_savings), "CO₂ red.": fmt_co2(co2_reduction),
        })
        fig_cf_pdf = go.Figure()
        fig_cf_pdf.add_trace(go.Bar(x=years, y=cf,
                                     marker_color=['#EF5350' if v < 0 else '#4CAF50' for v in cf]))
        fig_cf_pdf.update_layout(title="Novčani tok")
        pdf.add_plotly_chart(fig_cf_pdf, "Godišnji novčani tok")
        pdf_b = pdf.save()
        st.download_button("📄 Preuzmi PDF", pdf_b,
                           file_name=f"investicija_{tech.replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                           mime="application/pdf")


# ============================================================
# 5. SCENARIJ ANALIZA
# ============================================================
elif menu == "🔬 Scenarij analiza":
    section_header("🔬", "Scenarij analiza i osjetljivost")

    st.info("Analiza utjecaja promjene spot cijena i CO₂ na optimalnu strategiju opskrbe.", icon="🔬")

    if not MILP_OK:
        st.error("MILP optimizer nije dostupan.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: sc_batt_cap = st.number_input("Baterija kapacitet (MWh)", 0.1, 50.0, 6.0, 0.5, key="sc_bcap")
    with c2: sc_batt_pow = st.number_input("Baterija snaga (MW)", 0.1, 20.0, 1.0, 0.5, key="sc_bpow")
    with c3: sc_contr_vol = st.number_input("Ugovorena količina (MWh)", 0.0, 500.0, 100.0, 10.0, key="sc_cv")
    with c4: sc_contr_price = st.number_input("Ugovorena cijena (€/MWh)", 0.0, 200.0, 60.0, 1.0, key="sc_cp")

    tab_sc1, tab_sc2, tab_sc3 = st.tabs(["📊 Spot scenariji", "📈 CO₂ osjetljivost", "🎯 Monte Carlo"])

    with tab_sc1:
        st.markdown("#### Tri spot scenarija")
        c1, c2, c3 = st.columns(3)
        with c1: opt_factor = st.slider("Optimistični (% od realnog)", 50, 99, 80, key="sc_opt") / 100
        with c2: pess_factor = st.slider("Pesimistični (% od realnog)", 101, 200, 130, key="sc_pess") / 100
        with c3: init_soc_sc = st.slider("Početni SOC (%)", 0, 100, 50, key="sc_soc") / 100

        if st.button("🚀 Pokreni scenarij analizu", key="run_sc"):
            base = st.session_state.optimizer_spot
            scenarios = {
                '🟢 Optimistični': base * opt_factor,
                '🟡 Realni': base,
                '🔴 Pesimistični': base * pess_factor,
            }

            sc_results = {}
            progress = st.progress(0)
            for i, (sc_name, sc_spot) in enumerate(scenarios.items()):
                opt = MILPDayAheadOptimizer(
                    load=st.session_state.optimizer_load,
                    fne=st.session_state.optimizer_fne,
                    spot_price=sc_spot,
                    contracted_volume=sc_contr_vol,
                    contracted_price=sc_contr_price,
                    batt_capacity_mwh=sc_batt_cap,
                    batt_power_mw=sc_batt_pow,
                    co2_price=float(st.session_state.eua_price),
                    feedin_tariff=50.0,
                    solver_time_limit_s=45,
                )
                r = opt.optimize(init_soc_sc)
                sc_results[sc_name] = r
                progress.progress((i + 1) / len(scenarios))

            # Usporedba rezultata
            st.success("✅ Sva tri scenarija riješena!")
            cols = st.columns(3)
            for col, (sc_name, r) in zip(cols, sc_results.items()):
                with col:
                    st.markdown(f"**{sc_name}**")
                    if r.status == 'optimal':
                        st.metric("Ukupni trošak", fmt_eur(r.total_cost))
                        st.metric("CO₂ emisije", fmt_co2(r.co2_emissions_t))
                        st.metric("Samodostatnost", fmt_pct(r.self_sufficiency_pct))
                    else:
                        st.error("Neuspješan")

            # Usporedni grafikon
            if all(r.status == 'optimal' for r in sc_results.values()):
                hours = list(range(1, 25))
                fig_sc = go.Figure()
                colors_sc = ['#4CAF50', '#00BCD4', '#EF5350']
                for (sc_name, r), color in zip(sc_results.items(), colors_sc):
                    total_sup = r.spot_mwh + r.contract_mwh + st.session_state.optimizer_fne + r.batt_discharge_mwh
                    fig_sc.add_trace(go.Scatter(
                        x=hours, y=total_sup, name=sc_name,
                        line=dict(color=color, width=2.5),
                        mode='lines',
                    ))
                fig_sc.update_layout(title="Ukupna opskrba po scenariju (MWh/h)",
                                     **PLOTLY_LAYOUT, xaxis_title="Sat", yaxis_title="MWh", height=350)
                st.plotly_chart(fig_sc, use_container_width=True)

    with tab_sc2:
        st.markdown("#### Osjetljivost na cijenu CO₂")
        co2_min = st.slider("Min CO₂ cijena", 20, 100, 40, key="co2_min")
        co2_max = st.slider("Max CO₂ cijena", 100, 300, 200, key="co2_max")
        co2_steps = st.slider("Broj koraka", 3, 15, 7, key="co2_steps")

        if st.button("📊 Analiziraj CO₂ osjetljivost", key="run_co2_sens"):
            co2_prices = np.linspace(co2_min, co2_max, co2_steps)
            total_costs, co2_emissions, spot_usages = [], [], []

            progress2 = st.progress(0)
            for i, cp in enumerate(co2_prices):
                opt = MILPDayAheadOptimizer(
                    load=st.session_state.optimizer_load,
                    fne=st.session_state.optimizer_fne,
                    spot_price=st.session_state.optimizer_spot,
                    contracted_volume=sc_contr_vol,
                    contracted_price=sc_contr_price,
                    batt_capacity_mwh=sc_batt_cap,
                    batt_power_mw=sc_batt_pow,
                    co2_price=float(cp),
                    feedin_tariff=50.0,
                    solver_time_limit_s=30,
                )
                r = opt.optimize(0.5)
                if r.status == 'optimal':
                    total_costs.append(r.total_cost)
                    co2_emissions.append(r.co2_emissions_t)
                    spot_usages.append(float(np.sum(r.spot_mwh)))
                else:
                    total_costs.append(None)
                    co2_emissions.append(None)
                    spot_usages.append(None)
                progress2.progress((i + 1) / len(co2_prices))

            valid = [(p, c, e, s) for p, c, e, s in zip(co2_prices, total_costs, co2_emissions, spot_usages) if c is not None]
            if valid:
                ps, cs, es, ss = zip(*valid)
                fig_sens = go.Figure()
                fig_sens.add_trace(go.Scatter(x=list(ps), y=list(cs), name='Ukupni trošak €',
                                              line=dict(color='#EF5350', width=2.5), mode='lines+markers'))
                fig_sens.add_trace(go.Scatter(x=list(ps), y=[e * 100 for e in es], name='CO₂ × 100 tCO₂',
                                              line=dict(color='#4CAF50', width=2.5, dash='dash'), mode='lines+markers',
                                              yaxis='y2'))
                fig_sens.update_layout(
                    title="Osjetljivost na cijenu CO₂",
                    **PLOTLY_LAYOUT, xaxis_title="CO₂ cijena (€/tCO₂)", yaxis_title="Trošak (€)", height=380,
                    yaxis2=dict(title="CO₂ emisije × 100 (tCO₂)", overlaying='y', side='right',
                                color='#4CAF50', gridcolor='rgba(0,0,0,0)'),
                )
                st.plotly_chart(fig_sens, use_container_width=True)

    with tab_sc3:
        st.markdown("#### Monte Carlo simulacija spot cijena")
        st.info("Simulira N slučajnih scenarija spot cijena i analizira distribuciju troškova.", icon="🎲")

        mc_n = st.slider("Broj simulacija", 20, 200, 50, key="mc_n")
        mc_std = st.slider("Standardna devijacija spot (€/MWh)", 5, 30, 12, key="mc_std")

        if st.button("🎲 Pokreni Monte Carlo", key="run_mc"):
            np.random.seed(42)
            base_spot = st.session_state.optimizer_spot
            mc_costs = []
            mc_co2 = []

            progress3 = st.progress(0)
            for i in range(mc_n):
                noise = np.random.normal(0, mc_std, 24)
                sc_spot = np.clip(base_spot + noise, 5, 500)
                opt = MILPDayAheadOptimizer(
                    load=st.session_state.optimizer_load,
                    fne=st.session_state.optimizer_fne,
                    spot_price=sc_spot,
                    contracted_volume=sc_contr_vol,
                    contracted_price=sc_contr_price,
                    batt_capacity_mwh=sc_batt_cap,
                    batt_power_mw=sc_batt_pow,
                    co2_price=float(st.session_state.eua_price),
                    feedin_tariff=50.0,
                    solver_time_limit_s=20,
                )
                r = opt.optimize(0.5)
                if r.status == 'optimal':
                    mc_costs.append(r.total_cost)
                    mc_co2.append(r.co2_emissions_t)
                progress3.progress((i + 1) / mc_n)

            if mc_costs:
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(
                    x=mc_costs, nbinsx=20,
                    marker_color='#1565C0', opacity=0.8,
                    name='Distribucija troškova',
                ))
                fig_mc.add_vline(x=np.mean(mc_costs), line_dash="solid",
                                 line_color="#00BCD4", annotation_text=f"Prosjek: {fmt_eur(np.mean(mc_costs))}",
                                 annotation_font_color="#00BCD4")
                fig_mc.add_vline(x=np.percentile(mc_costs, 5), line_dash="dot",
                                 line_color="#4CAF50", annotation_text="P5",
                                 annotation_font_color="#4CAF50")
                fig_mc.add_vline(x=np.percentile(mc_costs, 95), line_dash="dot",
                                 line_color="#EF5350", annotation_text="P95",
                                 annotation_font_color="#EF5350")
                fig_mc.update_layout(title=f"Monte Carlo distribucija ({mc_n} scenarija)",
                                     **PLOTLY_LAYOUT, xaxis_title="Ukupni trošak (€)", yaxis_title="Frekvencija",
                                     height=380)
                st.plotly_chart(fig_mc, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Prosjek", fmt_eur(np.mean(mc_costs)))
                c2.metric("Medijan", fmt_eur(np.median(mc_costs)))
                c3.metric("P5 (optimistično)", fmt_eur(np.percentile(mc_costs, 5)))
                c4.metric("P95 (pesimistično)", fmt_eur(np.percentile(mc_costs, 95)))


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style="text-align:center; padding:0.8rem; font-size:0.75rem; color:#334155;">
    <strong style="color:#475569;">Danica Energy Optimizer PRO v6.0</strong> &nbsp;·&nbsp; 
    EKONERG – Institut za energetiku i zaštitu okoliša &nbsp;·&nbsp; © 2026 &nbsp;·&nbsp;
    <span style="color:#00BCD4;">MILP Enhanced</span>
</div>
""", unsafe_allow_html=True)
