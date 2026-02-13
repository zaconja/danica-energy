import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# IMPORTI – PDF I MILP OPTIMIZATOR
# ------------------------------------------------------------
from report_generator import PDFReport
from milp_optimizer import MILPDayAheadOptimizer

# ------------------------------------------------------------
# KONFIGURACIJA STRANICE
# ------------------------------------------------------------
st.set_page_config(
    page_title="Danica Energy Optimizer PRO",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS – isti kao prije
st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 700; color: #0B2F4D; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.2rem; color: #4A6572; margin-bottom: 1.5rem; }
    .card { background-color: white; border-radius: 12px; padding: 1.8rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #E8ECF0; margin-bottom: 1.5rem; }
    .metric-card { background: linear-gradient(145deg, #F8FAFC, #EFF2F5); border-radius: 12px; padding: 1.2rem; text-align: center; }
    .metric-label { font-size: 0.8rem; color: #5F6C80; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #0B2F4D; }
    .progress-container { margin-top: 1rem; background-color: #E9ECEF; border-radius: 10px; height: 20px; width: 100%; }
    .progress-fill { background: linear-gradient(90deg, #2E7D32, #43A047); height: 20px; border-radius: 10px; color: white; text-align: center; font-size: 0.8rem; line-height: 20px; }
    .delta-positive { color: #2E7D32; font-weight: 600; }
    .delta-negative { color: #C62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# POMOĆNE FUNKCIJE
# ------------------------------------------------------------
def format_eur(x):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M €"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.0f}k €"
    else:
        return f"{x:.0f} €"

def format_co2(x):
    if abs(x) >= 1e3:
        return f"{x/1e3:.1f}k tCO₂"
    else:
        return f"{x:.0f} tCO₂"

def metric_card(label, value, delta=None, delta_color="normal", suffix=""):
    if isinstance(value, (int, float)):
        val_str = f"{value:,.0f}{suffix}" if suffix else f"{value:,.0f}"
    else:
        val_str = str(value)
    delta_html = ""
    if delta is not None:
        delta_class = "delta-positive" if delta > 0 else "delta-negative"
        delta_sign = "+" if delta > 0 else ""
        delta_html = f'<div style="font-size:0.9rem;" class="{delta_class}">{delta_sign}{delta:,.0f}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val_str}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def progress_bar(value, max_value, label="", color="#2E7D32"):
    percent = min(100, value/max_value*100)
    st.markdown(f"""
    <div>
        <div style="display:flex; justify-content:space-between;">
            <span>{label}</span>
            <span>{value:,.0f} / {max_value:,.0f}</span>
        </div>
        <div class="progress-container">
            <div class="progress-fill" style="width:{percent}%; background:{color};">{percent:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# SESIJA – INICIJALIZACIJA PODATAKA (svi se mogu mijenjati UI)
# ------------------------------------------------------------
if 'portfolio_contracts' not in st.session_state:
    st.session_state.portfolio_contracts = [
        {"Energija": "Tranša 1", "Količina": 5000.0, "Jedinica": "MWh", "Cijena": 59.0, "Status": "Fiksno"},
        {"Energija": "Tranša 2", "Količina": 3000.0, "Jedinica": "MWh", "Cijena": 65.0, "Status": "Fiksno"},
        {"Energija": "Plin", "Količina": 1_500_000.0, "Jedinica": "m³", "Cijena": 35.0, "Status": "Fiksno"}
    ]
if 'portfolio_biomass' not in st.session_state:
    st.session_state.portfolio_biomass = [
        {"Količina": 1000.0, "Cijena": 120.0},
        {"Količina": 1500.0, "Cijena": 115.0}
    ]
if 'co2_total' not in st.session_state:
    st.session_state.co2_total = 8400.0
if 'co2_purchased' not in st.session_state:
    st.session_state.co2_purchased = 6500.0
if 'cropex_spot' not in st.session_state:
    st.session_state.cropex_spot = 78.5

if 'ob_now' not in st.session_state:
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
        'biomass_remaining': 120000.0
    }

if 'optimizer_load' not in st.session_state:
    np.random.seed(42)
    st.session_state.optimizer_load = np.random.normal(120, 20, 24).clip(min=80)
    st.session_state.optimizer_fne = np.random.normal(50, 15, 24).clip(min=0)
    st.session_state.optimizer_spot = np.random.normal(75, 10, 24).clip(min=40)
    st.session_state.optimizer_eua = np.random.normal(35, 8, 24).clip(min=20)

# ------------------------------------------------------------
# NASLOV I SIDEBAR
# ------------------------------------------------------------
st.markdown('<div class="main-title">⚡ Danica Energy Optimizer PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Napredna MILP optimizacija, analiza investicija i izvještavanje</div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/energy.png", width=80)
    st.markdown("## Navigacija")
    menu = st.radio(
        "Odaberi modul",
        ["📊 Pregled portfelja", "⚡ Operativna bilanca", "📅 Optimizacija D-1", "💰 Investicijski kalkulator"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Verzija:** 5.0 – MILP")
    st.markdown("**Status:** ✅ Spreman")

# ------------------------------------------------------------
# 1. PREGLED PORTFELJA
# ------------------------------------------------------------
if menu == "📊 Pregled portfelja":
    st.header("📊 Pregled energetskog portfelja")

    with st.expander("➕ Dodaj / Uredi ugovor", expanded=False):
        with st.form("contract_form"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                en_type = st.text_input("Energija", "Tranša 3")
            with col2:
                quantity = st.number_input("Količina", min_value=0.0, value=2000.0, step=100.0)
                unit = st.selectbox("Jedinica", ["MWh", "m³", "t"])
            with col3:
                price = st.number_input("Cijena (€/jed)", min_value=0.0, value=70.0, step=1.0)
            with col4:
                status = st.selectbox("Status", ["Fiksno", "Indeksirano"])
            submitted = st.form_submit_button("Dodaj ugovor")
            if submitted:
                st.session_state.portfolio_contracts.append({
                    "Energija": en_type,
                    "Količina": quantity,
                    "Jedinica": unit,
                    "Cijena": price,
                    "Status": status
                })
                st.success("Ugovor dodan!")
                st.rerun()

    st.subheader("📋 Ugovorene energije")
    df_contracts = pd.DataFrame(st.session_state.portfolio_contracts)
    edited_df = st.data_editor(df_contracts, use_container_width=True, num_rows="dynamic",
                              column_config={
                                  "Količina": st.column_config.NumberColumn(format="%.0f"),
                                  "Cijena": st.column_config.NumberColumn(format="%.2f")
                              })
    if not edited_df.equals(df_contracts):
        st.session_state.portfolio_contracts = edited_df.to_dict('records')
        st.rerun()

    with st.expander("🌱 Biomasa – zalihe", expanded=False):
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Postojeće zalihe")
            df_bio = pd.DataFrame(st.session_state.portfolio_biomass)
            if not df_bio.empty:
                st.dataframe(df_bio, use_container_width=True)
        with col2:
            st.subheader("Dodaj")
            q_bio = st.number_input("Količina (t)", min_value=0.0, value=500.0, step=100.0, key="bio_q")
            p_bio = st.number_input("Cijena (€/t)", min_value=0.0, value=110.0, step=5.0, key="bio_p")
            if st.button("Dodaj"):
                st.session_state.portfolio_biomass.append({"Količina": q_bio, "Cijena": p_bio})
                st.rerun()

    with st.expander("💨 CO₂ obveze", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.co2_total = st.number_input("Ukupne emisije (tCO₂)", min_value=0.0, value=st.session_state.co2_total, step=100.0)
        with col2:
            st.session_state.co2_purchased = st.number_input("Otkupljeno (tCO₂)", min_value=0.0, value=st.session_state.co2_purchased, step=100.0)
        remaining = st.session_state.co2_total - st.session_state.co2_purchased
        progress_bar(remaining, st.session_state.co2_total, "Preostala obveza", color="#C62828")

    st.subheader("📈 CROPEX Spot cijena")
    st.session_state.cropex_spot = st.number_input("Današnja cijena (€/MWh)", min_value=0.0, value=st.session_state.cropex_spot, step=1.0)

    # Izračun ključnih metrika
    df_el = edited_df[edited_df['Jedinica'] == 'MWh']
    total_mwh = df_el['Količina'].sum() if not df_el.empty else 0.0
    total_cost = (df_el['Količina'] * df_el['Cijena']).sum() if not df_el.empty else 0.0
    avg_price = total_cost / total_mwh if total_mwh > 0 else 0.0
    remaining = st.session_state.co2_total - st.session_state.co2_purchased

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Ukupno ugovoreno", total_mwh, suffix=" MWh")
    with col2:
        metric_card("Prosječna cijena", avg_price, suffix=" €/MWh")
    with col3:
        metric_card("CO₂ preostalo", remaining, suffix=" tCO₂")
    with col4:
        metric_card("CROPEX Spot", st.session_state.cropex_spot, suffix=" €/MWh")

    # Rezidual
    st.markdown("---")
    st.subheader("🔄 Rezidual za Spot tržište")
    total_demand = st.number_input("Planirana potrošnja (MWh)", min_value=0.0, value=10000.0, step=500.0, key="rez_demand")
    residual = max(0.0, total_demand - total_mwh)
    st.metric("Rezidual", f"{residual:,.0f} MWh", delta=f"{residual/total_demand:.1%}" if total_demand else "")

    # What-If analiza
    st.subheader("🔄 What-If analiza (Spot vs Fiksno)")
    col1, col2 = st.columns(2)
    with col1:
        wi_demand = st.number_input("Ukupna potrošnja (MWh)", min_value=0.0, value=10000.0, step=500.0, key="wi_demand")
        wi_spot = st.number_input("Spot cijena (€/MWh)", min_value=0.0, value=st.session_state.cropex_spot, step=1.0, key="wi_spot")
    with col2:
        wi_fixed_share = st.slider("Udio fiksnog dijela", 0.0, 1.0, 0.5, 0.01)
        st.markdown(f"**Spot udio:** {1-wi_fixed_share:.1%}")

    fixed_vol = wi_demand * wi_fixed_share
    scale = fixed_vol / total_mwh if total_mwh > 0 else 0.0
    new_fixed_cost = (df_el['Količina'] * df_el['Cijena'] * scale).sum() if not df_el.empty else 0.0
    new_spot_cost = (wi_demand - fixed_vol) * wi_spot
    new_total = new_fixed_cost + new_spot_cost
    savings = total_cost - new_total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Originalni trošak", format_eur(total_cost))
    col2.metric("Novi trošak", format_eur(new_total), delta=f"{savings:,.0f} €" if savings != 0 else None, delta_color="inverse")
    col3.metric("Izloženost tržištu", f"{(wi_demand-fixed_vol)/wi_demand:.1%}" if wi_demand else "0%")
    col4.metric("Prosječna cijena", f"{new_total/wi_demand:.2f} €/MWh" if wi_demand else "0")

    # Pie chart
    if not df_el.empty:
        fig = px.pie(df_el, values='Količina', names='Energija', title="Udjeli u portfelju (MWh)")
        st.plotly_chart(fig, use_container_width=True)

    # --- PDF IZVJEŠTAJ (PORTFELJ) ---
    if st.button("📥 Preuzmi PDF izvještaj (Portfelj)"):
        pdf = PDFReport("Izvještaj o energetskom portfelju")
        pdf.add_title()
        pdf.add_paragraph(
            "Metodologija: Prikaz svih ugovorenih energija s fiksnim cijenama, zaliha biomase i CO₂ obveza. "
            "What-If analiza temelji se na linearnoj promjeni udjela fiksnog i spot dijela."
        )
        metrics = {
            "Ukupno ugovoreno (MWh)": total_mwh,
            "Prosječna cijena (€/MWh)": avg_price,
            "CO₂ preostalo (t)": remaining,
            "CROPEX Spot (€/MWh)": st.session_state.cropex_spot,
        }
        pdf.add_metric_cards(metrics)
        pdf.add_dataframe(edited_df, "Ugovorene energije")
        
        wi_metrics = {
            "Originalni trošak (€)": total_cost,
            "Novi trošak (€)": new_total,
            "Ušteda (€)": savings,
            "Izloženost tržištu": f"{(wi_demand-fixed_vol)/wi_demand:.1%}" if wi_demand else "0%",
        }
        pdf.add_heading("What-If analiza", 3)
        pdf.add_metric_cards(wi_metrics)
        
        pdf_bytes = pdf.save()
        st.download_button(
            label="📄 Preuzmi PDF",
            data=pdf_bytes,
            file_name=f"portfelj_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

# ------------------------------------------------------------
# 2. OPERATIVNA BILANCA
# ------------------------------------------------------------
elif menu == "⚡ Operativna bilanca":
    st.header("⚡ Operativna energetska bilanca – Danica")
    st.markdown("### 📋 Trenutno stanje (možeš mijenjati)")

    with st.expander("✏️ Uredi trenutne vrijednosti", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.ob_now['fne_power'] = st.number_input("FNE (kW)", min_value=0.0, value=st.session_state.ob_now['fne_power'], step=10.0)
            st.session_state.ob_now['grid_import'] = st.number_input("Iz mreže (kW)", min_value=0.0, value=st.session_state.ob_now['grid_import'], step=10.0)
            st.session_state.ob_now['grid_export'] = st.number_input("U mrežu (kW)", min_value=0.0, value=st.session_state.ob_now['grid_export'], step=10.0)
        with col2:
            st.session_state.ob_now['bess_charge'] = st.number_input("BESS punjenje (kW)", min_value=0.0, value=st.session_state.ob_now['bess_charge'], step=10.0)
            st.session_state.ob_now['bess_discharge'] = st.number_input("BESS pražnjenje (kW)", min_value=0.0, value=st.session_state.ob_now['bess_discharge'], step=10.0)
            st.session_state.ob_now['thermal_power'] = st.number_input("Toplinski sustav (kW)", min_value=0.0, value=st.session_state.ob_now['thermal_power'], step=10.0)
        with col3:
            st.session_state.ob_now['co2_rate'] = st.number_input("CO₂ emisije (tCO₂/h)", min_value=0.0, value=st.session_state.ob_now['co2_rate'], step=0.1, format="%.2f")
            st.session_state.ob_now['plan_fne'] = st.number_input("Plan FNE (kWh)", min_value=0.0, value=st.session_state.ob_now['plan_fne'], step=100.0)
            st.session_state.ob_now['plan_bess'] = st.number_input("Plan BESS pražnjenje (%)", min_value=0.0, value=st.session_state.ob_now['plan_bess'], step=1.0)

    # Izračun bilance
    balance = (st.session_state.ob_now['fne_power'] + st.session_state.ob_now['bess_discharge']
               - st.session_state.ob_now['bess_charge'] - st.session_state.ob_now['grid_export']
               + st.session_state.ob_now['grid_import'])
    co2_daily = st.session_state.ob_now['co2_rate'] * 24

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Trenutno stanje bilance", balance, suffix=" kW")
    with col2:
        metric_card("CO₂ emisije (danas, procjena)", co2_daily, suffix=" tCO₂")
    with col3:
        metric_card("FNE proizvodnja", st.session_state.ob_now['fne_power'], suffix=" kW")

    # Plan vs stvarno
    st.subheader("📊 Plan vs. Stvarno")
    col1, col2 = st.columns(2)
    with col1:
        delta_fne = st.session_state.ob_now['fne_power'] * 24 - st.session_state.ob_now['plan_fne']
        st.metric("FNE", f"{st.session_state.ob_now['fne_power']*24:,.0f} kWh",
                  delta=f"{delta_fne:,.0f} kWh", delta_color="inverse")
    with col2:
        delta_bess = st.session_state.ob_now['bess_discharge'] - st.session_state.ob_now['plan_bess']
        st.metric("BESS pražnjenje", f"{st.session_state.ob_now['bess_discharge']:.0f} kW",
                  delta=f"{delta_bess:.0f} kW", delta_color="inverse")

    # Grafikon proizvodnje/potrošnje
    st.subheader("⚙️ Proizvodnja / potrošnja (trenutno)")
    df_power = pd.DataFrame({
        "Kategorija": ["FNE", "BESS pražnjenje", "Iz mreže", "U mrežu", "BESS punjenje", "Toplina"],
        "Snaga (kW)": [
            st.session_state.ob_now['fne_power'],
            st.session_state.ob_now['bess_discharge'],
            st.session_state.ob_now['grid_import'],
            -st.session_state.ob_now['grid_export'],
            -st.session_state.ob_now['bess_charge'],
            st.session_state.ob_now['thermal_power']
        ],
        "Tip": ["Proizvodnja", "Proizvodnja", "Proizvodnja", "Potrošnja", "Potrošnja", "Proizvodnja"]
    })
    fig = px.bar(df_power, x="Kategorija", y="Snaga (kW)", color="Tip",
                 color_discrete_map={"Proizvodnja": "#2E7D32", "Potrošnja": "#C62828"},
                 title="Trenutni tokovi energije")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Pie proizvodnja
    df_prod = df_power[df_power['Tip'] == 'Proizvodnja'].copy()
    df_prod['Snaga (kW)'] = df_prod['Snaga (kW)'].abs()
    fig2 = px.pie(df_prod, values='Snaga (kW)', names='Kategorija', title="Udio u proizvodnji")
    st.plotly_chart(fig2, use_container_width=True)

    # Dnevni profil
    st.subheader("📈 Dnevni profil (simulacija)")
    col1, col2 = st.columns(2)
    with col1:
        peak_load = st.slider("Maksimalna potrošnja (MWh/h)", 80.0, 200.0, 150.0)
        peak_fne = st.slider("Maksimalna FNE (MWh/h)", 30.0, 100.0, 70.0)
    with col2:
        load_pattern = st.selectbox("Obrazac potrošnje", ["Industrijski", "Uslužni", "Stambeni"])
        fne_pattern = st.selectbox("Obrazac FNE", ["Sunčano", "Oblačno", "Varijabilno"])

    hours = list(range(24))
    if load_pattern == "Industrijski":
        load_curve = 80 + 40 * np.sin(np.linspace(0, 2*np.pi, 24) + 0.5) + np.random.normal(0,5,24)
    elif load_pattern == "Uslužni":
        load_curve = 60 + 50 * (np.sin(np.linspace(-1.5, 1.5, 24))**2) + np.random.normal(0,5,24)
    else:
        load_curve = 50 + 30 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0,3,24)
    load_curve = np.clip(load_curve * peak_load/100, 50, peak_load+20)

    if fne_pattern == "Sunčano":
        fne_curve = peak_fne * np.array([0,0,0,0,0,5,30,60,85,95,100,95,85,70,50,30,15,5,0,0,0,0,0,0])/100
    elif fne_pattern == "Oblačno":
        fne_curve = peak_fne * np.array([0,0,0,0,0,2,15,35,50,60,55,45,35,25,20,12,5,1,0,0,0,0,0,0])/100
    else:
        fne_curve = peak_fne * np.array([0,0,0,0,0,5,30,70,90,70,50,30,80,90,60,30,10,5,0,0,0,0,0,0])/100
    fne_curve += np.random.normal(0,2,24)
    fne_curve = np.clip(fne_curve, 0, peak_fne)

    df_day = pd.DataFrame({
        'Sat': hours,
        'Potrošnja (MWh)': load_curve,
        'FNE (MWh)': fne_curve,
        'Neto (MWh)': load_curve - fne_curve
    })
    fig3 = px.line(df_day, x='Sat', y=['Potrošnja (MWh)', 'FNE (MWh)', 'Neto (MWh)'],
                   title="Simulirani dnevni profil", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Toplinska energija
    st.subheader("🔥 Toplinska energija & Plin/Biomasa")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Toplinski izvori**")
        st.session_state.ob_now['gas_boiler'] = st.number_input("Plinski kotao (MW)", min_value=0.0, value=st.session_state.ob_now['gas_boiler'], step=5.0)
        st.session_state.ob_now['biomass_boiler'] = st.number_input("Kotao na biomasu (MW)", min_value=0.0, value=st.session_state.ob_now['biomass_boiler'], step=5.0)
        total_heat = st.session_state.ob_now['gas_boiler'] + st.session_state.ob_now['biomass_boiler']
        st.metric("Ukupna toplinska snaga", f"{total_heat} MW")
        df_heat = pd.DataFrame({
            'Izvor': ['Plinski kotao', 'Biomasa'],
            'Snaga (MW)': [st.session_state.ob_now['gas_boiler'], st.session_state.ob_now['biomass_boiler']]
        })
        fig_heat = px.pie(df_heat, values='Snaga (MW)', names='Izvor', title="Udio u toplinskoj energiji")
        st.plotly_chart(fig_heat, use_container_width=True)
    with col2:
        st.markdown("**Zalihe**")
        st.session_state.ob_now['gas_remaining'] = st.number_input("Preostalo plina (m³)", min_value=0.0, value=st.session_state.ob_now['gas_remaining'], step=1000.0)
        st.session_state.ob_now['biomass_remaining'] = st.number_input("Preostalo biomase (t)", min_value=0.0, value=st.session_state.ob_now['biomass_remaining'], step=1000.0)
        progress_bar(st.session_state.ob_now['gas_remaining'], 200000.0, "Plin")
        progress_bar(st.session_state.ob_now['biomass_remaining'], 200000.0, "Biomasa")

    # --- PDF IZVJEŠTAJ (BILANCA) ---
    if st.button("📥 Preuzmi PDF izvještaj (Bilanca)"):
        pdf = PDFReport("Izvještaj o operativnoj bilanci")
        pdf.add_title()
        pdf.add_paragraph(
            "Metodologija: Prikaz trenutnih tokova energije, usporedba s planom, "
            "simulacija dnevnog profila potrošnje i proizvodnje FNE, "
            "te stanje toplinskih izvora i zaliha."
        )
        metrics = {
            "Stanje bilance (kW)": balance,
            "CO₂ emisije danas (t)": co2_daily,
            "FNE proizvodnja (kW)": st.session_state.ob_now['fne_power'],
            "BESS pražnjenje (kW)": st.session_state.ob_now['bess_discharge'],
        }
        pdf.add_metric_cards(metrics)
        pdf.add_dataframe(df_heat, "Toplinski izvori")
        fig_pdf = px.bar(df_power, x="Kategorija", y="Snaga (kW)", color="Tip",
                         color_discrete_map={"Proizvodnja": "#2E7D32", "Potrošnja": "#C62828"})
        pdf.add_plotly_chart(fig_pdf, "Tokovi energije")
        pdf_bytes = pdf.save()
        st.download_button(
            label="📄 Preuzmi PDF",
            data=pdf_bytes,
            file_name=f"bilanca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
        # ------------------------------------------------------------
# ------------------------------------------------------------
# 3. OPTIMIZACIJA D-1 – NAPREDNI MILP MODEL
# ------------------------------------------------------------
elif menu == "📅 Optimizacija D-1":
    st.header("📅 Optimizirani plan dan-unaprijed – MILP")
    st.markdown("##### *Mixed Integer Linear Programming – realistično modeliranje baterije*")

    with st.expander("📈 Uredi prognozu (24h)", expanded=False):
        st.markdown("**Cijene na CROPEX spot tržištu**")
        spot_vals = st.text_area(
            "Unesi cijene (odvojene zarezom, 24 vrijednosti)",
            value=",".join([f"{x:.1f}" for x in st.session_state.optimizer_spot])
        )
        try:
            new_spot = np.array([float(x.strip()) for x in spot_vals.split(",")])
            if len(new_spot) == 24:
                st.session_state.optimizer_spot = new_spot
            else:
                st.warning("Unesi točno 24 vrijednosti!")
        except:
            st.warning("Neispravan format. Koristi decimalne brojeve odvojene zarezom.")

        st.markdown("**Prognoza potrošnje (MWh/h)**")
        load_vals = st.text_area(
            "Potrošnja",
            value=",".join([f"{x:.1f}" for x in st.session_state.optimizer_load])
        )
        try:
            new_load = np.array([float(x.strip()) for x in load_vals.split(",")])
            if len(new_load) == 24:
                st.session_state.optimizer_load = new_load
        except:
            pass

        st.markdown("**Prognoza FNE (MWh/h)**")
        fne_vals = st.text_area(
            "FNE",
            value=",".join([f"{x:.1f}" for x in st.session_state.optimizer_fne])
        )
        try:
            new_fne = np.array([float(x.strip()) for x in fne_vals.split(",")])
            if len(new_fne) == 24:
                st.session_state.optimizer_fne = new_fne
        except:
            pass

        st.markdown("**EUA cijene (€/tCO₂)**")
        eua_vals = st.text_area(
            "EUA",
            value=",".join([f"{x:.1f}" for x in st.session_state.optimizer_eua])
        )
        try:
            new_eua = np.array([float(x.strip()) for x in eua_vals.split(",")])
            if len(new_eua) == 24:
                st.session_state.optimizer_eua = new_eua
        except:
            pass

    # --- PARAMETRI OPTIMIZACIJE ---
    col1, col2, col3 = st.columns(3)
    with col1:
        contracted_vol = st.number_input(
            "Ugovorena količina (MWh)",
            min_value=0.0, value=100.0, step=10.0
        )
        contracted_price = st.number_input(
            "Ugovorena cijena (€/MWh)",
            min_value=0.0, value=60.0, step=5.0
        )
    with col2:
        batt_cap = st.number_input(
            "Kapacitet baterije (MWh)",
            min_value=0.0, value=6.0, step=1.0
        )
        batt_pow = st.number_input(
            "Snaga baterije (MW)",
            min_value=0.0, value=1.0, step=0.5
        )
    with col3:
        co2_price = st.number_input(
            "Cijena EUA (€/tCO₂)",
            min_value=0.0, value=80.0, step=5.0
        )
        feedin = st.number_input(
            "Otkupna cijena viškova (€/MWh)",
            min_value=0.0, value=50.0, step=5.0
        )

    # --- NAPREDNE POSTAVKE ZA MILP ---
    st.markdown("### 🧠 Napredno modeliranje baterije (MILP)")
    col_adv1, col_adv2, col_adv3 = st.columns(3)
    with col_adv1:
        use_milp = st.checkbox("Koristi MILP (preporučeno)", value=True)
    with col_adv2:
        batt_min_power = st.number_input(
            "Min. snaga punjenja/pražnjenja (MW)",
            min_value=0.0, max_value=batt_pow, value=0.1, step=0.05
        )
    with col_adv3:
        batt_cycle_cost = st.number_input(
            "Trošak degradacije (€/MWh protoka)",
            min_value=0.0, value=5.0, step=1.0
        )

    # --- POKRETANJE OPTIMIZACIJE ---
    if st.button("🚀 Pokreni MILP optimizaciju", type="primary"):
        # ====== PRILAGODBA NA NOVI OPTIMIZATOR ======
        # 1. Ugovor (pretvorba u listu s jednim elementom)
        contracts = [{
            'volume_max': contracted_vol,
            'volume_min': 0.0,
            'price': contracted_price,
            'indexed': False,
            'must_take': False,
            'hours': list(range(24)),
        }]

        # 2. Baterija
        battery = {
            'capacity': batt_cap,
            'power': batt_pow,
            'efficiency': 0.9,
            'min_power': batt_min_power,
            'cycle_cost': batt_cycle_cost,
            'startup_cost': 0.0,
            'initial_soc': 0.0,
            'target_final_soc': None,
            'target_final_penalty': 0.0,
            'max_cycles': None,
        }

        # 3. Mreža
        grid = {
            'import_limit': float('inf'),
            'export_limit': float('inf'),
            'feedin_tariff': feedin,
        }

        # 4. CO₂
        co2 = {
            'intensity': 0.4,
            'price': co2_price,
            'cap': None,
        }

        # 5. Ostalo (isključeno)
        demand_response = None
        curtailment = False
        T = 24

        # Kreiraj optimizer
        optimizer = MILPDayAheadOptimizer(
            load=st.session_state.optimizer_load,
            fne=st.session_state.optimizer_fne,
            spot_price=st.session_state.optimizer_spot,
            contracts=contracts,
            battery=battery,
            grid=grid,
            co2=co2,
            demand_response=demand_response,
            curtailment=curtailment,
            T=T
        )

        with st.spinner("Rješavanje MILP modela..."):
            res = optimizer.optimize(initial_soc=0.0)

        if res['status'] == 'optimal':
            st.success("✅ MILP optimizacija uspješno završena!")

            # --- METRIKE ---
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                metric_card("Ukupni trošak", res['total_cost'], suffix=" €")
            with col_res2:
                metric_card("CO₂ emisije", res['co2_emissions'], suffix=" tCO₂")
            with col_res3:
                metric_card("Korištenje baterije", np.sum(res['batt_dis']), suffix=" MWh")
            with col_res4:
                metric_card("Prodaja u mrežu", np.sum(res['grid_sales']), suffix=" MWh")

            # --- TABLICA REZULTATA ---
            df_res = pd.DataFrame({
                'Sat': range(1, 25),
                'Spot (MWh)': res['spot_buy'],          # PROMJENA: res['spot'] -> res['spot_buy']
                'Tranše (MWh)': res['contr'],
                'Prodaja (MWh)': res['grid_sales'],
                'FNE (MWh)': st.session_state.optimizer_fne,
                'Punjenje (MWh)': res['batt_ch'],
                'Pražnjenje (MWh)': res['batt_dis'],
                'SOC (MWh)': res['soc']
            })

            # --- GRAFIKONI ---
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                fig1 = go.Figure()
                fig1.add_trace(go.Bar(name='Tranše', x=df_res['Sat'], y=df_res['Tranše (MWh)'], marker_color='#1E3A5F'))
                fig1.add_trace(go.Bar(name='Spot', x=df_res['Sat'], y=df_res['Spot (MWh)'], marker_color='#FF6B35'))
                fig1.add_trace(go.Bar(name='FNE', x=df_res['Sat'], y=df_res['FNE (MWh)'], marker_color='#2E7D32'))
                fig1.update_layout(
                    barmode='stack',
                    title='Optimizirani portfolio (MILP)',
                    xaxis_title='Sat',
                    yaxis_title='MWh',
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col_plot2:
                fig2 = px.line(
                    df_res, x='Sat', y='SOC (MWh)',
                    title='Stanje napunjenosti baterije',
                    markers=True
                )
                fig2.update_traces(line_color='#1E3A5F', line_width=3)
                st.plotly_chart(fig2, use_container_width=True)

            # --- DODATNI GRAFIKON: PUNJENJE/PRAŽNJENJE ---
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name='Pražnjenje', x=df_res['Sat'], y=df_res['Pražnjenje (MWh)'], marker_color='#2E7D32'))
            fig3.add_trace(go.Bar(name='Punjenje', x=df_res['Sat'], y=-df_res['Punjenje (MWh)'], marker_color='#C62828'))
            fig3.update_layout(
                barmode='relative',
                title='BESS – punjenje/pražnjenje',
                xaxis_title='Sat',
                yaxis_title='MWh',
                height=350
            )
            st.plotly_chart(fig3, use_container_width=True)

            # --- DETALJNA TABLICA (EXPANDER) ---
            with st.expander("📋 Detaljna tablica po satima"):
                st.dataframe(
                    df_res.style.format({
                        'Spot (MWh)': '{:.2f}',          # PROMJENA: format stupca
                        'Tranše (MWh)': '{:.2f}',
                        'Prodaja (MWh)': '{:.2f}',
                        'FNE (MWh)': '{:.2f}',
                        'Punjenje (MWh)': '{:.2f}',
                        'Pražnjenje (MWh)': '{:.2f}',
                        'SOC (MWh)': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            # --- PDF IZVJEŠTAJ (OPTIMIZACIJA) ---
            if st.button("📥 Preuzmi PDF izvještaj (Optimizacija)"):
                pdf = PDFReport("Izvještaj optimizacije dan-unaprijed – MILP")
                pdf.add_title()
                pdf.add_paragraph(
                    "Metodologija: Mixed Integer Linear Programming (MILP) s binarnim varijablama "
                    "za sprječavanje istovremenog punjenja i pražnjenja baterije. "
                    "Cilj minimizacije uključuje troškove nabave, CO₂ emisije, troškove degradacije baterije "
                    "i umanjen je za prihode od prodaje viškova. Ograničenja obuhvaćaju energetsku bilancu, "
                    "dinamiku baterije, te minimalne i maksimalne snage."
                )
                metrics = {
                    "Ukupni trošak (€)": res['total_cost'],
                    "CO₂ emisije (t)": res['co2_emissions'],
                    "Korištenje baterije (MWh)": np.sum(res['batt_dis']),
                    "Prodaja u mrežu (MWh)": np.sum(res['grid_sales']),
                }
                pdf.add_metric_cards(metrics)
                pdf.add_dataframe(df_res.round(2), "Rezultati po satima")
                pdf.add_plotly_chart(fig1, "Struktura opskrbe")
                pdf.add_plotly_chart(fig2, "Stanje napunjenosti baterije")
                pdf_bytes = pdf.save()
                st.download_button(
                    label="📄 Preuzmi PDF",
                    data=pdf_bytes,
                    file_name=f"optimizacija_milp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

        else:
            st.error(f"❌ MILP optimizacija nije uspjela: {res['message']}")
# ------------------------------------------------------------
# 4. INVESTICIJSKI KALKULATOR
# ------------------------------------------------------------
elif menu == "💰 Investicijski kalkulator":
    st.header("💰 Napredni investicijski kalkulator")
    st.markdown(
        '<div class="sub-title">Interaktivna analiza isplativosti – unesi vlastite parametre</div>',
        unsafe_allow_html=True
    )

    tech_defaults = {
        'BESS (baterija)': {
            'capex_kw': 400.0, 'opex_kw': 15.0, 'lifetime': 15,
            'co2': 0.1, 'prod': 0.0,
            'desc': 'Litij-ionski spremnik, 2h'
        },
        'FNE (solarna)': {
            'capex_kw': 700.0, 'opex_kw': 10.0, 'lifetime': 25,
            'co2': -0.8, 'prod': 1.2,
            'desc': 'Fotonaponska elektrana'
        },
        'Elektrokotao': {
            'capex_kw': 150.0, 'opex_kw': 5.0, 'lifetime': 20,
            'co2': -0.4, 'prod': 2.0,
            'desc': 'Zamjena za plinski kotao'
        },
        'FNE + BESS': {
            'capex_kw': 1100.0, 'opex_kw': 25.0, 'lifetime': 20,
            'co2': -1.0, 'prod': 1.2,
            'desc': 'Integrirani sustav'
        }
    }

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔧 Odabir tehnologije")
        tech = st.selectbox("Tip postrojenja", list(tech_defaults.keys()))
        st.caption(tech_defaults[tech]['desc'])

        capacity = st.number_input(
            "Instalirani kapacitet (kW)",
            min_value=1.0, value=1000.0, step=50.0
        )

        st.markdown("**Financijski parametri**")
        use_custom_capex = st.checkbox("Ručni unos CAPEX")
        if use_custom_capex:
            capex = st.number_input(
                "Ukupni CAPEX (€)",
                min_value=0.0,
                value=capacity * tech_defaults[tech]['capex_kw'],
                step=10000.0
            )
        else:
            capex = capacity * tech_defaults[tech]['capex_kw']
            st.metric("Preporučeni CAPEX", format_eur(capex))

        use_custom_opex = st.checkbox("Ručni unos OPEX")
        if use_custom_opex:
            opex = st.number_input(
                "Godišnji OPEX (€)",
                min_value=0.0,
                value=capacity * tech_defaults[tech]['opex_kw'],
                step=1000.0
            )
        else:
            opex = capacity * tech_defaults[tech]['opex_kw']
            st.metric("Preporučeni OPEX", format_eur(opex))

        lifetime = st.number_input(
            "Ekonomski vijek (god)",
            min_value=1, value=tech_defaults[tech]['lifetime'], step=1
        )
        discount = st.slider("Diskontna stopa (%)", 0.0, 15.0, 5.0, 0.5) / 100
        inflation = st.slider("Inflacija (%)", 0.0, 5.0, 2.0, 0.1) / 100

        st.markdown("**Energetski parametri**")
        if tech == 'Elektrokotao':
            prod_factor = st.number_input(
                "Potrošnja (MWh/kW/god)",
                min_value=0.0,
                value=float(tech_defaults[tech]['prod']),
                step=0.1
            )
            gas_price = st.number_input(
                "Cijena plina (€/MWh)",
                min_value=0.0, value=45.0, step=5.0
            )
            elec_price = 0.0
            self_cons = 1.0
            feedin = 0.0
        else:
            prod_factor = st.number_input(
                "Specifična proizvodnja (MWh/kW/god)",
                min_value=0.0,
                value=float(tech_defaults[tech]['prod']),
                step=0.1
            )
            elec_price = st.number_input(
                "Cijena el. energije (€/MWh)",
                min_value=0.0, value=80.0, step=5.0
            )
            self_cons = st.slider("Udio vlastite potrošnje", 0.0, 1.0, 0.8, 0.05)
            feedin = st.number_input(
                "Otkupna cijena viškova (€/MWh)",
                min_value=0.0, value=50.0, step=5.0
            )
            gas_price = 0.0

        st.markdown('</div>', unsafe_allow_html=True)

    # --- IZRAČUN POKAZATELJA ---
    annual_prod = capacity * prod_factor
    if tech == 'Elektrokotao':
        annual_savings = annual_prod * gas_price
    else:
        self_cons_energy = annual_prod * self_cons
        exported = annual_prod * (1 - self_cons)
        annual_savings = self_cons_energy * elec_price + exported * feedin

    # Novčani tokovi
    cf = np.zeros(lifetime + 1)
    cf[0] = -capex
    for t in range(1, lifetime + 1):
        cf[t] = annual_savings * (1 + inflation)**(t-1) - opex * (1 + inflation)**(t-1)

    t_vec = np.arange(lifetime + 1)
    npv = np.sum(cf / (1 + discount)**t_vec)

    try:
        irr = brentq(lambda r: np.sum(cf / (1 + r)**t_vec), -0.99, 1.0)
    except:
        irr = None

    cum = np.cumsum(cf)
    payback = float('inf')
    for i in range(1, len(cum)):
        if cum[i] >= 0:
            payback = i - cum[i-1] / (cum[i] - cum[i-1])
            break

    if annual_prod > 0:
        cost_cf = np.zeros_like(cf)
        cost_cf[0] = capex
        for t in range(1, lifetime + 1):
            cost_cf[t] = opex * (1 + inflation)**(t-1)
        total_cost_npv = np.sum(cost_cf / (1 + discount)**t_vec)
        lcoe = total_cost_npv / (annual_prod * lifetime)
    else:
        lcoe = 0.0

    co2_reduction = (
        -tech_defaults[tech]['co2'] * capacity
        if tech_defaults[tech]['co2'] < 0
        else tech_defaults[tech]['co2'] * capacity
    )

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Rezultati isplativosti")
        col1, col2, col3 = st.columns(3)
        col1.metric("NPV", format_eur(npv))
        col2.metric("IRR", f"{irr:.1%}" if irr else "n/a")
        col3.metric("Payback", f"{payback:.1f} god")

        col1, col2, col3 = st.columns(3)
        col1.metric("LCOE", f"{lcoe:.1f} €/MWh" if lcoe > 0 else "n/a")
        col2.metric("CO₂ redukcija", format_co2(co2_reduction))
        col3.metric("God. proizvodnja", f"{annual_prod:,.0f} MWh")
        st.metric("Godišnja ušteda", format_eur(annual_savings))
        st.markdown('</div>', unsafe_allow_html=True)

        # --- GRAFIKON NOVČANOG TOKA ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💰 Novčani tok")
        years = list(range(lifetime + 1))
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(
            x=years, y=cf,
            marker_color=['#C62828' if x < 0 else '#2E7D32' for x in cf]
        ))
        fig_cf.update_layout(
            title='Godišnji novčani tokovi',
            xaxis_title='Godina',
            yaxis_title='€',
            height=350
        )
        st.plotly_chart(fig_cf, use_container_width=True)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=years, y=np.cumsum(cf),
            mode='lines+markers',
            line=dict(color='#1E3A5F', width=3)
        ))
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_cum.update_layout(
            title='Kumulativni novčani tok',
            xaxis_title='Godina',
            yaxis_title='€',
            height=300
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- WATERFALL GRAFIKON (prema slici) ---
        st.subheader("📉 Promjena troškova (godišnje)")
        cost_items = {
            'Ušteda EE': -580000.0,
            'Smjena plina': -220000.0,
            'Trošak BESS': +300000.0,
            'Prihod od prodaje': -190000.0
        }
        df_waterfall = pd.DataFrame({
            'Stavka': list(cost_items.keys()),
            'Iznos (€)': list(cost_items.values())
        })
        fig_water = px.bar(
            df_waterfall, x='Stavka', y='Iznos (€)',
            color=['#2E7D32' if x < 0 else '#C62828' for x in df_waterfall['Iznos (€)']],
            title='Struktura godišnje promjene troška',
            color_discrete_sequence=['#2E7D32', '#C62828']
        )
        fig_water.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_water, use_container_width=True)

        # --- SPIDER CHART – USPOREDBA TEHNOLOGIJA ---
        st.subheader("🕸️ Usporedba tehnologija")
        if st.button("Generiraj usporedbu"):
            techs = list(tech_defaults.keys())
            npvs, irrs, paybacks, lcoes, co2s = [], [], [], [], []
            for t in techs:
                cap = 1000.0
                capex_t = cap * tech_defaults[t]['capex_kw']
                opex_t = cap * tech_defaults[t]['opex_kw']
                prod_t = cap * tech_defaults[t]['prod']
                if t == 'Elektrokotao':
                    save_t = prod_t * 45.0
                else:
                    save_t = prod_t * 0.8 * 80.0 + prod_t * 0.2 * 50.0
                cf_t = np.zeros(21)
                cf_t[0] = -capex_t
                for y in range(1, 21):
                    cf_t[y] = save_t - opex_t
                npv_t = np.sum(cf_t / (1 + 0.05)**np.arange(21))
                try:
                    irr_t = brentq(
                        lambda r: np.sum(cf_t / (1 + r)**np.arange(21)),
                        -0.99, 1.0
                    )
                except:
                    irr_t = 0.0
                cum_t = np.cumsum(cf_t)
                pb_t = next(
                    (
                        i - cum_t[i-1] / (cum_t[i] - cum_t[i-1])
                        for i in range(1, len(cum_t)) if cum_t[i] >= 0
                    ),
                    float('inf')
                )
                lcoe_t = (
                    capex_t / (prod_t * 20) + opex_t / prod_t
                    if prod_t > 0 else 0.0
                )
                co2_t = (
                    -tech_defaults[t]['co2'] * cap
                    if tech_defaults[t]['co2'] < 0
                    else tech_defaults[t]['co2'] * cap
                )
                npvs.append(npv_t)
                irrs.append(irr_t)
                paybacks.append(pb_t)
                lcoes.append(lcoe_t)
                co2s.append(co2_t)

            df_radar = pd.DataFrame({
                'Tehnologija': techs,
                'NPV (M€)': [x / 1e6 for x in npvs],
                'IRR (%)': [x * 100 for x in irrs],
                'Payback (god)': paybacks,
                'LCOE (€/MWh)': lcoes,
                'CO₂ red. (kt)': [x / 1000 for x in co2s]
            }).melt(id_vars='Tehnologija', var_name='Parametar', value_name='Vrijednost')

            fig_radar = px.line_polar(
                df_radar,
                r='Vrijednost',
                theta='Parametar',
                color='Tehnologija',
                line_close=True,
                title='Usporedba tehnologija (normalizirano)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # --- PDF IZVJEŠTAJ (INVESTICIJA) ---
    if st.button("📥 Preuzmi PDF izvještaj (Investicija)"):
        pdf = PDFReport("Izvještaj o isplativosti investicije")
        pdf.add_title()
        pdf.add_paragraph(
            "Metodologija: Izračun diskontiranih novčanih tokova uz uračunatu inflaciju. "
            "NPV – neto sadašnja vrijednost, IRR – interna stopa povrata, "
            "Payback – jednostavno razdoblje povrata, LCOE – nivelirani trošak energije. "
            "Sve vrijednosti izražene su u EUR, emisije CO₂ u tonama."
        )
        metrics = {
            "Tehnologija": tech,
            "Kapacitet (kW)": capacity,
            "CAPEX (€)": capex,
            "OPEX (€/god)": opex,
        }
        pdf.add_metric_cards(metrics)

        res_metrics = {
            "NPV (€)": npv,
            "IRR": f"{irr:.1%}" if irr else "n/a",
            "Payback (god)": f"{payback:.1f}",
            "LCOE (€/MWh)": f"{lcoe:.1f}" if lcoe > 0 else "n/a",
            "CO₂ redukcija (t/god)": co2_reduction,
            "God. ušteda (€)": annual_savings,
        }
        pdf.add_heading("Rezultati isplativosti", 3)
        pdf.add_metric_cards(res_metrics)

        years_plot = list(range(lifetime + 1))
        fig_cf_pdf = go.Figure()
        fig_cf_pdf.add_trace(go.Bar(
            x=years_plot, y=cf,
            marker_color=['#C62828' if x < 0 else '#2E7D32' for x in cf]
        ))
        fig_cf_pdf.update_layout(
            title="Godišnji novčani tokovi",
            xaxis_title="Godina",
            yaxis_title="€"
        )
        pdf.add_plotly_chart(fig_cf_pdf, "Novčani tok")

        pdf_bytes = pdf.save()
        st.download_button(
            label="📄 Preuzmi PDF",
            data=pdf_bytes,
            file_name=f"investicija_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )

# ------------------------------------------------------------
# KRAJ
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Izradio: EKONERG - Institut za energetiku i zaštitu okoliša | 2026")



