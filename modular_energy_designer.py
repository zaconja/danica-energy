import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Danica Energy Designer")

# ---- Inicijalizacija session_state ----
if 'modules' not in st.session_state:
    st.session_state.modules = []  # lista instanciranih modula

if 'connections' not in st.session_state:
    st.session_state.connections = []  # veze između modula

# ---- Definicije modula ----
MODULE_TYPES = {
    'PV': {'icon': '☀️', 'color': '#FFD700', 'params': ['capacity_kw', 'efficiency']},
    'Battery': {'icon': '🔋', 'color': '#2E7D32', 'params': ['capacity_kwh', 'power_kw']},
    'Electrolyzer': {'icon': '💧', 'color': '#1E3A5F', 'params': ['capacity_kw', 'efficiency']},
    'Hydrogen Tank': {'icon': '🧴', 'color': '#A9A9A9', 'params': ['volume_kg', 'pressure']},
    'Fuel Cell': {'icon': '⚡', 'color': '#FF6B35', 'params': ['power_kw', 'efficiency']},
    'Grid': {'icon': '🔌', 'color': '#4A6572', 'params': ['import_price', 'export_price']},
    'Load': {'icon': '📊', 'color': '#C62828', 'params': ['profile_type']},
}

# ---- LAYOUT ----
col_palette, col_canvas, col_props = st.columns([1, 3, 1])

# LIJEVO – paleta modula
with col_palette:
    st.markdown("### 📦 Moduli")
    for name, info in MODULE_TYPES.items():
        if st.button(f"{info['icon']} {name}", key=f"add_{name}"):
            new_id = len(st.session_state.modules)
            st.session_state.modules.append({
                'id': new_id,
                'type': name,
                'params': {p: 0.0 for p in info['params']},
                'pos': (100 + new_id*50, 100)  # privremena pozicija
            })
            st.rerun()

    if st.button("🗑️ Reset"):
        st.session_state.modules = []
        st.session_state.connections = []
        st.rerun()

# SREDINA – platno (simulirani prikaz)
with col_canvas:
    st.markdown("### 🖥️ Radni prostor")

    # Prikaz modula kao kartice (umjesto pravog platna)
    if not st.session_state.modules:
        st.info("Dodaj module s lijeve strane.")
    else:
        cols = st.columns(3)
        for i, mod in enumerate(st.session_state.modules):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**{MODULE_TYPES[mod['type']]['icon']} {mod['type']}**")
                    st.caption(f"ID: {mod['id']}")
                    if st.button(f"✏️ Uredi", key=f"edit_{mod['id']}"):
                        st.session_state.selected_module = mod['id']
                        st.rerun()

    # Gumb za izračun bilance
    if st.button("⚡ Izračunaj energetsku bilancu"):
        # Ovdje će ići poziv MILP optimizer-a s modulima
        st.success("Bilanca izračunata (simulacija).")

# DESNO – parametri odabranog modula
with col_props:
    st.markdown("### ⚙️ Parametri")
    if 'selected_module' in st.session_state:
        mod_id = st.session_state.selected_module
        mod = next((m for m in st.session_state.modules if m['id'] == mod_id), None)
        if mod:
            st.markdown(f"#### {MODULE_TYPES[mod['type']]['icon']} {mod['type']}")
            for param, value in mod['params'].items():
                new_val = st.number_input(param, value=float(value), key=f"param_{mod_id}_{param}")
                mod['params'][param] = new_val
            if st.button("Zatvori"):
                del st.session_state.selected_module
                st.rerun()
    else:
        st.info("Odaberi modul za uređivanje.")

# ---- PRIKAZ ENERGETSKIH TOKOVA (primjer) ----
if st.session_state.modules:
    st.markdown("---")
    st.markdown("### 📊 Trenutni tokovi (simulacija)")

    # Simulirani podaci za 24h
    hours = list(range(24))
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Dodaj proizvodnju (npr. PV)
    pv_present = any(m['type'] == 'PV' for m in st.session_state.modules)
    if pv_present:
        pv_profile = 100 * np.sin(np.linspace(0, np.pi, 24)) ** 2 + np.random.normal(0, 5, 24)
        fig.add_trace(go.Bar(x=hours, y=pv_profile, name="PV (kW)", marker_color='#FFD700'), secondary_y=False)

    # Dodaj potrošnju
    load_profile = 80 + 20 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0, 5, 24)
    fig.add_trace(go.Scatter(x=hours, y=load_profile, name="Potrošnja (kW)", line=dict(color='#C62828', width=3)), secondary_y=False)

    # Dodaj SOC baterije ako postoji
    battery_present = any(m['type'] == 'Battery' for m in st.session_state.modules)
    if battery_present:
        soc = 50 + 30 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0, 2, 24)
        fig.add_trace(go.Scatter(x=hours, y=soc, name="SOC (%)", line=dict(color='#2E7D32', dash='dot')), secondary_y=True)

    fig.update_layout(
        title="Simulirani dnevni profil",
        xaxis_title="Sat",
        hovermode='x unified',
        height=400
    )
    fig.update_yaxes(title_text="Snaga (kW)", secondary_y=False)
    fig.update_yaxes(title_text="SOC (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
