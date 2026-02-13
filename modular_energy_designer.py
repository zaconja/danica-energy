"""
MODULARNI ENERGETSKI DIZAJNER
==============================
Interaktivno sučelje za slaganje komponenti i optimizaciju tokova energije.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pulp import *

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

# ------------------------------------------------------------
# GLAVNA FUNKCIJA KOJA SE POZIVA IZ APP.PY
# ------------------------------------------------------------
def show_designer():
    st.header("🧩 Modularni energetski dizajner")
    st.markdown("Povlači komponente, spajaj ih i optimiziraj tokove energije.")

    # Inicijalizacija session_state za komponente
    if 'components' not in st.session_state:
        st.session_state.components = [
            {"id": 0, "type": "FNE", "x": 100, "y": 100, "capacity": 100, "production": 80},
            {"id": 1, "type": "Baterija", "x": 300, "y": 100, "capacity": 50, "soc": 25},
            {"id": 2, "type": "Potrošnja", "x": 500, "y": 100, "demand": 120},
            {"id": 3, "type": "Elektrolizator", "x": 300, "y": 250, "capacity": 30, "efficiency": 0.7},
        ]
        st.session_state.connections = [
            {"from": 0, "to": 1},
            {"from": 0, "to": 2},
            {"from": 1, "to": 2},
            {"from": 2, "to": 3},
        ]

    # LAYOUT: dva stupca – lijevo parametri, desno graf
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🔧 Komponente")
        # Prikaz komponenti s klizačima
        for comp in st.session_state.components:
            with st.expander(f"{comp['type']} (ID: {comp['id']})", expanded=False):
                if comp['type'] == "FNE":
                    comp['capacity'] = st.slider(
                        "Kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"cap_{comp['id']}"
                    )
                    comp['production'] = st.slider(
                        "Trenutna proizvodnja (kW)", 0, int(comp['capacity']), int(comp['production']), key=f"prod_{comp['id']}"
                    )
                elif comp['type'] == "Baterija":
                    comp['capacity'] = st.slider(
                        "Kapacitet (kWh)", 0, 200, int(comp['capacity']), key=f"bcap_{comp['id']}"
                    )
                    comp['soc'] = st.slider(
                        "Stanje napunjenosti (kWh)", 0, int(comp['capacity']), int(comp['soc']), key=f"soc_{comp['id']}"
                    )
                elif comp['type'] == "Potrošnja":
                    comp['demand'] = st.slider(
                        "Potrošnja (kW)", 0, 200, int(comp['demand']), key=f"dem_{comp['id']}"
                    )
                elif comp['type'] == "Elektrolizator":
                    comp['capacity'] = st.slider(
                        "Kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"ecap_{comp['id']}"
                    )
                    comp['efficiency'] = st.slider(
                        "Efikasnost", 0.0, 1.0, comp['efficiency'], 0.05, key=f"eeff_{comp['id']}"
                    )

        # Gumb za optimizaciju
        if st.button("⚡ Pokreni optimizaciju", use_container_width=True):
            run_optimization()

    with col_right:
        st.subheader("📊 Dijagram toka")
        # Prikaz sheme – koristimo jednostavan scatter + annotations
        fig = go.Figure()

        # Dodaj točke (komponente)
        for comp in st.session_state.components:
            color = {
                "FNE": "#2E7D32",
                "Baterija": "#1E3A5F",
                "Potrošnja": "#C62828",
                "Elektrolizator": "#FF6B35"
            }.get(comp['type'], "#888888")

            fig.add_trace(go.Scatter(
                x=[comp['x']],
                y=[comp['y']],
                mode='markers+text',
                marker=dict(size=30, color=color, line=dict(width=2, color='white')),
                text=[comp['type']],
                textposition="bottom center",
                name=comp['type'],
                hoverinfo='text',
                hovertext=f"{comp['type']}<br>ID: {comp['id']}"
            ))

        # Dodaj veze (strelicama)
        for conn in st.session_state.connections:
            from_comp = next(c for c in st.session_state.components if c['id'] == conn['from'])
            to_comp = next(c for c in st.session_state.components if c['id'] == conn['to'])
            fig.add_annotation(
                x=to_comp['x'], y=to_comp['y'],
                ax=from_comp['x'], ay=from_comp['y'],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='rgba(0,0,0,0.5)'
            )

        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Prostor za rezultate optimizacije
        if 'opt_results' in st.session_state:
            display_results()


def run_optimization():
    """Pokreće linearnu optimizaciju temeljenu na komponentama."""
    # Ovdje bi trebao biti pravi optimizacijski model.
    # Za sada – simulirani rezultati.
    np.random.seed(42)
    hours = 24
    results = pd.DataFrame({
        'Sat': range(1, hours+1),
        'FNE (kWh)': np.random.normal(80, 15, hours).clip(0, 150),
        'Baterija punjenje (kWh)': np.random.normal(20, 10, hours).clip(0, 50),
        'Baterija pražnjenje (kWh)': np.random.normal(15, 8, hours).clip(0, 40),
        'Elektrolizator (kWh)': np.random.normal(25, 12, hours).clip(0, 60),
        'Potrošnja (kWh)': np.random.normal(100, 20, hours).clip(50, 150),
        'Neto (kWh)': 0
    })
    results['Neto (kWh)'] = results['FNE (kWh)'] + results['Baterija pražnjenje (kWh)'] - results['Baterija punjenje (kWh)'] - results['Elektrolizator (kWh)'] - results['Potrošnja (kWh)']

    st.session_state.opt_results = results
    st.success("✅ Optimizacija završena!")


def display_results():
    """Prikazuje rezultate optimizacije u grafikonima i tablici."""
    results = st.session_state.opt_results

    col1, col2 = st.columns(2)
    with col1:
        fig = px.area(results, x='Sat', y=['FNE (kWh)', 'Baterija pražnjenje (kWh)', 'Elektrolizator (kWh)'],
                      title='Proizvodnja i potrošnja po satu',
                      color_discrete_sequence=['#2E7D32', '#1E3A5F', '#FF6B35'])
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(results, x='Sat', y='Neto (kWh)', title='Neto bilanca (višak/manjak)',
                     color=['#2E7D32' if x >= 0 else '#C62828' for x in results['Neto (kWh)']],
                     color_discrete_sequence=['#2E7D32', '#C62828'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Detaljna tablica"):
        st.dataframe(results.style.format("{:.1f}"), use_container_width=True)

    # Ukupne metrike
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ukupna FNE", f"{results['FNE (kWh)'].sum():.0f} kWh")
    col2.metric("Ukupna potrošnja", f"{results['Potrošnja (kWh)'].sum():.0f} kWh")
    col3.metric("Višak energije", f"{results[results['Neto (kWh)']>0]['Neto (kWh)'].sum():.0f} kWh")
    col4.metric("Manjak energije", f"{abs(results[results['Neto (kWh)']<0]['Neto (kWh)'].sum()):.0f} kWh")
