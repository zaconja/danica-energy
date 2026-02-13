"""
MODULARNI ENERGETSKI DIZAJNER – PREMIUM IZGLED
===============================================
Interaktivno sučelje sa stilskim ikonama i modernim grafikonima.
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

def get_icon(comp_type):
    """Vraća emoji ikonu za tip komponente."""
    icons = {
        "FNE": "☀️",
        "Baterija": "🔋",
        "Potrošnja": "💡",
        "Elektrolizator": "⚡"
    }
    return icons.get(comp_type, "❓")

def get_color(comp_type):
    """Vraća boju za tip komponente."""
    colors = {
        "FNE": "#2E7D32",
        "Baterija": "#1E3A5F",
        "Potrošnja": "#C62828",
        "Elektrolizator": "#FF6B35"
    }
    return colors.get(comp_type, "#888888")

# ------------------------------------------------------------
# GLAVNA FUNKCIJA
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
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.subheader("🔧 Komponente")
        # Prikaz komponenti s klizačima (u kartama)
        for comp in st.session_state.components:
            icon = get_icon(comp['type'])
            color = get_color(comp['type'])
            with st.expander(f"{icon} {comp['type']} (ID: {comp['id']})", expanded=False):
                if comp['type'] == "FNE":
                    comp['capacity'] = st.slider(
                        "☀️ Kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"cap_{comp['id']}"
                    )
                    comp['production'] = st.slider(
                        "⚡ Trenutna proizvodnja (kW)", 0, int(comp['capacity']), int(comp['production']), key=f"prod_{comp['id']}"
                    )
                elif comp['type'] == "Baterija":
                    comp['capacity'] = st.slider(
                        "🔋 Kapacitet (kWh)", 0, 200, int(comp['capacity']), key=f"bcap_{comp['id']}"
                    )
                    comp['soc'] = st.slider(
                        "📊 Stanje napunjenosti (kWh)", 0, int(comp['capacity']), int(comp['soc']), key=f"soc_{comp['id']}"
                    )
                elif comp['type'] == "Potrošnja":
                    comp['demand'] = st.slider(
                        "💡 Potrošnja (kW)", 0, 200, int(comp['demand']), key=f"dem_{comp['id']}"
                    )
                elif comp['type'] == "Elektrolizator":
                    comp['capacity'] = st.slider(
                        "⚡ Kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"ecap_{comp['id']}"
                    )
                    comp['efficiency'] = st.slider(
                        "🔁 Efikasnost", 0.0, 1.0, comp['efficiency'], 0.05, key=f"eeff_{comp['id']}"
                    )

        # Gumb za optimizaciju
        if st.button("⚡ Pokreni optimizaciju", use_container_width=True):
            run_optimization()

    with col_right:
        st.subheader("📊 Dijagram toka")
        # Prikaz sheme – koristimo scatter s tekstom kao ikonama
        fig = go.Figure()

        # Dodaj točke (komponente) s ikonama
        for comp in st.session_state.components:
            icon = get_icon(comp['type'])
            color = get_color(comp['type'])

            # Osnovni krug
            fig.add_trace(go.Scatter(
                x=[comp['x']],
                y=[comp['y']],
                mode='markers+text',
                marker=dict(
                    size=50,
                    color=color,
                    line=dict(width=3, color='white'),
                    symbol='circle'
                ),
                text=[icon],
                textfont=dict(size=24, color='white'),
                textposition="middle center",
                name=comp['type'],
                hoverinfo='text',
                hovertext=f"<b>{comp['type']}</b><br>ID: {comp['id']}<br>"
                          + (f"Proizvodnja: {comp['production']} kW" if 'production' in comp else '')
                          + (f"SOC: {comp['soc']} kWh" if 'soc' in comp else '')
                          + (f"Potrošnja: {comp['demand']} kW" if 'demand' in comp else '')
                          + (f"Kapacitet: {comp['capacity']} kW" if 'capacity' in comp else ''),
                hoverlabel=dict(bgcolor=color, font_size=14)
            ))

        # Dodaj veze (strelicama) – stilizirane
        for conn in st.session_state.connections:
            from_comp = next(c for c in st.session_state.components if c['id'] == conn['from'])
            to_comp = next(c for c in st.session_state.components if c['id'] == conn['to'])
            fig.add_annotation(
                x=to_comp['x'], y=to_comp['y'],
                ax=from_comp['x'], ay=from_comp['y'],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor='rgba(0,0,0,0.6)',
                standoff=15  # odmak od čvora
            )

        # Stilizacija grafikona
        fig.update_layout(
            showlegend=False,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[0, 600]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                visible=False,
                range=[0, 350]
            ),
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Prostor za rezultate optimizacije
        if 'opt_results' in st.session_state:
            display_results()


def run_optimization():
    """Pokreće linearnu optimizaciju (simulacija)."""
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
    """Prikazuje rezultate optimizacije – moderni grafikoni."""
    results = st.session_state.opt_results

    # 1. Stacked area chart za proizvodnju i potrošnju
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=results['Sat'], y=results['FNE (kWh)'],
        mode='lines',
        line=dict(width=0),
        stackgroup='one',
        name='FNE',
        fillcolor='rgba(46,125,50,0.7)'
    ))
    fig1.add_trace(go.Scatter(
        x=results['Sat'], y=results['Baterija pražnjenje (kWh)'],
        mode='lines',
        line=dict(width=0),
        stackgroup='one',
        name='Baterija (pražnjenje)',
        fillcolor='rgba(30,58,95,0.7)'
    ))
    fig1.add_trace(go.Scatter(
        x=results['Sat'], y=results['Potrošnja (kWh)'],
        mode='lines',
        line=dict(width=0),
        stackgroup='two',
        name='Potrošnja',
        fillcolor='rgba(198,40,40,0.7)'
    ))
    fig1.add_trace(go.Scatter(
        x=results['Sat'], y=results['Elektrolizator (kWh)'],
        mode='lines',
        line=dict(width=0),
        stackgroup='two',
        name='Elektrolizator',
        fillcolor='rgba(255,107,53,0.7)'
    ))
    fig1.update_layout(
        title=dict(text='📈 Proizvodnja i potrošnja', font=dict(size=16, color='#0B2F4D'), x=0.5),
        xaxis=dict(title='Sat', dtick=2),
        yaxis=dict(title='kWh'),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # 2. Bar chart za neto bilancu (višak/manjak)
    colors = ['#2E7D32' if x >= 0 else '#C62828' for x in results['Neto (kWh)']]
    fig2 = go.Figure(data=go.Bar(
        x=results['Sat'],
        y=results['Neto (kWh)'],
        marker_color=colors,
        marker_line_width=0,
        opacity=0.8
    ))
    fig2.update_layout(
        title=dict(text='⚖️ Neto bilanca', font=dict(size=16, color='#0B2F4D'), x=0.5),
        xaxis=dict(title='Sat', dtick=2),
        yaxis=dict(title='kWh'),
        hovermode='x',
        height=300,
        margin=dict(l=40, r=20, t=60, b=40)
    )

    # Prikaz u dva stupca
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # Detaljna tablica
    with st.expander("📋 Detaljna tablica"):
        st.dataframe(results.style.format("{:.1f}"), use_container_width=True)

    # Ukupne metrike u modernim karticama
    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("☀️ Ukupna FNE", f"{results['FNE (kWh)'].sum():.0f} kWh")
    cols[1].metric("💡 Ukupna potrošnja", f"{results['Potrošnja (kWh)'].sum():.0f} kWh")
    cols[2].metric("📈 Višak energije", f"{results[results['Neto (kWh)']>0]['Neto (kWh)'].sum():.0f} kWh")
    cols[3].metric("📉 Manjak energije", f"{abs(results[results['Neto (kWh)']<0]['Neto (kWh)'].sum()):.0f} kWh")
