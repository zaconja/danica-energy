"""
MODULARNI ENERGETSKI DIZAJNER – PRAVA SIMULACIJA
=================================================
Interaktivno sučelje sa stvarnim proračunom tokova energije.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# POMOĆNE FUNKCIJE
# ------------------------------------------------------------
def get_icon(comp_type):
    icons = {
        "FNE": "☀️",
        "Baterija": "🔋",
        "Potrošnja": "💡",
        "Elektrolizator": "⚡"
    }
    return icons.get(comp_type, "❓")

def get_color(comp_type):
    colors = {
        "FNE": "#2E7D32",
        "Baterija": "#1E3A5F",
        "Potrošnja": "#C62828",
        "Elektrolizator": "#FF6B35"
    }
    return colors.get(comp_type, "#888888")

# ------------------------------------------------------------
# REALISTIČNA SIMULACIJA (24 sata s dnevnim profilima)
# ------------------------------------------------------------
def run_simulation(components, hours=24):
    """
    Izračunava satne tokove energije na temelju komponenti.
    Koristi tipične dnevne profile za FNE i potrošnju.
    """
    # Izdvoji komponente
    fne = next(c for c in components if c['type'] == 'FNE')
    battery = next(c for c in components if c['type'] == 'Baterija')
    load = next(c for c in components if c['type'] == 'Potrošnja')
    electrolyzer = next(c for c in components if c['type'] == 'Elektrolizator')

    # Parametri
    P_fne_max = fne['capacity']                # kW (instalirani kapacitet)
    P_load_max = load['demand']                  # kW (maksimalna potrošnja)
    E_bat = battery['capacity']                  # kWh
    soc0 = battery['soc']                         # kWh (početno stanje)
    P_bat_max = E_bat / 2                         # pretpostavka: max snaga = pola kapaciteta
    P_ely_max = electrolyzer['capacity']          # kW
    eff_ely = electrolyzer['efficiency']

    # Tipični profili (normalizirani)
    # FNE – solarni profil (veća proizvodnja sredinom dana)
    solar_profile = np.array([
        0,0,0,0,0,0.1,0.3,0.5,0.7,0.9,1.0,0.95,
        0.9,0.8,0.6,0.4,0.2,0.1,0,0,0,0,0,0
    ])
    # Potrošnja – dva vrha (ujutro i navečer)
    load_profile_norm = np.array([
        0.6,0.5,0.5,0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,
        0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.5,0.5,0.6
    ])

    # Skaliraj prema korisničkim vrijednostima
    fne_profile = solar_profile * P_fne_max
    load_profile = load_profile_norm * P_load_max

    # Inicijalizacija rezultata
    soc = np.zeros(hours + 1)
    soc[0] = soc0
    ch = np.zeros(hours)
    dis = np.zeros(hours)
    ely = np.zeros(hours)
    grid_import = np.zeros(hours)
    grid_export = np.zeros(hours)

    for t in range(hours):
        net = fne_profile[t] - load_profile[t]
        # Prvo baterija
        if net > 0:  # višak
            # Možemo puniti bateriju
            charge_possible = min(net, P_bat_max, E_bat - soc[t])
            ch[t] = charge_possible
            net -= charge_possible
            # Ako još ima viška, ide u elektrolizator
            if net > 0:
                ely[t] = min(net, P_ely_max)
                net -= ely[t]
            # Preostalo ide u mrežu (izvoz)
            if net > 0:
                grid_export[t] = net
            soc[t+1] = soc[t] + ch[t]
        else:  # manjak
            deficit = -net
            # Možemo prazniti bateriju
            discharge_possible = min(deficit, P_bat_max, soc[t])
            dis[t] = discharge_possible
            soc[t+1] = soc[t] - dis[t]
            deficit -= discharge_possible
            # Ako još ima manjka, uvoz iz mreže
            if deficit > 0:
                grid_import[t] = deficit

    # Rezultati po satu
    df = pd.DataFrame({
        'Sat': range(1, hours+1),
        'FNE (kWh)': fne_profile,
        'Baterija punjenje (kWh)': ch,
        'Baterija pražnjenje (kWh)': dis,
        'SOC (kWh)': soc[:-1],  # stanje na početku sata
        'Elektrolizator (kWh)': ely,
        'Potrošnja (kWh)': load_profile,
        'Uvoz iz mreže (kWh)': grid_import,
        'Izvoz u mrežu (kWh)': grid_export,
        'Neto (kWh)': fne_profile - load_profile - ely + dis - ch  # bilanca nakon baterije
    })
    return df

# ------------------------------------------------------------
# GLAVNA FUNKCIJA
# ------------------------------------------------------------
def show_designer():
    st.header("🧩 Modularni energetski dizajner")
    st.markdown("Podešavaj komponente i pokreni simulaciju – grafikoni se ažuriraju!")

    # Inicijalizacija komponenti
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

    # LAYOUT
    col_left, col_right = st.columns([1.2, 1.8])

    with col_left:
        st.subheader("🔧 Komponente")
        for comp in st.session_state.components:
            icon = get_icon(comp['type'])
            with st.expander(f"{icon} {comp['type']} (ID: {comp['id']})", expanded=False):
                if comp['type'] == "FNE":
                    comp['capacity'] = st.slider(
                        "☀️ Instalirani kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"cap_{comp['id']}"
                    )
                    # Proizvodnja se sada računa iz profila, ne koristimo klizač za proizvodnju
                    # Možemo ostaviti samo kapacitet
                elif comp['type'] == "Baterija":
                    comp['capacity'] = st.slider(
                        "🔋 Kapacitet (kWh)", 0, 200, int(comp['capacity']), key=f"bcap_{comp['id']}"
                    )
                    comp['soc'] = st.slider(
                        "📊 Početno stanje (kWh)", 0, int(comp['capacity']), int(comp['soc']), key=f"soc_{comp['id']}"
                    )
                elif comp['type'] == "Potrošnja":
                    comp['demand'] = st.slider(
                        "💡 Prosječna dnevna potrošnja (kW)", 0, 200, int(comp['demand']), key=f"dem_{comp['id']}"
                    )
                elif comp['type'] == "Elektrolizator":
                    comp['capacity'] = st.slider(
                        "⚡ Kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"ecap_{comp['id']}"
                    )
                    comp['efficiency'] = st.slider(
                        "🔁 Efikasnost", 0.0, 1.0, comp['efficiency'], 0.05, key=f"eeff_{comp['id']}"
                    )

        if st.button("⚡ Pokreni simulaciju", use_container_width=True):
            df = run_simulation(st.session_state.components)
            st.session_state.opt_results = df
            st.success("✅ Simulacija završena!")

    with col_right:
        st.subheader("📊 Dijagram toka")
        fig = go.Figure()
        for comp in st.session_state.components:
            icon = get_icon(comp['type'])
            color = get_color(comp['type'])
            hover_text = f"<b>{comp['type']}</b><br>ID: {comp['id']}<br>"
            if 'capacity' in comp:
                hover_text += f"Kapacitet: {comp['capacity']} kW<br>"
            if 'soc' in comp:
                hover_text += f"SOC: {comp['soc']} kWh<br>"
            if 'demand' in comp:
                hover_text += f"Potrošnja: {comp['demand']} kW<br>"
            if 'efficiency' in comp:
                hover_text += f"Efikasnost: {comp['efficiency']:.1%}"
            fig.add_trace(go.Scatter(
                x=[comp['x']], y=[comp['y']],
                mode='markers+text',
                marker=dict(size=50, color=color, line=dict(width=3, color='white'), symbol='circle'),
                text=[icon],
                textfont=dict(size=24, color='white'),
                textposition="middle center",
                name=comp['type'],
                hoverinfo='text',
                hovertext=hover_text,
                hoverlabel=dict(bgcolor=color)
            ))
        for conn in st.session_state.connections:
            from_comp = next(c for c in st.session_state.components if c['id'] == conn['from'])
            to_comp = next(c for c in st.session_state.components if c['id'] == conn['to'])
            fig.add_annotation(
                x=to_comp['x'], y=to_comp['y'],
                ax=from_comp['x'], ay=from_comp['y'],
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=3,
                arrowcolor='rgba(0,0,0,0.6)', standoff=15
            )
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 600]),
            yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 350]),
            height=500, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'opt_results' in st.session_state:
            display_results(st.session_state.opt_results)


def display_results(df):
    """Prikazuje rezultate simulacije."""
    # Stacked area chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['Sat'], y=df['FNE (kWh)'], mode='lines', line=dict(width=0),
                              stackgroup='one', name='FNE', fillcolor='rgba(46,125,50,0.7)'))
    fig1.add_trace(go.Scatter(x=df['Sat'], y=df['Baterija pražnjenje (kWh)'], mode='lines', line=dict(width=0),
                              stackgroup='one', name='Baterija (pražnjenje)', fillcolor='rgba(30,58,95,0.7)'))
    fig1.add_trace(go.Scatter(x=df['Sat'], y=df['Potrošnja (kWh)'], mode='lines', line=dict(width=0),
                              stackgroup='two', name='Potrošnja', fillcolor='rgba(198,40,40,0.7)'))
    fig1.add_trace(go.Scatter(x=df['Sat'], y=df['Elektrolizator (kWh)'], mode='lines', line=dict(width=0),
                              stackgroup='two', name='Elektrolizator', fillcolor='rgba(255,107,53,0.7)'))
    fig1.update_layout(
        title='📈 Proizvodnja i potrošnja',
        xaxis_title='Sat', yaxis_title='kWh',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        height=350, margin=dict(l=40, r=20, t=60, b=40)
    )

    # Neto bilanca
    colors = ['#2E7D32' if x >= 0 else '#C62828' for x in df['Neto (kWh)']]
    fig2 = go.Figure(data=go.Bar(x=df['Sat'], y=df['Neto (kWh)'], marker_color=colors, marker_line_width=0, opacity=0.8))
    fig2.update_layout(
        title='⚖️ Neto bilanca (višak/manjak)',
        xaxis_title='Sat', yaxis_title='kWh',
        hovermode='x', height=300, margin=dict(l=40, r=20, t=60, b=40)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # SOC prikaz (zasebni grafikon)
    fig3 = go.Figure(data=go.Scatter(x=df['Sat'], y=df['SOC (kWh)'], mode='lines+markers',
                                     line=dict(color='#1E3A5F', width=3), marker=dict(size=6)))
    fig3.update_layout(
        title='🔋 Stanje napunjenosti baterije (SOC)',
        xaxis_title='Sat', yaxis_title='kWh',
        height=250, margin=dict(l=40, r=20, t=40, b=40)
    )
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Detaljna tablica"):
        st.dataframe(df.style.format("{:.1f}"), use_container_width=True)

    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("☀️ Ukupna FNE", f"{df['FNE (kWh)'].sum():.0f} kWh")
    cols[1].metric("💡 Ukupna potrošnja", f"{df['Potrošnja (kWh)'].sum():.0f} kWh")
    cols[2].metric("📈 Višak energije", f"{df[df['Neto (kWh)']>0]['Neto (kWh)'].sum():.0f} kWh")
    cols[3].metric("📉 Manjak energije", f"{abs(df[df['Neto (kWh)']<0]['Neto (kWh)'].sum()):.0f} kWh")
