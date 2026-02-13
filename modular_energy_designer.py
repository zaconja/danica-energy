"""
MODULARNI ENERGETSKI DIZAJNER – PREMIUM VIZUALNI PRIKAZ
========================================================
Interaktivno sučelje s modernim blok dijagramom i realnom simulacijom.
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
    Izračunava satne tokove energije i prosječne snage na vezama.
    """
    # Izdvoji komponente
    fne = next(c for c in components if c['type'] == 'FNE')
    battery = next(c for c in components if c['type'] == 'Baterija')
    load = next(c for c in components if c['type'] == 'Potrošnja')
    electrolyzer = next(c for c in components if c['type'] == 'Elektrolizator')

    # Parametri
    P_fne_max = fne['capacity']                # kW
    P_load_max = load['demand']                 # kW
    E_bat = battery['capacity']                 # kWh
    soc0 = battery['soc']                       # kWh
    P_bat_max = E_bat / 2                        # max snaga baterije
    P_ely_max = electrolyzer['capacity']         # kW
    eff_ely = electrolyzer['efficiency']

    # Tipični profili
    solar_profile = np.array([
        0,0,0,0,0,0.1,0.3,0.5,0.7,0.9,1.0,0.95,
        0.9,0.8,0.6,0.4,0.2,0.1,0,0,0,0,0,0
    ])
    load_profile_norm = np.array([
        0.6,0.5,0.5,0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,
        0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.5,0.5,0.6
    ])

    fne_profile = solar_profile * P_fne_max
    load_profile = load_profile_norm * P_load_max

    # Inicijalizacija
    soc = np.zeros(hours + 1)
    soc[0] = soc0
    ch = np.zeros(hours)
    dis = np.zeros(hours)
    ely = np.zeros(hours)
    grid_import = np.zeros(hours)
    grid_export = np.zeros(hours)

    # Praćenje tokova za svaku vezu (za prikaz prosjeka)
    flow_fne_bat = np.zeros(hours)
    flow_fne_load = np.zeros(hours)
    flow_bat_load = np.zeros(hours)
    flow_load_ely = np.zeros(hours)

    for t in range(hours):
        net = fne_profile[t] - load_profile[t]

        # Prvo baterija
        if net > 0:  # višak
            charge_possible = min(net, P_bat_max, E_bat - soc[t])
            ch[t] = charge_possible
            flow_fne_bat[t] = charge_possible
            net -= charge_possible

            if net > 0:
                ely[t] = min(net, P_ely_max)
                net -= ely[t]
                flow_load_ely[t] = ely[t]

            if net > 0:
                grid_export[t] = net

            soc[t+1] = soc[t] + ch[t]
        else:  # manjak
            deficit = -net
            discharge_possible = min(deficit, P_bat_max, soc[t])
            dis[t] = discharge_possible
            flow_bat_load[t] = discharge_possible
            soc[t+1] = soc[t] - dis[t]
            deficit -= discharge_possible

            if deficit > 0:
                grid_import[t] = deficit

        # FNE uvijek ide prema potrošnji (direktno)
        flow_fne_load[t] = min(fne_profile[t], load_profile[t])

    # Prosječne snage na vezama
    avg_flows = {
        (0, 1): np.mean(flow_fne_bat),   # FNE → Baterija
        (0, 2): np.mean(flow_fne_load),  # FNE → Potrošnja
        (1, 2): np.mean(flow_bat_load),  # Baterija → Potrošnja
        (2, 3): np.mean(flow_load_ely),  # Potrošnja → Elektrolizator
    }

    # Rezultati po satu
    df = pd.DataFrame({
        'Sat': range(1, hours+1),
        'FNE (kWh)': fne_profile,
        'Baterija punjenje (kWh)': ch,
        'Baterija pražnjenje (kWh)': dis,
        'SOC (kWh)': soc[:-1],
        'Elektrolizator (kWh)': ely,
        'Potrošnja (kWh)': load_profile,
        'Uvoz iz mreže (kWh)': grid_import,
        'Izvoz u mrežu (kWh)': grid_export,
        'Neto (kWh)': fne_profile - load_profile - ely + dis - ch
    })
    return df, avg_flows


# ------------------------------------------------------------
# MODERNI DIJAGRAM TOKA
# ------------------------------------------------------------
def create_flow_diagram(components, connections, avg_flows=None):
    """
    Kreira moderni dijagram toka s blokovima i zakrivljenim strelicama.
    """
    fig = go.Figure()

    # Dodaj blokove za svaku komponentu
    for comp in components:
        x, y = comp['x'], comp['y']
        icon = get_icon(comp['type'])
        color = get_color(comp['type'])

        # Tekst unutar bloka
        label = f"{icon} {comp['type']}"
        if 'capacity' in comp and comp['type'] != 'Potrošnja':
            label += f"<br>{comp['capacity']} kW"
        if 'demand' in comp:
            label += f"<br>{comp['demand']} kW"
        if 'soc' in comp:
            label += f"<br>{comp['soc']}/{comp['capacity']} kWh"

        # Pozadinski pravokutnik s gradientom
        fig.add_shape(
            type="rect",
            x0=x-60, y0=y-35, x1=x+60, y1=y+35,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.15,
            layer='below',
            name=f"bg_{comp['id']}"
        )

        # Tekst
        fig.add_annotation(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(size=12, color=color, family='Arial Black'),
            align='center',
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=color,
            borderwidth=2,
            borderpad=6,
            opacity=0.9
        )

    # Dodaj veze (zakrivljene strelice)
    for conn in connections:
        from_id = conn['from']
        to_id = conn['to']
        from_comp = next(c for c in components if c['id'] == from_id)
        to_comp = next(c for c in components if c['id'] == to_id)
        x0, y0 = from_comp['x'], from_comp['y']
        x1, y1 = to_comp['x'], to_comp['y']

        # Kontrolne točke za zakrivljenost
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        offset = 40
        if abs(x1 - x0) > abs(y1 - y0):
            mid_y += offset
        else:
            mid_x += offset

        # Linija
        fig.add_trace(go.Scatter(
            x=[x0, mid_x, x1],
            y=[y0, mid_y, y1],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.5)', width=2, shape='spline'),
            hoverinfo='none',
            showlegend=False
        ))

        # Strelica na kraju
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='rgba(0,0,0,0.6)'
        )

        # Ako imamo prosječni tok, prikaži ga na sredini
        if avg_flows and (from_id, to_id) in avg_flows:
            power = avg_flows[(from_id, to_id)]
            if power > 0.1:
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"{power:.1f} kW",
                    showarrow=False,
                    font=dict(size=10, color='white'),
                    bgcolor='rgba(0,0,0,0.6)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.8
                )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 600]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 350]),
        height=550,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        hovermode='closest'
    )
    return fig


# ------------------------------------------------------------
# GLAVNA FUNKCIJA
# ------------------------------------------------------------
def show_designer():
    st.header("🧩 Modularni energetski dizajner")
    st.markdown("Podešavaj komponente i pokreni simulaciju – grafikoni se ažuriraju!")

    # Inicijalizacija komponenti
    if 'components' not in st.session_state:
        st.session_state.components = [
            {"id": 0, "type": "FNE", "x": 100, "y": 100, "capacity": 100},
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
            with st.expander(f"{get_icon(comp['type'])} {comp['type']}", expanded=False):
                if comp['type'] == "FNE":
                    comp['capacity'] = st.slider(
                        "☀️ Instalirani kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"cap_{comp['id']}"
                    )
                elif comp['type'] == "Baterija":
                    comp['capacity'] = st.slider(
                        "🔋 Kapacitet (kWh)", 0, 200, int(comp['capacity']), key=f"bcap_{comp['id']}"
                    )
                    comp['soc'] = st.slider(
                        "📊 Početno stanje (kWh)", 0, int(comp['capacity']), int(comp['soc']), key=f"soc_{comp['id']}"
                    )
                elif comp['type'] == "Potrošnja":
                    comp['demand'] = st.slider(
                        "💡 Prosječna potrošnja (kW)", 0, 200, int(comp['demand']), key=f"dem_{comp['id']}"
                    )
                elif comp['type'] == "Elektrolizator":
                    comp['capacity'] = st.slider(
                        "⚡ Kapacitet (kW)", 0, 200, int(comp['capacity']), key=f"ecap_{comp['id']}"
                    )
                    comp['efficiency'] = st.slider(
                        "🔁 Efikasnost", 0.0, 1.0, comp['efficiency'], 0.05, key=f"eeff_{comp['id']}"
                    )

        if st.button("⚡ Pokreni simulaciju", use_container_width=True):
            df, avg_flows = run_simulation(st.session_state.components)
            st.session_state.opt_results = df
            st.session_state.avg_flows = avg_flows
            st.success("✅ Simulacija završena!")

    with col_right:
        st.subheader("📊 Dijagram toka")
        if 'avg_flows' in st.session_state:
            fig = create_flow_diagram(
                st.session_state.components,
                st.session_state.connections,
                st.session_state.avg_flows
            )
        else:
            fig = create_flow_diagram(st.session_state.components, st.session_state.connections)
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

    # SOC
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
