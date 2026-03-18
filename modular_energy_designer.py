"""
MODULARNI ENERGETSKI DIZAJNER – DARK PREMIUM
=============================================
Interaktivno sučelje s modernim blok dijagramom i realnom simulacijom.
Dark mode verzija za Danica Energy Optimizer PRO v7.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------
# DARK LAYOUT helper (local copy – ne ovisi o app.py)
# -----------------------------------------------------------------------
_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,18,35,0.6)',
    font=dict(family='Inter, sans-serif', color='#94A3B8', size=11),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#475569',
               linecolor='rgba(255,255,255,0.08)', showline=True),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', color='#475569',
               linecolor='rgba(255,255,255,0.08)', showline=True),
    legend=dict(bgcolor='rgba(10,18,35,0.85)', bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1, font=dict(color='#CBD5E1', size=11)),
    hoverlabel=dict(bgcolor='#0D1A30', font_size=12, bordercolor='rgba(0,188,212,0.3)'),
    margin=dict(l=50, r=25, t=50, b=45),
    colorway=['#00BCD4','#4CAF50','#FF6B35','#1565C0','#AB47BC','#FF7043'],
)

def _dark(fig, h=380, title=None):
    upd = dict(**_DARK, height=h)
    if title:
        upd['title'] = dict(text=title, font=dict(size=15, color='#CBD5E1'), x=0.5)
    fig.update_layout(**upd)
    return fig

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def get_icon(t):
    return {"FNE":"☀️","Baterija":"🔋","Potrošnja":"💡","Elektrolizator":"⚡"}.get(t,"❓")

def get_color(t):
    return {"FNE":"#4CAF50","Baterija":"#00BCD4","Potrošnja":"#EF5350","Elektrolizator":"#FF6B35"}.get(t,"#888")

# -----------------------------------------------------------------------
# SIMULACIJA 24h
# -----------------------------------------------------------------------
def run_simulation(components, hours=24):
    fne        = next(c for c in components if c['type']=='FNE')
    battery    = next(c for c in components if c['type']=='Baterija')
    load       = next(c for c in components if c['type']=='Potrošnja')
    electrolyzer = next(c for c in components if c['type']=='Elektrolizator')

    P_fne   = fne['capacity']
    E_bat   = battery['capacity']
    soc0    = battery['soc']
    P_bat   = E_bat / 2
    P_ely   = electrolyzer['capacity']

    solar  = np.array([0,0,0,0,0,0.1,0.3,0.5,0.7,0.9,1.0,0.95,
                       0.9,0.8,0.6,0.4,0.2,0.1,0,0,0,0,0,0])
    load_n = np.array([0.6,0.5,0.5,0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,
                       0.6,0.7,0.8,0.9,1.0,0.9,0.8,0.7,0.6,0.5,0.5,0.6])

    fne_p  = solar  * P_fne
    load_p = load_n * load['demand']

    soc = np.zeros(hours+1); soc[0] = soc0
    ch = dis = ely = grid_i = grid_e = np.zeros(hours)
    ch, dis, ely, grid_i, grid_e = [np.zeros(hours) for _ in range(5)]
    ff, fl, bl, le = [np.zeros(hours) for _ in range(4)]

    for t in range(hours):
        net = fne_p[t] - load_p[t]
        if net > 0:
            c_ = min(net, P_bat, E_bat - soc[t])
            ch[t] = c_; ff[t] = c_; net -= c_
            ely[t] = min(net, P_ely); net -= ely[t]; le[t] = ely[t]
            if net > 0: grid_e[t] = net
            soc[t+1] = soc[t] + ch[t]
        else:
            deficit = -net
            d_ = min(deficit, P_bat, soc[t])
            dis[t] = d_; bl[t] = d_; soc[t+1] = soc[t] - d_
            deficit -= d_
            if deficit > 0: grid_i[t] = deficit
        fl[t] = min(fne_p[t], load_p[t])

    avg = {(0,1): np.mean(ff), (0,2): np.mean(fl), (1,2): np.mean(bl), (2,3): np.mean(le)}

    df = pd.DataFrame({
        'Sat': range(1, hours+1),
        'FNE (kWh)': fne_p,
        'Baterija punjenje (kWh)': ch,
        'Baterija pražnjenje (kWh)': dis,
        'SOC (kWh)': soc[:-1],
        'Elektrolizator (kWh)': ely,
        'Potrošnja (kWh)': load_p,
        'Uvoz iz mreže (kWh)': grid_i,
        'Izvoz u mrežu (kWh)': grid_e,
        'Neto (kWh)': fne_p - load_p - ely + dis - ch,
    })
    return df, avg

# -----------------------------------------------------------------------
# DIJAGRAM TOKA – DARK
# -----------------------------------------------------------------------
def create_flow_diagram(components, connections, avg_flows=None):
    fig = go.Figure()

    for comp in components:
        x, y = comp['x'], comp['y']
        color = get_color(comp['type'])
        icon  = get_icon(comp['type'])

        label = f"{icon} {comp['type']}"
        if 'capacity' in comp and comp['type'] != 'Potrošnja':
            label += f"<br>{comp['capacity']} kW"
        if 'demand' in comp:
            label += f"<br>{comp['demand']} kW"
        if 'soc' in comp:
            label += f"<br>SOC: {comp['soc']}/{comp['capacity']} kWh"

        # glow rect
        fig.add_shape(
            type="rect",
            x0=x-65, y0=y-38, x1=x+65, y1=y+38,
            line=dict(color=color, width=2),
            fillcolor=color, opacity=0.08, layer='below',
        )
        fig.add_shape(
            type="rect",
            x0=x-63, y0=y-36, x1=x+63, y1=y+36,
            line=dict(color=color, width=1.5),
            fillcolor='rgba(10,18,35,0.92)', layer='below',
        )
        fig.add_annotation(
            x=x, y=y, text=label, showarrow=False,
            font=dict(size=12, color=color, family='Inter'),
            align='center',
            bgcolor='rgba(10,18,35,0.0)',
            borderwidth=0, borderpad=6,
        )

    for conn in connections:
        fc = next(c for c in components if c['id']==conn['from'])
        tc = next(c for c in components if c['id']==conn['to'])
        x0, y0, x1, y1 = fc['x'], fc['y'], tc['x'], tc['y']
        mx = (x0+x1)/2 + (40 if abs(x1-x0)<=abs(y1-y0) else 0)
        my = (y0+y1)/2 + (40 if abs(x1-x0)>abs(y1-y0) else 0)

        fig.add_trace(go.Scatter(
            x=[x0, mx, x1], y=[y0, my, y1],
            mode='lines',
            line=dict(color='rgba(0,188,212,0.35)', width=2, shape='spline'),
            hoverinfo='none', showlegend=False
        ))
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=3, arrowsize=1.5,
            arrowwidth=2, arrowcolor='rgba(0,188,212,0.6)'
        )
        if avg_flows and (conn['from'], conn['to']) in avg_flows:
            pwr = avg_flows[(conn['from'], conn['to'])]
            if pwr > 0.1:
                fig.add_annotation(
                    x=mx, y=my,
                    text=f"{pwr:.1f} kW",
                    showarrow=False,
                    font=dict(size=10, color='#00BCD4'),
                    bgcolor='rgba(10,18,35,0.85)',
                    bordercolor='rgba(0,188,212,0.4)',
                    borderwidth=1, borderpad=4, opacity=0.95
                )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,620]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0,360]),
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(10,18,35,0.0)',
        plot_bgcolor='rgba(10,18,35,0.0)',
    )
    return fig

# -----------------------------------------------------------------------
# REZULTATI
# -----------------------------------------------------------------------
def display_results(df):
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['Sat'], y=df['FNE (kWh)'], mode='lines', line=dict(width=0),
            stackgroup='one', name='FNE', fillcolor='rgba(76,175,80,0.7)'
        ))
        fig1.add_trace(go.Scatter(
            x=df['Sat'], y=df['Baterija pražnjenje (kWh)'], mode='lines', line=dict(width=0),
            stackgroup='one', name='BESS dis.', fillcolor='rgba(0,188,212,0.7)'
        ))
        fig1.add_trace(go.Scatter(
            x=df['Sat'], y=df['Potrošnja (kWh)'], mode='lines', line=dict(width=0),
            stackgroup='two', name='Potrošnja', fillcolor='rgba(239,83,80,0.6)'
        ))
        fig1.add_trace(go.Scatter(
            x=df['Sat'], y=df['Elektrolizator (kWh)'], mode='lines', line=dict(width=0),
            stackgroup='two', name='Elektrolizator', fillcolor='rgba(255,107,53,0.6)'
        ))
        _dark(fig1, 320, "Proizvodnja i potrošnja")
        fig1.update_layout(hovermode='x unified', xaxis_title="Sat", yaxis_title="kWh",
                           legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        colors_n = ['#4CAF50' if x>=0 else '#EF5350' for x in df['Neto (kWh)']]
        fig2 = go.Figure(go.Bar(
            x=df['Sat'], y=df['Neto (kWh)'],
            marker_color=colors_n, opacity=0.85
        ))
        _dark(fig2, 320, "Neto bilanca (višak / manjak)")
        fig2.update_layout(xaxis_title="Sat", yaxis_title="kWh", hovermode='x')
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure(go.Scatter(
        x=df['Sat'], y=df['SOC (kWh)'],
        mode='lines+markers', fill='tozeroy',
        fillcolor='rgba(0,188,212,0.1)',
        line=dict(color='#00BCD4', width=3),
        marker=dict(size=6, color='#00BCD4'),
    ))
    _dark(fig3, 240, "Stanje napunjenosti baterije (SOC)")
    fig3.update_layout(xaxis_title="Sat", yaxis_title="kWh")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Detaljna tablica"):
        st.dataframe(df.style.format("{:.1f}"), use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("☀️ Ukupna FNE", f"{df['FNE (kWh)'].sum():.0f} kWh")
    c2.metric("💡 Ukupna potrošnja", f"{df['Potrošnja (kWh)'].sum():.0f} kWh")
    c3.metric("📈 Višak energije", f"{df[df['Neto (kWh)']>0]['Neto (kWh)'].sum():.0f} kWh")
    c4.metric("📉 Manjak energije", f"{abs(df[df['Neto (kWh)']<0]['Neto (kWh)'].sum()):.0f} kWh")

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def show_designer():
    st.header("🧩 Modularni energetski dizajner")
    st.markdown('<div style="color:#64748B;font-size:0.9rem;margin-bottom:1.2rem;">'
                'Podešavaj komponente i pokreni simulaciju – grafikoni se ažuriraju u stvarnom vremenu.</div>',
                unsafe_allow_html=True)

    if 'components' not in st.session_state:
        st.session_state.components = [
            {"id": 0, "type": "FNE",           "x": 100, "y": 120, "capacity": 100},
            {"id": 1, "type": "Baterija",       "x": 310, "y": 120, "capacity": 50,  "soc": 25},
            {"id": 2, "type": "Potrošnja",      "x": 520, "y": 120, "demand": 120},
            {"id": 3, "type": "Elektrolizator", "x": 310, "y": 280, "capacity": 30,  "efficiency": 0.7},
        ]
        st.session_state.connections = [
            {"from": 0, "to": 1}, {"from": 0, "to": 2},
            {"from": 1, "to": 2}, {"from": 2, "to": 3},
        ]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🔧 Parametri komponenti")
        for comp in st.session_state.components:
            with st.expander(f"{get_icon(comp['type'])} {comp['type']}", expanded=False):
                if comp['type'] == "FNE":
                    comp['capacity'] = st.slider("☀️ Instalirani kapacitet (kW)", 0, 300,
                                                  int(comp['capacity']), key=f"cap_{comp['id']}")
                elif comp['type'] == "Baterija":
                    comp['capacity'] = st.slider("🔋 Kapacitet (kWh)", 0, 300,
                                                  int(comp['capacity']), key=f"bcap_{comp['id']}")
                    comp['soc'] = st.slider("📊 Početni SOC (kWh)", 0, int(comp['capacity']),
                                             min(int(comp['soc']), int(comp['capacity'])),
                                             key=f"soc_{comp['id']}")
                elif comp['type'] == "Potrošnja":
                    comp['demand'] = st.slider("💡 Prosj. potrošnja (kW)", 0, 300,
                                                int(comp['demand']), key=f"dem_{comp['id']}")
                elif comp['type'] == "Elektrolizator":
                    comp['capacity'] = st.slider("⚡ Kapacitet (kW)", 0, 200,
                                                  int(comp['capacity']), key=f"ecap_{comp['id']}")
                    comp['efficiency'] = st.slider("🔁 Efikasnost", 0.0, 1.0,
                                                    comp['efficiency'], 0.05, key=f"eeff_{comp['id']}")

        st.markdown("---")
        if st.button("⚡ Pokreni simulaciju", use_container_width=True, type="primary"):
            df_sim, avg = run_simulation(st.session_state.components)
            st.session_state.sim_df   = df_sim
            st.session_state.sim_avg  = avg
            st.success("✅ Simulacija završena!")

        # KPI sažetak ako imamo rezultate
        if 'sim_df' in st.session_state:
            df_s = st.session_state.sim_df
            ss_pct = float(np.minimum(df_s['FNE (kWh)']+df_s['Baterija pražnjenje (kWh)'],
                                       df_s['Potrošnja (kWh)']).sum()
                           / max(df_s['Potrošnja (kWh)'].sum(), 1) * 100)
            st.markdown("---")
            st.metric("Samodostatnost", f"{ss_pct:.1f}%")
            st.metric("Max FNE", f"{df_s['FNE (kWh)'].max():.0f} kWh")
            st.metric("Ukupni uvoz", f"{df_s['Uvoz iz mreže (kWh)'].sum():.0f} kWh")

    with col_right:
        st.subheader("📐 Dijagram toka energije")
        avg_flows = st.session_state.get('sim_avg')
        fig_flow = create_flow_diagram(
            st.session_state.components,
            st.session_state.connections,
            avg_flows
        )
        st.plotly_chart(fig_flow, use_container_width=True)

        if 'sim_df' in st.session_state:
            display_results(st.session_state.sim_df)
