"""
Dinamički Sankey dijagram toka energije – Dark Theme
Danica Energy Optimizer PRO v7
"""
import plotly.graph_objects as go


def create_energy_sankey(fne, load, batt_ch, batt_dis, electrolyzer=0):
    """
    Kreira interaktivni dark-mode Sankey dijagram toka energije.
    Vrijednosti u kW. Automatski računa uvoz/izvoz iz bilance.
    """
    neto = fne + batt_dis - load - batt_ch - electrolyzer
    grid_export = max(0, neto)
    grid_import = max(0, -neto)

    # ---- ČVOROVI ----
    labels = [
        "☀️ FNE",              # 0
        "🔌 Mreža (uvoz)",     # 1
        "🔋 BESS (pražnjenje)",# 2
        "⚡ Sabirnica",         # 3
        "💡 Potrošnja",        # 4
        "🔋 BESS (punjenje)",  # 5
        "🔌 Mreža (izvoz)",    # 6
        "🧪 Elektrolizator",   # 7
    ]

    node_colors = [
        "rgba(76,175,80,0.85)",    # FNE
        "rgba(33,150,243,0.85)",   # uvoz
        "rgba(0,188,212,0.85)",    # BESS pražnjenje
        "rgba(100,116,139,0.7)",   # sabirnica
        "rgba(239,83,80,0.85)",    # potrošnja
        "rgba(255,152,0,0.85)",    # BESS punjenje
        "rgba(70,130,180,0.85)",   # izvoz
        "rgba(171,71,188,0.85)",   # elektrolizator
    ]

    # ---- LINKOVI: izvori → sabirnica ----
    sources = [0,  1,          2]
    targets = [3,  3,          3]
    values  = [max(0.001, fne), max(0.001, grid_import), max(0.001, batt_dis)]

    link_colors = [
        "rgba(76,175,80,0.35)",
        "rgba(33,150,243,0.35)",
        "rgba(0,188,212,0.35)",
    ]

    # sabirnica → potrošači
    destinations = [
        (4, max(0.001, load),        "rgba(239,83,80,0.3)"),
        (5, max(0.001, batt_ch),     "rgba(255,152,0,0.3)"),
        (6, max(0.001, grid_export), "rgba(70,130,180,0.3)"),
        (7, max(0.001, electrolyzer),"rgba(171,71,188,0.3)"),
    ]
    for tgt, val, col in destinations:
        sources.append(3); targets.append(tgt); values.append(val); link_colors.append(col)

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=22,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=labels,
            color=node_colors,
            hoverlabel=dict(bgcolor="rgba(10,18,35,0.95)",
                            bordercolor="rgba(0,188,212,0.4)",
                            font=dict(color="#E2E8F0", size=12, family="Inter")),
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
            hoverlabel=dict(bgcolor="rgba(10,18,35,0.95)",
                            bordercolor="rgba(0,188,212,0.4)",
                            font=dict(color="#E2E8F0", size=12, family="Inter")),
        ),
    )])

    fig.update_layout(
        title=dict(
            text="Dinamički tok energije",
            font=dict(size=15, color="#CBD5E1", family="Inter"),
            x=0.5,
        ),
        paper_bgcolor="rgba(10,18,35,0.0)",
        plot_bgcolor="rgba(10,18,35,0.0)",
        font=dict(color="#94A3B8", family="Inter", size=12),
        width=820, height=440,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
