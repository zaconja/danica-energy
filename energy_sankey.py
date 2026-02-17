import plotly.graph_objects as go

def create_energy_sankey(fne, load, batt_ch, batt_dis, electrolyzer=0):
    """
    Kreira interaktivni Sankey dijagram toka energije.
    Sve vrijednosti su u kW.
    Automatski izračunava uvoz/izvoz iz mreže na temelju bilance.
    """
    # Izračun neto bilance
    neto = fne + batt_dis - load - batt_ch - electrolyzer
    if neto >= 0:
        grid_export = neto
        grid_import = 0
    else:
        grid_import = -neto
        grid_export = 0

    # Definicija čvorova
    labels = [
        "FNE",               # 0
        "Mreža (uvoz)",      # 1
        "Baterija (pražn.)", # 2
        "Sabirnica",         # 3
        "Potrošnja",         # 4
        "Baterija (punj.)",  # 5
        "Mreža (izvoz)",     # 6
        "Elektrolizator"     # 7
    ]

    node_colors = [
        "#FFD700",   # FNE – zlatna
        "#1E90FF",   # uvoz – plava
        "#32CD32",   # pražnjenje – zelena
        "#A9A9A9",   # sabirnica – siva
        "#FF6347",   # potrošnja – crvena
        "#FFA500",   # punjenje – narančasta
        "#4682B4",   # izvoz – tamnoplava
        "#8A2BE2"    # elektrolizator – ljubičasta
    ]

    # Linkovi: prvo izvori prema sabirnici
    sources = [0, 1, 2]
    targets = [3, 3, 3]
    values = [fne, grid_import, batt_dis]

    # Zatim sabirnica prema potrošačima
    sources += [3, 3, 3, 3]
    targets += [4, 5, 6, 7]
    values += [load, batt_ch, grid_export, electrolyzer]

    # Boje linkova (možeš mijenjati po želji)
    link_colors = [
        "rgba(255,215,0,0.5)",   # FNE
        "rgba(30,144,255,0.5)",  # uvoz
        "rgba(50,205,50,0.5)",   # pražnjenje
        "rgba(160,160,160,0.3)", # sabirnica->potrošnja
        "rgba(160,160,160,0.3)",
        "rgba(160,160,160,0.3)",
        "rgba(160,160,160,0.3)"
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text="Dinamički tok energije",
        font_size=12,
        width=800,
        height=500
    )
    return fig
