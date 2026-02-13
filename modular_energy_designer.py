import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import uuid

# ============================================================================
#  MODULARNI ENERGETSKI DIZAJNER
# ============================================================================
# Omogućuje vizualno slaganje komponenti (FNE, baterija, elektrolizator, potrošnja)
# i optimizaciju tokova energije.
# ============================================================================

class Component:
    """Bazna klasa za sve energetske komponente."""
    def __init__(self, name: str, comp_type: str, x: float = 0, y: float = 0):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.type = comp_type  # 'source', 'storage', 'converter', 'sink'
        self.x = x
        self.y = y
        self.inputs: List[str] = []      # ID komponenti koje ulaze u ovu
        self.outputs: List[str] = []     # ID komponenti koje izlaze iz ove
        self.parameters = {}
        self.results = {}
        
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'x': self.x,
            'y': self.y,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'parameters': self.parameters,
            'results': self.results
        }


class SolarPlant(Component):
    def __init__(self, name="Solarna elektrana", x=0, y=0):
        super().__init__(name, "source", x, y)
        self.parameters = {
            'capacity_kw': 1000.0,
            'profile': 'sunčano',  # sunčano, oblačno, varijabilno
            'cost_per_kwh': 0.0,
            'co2_intensity': 0.0
        }
        self.results = {
            'production_kwh': np.zeros(24),
            'revenue': 0.0
        }


class Battery(Component):
    def __init__(self, name="Baterija", x=0, y=0):
        super().__init__(name, "storage", x, y)
        self.parameters = {
            'capacity_kwh': 6000.0,
            'power_kw': 1000.0,
            'efficiency': 0.92,
            'initial_soc': 0.3,
            'min_soc': 0.1,
            'max_soc': 0.9,
            'cycle_cost': 0.02
        }
        self.results = {
            'soc': np.zeros(24),
            'charge_kwh': np.zeros(24),
            'discharge_kwh': np.zeros(24),
            'cycles': 0
        }


class Electrolyzer(Component):
    def __init__(self, name="Elektrolizator", x=0, y=0):
        super().__init__(name, "converter", x, y)
        self.parameters = {
            'capacity_kw': 500.0,
            'efficiency': 0.7,  # kWh elektriciteta -> kWh H2
            'h2_price': 3.0,     # €/kg (približno 33 kWh/kg)
            'min_load': 0.1
        }
        self.results = {
            'load_kw': np.zeros(24),
            'h2_production_kg': 0.0,
            'revenue': 0.0
        }


class Load(Component):
    def __init__(self, name="Potrošnja", x=0, y=0):
        super().__init__(name, "sink", x, y)
        self.parameters = {
            'base_load_kw': 500.0,
            'profile': 'industrijski',  # industrijski, uslužni, stambeni
            'peak_factor': 1.5,
            'shedding_allowed': False,
            'shedding_penalty': 200.0
        }
        self.results = {
            'load_kw': np.zeros(24),
            'shed_kw': np.zeros(24),
            'cost': 0.0
        }


class EnergyDesignerApp:
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        if 'designer_components' not in st.session_state:
            st.session_state.designer_components = []
        if 'designer_connections' not in st.session_state:
            st.session_state.designer_connections = []  # lista (from_id, to_id)
        if 'designer_next_id' not in st.session_state:
            st.session_state.designer_next_id = 0
            
    def run(self):
        st.markdown("""
        <style>
            .component-card {
                background: white;
                border-radius: 12px;
                padding: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                border: 1px solid #E8ECF0;
                margin-bottom: 12px;
                transition: all 0.2s;
                cursor: pointer;
            }
            .component-card:hover {
                box-shadow: 0 8px 24px rgba(0,0,0,0.1);
                transform: translateY(-2px);
            }
            .component-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                font-weight: 600;
            }
            .component-type {
                font-size: 0.8rem;
                color: #5F6C80;
                background: #F0F3F6;
                padding: 2px 8px;
                border-radius: 20px;
            }
            .connection-line {
                stroke: #2E7D32;
                stroke-width: 2;
                stroke-dasharray: 5,5;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.title("🧩 Modularni energetski dizajner")
        st.markdown("Povlači komponente i povezuj ih da bi kreirao vlastiti energetski sustav.")
        
        # Layout: lijevo – paleta komponenti, sredina – platno, desno – parametri
        col_left, col_canvas, col_right = st.columns([1.2, 2, 1.2])
        
        with col_left:
            self.render_palette()
            
        with col_canvas:
            self.render_canvas()
            
        with col_right:
            self.render_properties()
            
        # Donji dio – rezultati simulacije
        st.markdown("---")
        col_results, col_charts = st.columns([1, 1])
        
        with col_results:
            self.render_results_table()
            
        with col_charts:
            self.render_charts()
            
        # Gumb za pokretanje optimizacije
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("⚡ Pokreni optimizaciju", type="primary", use_container_width=True):
                self.run_optimization()
                
    def render_palette(self):
        st.markdown("### 📦 Paleta komponenti")
        st.markdown("Povuci na platno (klikni za dodavanje)")
        
        # Solarna elektrana
        with st.container():
            st.markdown("""
            <div class="component-card" onclick="alert('Dodaj solarnu')">
                <div class="component-header">
                    <span>☀️ Solarna elektrana</span>
                    <span class="component-type">izvor</span>
                </div>
                <div style="font-size:0.9rem; color:#5F6C80;">Proizvodi električnu energiju</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➕ Dodaj solarnu", key="add_solar", use_container_width=True):
                self.add_component(SolarPlant())
                st.rerun()
        
        # Baterija
        with st.container():
            st.markdown("""
            <div class="component-card">
                <div class="component-header">
                    <span>🔋 Baterija</span>
                    <span class="component-type">spremnik</span>
                </div>
                <div style="font-size:0.9rem; color:#5F6C80;">Pohranjuje energiju</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➕ Dodaj bateriju", key="add_battery", use_container_width=True):
                self.add_component(Battery())
                st.rerun()
        
        # Elektrolizator
        with st.container():
            st.markdown("""
            <div class="component-card">
                <div class="component-header">
                    <span>⚡ Elektrolizator</span>
                    <span class="component-type">pretvarač</span>
                </div>
                <div style="font-size:0.9rem; color:#5F6C80;">Elektricitet → vodik</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➕ Dodaj elektrolizator", key="add_ely", use_container_width=True):
                self.add_component(Electrolyzer())
                st.rerun()
        
        # Potrošnja
        with st.container():
            st.markdown("""
            <div class="component-card">
                <div class="component-header">
                    <span>🏭 Potrošnja</span>
                    <span class="component-type">potrošač</span>
                </div>
                <div style="font-size:0.9rem; color:#5F6C80;">Električno opterećenje</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("➕ Dodaj potrošnju", key="add_load", use_container_width=True):
                self.add_component(Load())
                st.rerun()
                
    def render_canvas(self):
        st.markdown("### 🖼️ Platno")
        
        if not st.session_state.designer_components:
            st.info("Platno je prazno. Dodaj komponente s lijeve strane.")
            return
            
        # Prikaz komponenti u gridu (pojednostavljeno – bez pravog drag&drop)
        cols = st.columns(3)
        for i, comp in enumerate(st.session_state.designer_components):
            with cols[i % 3]:
                self.render_component_card(comp)
                
    def render_component_card(self, comp):
        # Odabir ikone prema tipu
        icon = "☀️" if comp.type == "source" else "🔋" if comp.type == "storage" else "⚡" if comp.type == "converter" else "🏭"
        
        with st.container():
            st.markdown(f"""
            <div class="component-card">
                <div class="component-header">
                    <span>{icon} {comp.name}</span>
                    <span class="component-type">{comp.type}</span>
                </div>
                <div style="font-size:0.8rem; color:#0B2F4D;">ID: {comp.id}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Dugmad za brisanje i povezivanje
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Obriši", key=f"del_{comp.id}"):
                    self.remove_component(comp.id)
                    st.rerun()
            with col2:
                if st.button("🔗 Poveži", key=f"conn_{comp.id}"):
                    st.session_state['connecting_from'] = comp.id
                    st.rerun()
                    
    def render_properties(self):
        st.markdown("### ⚙️ Parametri")
        
        # Ako je odabrana komponenta za povezivanje
        if 'connecting_from' in st.session_state and st.session_state['connecting_from']:
            from_id = st.session_state['connecting_from']
            from_comp = self.find_component(from_id)
            if from_comp:
                st.info(f"Povezivanje od **{from_comp.name}** ({from_id})")
                
                # Lista mogućih odredišta (sve osim sebe)
                targets = [c for c in st.session_state.designer_components if c.id != from_id]
                target_options = {c.id: f"{c.name} ({c.type})" for c in targets}
                
                if target_options:
                    selected_target = st.selectbox(
                        "Odaberi odredište",
                        options=list(target_options.keys()),
                        format_func=lambda x: target_options[x]
                    )
                    
                    if st.button("✅ Spoji", use_container_width=True):
                        self.add_connection(from_id, selected_target)
                        del st.session_state['connecting_from']
                        st.rerun()
                else:
                    st.warning("Nema drugih komponenti za povezivanje.")
                    
                if st.button("❌ Odustani"):
                    del st.session_state['connecting_from']
                    st.rerun()
                    
        # Prikaz parametara odabrane komponente (ako nije u modu povezivanja)
        else:
            # Za sada prikazujemo parametre prve komponente – može se proširiti
            if st.session_state.designer_components:
                selected_id = st.selectbox(
                    "Odaberi komponentu",
                    options=[c.id for c in st.session_state.designer_components],
                    format_func=lambda x: self.find_component(x).name
                )
                comp = self.find_component(selected_id)
                if comp:
                    self.render_component_parameters(comp)
            else:
                st.info("Dodaj komponente za podešavanje parametara.")
                
    def render_component_parameters(self, comp):
        st.markdown(f"**{comp.name}** ({comp.type})")
        
        if comp.type == "source":
            comp.parameters['capacity_kw'] = st.number_input(
                "Kapacitet (kW)", 
                min_value=0.0, value=comp.parameters['capacity_kw'], step=50.0,
                key=f"param_{comp.id}_cap"
            )
            comp.parameters['profile'] = st.selectbox(
                "Profil proizvodnje",
                ["sunčano", "oblačno", "varijabilno"],
                index=["sunčano", "oblačno", "varijabilno"].index(comp.parameters['profile']),
                key=f"param_{comp.id}_prof"
            )
            
        elif comp.type == "storage":
            comp.parameters['capacity_kwh'] = st.number_input(
                "Kapacitet (kWh)", 
                min_value=0.0, value=comp.parameters['capacity_kwh'], step=100.0,
                key=f"param_{comp.id}_cap"
            )
            comp.parameters['power_kw'] = st.number_input(
                "Snaga (kW)", 
                min_value=0.0, value=comp.parameters['power_kw'], step=50.0,
                key=f"param_{comp.id}_pow"
            )
            comp.parameters['efficiency'] = st.slider(
                "Efikasnost", 0.5, 1.0, comp.parameters['efficiency'], 0.01,
                key=f"param_{comp.id}_eff"
            )
            
        elif comp.type == "converter":
            comp.parameters['capacity_kw'] = st.number_input(
                "Kapacitet (kW)", 
                min_value=0.0, value=comp.parameters['capacity_kw'], step=50.0,
                key=f"param_{comp.id}_cap"
            )
            comp.parameters['efficiency'] = st.slider(
                "Efikasnost", 0.5, 0.95, comp.parameters['efficiency'], 0.01,
                key=f"param_{comp.id}_eff"
            )
            comp.parameters['h2_price'] = st.number_input(
                "Cijena H₂ (€/kg)", 
                min_value=0.0, value=comp.parameters['h2_price'], step=0.5,
                key=f"param_{comp.id}_price"
            )
            
        elif comp.type == "sink":
            comp.parameters['base_load_kw'] = st.number_input(
                "Bazno opterećenje (kW)", 
                min_value=0.0, value=comp.parameters['base_load_kw'], step=50.0,
                key=f"param_{comp.id}_load"
            )
            comp.parameters['profile'] = st.selectbox(
                "Profil potrošnje",
                ["industrijski", "uslužni", "stambeni"],
                index=["industrijski", "uslužni", "stambeni"].index(comp.parameters['profile']),
                key=f"param_{comp.id}_prof"
            )
            
    def render_results_table(self):
        st.markdown("### 📊 Rezultati po komponenti")
        data = []
        for comp in st.session_state.designer_components:
            data.append({
                "Komponenta": comp.name,
                "Tip": comp.type,
                "Proizvodnja (kWh)": f"{comp.results.get('production_kwh', np.zeros(24)).sum():.0f}" if comp.type == "source" else "-",
                "Potrošnja (kWh)": f"{comp.results.get('load_kw', np.zeros(24)).sum():.0f}" if comp.type == "sink" else "-",
                "Punjenje (kWh)": f"{comp.results.get('charge_kwh', np.zeros(24)).sum():.0f}" if comp.type == "storage" else "-",
                "Pražnjenje (kWh)": f"{comp.results.get('discharge_kwh', np.zeros(24)).sum():.0f}" if comp.type == "storage" else "-",
                "H₂ (kg)": f"{comp.results.get('h2_production_kg', 0):.1f}" if comp.type == "converter" else "-"
            })
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            
    def render_charts(self):
        st.markdown("### 📈 Dijagrami")
        if st.session_state.designer_components:
            # Jednostavan prikaz – može se proširiti
            fig = go.Figure()
            for comp in st.session_state.designer_components:
                if comp.type == "source" and hasattr(comp.results, 'production_kwh'):
                    fig.add_trace(go.Scatter(
                        x=list(range(24)),
                        y=comp.results['production_kwh'],
                        mode='lines',
                        name=comp.name
                    ))
            if fig.data:
                fig.update_layout(
                    title="Proizvodnja po satu",
                    xaxis_title="Sat",
                    yaxis_title="kWh",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pokreni optimizaciju za prikaz grafova.")
        else:
            st.info("Dodaj komponente za početak.")
            
    def add_component(self, comp):
        st.session_state.designer_components.append(comp)
        
    def remove_component(self, comp_id):
        # Prvo ukloni sve veze koje uključuju ovu komponentu
        st.session_state.designer_connections = [
            (f, t) for f, t in st.session_state.designer_connections 
            if f != comp_id and t != comp_id
        ]
        # Zatim ukloni komponentu
        st.session_state.designer_components = [
            c for c in st.session_state.designer_components if c.id != comp_id
        ]
        
    def add_connection(self, from_id, to_id):
        # Spriječi duple veze
        if (from_id, to_id) not in st.session_state.designer_connections:
            st.session_state.designer_connections.append((from_id, to_id))
            # Ažuriraj inputs/outputs u komponentama
            from_comp = self.find_component(from_id)
            to_comp = self.find_component(to_id)
            if from_comp and to_comp:
                if to_id not in from_comp.outputs:
                    from_comp.outputs.append(to_id)
                if from_id not in to_comp.inputs:
                    to_comp.inputs.append(from_id)
                    
    def find_component(self, comp_id):
        for c in st.session_state.designer_components:
            if c.id == comp_id:
                return c
        return None
        
    def run_optimization(self):
        """Jednostavna simulacija – zamijenit ćemo pravom optimizacijom kasnije"""
        st.success("Optimizacija pokrenuta (simulacija)")
        
        # Generiraj profile za svaku komponentu
        for comp in st.session_state.designer_components:
            if comp.type == "source":
                # Solarni profil
                if comp.parameters['profile'] == "sunčano":
                    profile = np.array([0,0,0,0,0,5,30,60,85,95,100,95,85,70,50,30,15,5,0,0,0,0,0,0]) / 100
                elif comp.parameters['profile'] == "oblačno":
                    profile = np.array([0,0,0,0,0,2,15,35,50,60,55,45,35,25,20,12,5,1,0,0,0,0,0,0]) / 100
                else:
                    profile = np.array([0,0,0,0,0,5,30,70,90,70,50,30,80,90,60,30,10,5,0,0,0,0,0,0]) / 100
                comp.results['production_kwh'] = profile * comp.parameters['capacity_kw']
                
            elif comp.type == "sink":
                if comp.parameters['profile'] == "industrijski":
                    profile = 80 + 40 * np.sin(np.linspace(0, 2*np.pi, 24) + 0.5) + np.random.normal(0, 5, 24)
                elif comp.parameters['profile'] == "uslužni":
                    profile = 60 + 50 * (np.sin(np.linspace(-1.5, 1.5, 24))**2) + np.random.normal(0, 5, 24)
                else:
                    profile = 50 + 30 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.normal(0, 3, 24)
                profile = profile / 100 * comp.parameters['base_load_kw'] * comp.parameters.get('peak_factor', 1.0)
                profile = np.clip(profile, 0, None)
                comp.results['load_kw'] = profile
                comp.results['shed_kw'] = np.zeros(24)
                
        # Bilansiraj – pojednostavljeno
        st.rerun()


def show_designer():
    """Funkcija koja se poziva iz app.py za pokretanje dizajnera."""
    app = EnergyDesignerApp()
    app.run()


# Ako se modul pokreće samostalno (za testiranje)
if __name__ == "__main__":
    show_designer()
