"""
MILP OPTIMIZATOR ZA DAN-UNAPRIJED PLANIRANJE ENERGETSKOG PORTFELJA
====================================================================

Ova klasa implementira Mixed Integer Linear Programming (MILP) model za
optimalno planiranje dan-unaprijed u uvjetima tržišta električne energije.

Model uključuje:
    - Spot tržište (kupnja/prodaja uz cijenu CROPEX)
    - Višestruke bilateralne ugovore (fiksni volumen, indeksirani, opcionalni)
    - Baterijski spremnik s binarnim stanjima, minimalnom snagom, troškovima pokretanja i degradacije
    - Fotonaponsku elektranu (FNE) s opcijom redukcije (curtailment)
    - Potrošnju s opcijom prekida uz kaznu (demand response)
    - CO₂ emisije: intenzitet, cijena, gornja granica (cap)
    - Ograničenja priključka na mrežu (uvoz/izvoz)
    - Mogućnost zadavanja ciljanog stanja napunjenosti baterije na kraju dana

Autor:  (prilagođeno za Danica Energy Optimizer PRO)
Verzija: 6.0 – EXTREME MILP
Datum:   2026-02-13
"""

import numpy as np
import pulp as pl
from typing import List, Dict, Union, Optional, Tuple, Any

class MILPDayAheadOptimizer:
    """
    Napredni MILP optimizator za dan-unaprijed planiranje.
    
    Parametri
    ----------
    load : array_like
        Prognoza potrošnje [MWh/h] – 24 vrijednosti.
    fne : array_like
        Prognoza proizvodnje FNE [MWh/h] – 24 vrijednosti.
    spot_price : array_like
        Cijene na CROPEX spot tržištu [€/MWh] – 24 vrijednosti.
    
    contracts : list or dict, optional
        Lista ugovora. Svaki ugovor je dictionary s ključevima:
            'volume_max' : float – maksimalna količina [MWh] (obavezno)
            'volume_min' : float – minimalna količina [MWh] (default 0)
            'price' : float – fiksna cijena [€/MWh] (ako je 'indexed' = False)
            'indexed' : bool – ako je True, cijena = spot_price[t] * price (default False)
            'must_take' : bool – ako je True, volumen je fiksno jednak volume_max (default False)
            'hours' : list – dozvoljeni sati (default svi)
        Ako se preda samo dictionary, tretira se kao jedan ugovor.
    
    battery : dict
        Parametri baterije:
            'capacity' : float – kapacitet [MWh]
            'power' : float – maksimalna snaga punjenja/pražnjenja [MW]
            'efficiency' : float – round-trip efikasnost (0-1), default 0.9
            'min_power' : float – minimalna snaga kada je aktivna [MW], default 0.1
            'cycle_cost' : float – trošak degradacije [€/MWh protoka], default 0
            'startup_cost' : float – trošak pokretanja baterije [€], default 0
            'initial_soc' : float – početno stanje [MWh], default 0
            'target_final_soc' : float – željeno stanje na kraju [MWh], default None (ne zahtijeva se)
            'target_final_penalty' : float – kazna za odstupanje od cilja [€/MWh], default 0
            'max_cycles' : int – maksimalan broj ciklusa (punjenje+pražnjenje) u danu, default None
    
    grid : dict, optional
        Ograničenja priključka:
            'import_limit' : float – maksimalni uvoz iz mreže [MW], default inf
            'export_limit' : float – maksimalni izvoz u mrežu [MW], default inf
            'feedin_tariff' : float – otkupna cijena viškova [€/MWh], default 0
    
    co2 : dict, optional
        Parametri CO₂:
            'intensity' : float – emisijski faktor spot energije [tCO2/MWh], default 0.4
            'price' : float – cijena EUA [€/tCO2], default 0
            'cap' : float – gornja granica dnevnih emisija [tCO2], default None (neograničeno)
    
    demand_response : dict, optional
        Opcija prekida potrošnje:
            'max_shed' : float – maksimalna količina koja se može prekinuti [MWh/h]
            'penalty' : float – kazna za prekid [€/MWh]
    
    curtailment : bool, optional
        Dozvoli redukciju FNE (default False).
    
    T : int, optional
        Broj vremenskih intervala (default 24).
    
    """

    def __init__(self,
                 load: np.ndarray,
                 fne: np.ndarray,
                 spot_price: np.ndarray,
                 contracts: Union[Dict, List[Dict]],
                 battery: Dict,
                 grid: Optional[Dict] = None,
                 co2: Optional[Dict] = None,
                 demand_response: Optional[Dict] = None,
                 curtailment: bool = False,
                 T: int = 24):

        # ========== 1. ULAZNI PODACI ==========
        self.T = T
        self.load = np.array(load).flatten()
        self.fne = np.array(fne).flatten()
        self.spot_price = np.array(spot_price).flatten()

        # Validacija duljina
        if not (len(self.load) == len(self.fne) == len(self.spot_price) == self.T):
            raise ValueError(f"Svi vremenski nizovi moraju imati {self.T} elemenata.")

        # ========== 2. UGOVORI ==========
        # Osiguraj da je contracts lista
        if isinstance(contracts, dict):
            contracts = [contracts]
        self.contracts = contracts
        self.n_contracts = len(contracts)

        # ========== 3. BATERIJA ==========
        self.batt_cap = battery.get('capacity', 0)
        self.batt_pow = battery.get('power', 0)
        self.batt_eff = battery.get('efficiency', 0.9)
        self.batt_min_power = battery.get('min_power', 0.0)
        self.batt_cycle_cost = battery.get('cycle_cost', 0.0)
        self.batt_startup_cost = battery.get('startup_cost', 0.0)
        self.batt_initial_soc = battery.get('initial_soc', 0.0)
        self.batt_target_final_soc = battery.get('target_final_soc', None)
        self.batt_target_penalty = battery.get('target_final_penalty', 0.0)
        self.batt_max_cycles = battery.get('max_cycles', None)

        # ========== 4. MREŽA ==========
        grid = grid or {}
        self.grid_import_limit = grid.get('import_limit', float('inf'))
        self.grid_export_limit = grid.get('export_limit', float('inf'))
        self.feedin_tariff = grid.get('feedin_tariff', 0.0)

        # ========== 5. CO₂ ==========
        co2 = co2 or {}
        self.co2_intensity = co2.get('intensity', 0.4)
        self.co2_price = co2.get('price', 0.0)
        self.co2_cap = co2.get('cap', None)

        # ========== 6. DEMAND RESPONSE ==========
        dr = demand_response or {}
        self.dr_max_shed = dr.get('max_shed', 0.0)
        self.dr_penalty = dr.get('penalty', 0.0)

        # ========== 7. OSTALO ==========
        self.curtailment = curtailment

        # ========== 8. PROVJERA KONZISTENTNOSTI ==========
        if self.batt_min_power > self.batt_pow:
            raise ValueError("Minimalna snaga baterije ne može biti veća od maksimalne.")

        # ========== 9. INICIJALIZACIJA POMOĆNIH STRUKTURA ==========
        self._prob = None
        self._vars = {}
        self._constraints = {}
        self._solver_status = None

    # ------------------------------------------------------------------
    # METODA ZA OPTIMIZACIJU
    # ------------------------------------------------------------------
    def optimize(self, solver_msg: bool = False, return_duals: bool = False) -> Dict[str, Any]:
        """
        Pokreće MILP optimizaciju i vraća rješenje.

        Parameters
        ----------
        solver_msg : bool
            Ako je True, ispisuje log solvera (default False).
        return_duals : bool
            Ako je True i solver to podržava, vraća dualne varijable (samo za LpStatusOptimal).

        Returns
        -------
        dict
            Rječnik s rezultatima i statusom.
        """
        self._build_model()
        self._solve(solver_msg)

        if self._prob.status != pl.LpStatusOptimal:
            return {
                'status': 'failed',
                'message': f'Solver status: {pl.LpStatus[self._prob.status]}. '
                           'Provjerite ulazne podatke (npr. kapacitet baterije, potrošnja, ograničenja).'
            }

        return self._extract_results(return_duals)

    # ------------------------------------------------------------------
    # IZGRADNJA MODELA – DETALJNA FORMULACIJA
    # ------------------------------------------------------------------
    def _build_model(self):
        """Kreira problem, varijable, ograničenja i funkciju cilja."""
        self._prob = pl.LpProblem("DanAheadMILP", pl.LpMinimize)
        T = self.T
        K = self.n_contracts
        M = 1e6  # Big-M za logička ograničenja (dovoljno velik)

        # --------------------------------------------------------------
        # 1. VARIJABLE
        # --------------------------------------------------------------

        # --- Spot tržište ---
        # spot_buy[t]: energija kupljena na spotu [MWh]
        self._vars['spot_buy'] = pl.LpVariable.dicts(
            "SpotBuy", range(T), lowBound=0, cat='Continuous'
        )
        # spot_sell[t]: energija prodana na spotu [MWh]
        self._vars['spot_sell'] = pl.LpVariable.dicts(
            "SpotSell", range(T), lowBound=0, cat='Continuous'
        )

        # --- Ugovori ---
        # contract_power[k][t]: isporuka po ugovoru k u satu t [MWh]
        self._vars['contract'] = {
            k: pl.LpVariable.dicts(
                f"Contract_{k}", range(T), lowBound=0, cat='Continuous'
            )
            for k in range(K)
        }

        # Opcionalne binarne varijable za ugovore (ako imaju min_take)
        self._vars['contract_active'] = {}
        for k, c in enumerate(self.contracts):
            if c.get('volume_min', 0) > 0 and not c.get('must_take', False):
                self._vars['contract_active'][k] = pl.LpVariable.dicts(
                    f"ContractActive_{k}", range(T), cat='Binary'
                )

        # --- FNE (solarna) ---
        # fne_used[t]: iskorištena proizvodnja FNE [MWh] (ne može biti veća od prognoze)
        self._vars['fne_used'] = pl.LpVariable.dicts(
            "FNEused", range(T), lowBound=0, cat='Continuous'
        )
        if self.curtailment:
            # fne_curtailed[t]: višak koji se odbacuje
            self._vars['fne_curtailed'] = pl.LpVariable.dicts(
                "FNEcurtailed", range(T), lowBound=0, cat='Continuous'
            )

        # --- Baterija ---
        # ch[t]: punjenje [MWh]
        # dis[t]: pražnjenje [MWh]
        # soc[t]: stanje napunjenosti [MWh]
        self._vars['ch'] = pl.LpVariable.dicts(
            "Ch", range(T), lowBound=0, cat='Continuous'
        )
        self._vars['dis'] = pl.LpVariable.dicts(
            "Dis", range(T), lowBound=0, cat='Continuous'
        )
        self._vars['soc'] = pl.LpVariable.dicts(
            "SOC", range(T), lowBound=0, upBound=self.batt_cap, cat='Continuous'
        )

        # Binarne varijable za stanje baterije
        self._vars['u_ch'] = pl.LpVariable.dicts(
            "U_ch", range(T), cat='Binary'
        )
        self._vars['u_dis'] = pl.LpVariable.dicts(
            "U_dis", range(T), cat='Binary'
        )

        # Startup varijable (1 ako se baterija uključuje u satu t)
        if self.batt_startup_cost > 0:
            self._vars['startup_ch'] = pl.LpVariable.dicts(
                "Startup_ch", range(T), cat='Binary'
            )
            self._vars['startup_dis'] = pl.LpVariable.dicts(
                "Startup_dis", range(T), cat='Binary'
            )

        # --- Mreža: uvoz/izvoz limiti (već su ograničeni spot_sell/spot_buy uz dodatne limit varijable ako je potrebno) ---
        # Ne trebaju dodatne varijable, ograničenja će direktno limitirati spot_sell/spot_buy

        # --- Demand response ---
        if self.dr_max_shed > 0:
            self._vars['load_shed'] = pl.LpVariable.dicts(
                "LoadShed", range(T), lowBound=0, upBound=self.dr_max_shed, cat='Continuous'
            )
            # Opcionalno: binarna varijabla za aktivaciju prekida
            # self._vars['shed_active'] = pl.LpVariable.dicts("ShedActive", range(T), cat='Binary')
        else:
            self._vars['load_shed'] = {t: 0 for t in range(T)}

        # --- CO₂ cap (ako je definiran) ---
        # Nema dodatnih varijabli, ograničenje će biti linearno

        # --------------------------------------------------------------
        # 2. OGRANIČENJA
        # --------------------------------------------------------------

        # =================================================================
        # 2.1. BILANCA SNAGE (uz opciju prekida potrošnje)
        # =================================================================
        # Lijeva strana: proizvodnja + ugovori + spot kupnja + pražnjenje
        # Desna strana: potrošnja + punjenje + spot prodaja + prekid potrošnje (load shed)
        for t in range(T):
            lhs = (self._vars['fne_used'][t] +
                   sum(self._vars['contract'][k][t] for k in range(K)) +
                   self._vars['spot_buy'][t] +
                   self._vars['dis'][t])

            rhs = (self.load[t] +
                   self._vars['ch'][t] +
                   self._vars['spot_sell'][t] +
                   self._vars['load_shed'][t])

            self._prob += (lhs == rhs), f"EnergyBalance_{t}"

        # =================================================================
        # 2.2. FNE OGRANIČENJA
        # =================================================================
        for t in range(T):
            # Iskorištena FNE ne može premašiti prognozu
            self._prob += (self._vars['fne_used'][t] <= self.fne[t]), f"FNE_upper_{t}"
            if self.curtailment:
                # fne_used + fne_curtailed = prognoza
                self._vars['fne_curtailed'][t].setInitialValue(0)
                self._prob += (self._vars['fne_used'][t] + self._vars['fne_curtailed'][t] == self.fne[t]), f"FNE_total_{t}"
            else:
                # Ako curtailment nije dozvoljen, moramo iskoristiti svu proizvodnju
                self._prob += (self._vars['fne_used'][t] == self.fne[t]), f"FNE_full_{t}"

        # =================================================================
        # 2.3. UGOVORI – DETALJNA OGRANIČENJA
        # =================================================================
        for k, c in enumerate(self.contracts):
            vol_max = c.get('volume_max', 0)
            vol_min = c.get('volume_min', 0)
            price_type = 'fixed' if not c.get('indexed', False) else 'indexed'
            must_take = c.get('must_take', False)
            hours_allowed = c.get('hours', list(range(T)))  # default svi sati

            for t in range(T):
                # Ograničenje na dozvoljene sate
                if t not in hours_allowed:
                    self._prob += (self._vars['contract'][k][t] == 0), f"Contract_{k}_hour_{t}_forbidden"

                if must_take:
                    # Ugovor se mora ispuniti u potpunosti svaki sat (fiksna količina)
                    self._prob += (self._vars['contract'][k][t] == vol_max), f"Contract_{k}_musttake_{t}"
                else:
                    # Ograničenje maksimuma
                    self._prob += (self._vars['contract'][k][t] <= vol_max), f"Contract_{k}_max_{t}"
                    # Ograničenje minimuma – zahtijeva binarnu varijablu ako je vol_min > 0
                    if vol_min > 0:
                        active_var = self._vars['contract_active'][k][t]
                        self._prob += (self._vars['contract'][k][t] >= vol_min * active_var), f"Contract_{k}_min_{t}"
                        self._prob += (self._vars['contract'][k][t] <= vol_max * active_var), f"Contract_{k}_min_upper_{t}"

            # Dnevno ograničenje ukupne količine (ako je definirano)
            if 'total_max' in c:
                self._prob += (pl.lpSum(self._vars['contract'][k][t] for t in range(T)) <= c['total_max']), f"Contract_{k}_totalMax"
            if 'total_min' in c:
                self._prob += (pl.lpSum(self._vars['contract'][k][t] for t in range(T)) >= c['total_min']), f"Contract_{k}_totalMin"

        # =================================================================
        # 2.4. BATERIJA – DETALJNA OGRANIČENJA
        # =================================================================
        # Početno stanje
        self._prob += (self._vars['soc'][0] == self.batt_initial_soc), "SOC_initial"

        # Dinamika baterije (uz efikasnost)
        for t in range(T - 1):
            self._prob += (self._vars['soc'][t + 1] ==
                           self._vars['soc'][t] +
                           self.batt_eff * self._vars['ch'][t] -
                           self._vars['dis'][t] / self.batt_eff), f"SOC_dynamic_{t}"

        # Ograničenja snage i binarnih stanja
        for t in range(T):
            # Maksimalna snaga
            self._prob += (self._vars['ch'][t] <= self.batt_pow * self._vars['u_ch'][t]), f"Ch_max_{t}"
            self._prob += (self._vars['dis'][t] <= self.batt_pow * self._vars['u_dis'][t]), f"Dis_max_{t}"

            # Minimalna snaga (ako je definirana i > 0)
            if self.batt_min_power > 0:
                self._prob += (self._vars['ch'][t] >= self.batt_min_power * self._vars['u_ch'][t]), f"Ch_min_{t}"
                self._prob += (self._vars['dis'][t] >= self.batt_min_power * self._vars['u_dis'][t]), f"Dis_min_{t}"

            # Neistovremenost punjenja i pražnjenja
            self._prob += (self._vars['u_ch'][t] + self._vars['u_dis'][t] <= 1), f"ChDis_mutex_{t}"

            # Startup logika (ako je trošak pokretanja > 0)
            if self.batt_startup_cost > 0:
                # startup_ch[t] >= u_ch[t] - u_ch[t-1] (uz u_ch[-1] = 0)
                if t == 0:
                    self._prob += (self._vars['startup_ch'][t] >= self._vars['u_ch'][t]), f"Startup_ch_{t}"
                else:
                    self._prob += (self._vars['startup_ch'][t] >= self._vars['u_ch'][t] - self._vars['u_ch'][t - 1]), f"Startup_ch_{t}"
                self._prob += (self._vars['startup_ch'][t] >= 0)
                # Isto za pražnjenje
                if t == 0:
                    self._prob += (self._vars['startup_dis'][t] >= self._vars['u_dis'][t]), f"Startup_dis_{t}"
                else:
                    self._prob += (self._vars['startup_dis'][t] >= self._vars['u_dis'][t] - self._vars['u_dis'][t - 1]), f"Startup_dis_{t}"
                self._prob += (self._vars['startup_dis'][t] >= 0)

        # Ciljano stanje na kraju (ako je zadano)
        if self.batt_target_final_soc is not None:
            if self.batt_target_penalty > 0:
                # Meko ograničenje – uvesti varijable odstupanja
                dev_plus = pl.LpVariable("SOC_dev_plus", lowBound=0, cat='Continuous')
                dev_minus = pl.LpVariable("SOC_dev_minus", lowBound=0, cat='Continuous')
                self._prob += (self._vars['soc'][T - 1] - self.batt_target_final_soc == dev_plus - dev_minus), "SOC_target_deviation"
                # Trošak odstupanja dodat će se u cilj
                self._soc_dev_vars = (dev_plus, dev_minus)
            else:
                # Čvrsto ograničenje
                self._prob += (self._vars['soc'][T - 1] == self.batt_target_final_soc), "SOC_target"

        # Ograničenje broja ciklusa (jedan ciklus = 1 MWh protoka? ili broj prijelaza)
        if self.batt_max_cycles is not None:
            # Definirajmo ciklus kao svaki MWh protoka kroz bateriju
            total_throughput = pl.lpSum(self._vars['ch'][t] + self._vars['dis'][t] for t in range(T))
            self._prob += (total_throughput <= 2 * self.batt_cap * self.batt_max_cycles), "Battery_cycle_limit"

        # =================================================================
        # 2.5. MREŽA – OGRANIČENJA UVOZA/IZVOZA
        # =================================================================
        for t in range(T):
            self._prob += (self._vars['spot_buy'][t] <= self.grid_import_limit), f"GridImport_limit_{t}"
            self._prob += (self._vars['spot_sell'][t] <= self.grid_export_limit), f"GridExport_limit_{t}"

        # =================================================================
        # 2.6. CO₂ CAP
        # =================================================================
        if self.co2_cap is not None:
            total_co2 = self.co2_intensity * pl.lpSum(self._vars['spot_buy'][t] for t in range(T))
            self._prob += (total_co2 <= self.co2_cap), "CO2_cap"

        # =================================================================
        # 2.7. DEMAND RESPONSE – dodatna ograničenja
        # =================================================================
        if self.dr_max_shed > 0:
            for t in range(T):
                self._prob += (self._vars['load_shed'][t] <= self.load[t]), f"LoadShed_max_{t}"
                # Maksimalni prekid po satu već je upBound-om

        # --------------------------------------------------------------
        # 3. FUNKCIJA CILJA – MINIMIZACIJA UKUPNIH TROŠKOVA
        # --------------------------------------------------------------
        objective = 0

        # --- Trošak spot kupnje (uključuje CO₂) ---
        spot_cost = pl.lpSum(
            (self.spot_price[t] + self.co2_intensity * self.co2_price) * self._vars['spot_buy'][t]
            for t in range(T)
        )
        objective += spot_cost

        # --- Prihod od spot prodaje (feed-in) ---
        spot_revenue = pl.lpSum(
            self.feedin_tariff * self._vars['spot_sell'][t]
            for t in range(T)
        )
        objective -= spot_revenue  # oduzimamo prihod

        # --- Trošak ugovora ---
        for k, c in enumerate(self.contracts):
            price = c['price']
            if c.get('indexed', False):
                # Indeksirana cijena = spot_price[t] * price (multiplikator)
                contract_cost = pl.lpSum(
                    self.spot_price[t] * price * self._vars['contract'][k][t]
                    for t in range(T)
                )
            else:
                contract_cost = pl.lpSum(
                    price * self._vars['contract'][k][t]
                    for t in range(T)
                )
            objective += contract_cost

        # --- Trošak degradacije baterije ---
        if self.batt_cycle_cost > 0:
            degradation_cost = self.batt_cycle_cost * pl.lpSum(
                self._vars['ch'][t] + self._vars['dis'][t]
                for t in range(T)
            )
            objective += degradation_cost

        # --- Trošak pokretanja baterije ---
        if self.batt_startup_cost > 0:
            startup_cost = self.batt_startup_cost * pl.lpSum(
                self._vars['startup_ch'][t] + self._vars['startup_dis'][t]
                for t in range(T)
            )
            objective += startup_cost

        # --- Kazna za prekid potrošnje (demand response) ---
        if self.dr_max_shed > 0 and self.dr_penalty > 0:
            shed_penalty = self.dr_penalty * pl.lpSum(
                self._vars['load_shed'][t]
                for t in range(T)
            )
            objective += shed_penalty

        # --- Kazna za odstupanje od ciljanog SOC-a ---
        if self.batt_target_final_soc is not None and self.batt_target_penalty > 0:
            dev_plus, dev_minus = self._soc_dev_vars
            objective += self.batt_target_penalty * (dev_plus + dev_minus)

        # Postavi cilj
        self._prob += objective, "TotalCost"

        # --------------------------------------------------------------
        # 4. DODATNO: SPREMANJE OGRANIČENJA ZA DUAL (ako se traži)
        # --------------------------------------------------------------
        # Ovdje ne radimo ništa posebno, ali ćemo omogućiti dohvat dualnih varijabli
        # nakon rješavanja ako return_duals = True.

    # ------------------------------------------------------------------
    # RJEŠAVANJE MODELA
    # ------------------------------------------------------------------
    def _solve(self, msg: bool):
        """Pokreće CBC solver."""
        solver = pl.PULP_CBC_CMD(msg=msg, gapRel=0.001, timeLimit=60)  # gap 0.1%, max 60s
        self._prob.solve(solver)
        self._solver_status = self._prob.status

    # ------------------------------------------------------------------
    # EKSTRAKCIJA REZULTATA
    # ------------------------------------------------------------------
    def _extract_results(self, return_duals: bool = False) -> Dict[str, Any]:
        """Prikuplja vrijednosti varijabli i metrika u rječnik."""
        T = self.T
        K = self.n_contracts

        # Pomoćna funkcija za vađenje vrijednosti
        def val(var):
            return pl.value(var) if var is not None else 0.0

        # Osnovne varijable
        spot_buy_vals = [val(self._vars['spot_buy'][t]) for t in range(T)]
        spot_sell_vals = [val(self._vars['spot_sell'][t]) for t in range(T)]
        fne_used_vals = [val(self._vars['fne_used'][t]) for t in range(T)]
        ch_vals = [val(self._vars['ch'][t]) for t in range(T)]
        dis_vals = [val(self._vars['dis'][t]) for t in range(T)]
        soc_vals = [val(self._vars['soc'][t]) for t in range(T)]

        # Ugovori – agregiraj po vremenu (za kompatibilnost s app.py)
        contract_total_vals = [sum(val(self._vars['contract'][k][t]) for k in range(K)) for t in range(T)]

        # Ako je uključen curtailment
        if self.curtailment:
            curtailed_vals = [val(self._vars['fne_curtailed'][t]) for t in range(T)]
        else:
            curtailed_vals = [0.0] * T

        # Demand response
        if self.dr_max_shed > 0:
            shed_vals = [val(self._vars['load_shed'][t]) for t in range(T)]
        else:
            shed_vals = [0.0] * T

        # Ukupni trošak
        total_cost = val(self._prob.objective)

        # CO₂ emisije
        total_spot_buy = np.sum(spot_buy_vals)
        co2_emissions = total_spot_buy * self.co2_intensity

        # Baterija – korištenje
        total_discharge = np.sum(dis_vals)
        total_charge = np.sum(ch_vals)

        # Zadovoljena potrošnja (ukupno)
        total_load_satisfied = np.sum(self.load) - np.sum(shed_vals)

        # Priprema povratnog rječnika
        result = {
            'status': 'optimal',
            'spot_buy': np.array(spot_buy_vals),
            'spot_sell': np.array(spot_sell_vals),
            'spot_net': np.array(spot_buy_vals) - np.array(spot_sell_vals),  # neto kupnja
            'contract': np.array(contract_total_vals),
            'fne_used': np.array(fne_used_vals),
            'fne_curtailed': np.array(curtailed_vals),
            'batt_ch': np.array(ch_vals),
            'batt_dis': np.array(dis_vals),
            'soc': np.array(soc_vals),
            'load_shed': np.array(shed_vals),
            'total_cost': total_cost,
            'co2_emissions': co2_emissions,
            'battery_throughput': total_discharge + total_charge,
            'battery_cycles': (total_discharge + total_charge) / (2 * self.batt_cap) if self.batt_cap > 0 else 0,
            'load_satisfied': total_load_satisfied,
            'load_shed_total': np.sum(shed_vals),
        }

        # Dodaj dualne varijable ako se traže
        if return_duals:
            duals = {}
            for name, constraint in self._prob.constraints.items():
                pi = constraint.pi
                if pi is not None:
                    duals[name] = pi
            result['duals'] = duals

        return result

    # ------------------------------------------------------------------
    # POMOĆNE METODE
    # ------------------------------------------------------------------
    @property
    def status(self):
        """Vraća status zadnjeg rješavanja."""
        return self._solver_status

    def get_variable_values(self, var_name: str) -> np.ndarray:
        """Dohvaća vrijednosti varijable po vremenu (ako postoji)."""
        if var_name in self._vars:
            var_dict = self._vars[var_name]
            if isinstance(var_dict, dict):
                # Preuzmi sortirano po t
                return np.array([pl.value(var_dict[t]) for t in range(self.T)])
        raise KeyError(f"Varijabla '{var_name}' nije pronađena.")

    def get_contract_values(self, contract_index: int) -> np.ndarray:
        """Dohvaća profil isporuke za pojedini ugovor."""
        if 0 <= contract_index < self.n_contracts:
            var_dict = self._vars['contract'][contract_index]
            return np.array([pl.value(var_dict[t]) for t in range(self.T)])
        else:
            raise IndexError(f"Ugovor {contract_index} ne postoji.")


# ------------------------------------------------------------------
# PRIMJER UPOTREBE (samo ako se pokreće direktno)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Testni primjer s detaljnim postavkama
    T = 24
    np.random.seed(42)
    load = np.random.normal(120, 20, T).clip(min=80)
    fne = np.random.normal(50, 15, T).clip(min=0)
    spot = np.random.normal(75, 10, T).clip(min=40)

    # Definiraj višestruke ugovore
    contracts = [
        {
            'volume_max': 100.0,
            'volume_min': 50.0,
            'price': 60.0,
            'indexed': False,
            'must_take': False,
            'hours': list(range(24)),
            'total_max': 1500.0
        },
        {
            'volume_max': 200.0,
            'volume_min': 0.0,
            'price': 1.1,  # multiplikator
            'indexed': True,
            'must_take': False,
            'hours': list(range(8, 20))  # samo radno vrijeme
        }
    ]

    battery = {
        'capacity': 6.0,
        'power': 2.0,
        'efficiency': 0.92,
        'min_power': 0.2,
        'cycle_cost': 2.0,
        'startup_cost': 5.0,
        'initial_soc': 3.0,
        'target_final_soc': 3.0,
        'target_final_penalty': 10.0,
        'max_cycles': 3
    }

    grid = {
        'import_limit': 200.0,
        'export_limit': 150.0,
        'feedin_tariff': 45.0
    }

    co2 = {
        'intensity': 0.4,
        'price': 80.0,
        'cap': 500.0
    }

    dr = {
        'max_shed': 30.0,
        'penalty': 200.0
    }

    optimizer = MILPDayAheadOptimizer(
        load=load,
        fne=fne,
        spot_price=spot,
        contracts=contracts,
        battery=battery,
        grid=grid,
        co2=co2,
        demand_response=dr,
        curtailment=True,
        T=T
    )

    result = optimizer.optimize(solver_msg=True, return_duals=False)

    if result['status'] == 'optimal':
        print(f"Optimizacija uspješna! Ukupni trošak: {result['total_cost']:.2f} €")
        print(f"CO₂ emisije: {result['co2_emissions']:.2f} t")
        print(f"Iskorištenje baterije: {result['battery_throughput']:.2f} MWh")
        print(f"Broj ciklusa (approx): {result['battery_cycles']:.2f}")
    else:
        print(f"Optimizacija neuspješna: {result['message']}")
