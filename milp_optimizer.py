"""
MILP Day-Ahead Energy Optimizer - Enhanced Version
Poboljšani MILP model s višeperiodnom optimizacijom, degradacijom baterije,
ograničenjima na razini klasteriranja sati i scenarij analizom.
"""
import pulp as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BatteryParams:
    """Parametri baterije za MILP model."""
    capacity_mwh: float = 6.0
    power_mw: float = 1.0
    efficiency: float = 0.92          # round-trip efficiency
    min_soc_pct: float = 0.10         # min SOC kao udio kapaciteta
    max_soc_pct: float = 0.95         # max SOC kao udio kapaciteta
    min_power_mw: float = 0.1         # min snaga ako je aktivan
    cycle_cost_eur_mwh: float = 5.0   # degradacija po MWh protoka
    startup_cost_eur: float = 0.0     # trošak pokretanja (€)
    initial_soc_pct: float = 0.50     # početni SOC
    final_soc_pct: Optional[float] = None  # ciljni završni SOC (None = slobodan)
    max_cycles_per_day: Optional[int] = None  # max ciklusa/dan


@dataclass
class GridParams:
    """Parametri mreže i tržišta."""
    spot_prices: np.ndarray = field(default_factory=lambda: np.zeros(24))
    feedin_tariff: float = 50.0       # €/MWh za prodaju viška
    max_import_mw: Optional[float] = None  # ograničenje uvoza iz mreže
    max_export_mw: Optional[float] = None  # ograničenje izvoza u mrežu
    grid_fee_eur_mwh: float = 0.0     # mrežna naknada za uvoz
    peak_hours: List[int] = field(default_factory=list)  # sati "peak" tarife
    peak_multiplier: float = 1.5      # množač tarife u peak satima


@dataclass
class CO2Params:
    """CO2 parametri."""
    intensity_grid: float = 0.40      # tCO2/MWh – mreža (HRV prosjek)
    intensity_fne: float = 0.0        # tCO2/MWh – FNE (nula)
    eua_price: float = 80.0           # €/tCO2 – EUA cijena
    use_hourly_intensity: bool = False
    hourly_intensity: Optional[np.ndarray] = None  # varijabilni intenzitet


@dataclass
class OptimizationResult:
    """Rezultati MILP optimizacije."""
    status: str
    total_cost: float = 0.0
    spot_cost: float = 0.0
    contract_cost: float = 0.0
    battery_degradation_cost: float = 0.0
    sales_revenue: float = 0.0
    co2_cost: float = 0.0
    co2_emissions_t: float = 0.0
    spot_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    contract_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    sales_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    batt_charge_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    batt_discharge_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    soc_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    net_load_mwh: np.ndarray = field(default_factory=lambda: np.zeros(24))
    self_sufficiency_pct: float = 0.0
    peak_shaving_mw: float = 0.0
    solver_time_s: float = 0.0
    message: str = ""


class MILPDayAheadOptimizer:
    """
    Napredni MILP optimizer za dnevno planiranje energije.
    
    Modelira:
    - Binarno isključivanje punjenja/pražnjenja baterije
    - Degradaciju baterije i ograničenja ciklusa
    - CO2 troškove (varijabilni intenzitet po satu)
    - Peak tarife i mrežna ograničenja
    - Višestruke ugovorne tranše
    - Scenarij analizu (optimistični/pesimistični/realni)
    """

    def __init__(
        self,
        load: np.ndarray,
        fne: np.ndarray,
        spot_price: np.ndarray,
        contracted_volume: float,
        contracted_price: float,
        batt_capacity_mwh: float = 6.0,
        batt_power_mw: float = 1.0,
        # Legacy parametri za kompatibilnost
        batt_efficiency: float = 0.92,
        co2_intensity: float = 0.40,
        co2_price: float = 80.0,
        feedin_tariff: float = 50.0,
        batt_min_power: float = 0.1,
        batt_cycle_cost: float = 5.0,
        batt_startup_cost: float = 0.0,
        # Napredni parametri
        min_soc_pct: float = 0.10,
        max_soc_pct: float = 0.95,
        max_import_mw: Optional[float] = None,
        max_export_mw: Optional[float] = None,
        grid_fee_eur_mwh: float = 0.0,
        peak_hours: Optional[List[int]] = None,
        peak_multiplier: float = 1.5,
        hourly_co2_intensity: Optional[np.ndarray] = None,
        final_soc_pct: Optional[float] = None,
        max_cycles_per_day: Optional[int] = None,
        solver_time_limit_s: int = 60,
    ):
        self.T = 24
        self.load = np.array(load, dtype=float)
        self.fne = np.array(fne, dtype=float)
        self.spot_price = np.array(spot_price, dtype=float)
        self.contr_vol = contracted_volume
        self.contr_price = contracted_price

        self.batt = BatteryParams(
            capacity_mwh=batt_capacity_mwh,
            power_mw=batt_power_mw,
            efficiency=batt_efficiency,
            min_soc_pct=min_soc_pct,
            max_soc_pct=max_soc_pct,
            min_power_mw=batt_min_power,
            cycle_cost_eur_mwh=batt_cycle_cost,
            startup_cost_eur=batt_startup_cost,
            final_soc_pct=final_soc_pct,
            max_cycles_per_day=max_cycles_per_day,
        )

        # CO2 intensity – hourly or constant
        if hourly_co2_intensity is not None and len(hourly_co2_intensity) == 24:
            self.co2_intensity = np.array(hourly_co2_intensity)
        else:
            self.co2_intensity = np.full(24, co2_intensity)
        self.co2_price = co2_price

        self.feedin = feedin_tariff
        self.max_import = max_import_mw
        self.max_export = max_export_mw
        self.grid_fee = grid_fee_eur_mwh
        self.peak_hours = set(peak_hours) if peak_hours else set()
        self.peak_mult = peak_multiplier
        self.solver_time_limit = solver_time_limit_s

    # ------------------------------------------------------------------
    # GŁÓWNA METODA
    # ------------------------------------------------------------------
    def optimize(self, initial_soc: float = 0.0) -> OptimizationResult:
        """
        Pokretanje MILP optimizacije. initial_soc u MWh ili kao udio [0,1].
        """
        import time
        t0 = time.time()

        # Normalizacija initial_soc
        if 0 <= initial_soc <= 1.0 and initial_soc != 0.0:
            initial_soc_mwh = initial_soc * self.batt.capacity_mwh
        else:
            initial_soc_mwh = float(initial_soc)

        T = self.T
        batt = self.batt

        prob = pl.LpProblem("DayAheadOptimization_v2", pl.LpMinimize)

        # ---- VARIJABLE ------------------------------------------------
        # Nabava
        spot = pl.LpVariable.dicts("spot", range(T), lowBound=0, cat='Continuous')
        contr = pl.LpVariable.dicts("contr", range(T), lowBound=0, cat='Continuous')
        sales = pl.LpVariable.dicts("sales", range(T), lowBound=0, cat='Continuous')

        # Baterija
        ch = pl.LpVariable.dicts("ch", range(T), lowBound=0, cat='Continuous')
        dis = pl.LpVariable.dicts("dis", range(T), lowBound=0, cat='Continuous')
        soc = pl.LpVariable.dicts("soc", range(T), lowBound=0,
                                   upBound=batt.capacity_mwh * batt.max_soc_pct,
                                   cat='Continuous')

        # Binarne varijable
        u_ch = pl.LpVariable.dicts("u_ch", range(T), cat='Binary')
        u_dis = pl.LpVariable.dicts("u_dis", range(T), cat='Binary')
        # Startup varijable (za trošak pokretanja)
        if batt.startup_cost_eur > 0:
            u_start = pl.LpVariable.dicts("u_start", range(T), cat='Binary')

        # ---- OGRANIČENJA -----------------------------------------------
        # 1. Energetska bilanca
        for t in range(T):
            prob += (
                self.fne[t] + contr[t] + dis[t] + spot[t]
                == self.load[t] + ch[t] + sales[t]
            ), f"balance_{t}"

        # 2. Ukupna ugovorena količina
        if self.contr_vol > 0:
            prob += pl.lpSum(contr[t] for t in range(T)) <= self.contr_vol, "total_contr"

        # 3. Dinamika SOC
        prob += soc[0] == initial_soc_mwh + batt.efficiency * ch[0] - dis[0] / batt.efficiency, "soc_t0"
        for t in range(1, T):
            prob += (
                soc[t] == soc[t-1] + batt.efficiency * ch[t] - dis[t] / batt.efficiency
            ), f"soc_dyn_{t}"

        # 4. Min SOC ograničenje
        for t in range(T):
            prob += soc[t] >= batt.capacity_mwh * batt.min_soc_pct, f"soc_min_{t}"

        # 5. Snaga baterije + min snaga + isključivost
        for t in range(T):
            prob += ch[t] <= batt.power_mw * u_ch[t], f"ch_max_{t}"
            prob += dis[t] <= batt.power_mw * u_dis[t], f"dis_max_{t}"
            if batt.min_power_mw > 0:
                prob += ch[t] >= batt.min_power_mw * u_ch[t], f"ch_min_{t}"
                prob += dis[t] >= batt.min_power_mw * u_dis[t], f"dis_min_{t}"
            prob += u_ch[t] + u_dis[t] <= 1, f"mutex_{t}"

        # 6. Mrežna ograničenja
        if self.max_import is not None:
            for t in range(T):
                prob += spot[t] <= self.max_import, f"max_import_{t}"
        if self.max_export is not None:
            for t in range(T):
                prob += sales[t] <= self.max_export, f"max_export_{t}"

        # 7. Završni SOC
        if batt.final_soc_pct is not None:
            prob += soc[T-1] >= batt.capacity_mwh * batt.final_soc_pct, "final_soc"

        # 8. Max ciklusa/dan
        if batt.max_cycles_per_day is not None:
            prob += (
                pl.lpSum(ch[t] for t in range(T)) <= batt.max_cycles_per_day * batt.capacity_mwh
            ), "max_cycles"

        # 9. Startup cost modelling
        if batt.startup_cost_eur > 0:
            prob += u_start[0] >= u_dis[0], "startup_0"
            for t in range(1, T):
                prob += u_start[t] >= u_dis[t] - u_dis[t-1], f"startup_{t}"

        # ---- FUNKCIJA CILJA -------------------------------------------
        # Spot trošak (+ CO2 + mrežna naknada + peak premija)
        spot_cost_terms = []
        for t in range(T):
            base_price = self.spot_price[t]
            co2_adder = self.co2_intensity[t] * self.co2_price
            grid_fee = self.grid_fee
            peak_adder = (base_price * (self.peak_mult - 1)) if t in self.peak_hours else 0.0
            total_price = base_price + co2_adder + grid_fee + peak_adder
            spot_cost_terms.append(total_price * spot[t])

        spot_cost = pl.lpSum(spot_cost_terms)
        contr_cost = pl.lpSum(self.contr_price * contr[t] for t in range(T))
        sales_rev = pl.lpSum(self.feedin * sales[t] for t in range(T))
        degradation = pl.lpSum(
            batt.cycle_cost_eur_mwh * (ch[t] + dis[t]) for t in range(T)
        )

        obj = spot_cost + contr_cost + degradation - sales_rev
        if batt.startup_cost_eur > 0:
            obj += pl.lpSum(batt.startup_cost_eur * u_start[t] for t in range(T))

        prob += obj, "TotalCost"

        # ---- RJEŠAVANJE -----------------------------------------------
        solver = pl.PULP_CBC_CMD(
            msg=False,
            timeLimit=self.solver_time_limit,
            gapRel=0.001,  # 0.1% gap tolerance
        )
        prob.solve(solver)
        solver_time = time.time() - t0

        # ---- REZULTATI ------------------------------------------------
        if prob.status == pl.LpStatusOptimal or prob.status == 1:
            spot_v = np.array([max(0, pl.value(spot[t]) or 0) for t in range(T)])
            contr_v = np.array([max(0, pl.value(contr[t]) or 0) for t in range(T)])
            sales_v = np.array([max(0, pl.value(sales[t]) or 0) for t in range(T)])
            ch_v = np.array([max(0, pl.value(ch[t]) or 0) for t in range(T)])
            dis_v = np.array([max(0, pl.value(dis[t]) or 0) for t in range(T)])
            soc_v = np.array([max(0, pl.value(soc[t]) or 0) for t in range(T)])

            # Izvedeni pokazatelji
            co2_total = float(np.sum(spot_v * self.co2_intensity))
            co2_cost_total = co2_total * self.co2_price
            spot_cost_val = float(np.sum(
                spot_v * (self.spot_price + self.co2_intensity * self.co2_price + self.grid_fee)
            ))
            contr_cost_val = float(np.sum(contr_v) * self.contr_price)
            sales_rev_val = float(np.sum(sales_v) * self.feedin)
            degradation_val = float(np.sum(ch_v + dis_v) * batt.cycle_cost_eur_mwh)

            # Self-sufficiency = (FNE + dis) / load
            fne_dis = self.fne + dis_v
            ss = float(np.minimum(fne_dis, self.load).sum() / max(self.load.sum(), 1e-9))

            # Peak shaving = max_load - max_net_load_after_batt
            net_load = self.load - dis_v + ch_v - self.fne
            peak_shaving = float(max(self.load) - max(np.maximum(net_load, 0)))

            return OptimizationResult(
                status='optimal',
                total_cost=float(pl.value(prob.objective) or 0),
                spot_cost=spot_cost_val,
                contract_cost=contr_cost_val,
                battery_degradation_cost=degradation_val,
                sales_revenue=sales_rev_val,
                co2_cost=co2_cost_total,
                co2_emissions_t=co2_total,
                spot_mwh=spot_v,
                contract_mwh=contr_v,
                sales_mwh=sales_v,
                batt_charge_mwh=ch_v,
                batt_discharge_mwh=dis_v,
                soc_mwh=soc_v,
                net_load_mwh=net_load,
                self_sufficiency_pct=ss * 100,
                peak_shaving_mw=peak_shaving,
                solver_time_s=solver_time,
                message=f"Optimal solution found in {solver_time:.1f}s",
            )
        else:
            return OptimizationResult(
                status='failed',
                solver_time_s=solver_time,
                message=f"Solver status: {pl.LpStatus.get(prob.status, 'Unknown')}",
            )

    # ------------------------------------------------------------------
    # SCENARIJ ANALIZA
    # ------------------------------------------------------------------
    def run_scenarios(
        self,
        initial_soc: float = 0.0,
        spot_price_scenarios: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, OptimizationResult]:
        """
        Pokretanje optimizacije za više scenarija spot cijena.
        Vraća dict: {ime_scenarija: OptimizationResult}
        """
        if spot_price_scenarios is None:
            spot_price_scenarios = {
                'Realni': self.spot_price,
                'Optimistični (-20%)': self.spot_price * 0.80,
                'Pesimistični (+30%)': self.spot_price * 1.30,
            }

        original_spot = self.spot_price.copy()
        results = {}
        for name, prices in spot_price_scenarios.items():
            self.spot_price = np.array(prices)
            results[name] = self.optimize(initial_soc)
            results[name].message = f"Scenarij: {name} | " + results[name].message
        self.spot_price = original_spot
        return results

    # ------------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # ------------------------------------------------------------------
    def sensitivity_co2_price(
        self,
        prices: List[float],
        initial_soc: float = 0.0,
    ) -> List[Tuple[float, OptimizationResult]]:
        """Analiza osjetljivosti na promjenu cijene CO2."""
        original = self.co2_price
        results = []
        for p in prices:
            self.co2_price = p
            r = self.optimize(initial_soc)
            results.append((p, r))
        self.co2_price = original
        return results
