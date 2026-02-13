"""
MILP OPTIMIZATOR ZA DAN-UNAPRIJED PLANIRANJE
==============================================
Potpuno kompatibilna zamjena za tvoj stari optimizer.
Isti konstruktor, isti povratni ključevi – ali unutra je moćan MILP model.
"""

import numpy as np
import pulp as pl
from typing import List, Dict, Optional, Any

class MILPDayAheadOptimizer:
    """
    Potpuno kompatibilan s originalnim konstruktorom.
    """
    def __init__(self,
                 load,
                 fne,
                 spot_price,
                 contracted_volume,
                 contracted_price,
                 batt_capacity_mwh,
                 batt_power_mw,
                 batt_efficiency=0.9,
                 co2_intensity=0.4,
                 co2_price=80,
                 feedin_tariff=50,
                 batt_min_power=0.1,
                 batt_cycle_cost=5,
                 batt_startup_cost=0):

        self.load = np.array(load).flatten()
        self.fne = np.array(fne).flatten()
        self.spot_price = np.array(spot_price).flatten()

        self.contracted_volume = contracted_volume
        self.contracted_price = contracted_price

        self.batt_cap = batt_capacity_mwh
        self.batt_pow = batt_power_mw
        self.batt_eff = batt_efficiency
        self.batt_min_power = batt_min_power
        self.batt_cycle_cost = batt_cycle_cost
        self.batt_startup_cost = batt_startup_cost

        self.co2_intensity = co2_intensity
        self.co2_price = co2_price
        self.feedin_tariff = feedin_tariff

        # Dodatne napredne opcije – isključene za kompatibilnost
        self.co2_cap = None
        self.dr_max_shed = 0.0
        self.dr_penalty = 0.0
        self.curtailment = False
        self.grid_import_limit = None
        self.grid_export_limit = None

        self.T = 24

    def optimize(self, initial_soc=0.0):
        contracts = [{
            'volume_max': self.contracted_volume,
            'volume_min': 0.0,
            'price': self.contracted_price,
            'indexed': False,
            'must_take': False,
            'hours': list(range(self.T)),
        }]

        battery = {
            'capacity': self.batt_cap,
            'power': self.batt_pow,
            'efficiency': self.batt_eff,
            'min_power': self.batt_min_power,
            'cycle_cost': self.batt_cycle_cost,
            'startup_cost': self.batt_startup_cost,
            'initial_soc': initial_soc,
            'target_final_soc': None,
            'target_final_penalty': 0.0,
            'max_cycles': None,
        }

        grid = {
            'import_limit': self.grid_import_limit,
            'export_limit': self.grid_export_limit,
            'feedin_tariff': self.feedin_tariff,
        }

        co2 = {
            'intensity': self.co2_intensity,
            'price': self.co2_price,
            'cap': self.co2_cap,
        }

        demand_response = None

        optimizer = _MILPCore(
            load=self.load,
            fne=self.fne,
            spot_price=self.spot_price,
            contracts=contracts,
            battery=battery,
            grid=grid,
            co2=co2,
            demand_response=demand_response,
            curtailment=self.curtailment,
            T=self.T
        )

        result = optimizer.optimize(initial_soc=initial_soc)

        if result['status'] != 'optimal':
            return {
                'status': 'failed',
                'message': result.get('message', 'Optimizacija nije uspjela.')
            }

        return {
            'spot': result['spot_buy'],
            'contr': result['contract'],
            'grid_sales': result['grid_sales'],
            'batt_ch': result['batt_ch'],
            'batt_dis': result['batt_dis'],
            'soc': result['soc'],
            'total_cost': result['total_cost'],
            'co2_emissions': result['co2_emissions'],
            'status': 'optimal'
        }


class _MILPCore:
    """
    Interna klasa koja obavlja MILP optimizaciju.
    """
    def __init__(self,
                 load: np.ndarray,
                 fne: np.ndarray,
                 spot_price: np.ndarray,
                 contracts: List[Dict],
                 battery: Dict,
                 grid: Dict,
                 co2: Dict,
                 demand_response: Optional[Dict] = None,
                 curtailment: bool = False,
                 T: int = 24):

        self.T = T
        self.load = load
        self.fne = fne
        self.spot_price = spot_price
        self.contracts = contracts
        self.n_contracts = len(contracts)

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

        self.grid_import_limit = grid.get('import_limit')
        self.grid_export_limit = grid.get('export_limit')
        self.feedin_tariff = grid.get('feedin_tariff', 0.0)

        self.co2_intensity = co2.get('intensity', 0.4)
        self.co2_price = co2.get('price', 0.0)
        self.co2_cap = co2.get('cap', None)

        dr = demand_response or {}
        self.dr_max_shed = dr.get('max_shed', 0.0)
        self.dr_penalty = dr.get('penalty', 0.0)

        self.curtailment = curtailment

        self._prob = None
        self._vars = {}
        self._soc_dev_vars = None

    def optimize(self, initial_soc: float = 0.0) -> Dict[str, Any]:
        self._build_model(initial_soc)
        self._solve()

        if self._prob.status != pl.LpStatusOptimal:
            return {
                'status': 'failed',
                'message': f'Solver status: {pl.LpStatus[self._prob.status]}'
            }

        return self._extract_results()

    def _build_model(self, initial_soc: float):
        self._prob = pl.LpProblem("DayAheadMILP", pl.LpMinimize)
        T = self.T
        K = self.n_contracts

        # VARIJABLE
        self._vars['spot_buy'] = pl.LpVariable.dicts("SpotBuy", range(T), lowBound=0, cat='Continuous')
        self._vars['spot_sell'] = pl.LpVariable.dicts("SpotSell", range(T), lowBound=0, cat='Continuous')

        self._vars['contract'] = {
            k: pl.LpVariable.dicts(f"Contract_{k}", range(T), lowBound=0, cat='Continuous')
            for k in range(K)
        }
        self._vars['contract_active'] = {}
        for k, c in enumerate(self.contracts):
            if c.get('volume_min', 0) > 0 and not c.get('must_take', False):
                self._vars['contract_active'][k] = pl.LpVariable.dicts(
                    f"ContractActive_{k}", range(T), cat='Binary'
                )

        self._vars['fne_used'] = pl.LpVariable.dicts("FNEused", range(T), lowBound=0, cat='Continuous')
        if self.curtailment:
            self._vars['fne_curtailed'] = pl.LpVariable.dicts("FNEcurtailed", range(T), lowBound=0, cat='Continuous')

        self._vars['ch'] = pl.LpVariable.dicts("Ch", range(T), lowBound=0, cat='Continuous')
        self._vars['dis'] = pl.LpVariable.dicts("Dis", range(T), lowBound=0, cat='Continuous')
        self._vars['soc'] = pl.LpVariable.dicts("SOC", range(T), lowBound=0, upBound=self.batt_cap, cat='Continuous')
        self._vars['u_ch'] = pl.LpVariable.dicts("U_ch", range(T), cat='Binary')
        self._vars['u_dis'] = pl.LpVariable.dicts("U_dis", range(T), cat='Binary')
        if self.batt_startup_cost > 0:
            self._vars['startup_ch'] = pl.LpVariable.dicts("Startup_ch", range(T), cat='Binary')
            self._vars['startup_dis'] = pl.LpVariable.dicts("Startup_dis", range(T), cat='Binary')

        if self.dr_max_shed > 0:
            self._vars['load_shed'] = pl.LpVariable.dicts(
                "LoadShed", range(T), lowBound=0, upBound=self.dr_max_shed, cat='Continuous'
            )
        else:
            self._vars['load_shed'] = {t: 0 for t in range(T)}

        # OGRANIČENJA
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

        for t in range(T):
            self._prob += (self._vars['fne_used'][t] <= self.fne[t]), f"FNE_upper_{t}"
            if self.curtailment:
                self._prob += (self._vars['fne_used'][t] + self._vars['fne_curtailed'][t] == self.fne[t]), f"FNE_total_{t}"
            else:
                self._prob += (self._vars['fne_used'][t] == self.fne[t]), f"FNE_full_{t}"

        for k, c in enumerate(self.contracts):
            vol_max = c.get('volume_max', 0)
            vol_min = c.get('volume_min', 0)
            must_take = c.get('must_take', False)
            hours_allowed = c.get('hours', list(range(T)))
            for t in range(T):
                if t not in hours_allowed:
                    self._prob += (self._vars['contract'][k][t] == 0), f"Contract_{k}_hour_{t}_forbidden"
                if must_take:
                    self._prob += (self._vars['contract'][k][t] == vol_max), f"Contract_{k}_musttake_{t}"
                else:
                    self._prob += (self._vars['contract'][k][t] <= vol_max), f"Contract_{k}_max_{t}"
                    if vol_min > 0:
                        active = self._vars['contract_active'][k][t]
                        self._prob += (self._vars['contract'][k][t] >= vol_min * active), f"Contract_{k}_min_{t}"
                        self._prob += (self._vars['contract'][k][t] <= vol_max * active), f"Contract_{k}_min_upper_{t}"
            if 'total_max' in c:
                self._prob += (pl.lpSum(self._vars['contract'][k][t] for t in range(T)) <= c['total_max']), f"Contract_{k}_totalMax"
            if 'total_min' in c:
                self._prob += (pl.lpSum(self._vars['contract'][k][t] for t in range(T)) >= c['total_min']), f"Contract_{k}_totalMin"

        self._prob += (self._vars['soc'][0] == initial_soc), "SOC_initial"
        for t in range(T - 1):
            self._prob += (self._vars['soc'][t + 1] ==
                           self._vars['soc'][t] +
                           self.batt_eff * self._vars['ch'][t] -
                           self._vars['dis'][t] / self.batt_eff), f"SOC_dynamic_{t}"
        for t in range(T):
            self._prob += (self._vars['ch'][t] <= self.batt_pow * self._vars['u_ch'][t]), f"Ch_max_{t}"
            self._prob += (self._vars['dis'][t] <= self.batt_pow * self._vars['u_dis'][t]), f"Dis_max_{t}"
            if self.batt_min_power > 0:
                self._prob += (self._vars['ch'][t] >= self.batt_min_power * self._vars['u_ch'][t]), f"Ch_min_{t}"
                self._prob += (self._vars['dis'][t] >= self.batt_min_power * self._vars['u_dis'][t]), f"Dis_min_{t}"
            self._prob += (self._vars['u_ch'][t] + self._vars['u_dis'][t] <= 1), f"ChDis_mutex_{t}"
            if self.batt_startup_cost > 0:
                if t == 0:
                    self._prob += (self._vars['startup_ch'][t] >= self._vars['u_ch'][t]), f"Startup_ch_{t}"
                    self._prob += (self._vars['startup_dis'][t] >= self._vars['u_dis'][t]), f"Startup_dis_{t}"
                else:
                    self._prob += (self._vars['startup_ch'][t] >= self._vars['u_ch'][t] - self._vars['u_ch'][t - 1]), f"Startup_ch_{t}"
                    self._prob += (self._vars['startup_dis'][t] >= self._vars['u_dis'][t] - self._vars['u_dis'][t - 1]), f"Startup_dis_{t}"
        if self.batt_target_final_soc is not None:
            if self.batt_target_penalty > 0:
                dev_plus = pl.LpVariable("SOC_dev_plus", lowBound=0, cat='Continuous')
                dev_minus = pl.LpVariable("SOC_dev_minus", lowBound=0, cat='Continuous')
                self._prob += (self._vars['soc'][T - 1] - self.batt_target_final_soc == dev_plus - dev_minus), "SOC_target_deviation"
                self._soc_dev_vars = (dev_plus, dev_minus)
            else:
                self._prob += (self._vars['soc'][T - 1] == self.batt_target_final_soc), "SOC_target"
        if self.batt_max_cycles is not None:
            total_throughput = pl.lpSum(self._vars['ch'][t] + self._vars['dis'][t] for t in range(T))
            self._prob += (total_throughput <= 2 * self.batt_cap * self.batt_max_cycles), "Battery_cycle_limit"

        if self.grid_import_limit is not None:
            for t in range(T):
                self._prob += (self._vars['spot_buy'][t] <= self.grid_import_limit), f"GridImport_limit_{t}"
        if self.grid_export_limit is not None:
            for t in range(T):
                self._prob += (self._vars['spot_sell'][t] <= self.grid_export_limit), f"GridExport_limit_{t}"

        if self.co2_cap is not None:
            total_co2 = self.co2_intensity * pl.lpSum(self._vars['spot_buy'][t] for t in range(T))
            self._prob += (total_co2 <= self.co2_cap), "CO2_cap"

        objective = 0
        spot_cost = pl.lpSum(
            (self.spot_price[t] + self.co2_intensity * self.co2_price) * self._vars['spot_buy'][t]
            for t in range(T)
        )
        objective += spot_cost
        spot_revenue = pl.lpSum(
            self.feedin_tariff * self._vars['spot_sell'][t]
            for t in range(T)
        )
        objective -= spot_revenue
        for k, c in enumerate(self.contracts):
            price = c['price']
            if c.get('indexed', False):
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
        if self.batt_cycle_cost > 0:
            degradation_cost = self.batt_cycle_cost * pl.lpSum(
                self._vars['ch'][t] + self._vars['dis'][t] for t in range(T)
            )
            objective += degradation_cost
        if self.batt_startup_cost > 0:
            startup_cost = self.batt_startup_cost * pl.lpSum(
                self._vars['startup_ch'][t] + self._vars['startup_dis'][t] for t in range(T)
            )
            objective += startup_cost
        if self.dr_max_shed > 0 and self.dr_penalty > 0:
            shed_penalty = self.dr_penalty * pl.lpSum(
                self._vars['load_shed'][t] for t in range(T)
            )
            objective += shed_penalty
        if self.batt_target_final_soc is not None and self.batt_target_penalty > 0:
            dev_plus, dev_minus = self._soc_dev_vars
            objective += self.batt_target_penalty * (dev_plus + dev_minus)

        self._prob += objective, "TotalCost"

    def _solve(self):
        solver = pl.PULP_CBC_CMD(msg=False, gapRel=0.001, timeLimit=60)
        self._prob.solve(solver)

    def _extract_results(self) -> Dict[str, Any]:
        T = self.T
        K = self.n_contracts
        def val(var):
            return pl.value(var) if var is not None else 0.0

        spot_buy_vals = [val(self._vars['spot_buy'][t]) for t in range(T)]
        spot_sell_vals = [val(self._vars['spot_sell'][t]) for t in range(T)]
        fne_used_vals = [val(self._vars['fne_used'][t]) for t in range(T)]
        ch_vals = [val(self._vars['ch'][t]) for t in range(T)]
        dis_vals = [val(self._vars['dis'][t]) for t in range(T)]
        soc_vals = [val(self._vars['soc'][t]) for t in range(T)]
        contract_total_vals = [sum(val(self._vars['contract'][k][t]) for k in range(K)) for t in range(T)]

        if self.curtailment:
            curtailed_vals = [val(self._vars['fne_curtailed'][t]) for t in range(T)]
        else:
            curtailed_vals = [0.0] * T

        if self.dr_max_shed > 0:
            shed_vals = [val(self._vars['load_shed'][t]) for t in range(T)]
        else:
            shed_vals = [0.0] * T

        total_cost = val(self._prob.objective)
        total_spot_buy = np.sum(spot_buy_vals)
        co2_emissions = total_spot_buy * self.co2_intensity

        return {
            'spot_buy': np.array(spot_buy_vals),
            'spot_sell': np.array(spot_sell_vals),
            'grid_sales': np.array(spot_sell_vals),
            'contract': np.array(contract_total_vals),
            'fne_used': np.array(fne_used_vals),
            'fne_curtailed': np.array(curtailed_vals),
            'batt_ch': np.array(ch_vals),
            'batt_dis': np.array(dis_vals),
            'soc': np.array(soc_vals),
            'load_shed': np.array(shed_vals),
            'total_cost': total_cost,
            'co2_emissions': co2_emissions,
            'status': 'optimal'
        }
