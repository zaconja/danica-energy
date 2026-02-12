import pulp as pl
import numpy as np

class MILPDayAheadOptimizer:
    def __init__(self, load, fne, spot_price,
                 contracted_volume, contracted_price,
                 batt_capacity_mwh, batt_power_mw,
                 batt_efficiency=0.9,
                 co2_intensity=0.4, co2_price=80,
                 feedin_tariff=50,
                 batt_min_power=0.1,
                 batt_cycle_cost=5,
                 batt_startup_cost=0):
        self.T = 24
        self.load = np.array(load)
        self.fne = np.array(fne)
        self.spot_price = np.array(spot_price)
        self.contr_vol = contracted_volume
        self.contr_price = contracted_price
        self.batt_cap = batt_capacity_mwh
        self.batt_pow = batt_power_mw
        self.eff = batt_efficiency
        self.co2_intensity = co2_intensity
        self.co2_price = co2_price
        self.feedin = feedin_tariff
        self.batt_min_power = batt_min_power
        self.batt_cycle_cost = batt_cycle_cost
        self.batt_startup_cost = batt_startup_cost

    def optimize(self, initial_soc=0.0):
        T = self.T
        prob = pl.LpProblem("DayAheadOptimization", pl.LpMinimize)

        # Varijable
        spot = pl.LpVariable.dicts("spot", range(T), lowBound=0, cat='Continuous')
        contr = pl.LpVariable.dicts("contr", range(T), lowBound=0, cat='Continuous')
        sales = pl.LpVariable.dicts("sales", range(T), lowBound=0, cat='Continuous')
        ch = pl.LpVariable.dicts("ch", range(T), lowBound=0, cat='Continuous')
        dis = pl.LpVariable.dicts("dis", range(T), lowBound=0, cat='Continuous')
        soc = pl.LpVariable.dicts("soc", range(T), lowBound=0, upBound=self.batt_cap, cat='Continuous')

        # Binarne varijable za stanje baterije
        u_ch = pl.LpVariable.dicts("u_ch", range(T), cat='Binary')
        u_dis = pl.LpVariable.dicts("u_dis", range(T), cat='Binary')

        # Ograničenja
        # 1. Bilanca
        for t in range(T):
            prob += (self.fne[t] + contr[t] + dis[t] + spot[t] ==
                     self.load[t] + ch[t] + sales[t]), f"balance_{t}"

        # 2. Ukupna ugovorena količina
        prob += pl.lpSum(contr[t] for t in range(T)) <= self.contr_vol, "total_contract"

        # 3. Dinamika baterije
        prob += soc[0] == initial_soc, "soc_initial"
        for t in range(T-1):
            prob += (soc[t+1] == soc[t] + self.eff * ch[t] - dis[t] / self.eff), f"soc_dynamic_{t}"

        # 4. Ograničenja baterije (snaga, min/max, isključivost)
        for t in range(T):
            # Maksimalna snaga
            prob += ch[t] <= self.batt_pow * u_ch[t], f"ch_max_{t}"
            prob += dis[t] <= self.batt_pow * u_dis[t], f"dis_max_{t}"
            # Minimalna snaga (ako je definirana)
            if self.batt_min_power > 0:
                prob += ch[t] >= self.batt_min_power * u_ch[t], f"ch_min_{t}"
                prob += dis[t] >= self.batt_min_power * u_dis[t], f"dis_min_{t}"
            # Ne istovremeno punjenje i pražnjenje
            prob += u_ch[t] + u_dis[t] <= 1, f"ch_dis_mutex_{t}"

        # 5. Funkcija cilja
        spot_cost = pl.lpSum((self.spot_price[t] + self.co2_intensity * self.co2_price) * spot[t]
                             for t in range(T))
        contr_cost = pl.lpSum(self.contr_price * contr[t] for t in range(T))
        sales_revenue = pl.lpSum(self.feedin * sales[t] for t in range(T))
        degradation_cost = self.batt_cycle_cost * pl.lpSum(ch[t] + dis[t] for t in range(T))

        prob += spot_cost + contr_cost + degradation_cost - sales_revenue, "TotalCost"

        # Rješavanje
        solver = pl.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if prob.status == pl.LpStatusOptimal:
            spot_vals = [pl.value(spot[t]) for t in range(T)]
            contr_vals = [pl.value(contr[t]) for t in range(T)]
            sales_vals = [pl.value(sales[t]) for t in range(T)]
            ch_vals = [pl.value(ch[t]) for t in range(T)]
            dis_vals = [pl.value(dis[t]) for t in range(T)]
            soc_vals = [pl.value(soc[t]) for t in range(T)]

            return {
                'spot': np.array(spot_vals),
                'contr': np.array(contr_vals),
                'grid_sales': np.array(sales_vals),
                'batt_ch': np.array(ch_vals),
                'batt_dis': np.array(dis_vals),
                'soc': np.array(soc_vals),
                'total_cost': pl.value(prob.objective),
                'co2_emissions': np.sum(spot_vals) * self.co2_intensity,
                'status': 'optimal'
            }
        else:
            return {'status': 'failed', 'message': f'Solver status: {pl.LpStatus[prob.status]}'}
