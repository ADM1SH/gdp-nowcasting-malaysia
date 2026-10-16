from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import Ridge

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class GDPNowcastResult(BaseModel):
    year: int
    nowcast_point_pct: float
    ci_lower_pct: float
    ci_upper_pct: float
    mof_official_range: str
    within_official_range: bool
    services_growth_pct: float
    ipi_growth_pct: float

def run_gdp_nowcast_model(
    services_2027_growth_pct: float = 5.2,
    ipi_2027_growth_pct: float = 4.3,
) -> List[GDPNowcastResult]:
    df_gdp = pd.read_csv(DATA_DIR / "table_3_1_gdp_sector.csv")
    df_srv = pd.read_csv(DATA_DIR / "table_3_2_index_of_services.csv")
    df_ipi = pd.read_csv(DATA_DIR / "table_3_3_ipi.csv")

    years = [2022, 2023, 2024, 2025, 2026]
    srv_total = df_srv[df_srv["subsector"] == "Services Total"].iloc[0]
    ipi_total = df_ipi[df_ipi["subsector"] == "Total Industrial Production"].iloc[0]
    gdp_total = df_gdp[df_gdp["sector"] == "GDP at purchasers' prices"].iloc[0]

    srv_hist = [float(srv_total[f"y{y}"]) for y in years]
    ipi_hist = [float(ipi_total[f"y{y}"]) for y in years]
    gdp_hist = [float(gdp_total[f"g_{y}"]) for y in years]

    X_train = np.column_stack([srv_hist, ipi_hist])
    y_train = np.array(gdp_hist)

    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)

    residuals = y_train - model.predict(X_train)
    rmse = max(float(np.std(residuals)), 0.22)

    all_years = years + [2027]
    all_srv = srv_hist + [services_2027_growth_pct]
    all_ipi = ipi_hist + [ipi_2027_growth_pct]

    results: List[GDPNowcastResult] = []
    for y, s_val, i_val in zip(all_years, all_srv, all_ipi):
        pred = float(model.predict([[s_val, i_val]])[0])
        ci_low = round(pred - 1.96 * rmse, 2)
        ci_high = round(pred + 1.96 * rmse, 2)
        pred_round = round(pred, 2)

        if y == 2026:
            official = "4.8% - 5.3%"
            in_range = 4.8 <= pred_round <= 5.3
        elif y == 2027:
            official = "4.2% - 5.2%"
            in_range = 4.2 <= pred_round <= 5.2
        else:
            actual_g = float(gdp_total[f"g_{y}"])
            official = f"{actual_g:.1f}% (Actual)"
            in_range = abs(pred_round - actual_g) <= 0.6

        results.append(
            GDPNowcastResult(
                year=y,
                nowcast_point_pct=pred_round,
                ci_lower_pct=ci_low,
                ci_upper_pct=ci_high,
                mof_official_range=official,
                within_official_range=bool(in_range),
                services_growth_pct=round(s_val, 2),
                ipi_growth_pct=round(i_val, 2),
            )
        )
    return results
