from instruments import Swaption
from models import SwaptionPricer
from market_data import df_swaption
from utils import display_3d_grid, actual_360, display_cube
import pandas as pd

vols = []

swaption_pricer = SwaptionPricer()

for _, row in df_swaption.iterrows():
    swpt = Swaption(actual_360, row.option_maturity, row.swap_tenor, row.strike, True)
    vol = swaption_pricer.implied_vol(swpt, row.option_maturity, row.price, row.forward_rate , row.discount_factor)
    vols.append(vol)

df_swaption["implied_vol"] = vols

X_list, Y_list, Z_list, titles = [], [], [], []

for tenor in df_swaption["swap_tenor"].unique():
    df_t = df_swaption[df_swaption["swap_tenor"] == tenor]
    X_list.append(df_t["option_maturity"].values)
    Y_list.append(df_t["strike"].values)
    Z_list.append(df_t["implied_vol"].values)
    titles.append(f"Tenor = {tenor}Y")

def display_vols():
    display_3d_grid(
    X_list, Y_list, Z_list,
    titles=titles,
    xlabels="Maturity", ylabels="Strike", zlabels="Volatility (%)",
    ncols=3
    )

    display_cube(
        df_swaption["option_maturity"],
        df_swaption["swap_tenor"],
        df_swaption["strike"],
        df_swaption["implied_vol"],
        xlabel="Maturity",
        ylabel="Tenor",
        zlabel="Strike",
        value_label="Implied Volatility",
        title="Implied Volatility Cube"
    )