from instruments import Swaption
from models import SwaptionPricer, sabr_fit
from market_data import df_swaption, strikes, market_vols
from utils import display_3d_grid, actual_360, display_cube, display_tabular
import pandas as pd

vols = []

swaption_pricer = SwaptionPricer()

for _, row in df_swaption.iterrows():
    swpt = Swaption(actual_360, row.option_maturity, row.swap_tenor, row.strike, True)
    vol = swaption_pricer.implied_vol(swpt, row.option_maturity, row.price, row.forward_rate , row.discount_factor)
    vols.append(vol)

df_swaption["implied_vol"] = vols

X_list, Y_list, Z_list, titles = [], [], [], []

for strike in df_swaption["strike"].unique():
    df_t = df_swaption[df_swaption["strike"] == strike]
    X_list.append(df_t["option_maturity"].values)
    Y_list.append(df_t["swap_tenor"].values)
    Z_list.append(df_t["implied_vol"].values)
    titles.append(f"strike = {strike}Y")

def display_vols():
    display_3d_grid(
    [X_list], [Y_list], [Z_list],
    titles=titles,
    xlabels="Maturity", ylabels="Tenor", zlabels="Volatility (%)",
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

def compute_sabr_fit():
    F = 0.025  
    T = 5.0    

    strikes_array = strikes
    market_vols_array = market_vols

    popt = sabr_fit(F, T, strikes_array, market_vols_array)
    
    display_tabular([["Alpha", "Beta", "Rho", "Nu"], popt],
                    headers=["Parameter", "Value"],)
    
    return popt