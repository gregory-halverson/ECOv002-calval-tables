
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import os
import pandas as pd
from typing import Optional
from . import error_funcs

def quick_look_plot_single_model(
    big_df_ss: pd.DataFrame,
    time: str,
    model_col: str,
    model_name: str,
    LE_var: str = 'LEcorr50',
    output_filename: Optional[str] = None
) -> None:
    """
    Plots the results for a single model against flux tower LE.
    Parameters:
        big_df_ss: pd.DataFrame containing all data
        time: string for output file naming
        model_col: column name for model output
        model_name: display name for the model
        LE_var: column name for flux tower LE (default 'LEcorr50')
        output_filename: Optional path for output figure file. If None, figure is not saved.
    """
    colors = {
        'CRO': '#FFEC8B', 'CSH': '#AB82FF', 'CVM': '#8B814C', 
        'DBF': '#98FB98', 'EBF': '#7FFF00', 'ENF': '#006400', 
        'GRA': '#FFA54F', 'MF': '#8FBC8F', 'OSH': '#FFE4E1', 
        'SAV': '#FFD700', 'WAT': '#98F5FF', 'WET': '#4169E1', 
        'WSA': '#CDAA7D'
    }
    scatter_colors = [colors.get(veg, 'gray') for veg in big_df_ss['vegetation']]
    one2one = np.arange(-250, 1200, 5)
    def calculate_metrics(x, y):
        rmse = error_funcs.rmse(y, x)
        r2 = error_funcs.R2_fun(y, x)
        slope, intercept = error_funcs.lin_regress(y, x)
        bias = error_funcs.BIAS_fun(y,x)
        return rmse, r2, slope, intercept, bias
    x = big_df_ss[LE_var].to_numpy()
    y = big_df_ss[model_col].to_numpy()
    err = big_df_ss['ETinstUncertainty'].to_numpy() if 'ETinstUncertainty' in big_df_ss.columns else None
    xerr = big_df_ss[['LE_filt', 'LEcorr50', 'LEcorr_ann']].std(axis=1).to_numpy() if all(col in big_df_ss.columns for col in ['LE_filt', 'LEcorr50', 'LEcorr_ann']) else None
    rmse, r2, slope, intercept, bias = calculate_metrics(x, y)
    number_of_points = np.sum(~np.isnan(y) & ~np.isnan(x))
    fig, ax = plt.subplots(figsize=(6, 6))
    if err is not None and xerr is not None:
        ax.errorbar(x, y, yerr=err, xerr=xerr, fmt='none', ecolor='lightgray')
    ax.scatter(x, y, c=scatter_colors, marker='o', s=6, zorder=4)
    ax.plot(one2one, one2one, '--', c='k')
    ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
    ax.set_title(model_name)
    ax.set_xlim([-250, 1200])
    ax.set_ylim([-250, 1200])
    ax.set_ylabel('Model LE Wm$^{-2}$',fontsize=14)
    ax.set_xlabel('Flux Tower LE Wm$^{-2}$',fontsize=14)
    ax.text(-0.1, 1.1, 'a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    ax.text(500, -200, f'y = {slope:.1f}x + {intercept:.1f} \nRMSE: {rmse:.1f} Wm$^-$² \nbias: {bias:.1f} Wm$^-$² \nR$^2$: {r2:.2f}', fontsize=12)
    scatter_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=6, label=veg) for veg, color in colors.items()]
    fig.legend(handles=scatter_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, title='Vegetation Type',fontsize=10)
    fig.subplots_adjust(bottom=0.2)
    fig.tight_layout()
    if output_filename is not None:
        fig.savefig(output_filename, dpi=600, bbox_inches='tight')
    # Return the figure object for notebook display
    return fig
