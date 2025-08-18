
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import os
from . import error_funcs

import pandas as pd
from typing import Optional

def quick_look_plot_single_model(
    calval_table_df: pd.DataFrame,
    model_output_variable_name: str,
    model_name: str,
    insitu_variable_name: str = 'LEcorr50'
) -> plt.Figure:
    """
    Generate a scatter plot comparing model output to flux tower latent energy (LE) measurements, color-coded by vegetation type, with error bars and annotated performance metrics.

    This function is designed for visual evaluation of model performance against observed data, and is suitable for use in Jupyter notebooks (returns a matplotlib Figure object for direct display).

    Parameters
    ----------
    calval_table_df : pd.DataFrame
        DataFrame containing all relevant data for plotting and analysis. Must include columns for vegetation type, model output, flux tower LE, and optionally uncertainty/error columns.
    model_output_variable_name : str
        Column name in `calval_table_df` for the model output to be plotted on the y-axis.
    model_name : str
        Display name for the model, used as the plot title.
    insitu_variable_name : str, optional
        Column name for the flux tower LE values to be plotted on the x-axis (default is 'LEcorr50').

    Returns
    -------
    matplotlib.figure.Figure
        The generated matplotlib Figure object, ready for display in a Jupyter notebook or further customization.

    Notes
    -----
    - Points are color-coded by vegetation type using a predefined color map.
    - Error bars are included if uncertainty columns are present in the DataFrame.
    - The plot includes a 1:1 reference line, a regression line, and annotations for RMSE, bias, and R².
    - A legend for vegetation types is included below the plot.

    Example
    -------
    >>> fig = quick_look_plot_single_model(df, 'model_output', 'My Model')
    >>> fig.show()
    """
    # Define colors for each vegetation type for scatter plot visualization
    colors = {
        'CRO': '#FFEC8B', 'CSH': '#AB82FF', 'CVM': '#8B814C', 
        'DBF': '#98FB98', 'EBF': '#7FFF00', 'ENF': '#006400', 
        'GRA': '#FFA54F', 'MF': '#8FBC8F', 'OSH': '#FFE4E1', 
        'SAV': '#FFD700', 'WAT': '#98F5FF', 'WET': '#4169E1', 
        'WSA': '#CDAA7D'
    }
    # Assign a color to each data point based on its vegetation type
    scatter_colors = [colors.get(veg, 'gray') for veg in calval_table_df['vegetation']]
    # Create a reference 1:1 line for the plot (where model output equals flux tower measurement)
    one2one = np.arange(-250, 1200, 5)
    # Helper function to calculate performance metrics between model and observed data
    def calculate_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
        # RMSE: Root Mean Square Error
        rmse = error_funcs.rmse(y, x)
        # R2: Coefficient of Determination
        r2 = error_funcs.R2_fun(y, x)
        # Slope and intercept from linear regression
        slope, intercept = error_funcs.lin_regress(y, x)
        # Bias: Mean difference between model and observed
        bias = error_funcs.BIAS_fun(y,x)
        return rmse, r2, slope, intercept, bias
    # Extract flux tower LE values and model output values from DataFrame
    x = calval_table_df[insitu_variable_name].to_numpy()
    y = calval_table_df[model_output_variable_name].to_numpy()
    # Extract uncertainty in model output if available
    err = calval_table_df['ETinstUncertainty'].to_numpy() if 'ETinstUncertainty' in calval_table_df.columns else None
    # Calculate standard deviation across three LE columns for x error bars if all columns are present
    xerr = calval_table_df[['LE_filt', 'LEcorr50', 'LEcorr_ann']].std(axis=1).to_numpy() if all(col in calval_table_df.columns for col in ['LE_filt', 'LEcorr50', 'LEcorr_ann']) else None
    # Compute performance metrics for the plot annotation
    rmse, r2, slope, intercept, bias = calculate_metrics(x, y)
    # Count number of valid (non-NaN) data points
    number_of_points = np.sum(~np.isnan(y) & ~np.isnan(x))
    # Create the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot error bars if uncertainty data is available
    if err is not None and xerr is not None:
        ax.errorbar(x, y, yerr=err, xerr=xerr, fmt='', ecolor='lightgray')
    # Plot the scatter points, colored by vegetation type
    ax.scatter(x, y, c=scatter_colors, marker='o', s=6, zorder=4)
    # Plot the 1:1 reference line (ideal fit)
    ax.plot(one2one, one2one, '--', c='k')
    # Plot the regression line (actual fit)
    ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
    # Set plot title and axis limits
    ax.set_title(model_name)
    ax.set_xlim([-250, 1200])
    ax.set_ylim([-250, 1200])
    # Set axis labels
    ax.set_ylabel('Model LE Wm$^{-2}$',fontsize=14)
    ax.set_xlabel('Flux Tower LE Wm$^{-2}$',fontsize=14)
    # Add subplot label
    ax.text(-0.1, 1.1, 'a)', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')
    # Annotate plot with regression equation, RMSE, bias, and R2
    ax.text(500, -200, f'y = {slope:.1f}x + {intercept:.1f} \nRMSE: {rmse:.1f} Wm$^-$² \nbias: {bias:.1f} Wm$^-$² \nR$^2$: {r2:.2f}', fontsize=12)
    # Create legend handles for each vegetation type
    scatter_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=6, label=veg) for veg, color in colors.items()]
    # Add legend to the figure
    fig.legend(handles=scatter_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, title='Vegetation Type',fontsize=10)
    # Adjust layout for better appearance
    fig.tight_layout()
    # Return the figure object for display in Jupyter notebooks
    return fig
