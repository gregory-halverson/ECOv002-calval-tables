"""
This module contains functions for plotting and evaluating meteorological and
flux tower data.

It provides several plotting functions for a quick look at data, including
scatter plots for evaluating different meteorological variables and energy
fluxes, and a function for quality assurance and quality control (QAQC) plots.
The module relies on libraries like pandas, numpy, and matplotlib for data
manipulation and visualization.
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pandas as pd
import sys
from matplotlib.dates import DateFormatter

# Assuming 'error_funcs' is a local module in the same directory or on the
# Python path
from . import error_funcs

REL_PATH = os.getcwd() + '/'
FIG_PATH = REL_PATH + 'results/figures/'
LIB_PATH = REL_PATH + 'src'
sys.path.insert(0, LIB_PATH)


def quick_look_plots_met(big_df_ss, time):
    """
    Generates a set of 3x2 scatter plots comparing various meteorological
    variables from a model against observed data.

    The function plots Net Radiation, Downwelling Shortwave Radiation, Air
    Temperature, Relative Humidity, Surface Soil Moisture, and Root Zone Soil
    Moisture. It calculates and displays key statistical metrics (RMSE, R²,
    slope, intercept, and bias) for each variable.

    Args:
        big_df_ss (pd.DataFrame): A DataFrame containing both model and
                                    observed data.
        time (str): A string representing the time period for the plot file
                    name.
    """
    # Set plotting style and parameters
    plt.rc('lines', linestyle='None')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    one2one = np.arange(-250, 1200, 5)

    def plot_metric(
        ax,
        x_data,
        y_data,
        title,
        xlabel,
        ylabel,
        color,
        ylim,
        xlim,
        plot_label,
        text_x,
        text_y,
    ):
        """Helper function to plot a single metric and its statistics."""
        rmse = error_funcs.rmse(y_data, x_data)
        r2 = error_funcs.R2_fun(y_data, x_data)
        slope, intercept = error_funcs.lin_regress(y_data, x_data)
        bias = error_funcs.BIAS_fun(y_data, x_data)

        ax.scatter(x_data, y_data, c=color, marker='o', s=4)
        ax.plot(one2one, one2one, '--', c='k')
        ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.text(
            text_x,
            text_y,
            f'y = {round(slope, 2)}x + {round(intercept, 1)}\n'
            f'RMSE: {round(rmse, 1)} {ylabel.split(" ")[-1]}\n'
            f'bias: {round(bias, 1)} {ylabel.split(" ")[-1]}\n'
            f'R²: {round(r2, 2)}',
        )
        ax.text(
            -0.2,
            1.05,
            f'{plot_label})',
            transform=ax.transAxes,
            fontsize=14,
            weight='bold',
        )

    # Net Radiation Plot
    plot_metric(
        axs[0, 0],
        big_df_ss.NETRAD_filt.to_numpy(),
        big_df_ss.Rn.to_numpy(),
        'Net Radiation',
        'Obs Rn Wm$^{-2}$',
        'Model Rn Wm$^{-2}$',
        'darkorange',
        [-200, 1000],
        [-200, 1000],
        'a',
        -150,
        600,
    )

    # Downwelling Shortwave Radiation Plot
    plot_metric(
        axs[0, 1],
        big_df_ss.SW_IN.to_numpy(),
        big_df_ss.Rg.to_numpy(),
        'Downwelling Shortwave Radiation',
        'Obs R$_{SD}$ Wm$^{-2}$',
        'Model R$_{SD}$ Wm$^{-2}$',
        'orange',
        [-200, 1500],
        [-200, 1500],
        'b',
        -150,
        950,
    )

    # Air Temperature Plot
    plot_metric(
        axs[1, 0],
        big_df_ss.AirTempC.to_numpy(),
        big_df_ss.Ta.to_numpy(),
        'Air Temp (C)',
        'Obs Ta $^{o}$C',
        'Model Ta $^{o}$C',
        'darkred',
        [-25, 40],
        [-25, 40],
        'c',
        10,
        -22,
    )

    # Relative Humidity Plot
    plot_metric(
        axs[1, 1],
        big_df_ss.RH_percentage.to_numpy(),
        big_df_ss.RH.to_numpy(),
        'RH',
        'Obs RH',
        'Model RH',
        'royalblue',
        [0, 1],
        [0, 1],
        'd',
        0.5,
        0.05,
    )

    # Surface Soil Moisture Plot
    big_df_ss.loc[big_df_ss.SM_surf > 0.60, 'SM_surf'] = np.nan
    plot_metric(
        axs[2, 0],
        big_df_ss.SM_surf.to_numpy(),
        big_df_ss.SM.to_numpy(),
        'SM$_{surf}$',
        'Obs VWC m$^{3}$m$^{-3}$',
        'Model VWC m$^{3}$m$^{-3}$',
        'lightblue',
        [0, 0.8],
        [0, 0.8],
        'e',
        0.1,
        0.55,
    )

    # Root Zone Soil Moisture Plot
    big_df_ss.loc[big_df_ss.SM_rz > 0.60, 'SM_rz'] = np.nan
    plot_metric(
        axs[2, 1],
        big_df_ss.SM_rz.to_numpy(),
        big_df_ss.SM.to_numpy(),
        'SM$_{rz}$',
        'Obs VWC m$^{3}$m$^{-3}$',
        'Model VWC m$^{3}$m$^{-3}$',
        'darkblue',
        [0, 0.8],
        [0, 0.8],
        'f',
        0.1,
        0.55,
    )

    fig.tight_layout()
    fig.savefig(FIG_PATH + 'auxiliary/auxiliary_eval_' + time + '.png', dpi=300)


def quick_look_plots(big_df_ss, time, LE_var='LEcorr50'):
    """
    Generates a set of 3x2 scatter plots comparing various energy flux models
    against a chosen reference variable, with optional error bars and a legend
    for vegetation types.

    The function plots Latent Heat (LE) fluxes from six different models:
    PT-JPL (C1), JET, PT-JPL_SM, BESS, STIC, and MOD16. It calculates and
    displays key statistical metrics (RMSE, R², slope, intercept, and bias) for
    each model's performance.

    Args:
        big_df_ss (pd.DataFrame): A DataFrame containing model and observed
                                    data.
        time (str): A string representing the time period for the plot file
                    name.
        LE_var (str, optional): The name of the column in big_df_ss to use as
                                the reference LE variable. Defaults to
                                'LEcorr50'.
    """
    plt.rc('lines', linestyle='None')
    plt.style.use('seaborn-v0_8-whitegrid')

    colors = {
        'CRO': '#FFEC8B', 'CSH': '#AB82FF', 'CVM': '#8B814C',
        'DBF': '#98FB98', 'EBF': '#7FFF00', 'ENF': '#006400',
        'GRA': '#FFA54F', 'MF': '#8FBC8F', 'OSH': '#FFE4E1',
        'SAV': '#FFD700', 'WAT': '#98F5FF', 'WET': '#4169E1',
        'WSA': '#CDAA7D',
    }

    scatter_colors = [colors.get(veg, 'gray') for veg in big_df_ss['vegetation']]
    one2one = np.arange(-250, 1200, 5)

    def calculate_metrics(x, y):
        """Helper function to calculate error metrics."""
        rmse = error_funcs.rmse(y, x)
        r2 = error_funcs.R2_fun(y, x)
        slope, intercept = error_funcs.lin_regress(y, x)
        bias = error_funcs.BIAS_fun(y, x)
        return rmse, r2, slope, intercept, bias

    metrics = {
        'ETinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.ETinst),
        'JET': calculate_metrics(big_df_ss[LE_var], big_df_ss.JET),
        'PTJPLSMinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.PTJPLSMinst),
        'BESSinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.BESSinst),
        'STICinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.STICinst),
        'MOD16inst': calculate_metrics(big_df_ss[LE_var], big_df_ss.MOD16inst),
    }

    model_names = {
        'ETinst': 'PT-JPL (C1)',
        'JET': 'JET',
        'PTJPLSMinst': 'PT-JPL$_{SM}$',
        'BESSinst': 'BESS',
        'STICinst': 'STIC',
        'MOD16inst': 'MOD16',
    }

    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    plt.rcParams.update({'font.size': 14})
    subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    for i, (key, (rmse, r2, slope, intercept, bias)) in enumerate(metrics.items()):
        x = big_df_ss[LE_var].to_numpy()
        y = big_df_ss[f'{key}'].to_numpy()
        err = big_df_ss['ETinstUncertainty'].to_numpy()
        xerr = big_df_ss[['LE_filt', 'LEcorr50', 'LEcorr_ann']].std(axis=1).to_numpy()

        ax = axs[i // 2, i % 2]
        ax.errorbar(x, y, yerr=err, xerr=xerr, fmt='', ecolor='lightgray')
        ax.scatter(x, y, c=scatter_colors, marker='o', s=4, zorder=4)
        ax.plot(one2one, one2one, '--', c='k')
        ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
        ax.set_title(model_names[key])
        ax.set_xlim([-250, 1200])
        ax.set_ylim([-250, 1200])
        if i % 2 == 0:
            ax.set_ylabel('Model LE Wm$^{-2}$', fontsize=14)

        ax.text(
            -0.1,
            1.1,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=16,
            fontweight='bold',
            va='top',
            ha='right',
        )
        ax.text(
            500,
            -200,
            f'y = {slope:.1f}x + {intercept:.1f} \n'
            f'RMSE: {rmse:.1f} Wm$^-$² \n'
            f'bias: {bias:.1f} Wm$^-$² \n'
            f'R$^2$: {r2:.2f}',
            fontsize=12,
        )
        if i // 2 == 2:
            ax.set_xlabel('Flux Tower LE Wm$^{-2}$', fontsize=14)

    # Create legend
    scatter_handles = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker='o',
            linestyle='None',
            markersize=6,
            label=veg,
        )
        for veg, color in colors.items()
    ]
    fig.legend(
        handles=scatter_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=7,
        title='Vegetation Type',
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(
        f'{FIG_PATH}/le_fluxes/le_eval_{LE_var}_{time}.png',
        dpi=600,
        bbox_inches='tight',
    )


def quick_look_plots_filt(big_df_ss, time, LE_var='LEcorr50'):
    """
    Generates a set of 3x2 scatter plots similar to `quick_look_plots`, but with
    additional filtering on BESS and STIC data.

    This function is a variant of `quick_look_plots` that specifically handles
    filtering out zero and extreme values from 'BESSinst' and 'STICinst' data
    before plotting, and presents the results for the same set of energy flux
    models.

    Args:
        big_df_ss (pd.DataFrame): A DataFrame containing model and observed
                                    data.
        time (str): A string representing the time period for the plot file
                    name.
        LE_var (str, optional): The name of the column in big_df_ss to use as
                                the reference LE variable. Defaults to
                                'LEcorr50'.
    """
    big_df_ss['BESSinst'].replace(0, np.nan, inplace=True)
    big_df_ss['BESSinst'].replace(1000, np.nan, inplace=True)
    big_df_ss['STICinst'].replace(0, np.nan, inplace=True)

    plt.rc('lines', linestyle='None')
    plt.style.use('seaborn-v0_8-whitegrid')

    colors = {
        'CRO': '#FFEC8B', 'CSH': '#AB82FF', 'CVM': '#8B814C',
        'DBF': '#98FB98', 'EBF': '#7FFF00', 'ENF': '#006400',
        'GRA': '#FFA54F', 'MF': '#8FBC8F', 'OSH': '#FFE4E1',
        'SAV': '#FFD700', 'WAT': '#98F5FF', 'WET': '#4169E1',
        'WSA': '#CDAA7D',
    }

    scatter_colors = [colors.get(veg, 'gray') for veg in big_df_ss['vegetation']]
    one2one = np.arange(-250, 1200, 5)

    def calculate_metrics(x, y):
        """Helper function to calculate error metrics."""
        rmse = error_funcs.rmse(y, x)
        r2 = error_funcs.R2_fun(y, x)
        slope, intercept = error_funcs.lin_regress(y, x)
        bias = error_funcs.BIAS_fun(y, x)
        return rmse, r2, slope, intercept, bias

    metrics = {
        'ETinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.ETinst),
        'PTJPLSMinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.PTJPLSMinst),
        'BESSinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.BESSinst),
        'STICinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.STICinst),
        'MOD16inst': calculate_metrics(big_df_ss[LE_var], big_df_ss.MOD16inst),
        'JET': calculate_metrics(big_df_ss[LE_var], big_df_ss.JET),
    }

    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    plt.rcParams.update({'font.size': 14})

    for i, (key, (rmse, r2, slope, intercept, bias)) in enumerate(metrics.items()):
        x = big_df_ss[LE_var].to_numpy()
        y = big_df_ss[f'{key}'].to_numpy()
        err = big_df_ss['ETinstUncertainty'].to_numpy()
        xerr = big_df_ss[['LE_filt', 'LEcorr50', 'LEcorr_ann']].std(axis=1).to_numpy()

        ax = axs[i // 2, i % 2]
        ax.errorbar(x, y, yerr=err, xerr=xerr, fmt='', ecolor='lightgray')
        ax.scatter(x, y, c=scatter_colors, marker='o', s=4, zorder=4)
        ax.plot(one2one, one2one, '--', c='k')
        ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
        ax.set_title(f'{key}')
        ax.set_xlim([-250, 1200])
        ax.set_ylim([-250, 1200])
        if i % 2 == 0:
            ax.set_ylabel('Model LE Wm$^{-2}$', fontsize=14)

        ax.text(
            500,
            -200,
            f'y = {slope:.2f}x + {intercept:.2f}\n'
            f'RMSE: {rmse:.2f} Wm$^{-2}$\n'
            f'R$^2$: {r2:.3f}\n'
            f'bias: {bias:.2f}Wm$^{-2}',
            fontsize=12,
        )
        if i // 2 == 2:
            ax.set_xlabel('Flux Tower LE Wm$^{-2}$', fontsize=14)

    fig.tight_layout()
    fig.savefig(
        f'{FIG_PATH}/le_fluxes/le_eval_filt_{time}.png', dpi=600,
    )


def plot_colocated(ground_site_df, eco_site_i_df, site, utc_offset):
    """
    Generates a series of diurnal plots comparing ground-based flux data with
    colocated model data for a specific site.

    This function plots various observed energy fluxes (NETRAD, LE, H, G) for
    each day and superimposes a point for the 'JET' model's LE flux at the
    corresponding observation time, including an uncertainty bar.

    Args:
        ground_site_df (pd.DataFrame): DataFrame containing ground-based
                                        flux data.
        eco_site_i_df (pd.DataFrame): DataFrame containing colocated model data.
        site (str): The name of the site.
        utc_offset (int): The UTC offset for the site.
    """
    eco_site_i_df['JET'] = eco_site_i_df[
        ['PTJPLSMinst', 'BESSinst', 'STICinst', 'MOD16inst']
    ].median(axis=1)

    eco_site_i_df['solar_time'] = eco_site_i_df.index + pd.DateOffset(
        hours=utc_offset,
    )
    ground_site_df['solar_time'] = ground_site_df.index + pd.DateOffset(
        hours=utc_offset,
    )

    for idx in eco_site_i_df.index:
        solar_day = (idx + pd.DateOffset(hours=utc_offset)).normalize()
        df_ground_day = ground_site_df[
            ground_site_df['solar_time'].dt.normalize() == solar_day
        ]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['NETRAD_filt'],
            label='NETRAD_filt',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['LE_filt'],
            label='LE_filt',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['LEcorr50'],
            label='LEcorr50',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['H_filt'],
            label='H_filt',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['G_filt'],
            label='G_filt',
        )

        y_value = eco_site_i_df.loc[idx, 'JET']
        yerr_value = eco_site_i_df.loc[idx, 'ETinstUncertainty']
        ax.errorbar(
            eco_site_i_df.loc[idx, 'solar_time'],
            y_value,
            yerr=yerr_value,
            fmt='ro',
            label=f'JET {idx.time()}',
        )

        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        ax.set_ylabel('Wm$^{-2}$')
        ax.legend(fontsize='x-small')
        plt.title(site + ' ' + f'{idx}')
        plt.savefig(
            FIG_PATH
            + 'supplementary/diurnal_observations/'
            + site
            + '_'
            + f'{idx}'
            + '.png',
            dpi=300,
        )
        plt.close(fig)


def plot_blind_filter(ground_site_df, all_site_eco_data_df, site, utc_offset):
    """
    Generates a series of diurnal plots for a specific site, comparing
    ground-based flux data with a model's 'JET' flux.

    This function plots observed energy fluxes (NETRAD, LE, H, G) for each day
    and adds a vertical line at the observation time of the 'JET' model to
    visualize the model's output in the context of the ground observations.

    Args:
        ground_site_df (pd.DataFrame): DataFrame containing ground-based
                                        flux data.
        all_site_eco_data_df (pd.DataFrame): DataFrame containing model data
                                                for all sites.
        site (str): The name of the site.
        utc_offset (int): The UTC offset for the site.
    """
    all_site_eco_data_df['JET'] = all_site_eco_data_df[
        ['PTJPLSMinst', 'BESSinst', 'STICinst', 'MOD16inst']
    ].median(axis=1)

    all_site_eco_data_df['solar_time'] = all_site_eco_data_df.index + pd.DateOffset(
        hours=utc_offset,
    )
    ground_site_df['solar_time'] = ground_site_df.index + pd.DateOffset(
        hours=utc_offset,
    )

    for idx in all_site_eco_data_df.index:
        solar_day = (idx + pd.DateOffset(hours=utc_offset)).normalize()
        df_ground_day = ground_site_df[
            ground_site_df['solar_time'].dt.normalize() == solar_day
        ]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['NETRAD_filt'],
            label='NETRAD_filt',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['LE_filt'],
            label='LE_filt',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['LEcorr50'],
            label='LEcorr50',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['H_filt'],
            label='H_filt',
        )
        ax.plot(
            df_ground_day['solar_time'],
            df_ground_day['G_filt'],
            label='G_filt',
        )

        ax.axvline(
            x=all_site_eco_data_df.loc[idx, 'solar_time'],
            color='red',
            linestyle='--',
            label='Observation time',
        )

        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        ax.set_ylabel('Wm$^{-2}$')
        ax.legend(fontsize='x-small')
        plt.title(site + ' ' + f'{idx}')
        plt.savefig(
            FIG_PATH
            + 'supplementary/blind_filter/'
            + site
            + '_'
            + f'{idx}'
            + '.png',
            dpi=300,
        )
        plt.close(fig)


def qaqc_plots(site):
    """
    Generates a series of quality assurance and quality control (QAQC) plots
    for a specified site's flux data.

    This function reads a filtered flux tower data file and creates plots for
    NETRAD, LE, H, and G. The final subplot provides text-based statistics
    about energy balance closure ratios and data availability after filtering.

    Args:
        site (str): The name of the site for which to generate the plots.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    data_path = REL_PATH + 'data/cleaned_data/'
    file_name = data_path + site + '_qaqc_filt_ebc.csv'
    site_df = pd.read_csv(file_name)

    site_df['index_time'] = pd.to_datetime(site_df['local_time'])
    site_df.set_index('index_time', inplace=True)

    fig, axs = plt.subplots(5, 1, figsize=(12, 12))

    def plot_data(ax, data, label, ylabel):
        """Helper function to plot a single data series."""
        if data is not None and not data.empty:
            data.plot(label=label, ax=ax, x_compat=True)
            ax.legend()
            ax.set_ylabel(ylabel)
        else:
            ax.set_xticks([])

    plot_data(axs[0], site_df.get('NETRAD_filt'), 'NETRAD', 'NETRAD Wm-2')

    if 'LE' in site_df.columns and 'LE_filt' in site_df.columns:
        plot_data(axs[1], site_df['LE'], 'LE', 'LE Wm-2')
        plot_data(axs[1], site_df['LE_filt'], 'LE_filt', 'LE_filt Wm-2')

    plot_data(axs[2], site_df.get('H_filt'), 'H', 'H Wm-2')
    plot_data(axs[3], site_df.get('G_filt'), 'G', 'G Wm-2')

    axs[4].set_title('Statistics')
    axs[4].axis('off')

    if all(
        col in site_df.columns
        for col in ['LE_filt', 'LEcorr_ann', 'LEcorr50', 'LEcorr25', 'LEcorr75']
    ):
        closure_ratio = round(np.mean(site_df['LE_filt'] / site_df['LEcorr_ann']), 2)
        le_corr50_mean = round(np.mean(site_df['LE_filt'] / site_df['LEcorr50']), 2)
        le_corr25_mean = round(np.mean(site_df['LE_filt'] / site_df['LEcorr25']), 2)
        le_corr75_mean = round(np.mean(site_df['LE_filt'] / site_df['LEcorr75']), 2)

        data_availability = {
            'NETRAD': round(site_df['NETRAD_filt'].count() / len(site_df.index), 2),
            'LE': round(site_df['LE_filt'].count() / len(site_df.index), 2),
            'H': round(site_df['H_filt'].count() / len(site_df.index), 3),
            'G': round(site_df['G_filt'].count() / len(site_df.index), 3),
            'SM': round(site_df['SM_surf'].count() / len(site_df.index), 2),
            'Ta': round(site_df['AirTempC'].count() / len(site_df.index), 2),
            'RH': round(site_df['RH'].count() / len(site_df.index), 3),
        }

        data_availability = {
            k: v for k, v in data_availability.items() if not np.isnan(v)
        }

        axs[4].text(0.0, 0.65, 'Closure Ratio')
        axs[4].text(0.0, 0.5, f'Annual Closure: {closure_ratio}')
        axs[4].text(0.0, 0.35, f'LEcorr50 mean: {le_corr50_mean}')
        axs[4].text(0.0, 0.2, f'LEcorr25 mean: {le_corr25_mean}')
        axs[4].text(0.0, 0.05, f'LEcorr75 mean: {le_corr75_mean}')
        axs[4].text(0.25, 0.65, 'Data Availability After Filtering')
        for i, (label, value) in enumerate(data_availability.items()):
            axs[4].text(0.25, 0.5 - i * 0.15, f'{label}: {value}')

    fig.tight_layout()
    fig.savefig(
        FIG_PATH + 'supplementary/AMF_qaqc/' + site + '_qaqc.png', dpi=250,
    )
    plt.close(fig)


def bias_eval(big_df_ss, var):
    """
    Generates a scatter plot to evaluate the bias of different models against
    the 'LEcorr50' reference, as a function of another variable.

    For each model, the function calculates the bias (model - reference) and
    plots it against a specified variable (e.g., 'NDVI-UQ'). This helps
    visualize how model bias might be correlated with other environmental or
    site characteristics.

    Args:
        big_df_ss (pd.DataFrame): A DataFrame containing model data,
                                    'LEcorr50' reference data, and the
                                    variable of interest.
        var (str): The name of the column in big_df_ss to use for the x-axis.
    """
    models = ['JET', 'STICinst', 'PTJPLSMinst', 'MOD16inst', 'BESSinst', 'ETinst']
    for model in models:
        bias = big_df_ss[model] - big_df_ss['LEcorr50']
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(10, 6))
        plt.scatter(big_df_ss[var], bias, c='blue', marker='o', alpha=0.5)
        plt.title('Bias vs ' + var)
        plt.xlabel(var)
        plt.ylabel(f'Bias ({model} - LEcorr50)')
        plt.grid(True)
        plt.savefig(
            FIG_PATH + 'supplementary/bias_eval/' + model + '_' + var + '.png',
            dpi=250,
        )