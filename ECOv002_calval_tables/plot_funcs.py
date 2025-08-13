### Plot towers based on meta_data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.dates import DateFormatter
import matplotlib.lines as mlines
import sys
from . import error_funcs

rel_path = os.getcwd()+'/'
fig_path = rel_path+'/results/figures/'
lib_path = rel_path+'src'
sys.path.insert(0,lib_path)

def quick_look_plots_met(big_df_ss, time):

    lines = {'linestyle': 'None'}
    plt.rc('lines', **lines)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Data extraction
    x00 = big_df_ss.NETRAD_filt.to_numpy()
    y00 = big_df_ss.Rn.to_numpy()
    rnet_rmse = error_funcs.rmse(y00, x00)
    rnet_r2 = error_funcs.R2_fun(y00, x00)
    rnet_slope, rnet_int = error_funcs.lin_regress(y00, x00)
    rnet_bias = error_funcs.BIAS_fun(y00,x00)

    x01 = big_df_ss.AirTempC.to_numpy()
    y01 = big_df_ss.Ta.to_numpy()
    ta_rmse = error_funcs.rmse(y01, x01)
    ta_r2 = error_funcs.R2_fun(y01, x01)
    ta_slope, ta_int = error_funcs.lin_regress(y01, x01)
    ta_bias = error_funcs.BIAS_fun(y01,x01)

    big_df_ss.loc[big_df_ss.SM_surf > 0.60, 'SM_surf'] = np.nan
    x10 = big_df_ss.SM_surf.to_numpy()
    y10 = big_df_ss.SM.to_numpy()
    sm_rmse = error_funcs.rmse(y10, x10)
    sm_r2 = error_funcs.R2_fun(y10, x10)
    sm_slope, sm_int = error_funcs.lin_regress(y10, x10)
    sm_bias = error_funcs.BIAS_fun(y10,x10)

    big_df_ss.loc[big_df_ss.SM_rz > 0.60, 'SM_rz'] = np.nan
    x11 = big_df_ss.SM_rz.to_numpy()
    y11 = big_df_ss.SM.to_numpy()
    smrz_rmse = error_funcs.rmse(y11, x11)
    smrz_r2 = error_funcs.R2_fun(y11, x11)
    smrz_slope, smrz_int = error_funcs.lin_regress(y11, x11)
    smrz_bias = error_funcs.BIAS_fun(y11,x11)

    x20 = big_df_ss.RH_percentage.to_numpy()
    y20 = big_df_ss.RH.to_numpy()
    rh_rmse = error_funcs.rmse(y20, x20)
    rh_r2 = error_funcs.R2_fun(y20, x20)
    rh_slope, rh_int = error_funcs.lin_regress(y20, x20)
    rh_bias = error_funcs.BIAS_fun(y20,x20)

    x21 = big_df_ss.SW_IN.to_numpy()
    y21 = big_df_ss.Rg.to_numpy()
    rg_rmse = error_funcs.rmse(y21, x21)
    rg_r2 = error_funcs.R2_fun(y21, x21)
    rg_slope, rg_int = error_funcs.lin_regress(y21, x21)
    rg_bias = error_funcs.BIAS_fun(y21,x21)

    one2one = np.arange(-250, 1200, 5)

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(3, 2, figsize=(9, 12))

    # Net Radiation Plot
    axs[0, 0].scatter(x00, y00, c='darkorange', marker='o', s=4)
    axs[0, 0].plot(one2one, one2one, '--', c='k')
    axs[0, 0].plot(one2one, one2one * rnet_slope + rnet_int, '--', c='gray')
    axs[0, 0].set_ylim([-200, 1000])
    axs[0, 0].set_xlim([-200, 1000])
    axs[0, 0].set_title('Net Radiation')
    axs[0, 0].set_ylabel('Model Rn Wm$^{-2}$')
    axs[0, 0].set_xlabel('Obs Rn Wm$^{-2}$')
    axs[0, 0].text(-150, 600, f'y = {round(rnet_slope, 2)}x + {round(rnet_int, 1)}\nRMSE: {round(rnet_rmse, 1)} Wm$^-$²\nbias: {round(rnet_bias, 1)} Wm$^-$²\nR²: {round(rnet_r2, 2)}')
    axs[0, 0].text(-0.2, 1.05, 'a)', transform=axs[0, 0].transAxes, fontsize=14, weight='bold')

    # Rg vs. SW_IN Plot
    axs[0, 1].scatter(x21, y21, c='orange', marker='o', s=4)
    axs[0, 1].plot(one2one, one2one, '--', c='k')
    axs[0, 1].plot(one2one, one2one * rg_slope + rg_int, '--', c='gray')
    axs[0, 1].set_ylim([-200, 1500])
    axs[0, 1].set_xlim([-200, 1500])
    axs[0, 1].set_title('Downwelling Shortwave Radiation')
    axs[0, 1].set_ylabel('Model R$_{SD}$ Wm$^{-2}$')
    axs[0, 1].set_xlabel('Obs R$_{SD}$ Wm$^{-2}$')
    axs[0, 1].text(-150, 950, f'y = {round(rg_slope, 2)}x + {round(rg_int, 1)}\nRMSE: {round(rg_rmse, 1)} Wm$^-$²\nbias: {round(rg_bias, 1)} Wm$^-$²\nR²: {round(rg_r2, 2)}')
    axs[0, 1].text(-0.2, 1.05, 'b)', transform=axs[0, 1].transAxes, fontsize=14, weight='bold')

    # Air Temperature Plot
    axs[1, 0].scatter(x01, y01, c='darkred', marker='o', s=4)
    axs[1, 0].plot(one2one, one2one, '--', c='k')
    axs[1, 0].plot(one2one, one2one * ta_slope + ta_int, '--', c='gray')
    axs[1, 0].set_ylim([-25, 40])
    axs[1, 0].set_xlim([-25, 40])
    axs[1, 0].set_title('Air Temp (C)')
    axs[1, 0].set_ylabel('Model Ta $^{o}$C')
    axs[1, 0].set_xlabel('Obs Ta $^{o}$C')
    axs[1, 0].text(10, -22, f'y = {round(ta_slope, 2)}x + {round(ta_int, 2)}\nRMSE: {round(ta_rmse, 2)} C\nbias: {round(ta_bias, 2)} C\nR²: {round(ta_r2, 2)}')
    axs[1, 0].text(-0.2, 1.05, 'c)', transform=axs[1, 0].transAxes, fontsize=14, weight='bold')

    # Relative Humidity Plot
    axs[1, 1].scatter(x20, y20, c='royalblue', marker='o', s=4)
    axs[1, 1].plot(one2one, one2one, '--', c='k')
    axs[1, 1].plot(one2one, one2one * rh_slope + rh_int, '--', c='gray')
    axs[1, 1].set_ylim([0, 1])
    axs[1, 1].set_xlim([0, 1])
    axs[1, 1].set_title('RH')
    axs[1, 1].set_ylabel('Model RH')
    axs[1, 1].set_xlabel('Obs RH')
    axs[1, 1].text(0.5, 0.05, f'y = {round(rh_slope, 2)}x + {round(rh_int, 2)}\nRMSE: {round(rh_rmse, 2)}\nbias: {round(rh_bias, 2)}\nR²: {round(rh_r2, 2)}')
    axs[1, 1].text(-0.2, 1.05, 'd)', transform=axs[1, 1].transAxes, fontsize=14, weight='bold')

    # Surface Soil Moisture Plot
    axs[2, 0].scatter(x10, y10, c='lightblue', marker='o', s=4)
    axs[2, 0].plot(one2one, one2one, '--', c='k')
    axs[2, 0].plot(one2one, one2one * sm_slope + sm_int, '--', c='gray')
    axs[2, 0].set_ylim([0, 0.8])
    axs[2, 0].set_xlim([0, 0.8])
    axs[2, 0].set_title('SM$_{surf}$')
    axs[2, 0].set_ylabel('Model VWC m$^{3}$m$^{-3}$')
    axs[2, 0].set_xlabel('Obs VWC m$^{3}$m$^{-3}$')
    axs[2, 0].text(0.1, 0.55, f'y = {round(sm_slope, 2)}x + {round(sm_int, 2)}\nRMSE: {round(sm_rmse, 2)}\nbias: {round(sm_bias, 2)}\nR²: {round(sm_r2, 2)}')
    axs[2, 0].text(-0.2, 1.05, 'e)', transform=axs[2, 0].transAxes, fontsize=14, weight='bold')

    # Root Zone Soil Moisture Plot
    axs[2, 1].scatter(x11, y11, c='darkblue', marker='o', s=4)
    axs[2, 1].plot(one2one, one2one, '--', c='k')
    axs[2, 1].plot(one2one, one2one * smrz_slope + smrz_int, '--', c='gray')
    axs[2, 1].set_ylim([0, 0.8])
    axs[2, 1].set_xlim([0, 0.8])
    axs[2, 1].set_title('SM$_{rz}$')
    axs[2, 1].set_ylabel('Model VWC m$^{3}$m$^{-3}$')
    axs[2, 1].set_xlabel('Obs VWC m$^{3}$m$^{-3}$')
    axs[2, 1].text(0.1, 0.55, f'y = {round(smrz_slope, 2)}x + {round(smrz_int, 2)}\nRMSE: {round(smrz_rmse, 2)}\nbias: {round(smrz_bias, 2)}\nR²: {round(smrz_r2, 2)}')
    axs[2, 1].text(-0.2, 1.05, 'f)', transform=axs[2, 1].transAxes, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.savefig(fig_path + 'auxiliary/auxiliary_eval_' + time + '.png', dpi=300)

def quick_look_plots(big_df_ss, time,LE_var='LEcorr50'):
    plt.rc('lines', linestyle='None')
    plt.style.use('seaborn-v0_8-whitegrid')

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

    metrics = {
        'ETinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.ETinst),
        'JET': calculate_metrics(big_df_ss[LE_var], big_df_ss.JET),
        'PTJPLSMinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.PTJPLSMinst),
        'BESSinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.BESSinst),
        'STICinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.STICinst),
        'MOD16inst': calculate_metrics(big_df_ss[LE_var], big_df_ss.MOD16inst)
    }

    model_names = {
        'ETinst': 'PT-JPL (C1)',
        'JET': 'JET',
        'PTJPLSMinst': 'PT-JPL$_{SM}$',
        'BESSinst': 'BESS',
        'STICinst': 'STIC',
        'MOD16inst': 'MOD16'
    }

    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    plt.rcParams.update({'font.size': 14})

    subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']  # Subplot labels


    for i, (key, (rmse, r2, slope, intercept,bias)) in enumerate(metrics.items()):
        x = big_df_ss[LE_var].to_numpy()
        y = big_df_ss[f'{key}'].to_numpy()
        err = big_df_ss['ETinstUncertainty'].to_numpy()
        xerr = big_df_ss[['LE_filt', 'LEcorr50', 'LEcorr_ann']].std(axis=1).to_numpy()

        number_of_points = np.sum(~np.isnan(y) & ~np.isnan(x))

        ax = axs[i // 2, i % 2]
        ax.errorbar(x, y, yerr=err, xerr=xerr, fmt='', ecolor='lightgray')
        ax.scatter(x, y, c=scatter_colors, marker='o', s=4, zorder=4)
        ax.plot(one2one, one2one, '--', c='k')
        ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
        #ax.set_title(f'{key}')
        ax.set_title(model_names[key])
        ax.set_xlim([-250, 1200])
        ax.set_ylim([-250, 1200])
        if i % 2 == 0:
            ax.set_ylabel('Model LE Wm$^{-2}$',fontsize=14)
        
        # Adding subplot label 'a', 'b', 'c', etc.
        ax.text(-0.1, 1.1, subplot_labels[i], transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='right')
        # Adding statistics directly to the plot
        #ax.text(500, 0, f'N = {number_of_points}',fontsize=12)
        ax.text(500, -200, f'y = {slope:.1f}x + {intercept:.1f} \nRMSE: {rmse:.1f} Wm$^-$² \nbias: {bias:.1f} Wm$^-$² \nR$^2$: {r2:.2f}', fontsize=12)
        # ax.text(500, -150, f'RMSE: {rmse:.2f} Wm$^{-2}$', fontsize=12)
        # ax.text(500, -250, f'R$^2$: {r2:.3f}', fontsize=12)
        # Add xlabel for the bottom row of subplots
        if i // 2 == 2:
            ax.set_xlabel('Flux Tower LE Wm$^{-2}$',fontsize=14)

    # Creating the color legend next to the second subplot (first row, second column)
    scatter_handles = [mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=6, label=veg) for veg, color in colors.items()]
    fig.legend(handles=scatter_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, title='Vegetation Type',fontsize=10)

    fig.tight_layout()
    fig.savefig(f'{fig_path}/le_fluxes/le_eval_{LE_var}_{time}.png', dpi=600,bbox_inches='tight')

def quick_look_plots_filt(big_df_ss, time,LE_var='LEcorr50'):
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

    metrics = {
        'ETinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.ETinst),
        'PTJPLSMinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.PTJPLSMinst),
        'BESSinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.BESSinst),
        'STICinst': calculate_metrics(big_df_ss[LE_var], big_df_ss.STICinst),
        'MOD16inst': calculate_metrics(big_df_ss[LE_var], big_df_ss.MOD16inst),
        'JET': calculate_metrics(big_df_ss[LE_var], big_df_ss.JET)
    }

    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    plt.rcParams.update({'font.size': 14})

    for i, (key, (rmse, r2, slope, intercept,bias)) in enumerate(metrics.items()):
        x = big_df_ss[LE_var].to_numpy()
        y = big_df_ss[f'{key}'].to_numpy()
        err = big_df_ss['ETinstUncertainty'].to_numpy()
        xerr = big_df_ss[['LE_filt', 'LEcorr50', 'LEcorr_ann']].std(axis=1).to_numpy()

        number_of_points = np.sum(~np.isnan(y) & ~np.isnan(x))

        ax = axs[i // 2, i % 2]
        ax.errorbar(x, y, yerr=err, xerr=xerr, fmt='', ecolor='lightgray')
        ax.scatter(x, y, c=scatter_colors, marker='o', s=4, zorder=4)
        ax.plot(one2one, one2one, '--', c='k')
        ax.plot(one2one, one2one * slope + intercept, '--', c='gray')
        ax.set_title(f'{key}')
        ax.set_xlim([-250, 1200])
        ax.set_ylim([-250, 1200])
        if i % 2 == 0:
            ax.set_ylabel('Model LE Wm$^{-2}$',fontsize=14)
        
        # Adding statistics directly to the plot
        ax.text(500, -200, f'y = {slope:.2f}x + {intercept:.2f}\nRMSE: {rmse:.2f} Wm$^{-2}$\nR$^2$: {r2:.3f}\nbias: {bias:.2f}Wm$^{-2}$', fontsize=12)
        # Add xlabel for the bottom row of subplots
        if i // 2 == 2:
            ax.set_xlabel('Flux Tower LE Wm$^{-2}$',fontsize=14)

    fig.tight_layout()
    fig.savefig(f'{fig_path}/le_fluxes/le_eval_filt_{time}.png', dpi=600)

def plot_colocated(ground_site_df, eco_site_i_df,site,utc_offset):
    eco_site_i_df['JET']= eco_site_i_df[['PTJPLSMinst','BESSinst','STICinst','MOD16inst']].median(axis=1)

    eco_site_i_df['solar_time'] = eco_site_i_df.index + pd.DateOffset(hours=utc_offset)
    ground_site_df['solar_time'] = ground_site_df.index + pd.DateOffset(hours=utc_offset)

    for idx in eco_site_i_df.index:
        solar_day = (idx + pd.DateOffset(hours=utc_offset)).normalize()  # Extract the day part (year, month, day) of the solar time
        df_ground_day = ground_site_df[ground_site_df['solar_time'].dt.normalize() == solar_day]

        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot the data using solar_time for the entire day
        ax.plot(df_ground_day['solar_time'], df_ground_day['NETRAD_filt'], label='NETRAD_filt')
        ax.plot(df_ground_day['solar_time'], df_ground_day['LE_filt'], label='LE_filt')
        ax.plot(df_ground_day['solar_time'], df_ground_day['LEcorr50'], label='LEcorr50')
        ax.plot(df_ground_day['solar_time'], df_ground_day['H_filt'], label='H_filt')
        ax.plot(df_ground_day['solar_time'], df_ground_day['G_filt'], label='G_filt')

        y_value = eco_site_i_df.loc[idx, 'JET']
        yerr_value = eco_site_i_df.loc[idx, 'ETinstUncertainty']
        ax.errorbar(eco_site_i_df.loc[idx,'solar_time'], y_value, yerr=yerr_value, fmt='ro', label=f"JET {idx.time()}")
        # Add vertical line at solar_hour
        #ax.axvline(x=eco_site_i_df.loc[idx,'solar_time'], color='red', linestyle='--', label='Observation time')

        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
        ax.set_ylabel('Wm$^{-2}$')
        ax.legend(fontsize='x-small')
        plt.title(site + ' ' + f"{idx}")

        # Save figure
        plt.savefig(fig_path+'supplementary/diurnal_observations/'+site+'_'+f"{idx}"+'.png',dpi=300)
        plt.close(fig)

def plot_blind_filter(ground_site_df, all_site_eco_data_df,site,utc_offset):
    all_site_eco_data_df['JET']= all_site_eco_data_df[['PTJPLSMinst','BESSinst','STICinst','MOD16inst']].median(axis=1)

    all_site_eco_data_df['solar_time'] = all_site_eco_data_df.index + pd.DateOffset(hours=utc_offset)
    ground_site_df['solar_time'] = ground_site_df.index + pd.DateOffset(hours=utc_offset)

    for idx in all_site_eco_data_df.index:
        solar_day = (idx + pd.DateOffset(hours=utc_offset)).normalize()  # Extract the day part (year, month, day) of the solar time
        df_ground_day = ground_site_df[ground_site_df['solar_time'].dt.normalize() == solar_day]

        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot the data using solar_time for the entire day
        ax.plot(df_ground_day['solar_time'], df_ground_day['NETRAD_filt'], label='NETRAD_filt')
        ax.plot(df_ground_day['solar_time'], df_ground_day['LE_filt'], label='LE_filt')
        ax.plot(df_ground_day['solar_time'], df_ground_day['LEcorr50'], label='LEcorr50')
        ax.plot(df_ground_day['solar_time'], df_ground_day['H_filt'], label='H_filt')
        ax.plot(df_ground_day['solar_time'], df_ground_day['G_filt'], label='G_filt')

        y_value = all_site_eco_data_df.loc[idx, 'JET']
        # yerr_value = eco_site_i_df.loc[idx, 'ETinstUncertainty'][0]
        #ax.errorbar(eco_site_i_df.loc[idx,'solar_time'], y_value, yerr=yerr_value, fmt='ro', label=f"JET {idx.time()}")
        # Add vertical line at solar_hour
        ax.axvline(x=all_site_eco_data_df.loc[idx,'solar_time'], color='red', linestyle='--', label='Observation time')

        ax.set_xlabel('Time')
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
        ax.set_ylabel('Wm$^{-2}$')
        ax.legend(fontsize='x-small')
        plt.title(site + ' ' + f"{idx}")

        # Save figure
        plt.savefig(fig_path+'supplementary/blind_filter/'+site+'_'+f"{idx}"+'.png',dpi=300)
        plt.close(fig)

def qaqc_plots(site):
    plt.style.use('seaborn-v0_8-whitegrid')

    data_path = rel_path+'data/cleaned_data/'
    file_name = data_path+site+'_qaqc_filt_ebc.csv'
    site_df = pd.read_csv(file_name)

    # Convert 'local_time' column to datetime and set as index
    site_df['index_time'] = pd.to_datetime(site_df['local_time'])
    site_df.set_index('index_time', inplace=True)

        # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 12))

    # Define a function to plot data if it exists
    def plot_data(ax, data, label, ylabel):
        if data is not None and not data.empty:  # Check if data is not None and not empty
            data.plot(label=label, ax=ax, x_compat=True)
            ax.legend()
            ax.set_ylabel(ylabel)
            # ax.set_xticks([])
        else:
            ax.set_xticks([])  # Hide x-axis ticks if no data

    # Plot NETRAD data
    plot_data(axs[0], site_df.get('NETRAD_filt'), 'NETRAD', 'NETRAD Wm-2')

    # Plot LE data
    if 'LE' in site_df.columns and 'LE_filt' in site_df.columns:
        plot_data(axs[1], site_df['LE'], 'LE', 'LE Wm-2')
        plot_data(axs[1], site_df['LE_filt'], 'LE_filt', 'LE_filt Wm-2')

    # Plot H data
    plot_data(axs[2], site_df.get('H_filt'), 'H', 'H Wm-2')

    # Plot G data
    plot_data(axs[3], site_df.get('G_filt'), 'G', 'G Wm-2')

    # Add statistics to the last subplot
    axs[4].set_title('Statistics')
    axs[4].axis('off')

    # Calculate statistics if the necessary columns exist
    if all(col in site_df.columns for col in ['LE_filt', 'LEcorr_ann', 'LEcorr50', 'LEcorr25', 'LEcorr75']):
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
            'RH': round(site_df['RH'].count() / len(site_df.index), 3)
        }

        for key in list(data_availability):
            if np.isnan(data_availability[key]):
                del data_availability[key]

        axs[4].text(0.0, 0.65, 'Closure Ratio')
        axs[4].text(0.0, 0.5, f'Annual Closure: {closure_ratio}')
        axs[4].text(0.0, 0.35, f'LEcorr50 mean: {le_corr50_mean}')
        axs[4].text(0.0, 0.2, f'LEcorr25 mean: {le_corr25_mean}')
        axs[4].text(0.0, 0.05, f'LEcorr75 mean: {le_corr75_mean}')
        axs[4].text(0.25, 0.65, 'Data Availability After Filtering')
        for i, (label, value) in enumerate(data_availability.items()):
            axs[4].text(0.25, 0.5 - i * 0.15, f'{label}: {value}')

    # Adjust layout and save the figure
    fig.tight_layout()
    fig.savefig(fig_path + 'supplementary/AMF_qaqc/' + site + '_qaqc.png', dpi=250)
    plt.close(fig)

def bias_eval(big_df_ss,var):

    models = ['JET','STICinst','PTJPLSMinst','MOD16inst','BESSinst','ETinst']
    for model in models:
        bias = big_df_ss[model]-big_df_ss['LEcorr50']

        plt.style.use('seaborn-v0_8-whitegrid')

        # Create a scatter plot of bias vs big_df_ss['NDVI-UQ']
        plt.figure(figsize=(10, 6))
        plt.scatter(big_df_ss[var], bias, c='blue', marker='o', alpha=0.5)
        plt.title('Bias vs '+var)
        plt.xlabel(var)
        plt.ylabel('Bias ('+model+' - LEcorr50)')
        plt.grid(True)

        plt.savefig(fig_path + 'supplementary/bias_eval/'+model+'_' + var + '.png', dpi=250)
