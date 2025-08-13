"""
This module contains functions for processing and quality-controlling
AmeriFlux eddy covariance data.

It includes utilities for filtering sites based on metadata, handling time
conversions (UTC, local, and solar time), and performing energy balance
closure corrections. The module also provides functions for reading and
cleaning raw AmeriFlux data, as well as converting latent heat flux to
evapotranspiration.
"""
import os
import sys
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from tables import NaturalNameWarning

# Ignore specific warnings to prevent clutter
warnings.filterwarnings(action='ignore', category=NaturalNameWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
pd.options.mode.chained_assignment = None

REL_PATH = os.getcwd() + '/'
DATA_PATH = REL_PATH + 'data/AMF_metadata/'


# --- SITE METADATA FILTERING ---
def limit_cols(sites):
    """
    Limits a DataFrame of AmeriFlux sites to useful information.

    Cleans up column names and sets a new index based on the site ID.

    Args:
        sites (pd.DataFrame): DataFrame of AmeriFlux sites with metadata.

    Returns:
        pd.DataFrame: A new DataFrame with a limited set of columns and a
                      cleaned index.
    """
    new_index = [s.replace('\xa0', '') for s in sites.index]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sites['new_index'] = new_index
    sites.set_index(sites.new_index, inplace=True)
    out_df = sites[
        ['Name', 'Lat', 'Long', 'Elev_(m)', 'Clim', 'Veg', 'MAT_(°C)', 'MAP_(mm)']
    ]
    return out_df


def filter_sites(filename=DATA_PATH + 'ameriflux_meta.csv'):
    """
    Filters AmeriFlux sites based on specific criteria for ECOSTRESS
    observations.

    The criteria are:
    - Latitude between 53.6N and 53.6S.
    - Open Access License CC-By-4.0.
    - End date is NaN or more recent than 2018.

    Args:
        filename (str): Path to the AmeriFlux metadata file.

    Returns:
        pd.DataFrame: A DataFrame of filtered sites.
    """
    table = pd.read_csv(filename)
    table.set_index(table.columns[0], inplace=True)

    lat_f_sites = table[(table[table.columns[2]] > -53.6) & (table[table.columns[2]] < 53.6)]
    out_cols = [c.split('\xa0')[0].replace(' ', '_') for c in lat_f_sites.columns]
    lat_f_sites.columns = out_cols

    license_filter = lat_f_sites['Data_Use_Policy1'] == 'CC-BY-4.0'
    lat_lic_sites = lat_f_sites[license_filter]
    out_df = limit_cols(lat_lic_sites)
    out_df.index.rename('Sites', inplace=True)

    return out_df


def get_dois(in_path=DATA_PATH):
    """
    Retrieves the DOI for each AmeriFlux site.

    Args:
        in_path (str): Path to the metadata directory.

    Returns:
        pd.DataFrame: A DataFrame with site IDs as the index and a 'doi' column.
    """
    citation_name = in_path + 'ameriflux_citations.csv'
    cite_meta = pd.read_csv(citation_name)
    cite_meta.set_index('site_id', inplace=True)
    return cite_meta[['doi']]


def create_table_1(save_to_csv=False, out_dir=''):
    """
    Generates a table of sites, merging filtered metadata with DOIs.

    Args:
        save_to_csv (bool): If True, saves the table to a CSV file.
        out_dir (str): The directory to save the CSV file.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    site_df = filter_sites()
    doi_df = get_dois()
    table1 = pd.merge(site_df, doi_df, left_index=True, right_index=True)

    if save_to_csv:
        table1.to_csv(out_dir + 'table1.csv')
    return table1


# --- TIME CONVERSION FUNCTIONS ---
def get_utc_hr_offset(site_meta_fname):
    """
    Returns the UTC offset in hours from a site's metadata file.

    Args:
        site_meta_fname (str): Path to the site metadata file.

    Returns:
        int: The UTC offset in hours.
    """
    site_meta = pd.read_excel(site_meta_fname)
    utc_offset_s = site_meta.DATAVALUE[site_meta['VARIABLE'] == 'UTC_OFFSET']
    utc_offset_it = iter(np.array(utc_offset_s))
    utc_offset_first = next(utc_offset_it)
    utc_offset = int(float(utc_offset_first))
    print(f'\tutc offset is:\t{utc_offset}')
    return utc_offset


def change_to_utc(times, utc_offset):
    """
    Converts a time series to UTC by subtracting the UTC offset.

    Args:
        times (pd.DatetimeIndex): Time series to convert.
        utc_offset (int): The UTC offset in hours.

    Returns:
        pd.DatetimeIndex: The converted time series in UTC.
    """
    return times - pd.DateOffset(hours=utc_offset)


def change_to_local(times, utc_offset):
    """
    Converts a time series to local time by adding the UTC offset.

    Args:
        times (pd.DatetimeIndex): Time series to convert.
        utc_offset (int): The UTC offset in hours.

    Returns:
        pd.DatetimeIndex: The converted time series in local time.
    """
    print('creating local time columns')
    return times + pd.DateOffset(hours=utc_offset)


def get_lon(site_meta_fname):
    """
    Returns the longitude of a site from its metadata file.

    Args:
        site_meta_fname (str): Path to the site metadata file.

    Returns:
        float: The longitude in degrees.
    """
    site_meta = pd.read_excel(site_meta_fname)
    long = site_meta.DATAVALUE[site_meta['VARIABLE'] == 'LOCATION_LONG']
    return np.array(long).astype(float)[0]


def longitude_to_offset(longitude_deg):
    """
    Converts longitude to a time offset.

    Args:
        longitude_deg (float): Longitude in degrees.

    Returns:
        timedelta: The time offset corresponding to the longitude.
    """
    return timedelta(hours=(np.radians(longitude_deg) / np.pi * 12))


def utc_to_solar(datetime_utc, longitude_deg):
    """
    Converts UTC datetime to solar apparent time.

    Args:
        datetime_utc (pd.DatetimeIndex): Time series in UTC.
        longitude_deg (float): Longitude in degrees.

    Returns:
        pd.DatetimeIndex: The converted time series in solar time.
    """
    return datetime_utc + longitude_to_offset(longitude_deg)


# --- VARIABLE CALCULATION FUNCTIONS ---
def calc_SWin(in_df):
    """
    Calculates the mean shortwave incoming radiation from available columns.
    """
    print('\treading SW_IN')
    final_list = [c for c in in_df.columns if c.startswith('SW_IN') and '_F' not in c]
    return in_df[final_list].mean(axis=1).values


def calc_H(in_df):
    """
    Calculates the mean sensible heat flux (H) from available columns.
    """
    print('\treading H')
    final_list = [c for c in in_df.columns if c.startswith('H')]
    final_list_filt = [c for c in final_list if 'H2O' not in c]
    final_list_filt2 = [c for c in final_list_filt if 'SSITC' not in c]
    final_list_filt3 = [c for c in final_list_filt2 if '_F' not in c]
    return in_df[final_list_filt3].mean(axis=1).values


def calc_G(in_df):
    """
    Calculates the mean ground heat flux (G) from available columns.
    """
    print('\treading G')
    final_list = [c for c in in_df.columns if c.startswith('G')]
    final_list_filt = [c for c in final_list if 'GPP' not in c]
    final_list_filt2 = [c for c in final_list_filt if '_F' not in c]
    return in_df[final_list_filt2].mean(axis=1).values


def calc_NETRAD(in_df):
    """
    Calculates the mean net radiation (NETRAD) from available columns.
    """
    print('\treading NETRAD')
    final_list = [c for c in in_df.columns if c.startswith('NETRAD')]
    final_list_filt = [c for c in final_list if '_F' not in c]
    return in_df[final_list_filt].mean(axis=1).values


def calc_LE(in_df):
    """
    Calculates the mean latent heat flux (LE) from available columns.
    """
    print('\treading LE')
    final_list = [c for c in in_df.columns if c.startswith('LE')]
    final_list_filt2 = [c for c in final_list if 'SSITC' not in c]
    final_list_filt3 = [c for c in final_list_filt2 if 'LEAF' not in c]
    final_list_filt4 = [c for c in final_list_filt3 if '_F' not in c]
    LE = in_df[final_list_filt4].mean(axis=1).values
    print(LE)
    return LE


def calc_SWC(in_df):
    """
    Calculates the mean surface soil water content (SWC) from available
    columns.
    """
    print('\treading SWC surface')
    final_list = []
    for i in np.arange(1, 9):
        try:
            final_list.append(
                list(in_df.columns[(in_df.columns.str.startswith(f'SWC_{i}_1'))])[0],
            )
        except IndexError:
            continue
    final_list_filt2 = [c for c in final_list if '_PI' not in c]
    return in_df[final_list_filt2].mean(axis=1).values


def calc_all_SWC(in_df):
    """
    Calculates the mean soil water content (SWC) for all observations.
    """
    print('\treading SWC all')
    final_list = in_df.columns[in_df.columns.str.startswith('SWC_')]
    final_list_filt2 = [c for c in final_list if '_PI' not in c]
    return in_df[final_list_filt2].mean(axis=1).values


def calc_RH(in_df):
    """
    Calculates the mean relative humidity (RH) from available columns.
    """
    print('\treading RH')
    final_list = [c for c in in_df.columns if c.startswith('RH')]
    final_list_filt2 = [c for c in final_list if '_PI' not in c]
    return in_df[final_list_filt2].mean(axis=1).values


def calc_AirTemp(in_df):
    """
    Calculates the mean air temperature from available columns.
    """
    print('\treading Air Temperature')
    final_list = [c for c in in_df.columns if c.startswith('TA')]
    final_list_filt2 = [c for c in final_list if 'TAU' not in c]
    final_list_filt3 = [c for c in final_list_filt2 if '_PI' not in c]
    return in_df[final_list_filt3].mean(axis=1).values


# --- QAQC AND ENERGY BALANCE CLOSURE ---
def remove_spikes(in_df, varnames=['LE'], z=6.5):
    """
    Removes spikes in data using the median of absolute deviation about the
    median, as described in Papale et al. (2006).

    Args:
        in_df (pd.DataFrame): DataFrame with AmeriFlux data.
        varnames (list): List of variable names to filter.
        z (float): The threshold for outlier detection. Larger numbers are
                   more conservative.

    Returns:
        pd.DataFrame: The DataFrame with spikes removed, creating new
                      filtered columns (e.g., 'LE_filt').
    """
    df_temp = in_df.copy()
    df_day = df_temp[
        (df_temp.NETRAD > 0)
        | (df_temp.NETRAD.isnull())
        & ((df_temp.index.hour >= 7) & (df_temp.index.hour < 17))
    ]
    df_night = df_temp[
        (df_temp.NETRAD <= 0)
        | (df_temp.NETRAD.isnull())
        & ((df_temp.index.hour < 7) | ((df_temp.index.hour >= 17)))
    ]

    for var in varnames:
        di_n = df_night[var].diff() - (df_night[var].diff(periods=-1) * -1.0)
        di_d = df_day[var].diff() - (df_day[var].diff(periods=-1) * -1.0)
        md_n = np.nanmedian(di_n)
        md_d = np.nanmedian(di_d)
        mad_n = np.nanmedian(np.abs(di_n - md_n))
        mad_d = np.nanmedian(np.abs(di_d - md_d))

        mask_nh = di_n < md_n - (z * mad_n / 0.6745)
        mask_nl = di_n > md_n + (z * mad_n / 0.6745)
        df_night.loc[mask_nh | mask_nl, var] = np.nan

        mask_dh = di_d < md_d - (z * mad_d / 0.6745)
        mask_dl = di_d > md_d + (z * mad_d / 0.6745)
        df_day.loc[mask_dh | mask_dl, var] = np.nan

    df_out = pd.concat([df_night, df_day], verify_integrity=True).sort_index()
    vnameout = var + '_filt'
    in_df[vnameout] = df_out[var]
    print(f'\t{var}_filt created')
    return in_df


def rolling_quantile_filter(in_df, _var_='LE'):
    """
    Applies a conservative rolling 15-day quantile filter to remove outliers
    that weren't caught by the spike removal algorithm.
    """
    df = in_df.copy()
    df['IQR'] = (
        df[_var_].rolling('15D', min_periods=int(48 * 5)).quantile(0.75)
        - df[_var_].rolling('15D', min_periods=int(48 * 5)).quantile(0.25)
    )
    df['max'] = (
        df['IQR'] * 2.5 + df[_var_].rolling('15D', min_periods=int(48 * 5)).quantile(0.75)
    )
    df['min'] = (
        df[_var_].rolling('15D', min_periods=int(48 * 5)).quantile(0.25) - df['IQR'] * 2.5
    )

    df.loc[df[_var_] > df['max'], _var_] = np.nan
    df.loc[df[_var_] < df['min'], _var_] = np.nan
    df.drop(['IQR', 'max', 'min'], inplace=True, axis=1)

    return df


def filter_based_on_threshs(
    in_df,
    LE_threshes=[-150, 1200],
    H_threshes=[-150, 1200],
    NETRAD_threshes=[-250, 1400],
    G_threshes=[-250, 500],
    filtered=True,
):
    """
    Removes data that falls outside of specified physical thresholds.

    Args:
        in_df (pd.DataFrame): DataFrame containing flux data.
        LE_threshes (list): Min and max thresholds for LE.
        H_threshes (list): Min and max thresholds for H.
        NETRAD_threshes (list): Min and max thresholds for NETRAD.
        G_threshes (list): Min and max thresholds for G.
        filtered (bool): If True, applies filters to '_filt' columns.

    Returns:
        pd.DataFrame: The DataFrame with values outside the thresholds set
                      to NaN.
    """
    _f_ = '_filt' if filtered else ''
    df_amf = in_df.copy()

    df_amf.loc[df_amf['LE' + _f_] < LE_threshes[0], 'LE' + _f_] = np.nan
    df_amf.loc[df_amf['LE' + _f_] > LE_threshes[1], 'LE' + _f_] = np.nan
    df_amf.loc[df_amf['NETRAD' + _f_] < NETRAD_threshes[0], 'NETRAD' + _f_] = np.nan
    df_amf.loc[df_amf['NETRAD' + _f_] > NETRAD_threshes[1], 'NETRAD' + _f_] = np.nan
    df_amf.loc[df_amf['G' + _f_] < G_threshes[0], 'G' + _f_] = np.nan
    df_amf.loc[df_amf['G' + _f_] > G_threshes[1], 'G' + _f_] = np.nan
    df_amf.loc[df_amf['H' + _f_] < LE_threshes[0], 'H' + _f_] = np.nan
    df_amf.loc[df_amf['H' + _f_] > LE_threshes[1], 'H' + _f_] = np.nan

    return df_amf


def force_close_fluxnet(in_df, filtered=False, verbose=True):
    """
    Performs Energy Balance Forced Closure according to Fluxnet methods.

    This function calculates a correction factor and applies it to LE and H.

    Args:
        in_df (pd.DataFrame): AmeriFlux data frame with energy balance variables.
        filtered (bool): If True, uses filtered columns.
        verbose (bool): If True, prints status messages.

    Returns:
        pd.DataFrame: DataFrame with adjusted LE and H variables.
    """
    _f_ = '_filt' if filtered else ''
    df = in_df.copy()

    vars_to_use = ['LE' + _f_, 'H' + _f_, 'NETRAD' + _f_, 'G' + _f_]
    df = df[vars_to_use].astype(float).copy()

    if int(df['G' + _f_].count()) == 0 or df['G' + _f_].count() / len(df.index) < 0.3:
        df['_RadFlux_'] = df['NETRAD' + _f_]
        df['no_G_flag'] = 1
        if verbose:
            print('\tno valid G data available')
    else:
        df['_RadFlux_'] = df['NETRAD' + _f_] - df['G' + _f_]
        df['no_G_flag'] = 0

    df['ebc_cf'] = df['_RadFlux_'] / (df['H' + _f_] + df['LE' + _f_])
    Q1 = df['ebc_cf'].quantile(0.25)
    Q3 = df['ebc_cf'].quantile(0.75)
    IQR = Q3 - Q1

    filtered_df = df.query('(@Q1 - 1.5 * @IQR) <= ebc_cf <= (@Q3 + 1.5 * @IQR)')
    removed_mask = set(df.index) - set(filtered_df.index)
    removed_mask = pd.to_datetime(list(removed_mask))
    df.ebc_cf.loc[removed_mask] = np.nan

    if verbose:
        print(f'\tmean correction factor is: {np.round(np.nanmean(df.ebc_cf.values), 2)}')
        print(f'\tclosure ratio mean is: {1 / df.ebc_cf.mean()}')
        print(
            f'\tpercent of valid closure crs is: {100 * df["ebc_cf"].count() / len(df.index)}',
        )

    df['ebc_cf_all'] = df.ebc_cf.median()
    df['ebc_cf_stable'] = df.ebc_cf.copy()

    min_period_thresh = 48
    night_or_day_mask = (df.index.hour > 20) | (df.index.hour <= 3) | ((df.index.hour > 10) & (df.index.hour <= 14))
    df.loc[~night_or_day_mask, 'ebc_cf_stable'] = np.nan
    df['ebc_cf_25'] = df.ebc_cf_stable.rolling('15D', min_periods=min_period_thresh, center=True).quantile(
        0.25,
        interpolation='nearest',
    )
    df['ebc_cf_50'] = df.ebc_cf_stable.rolling('15D', min_periods=min_period_thresh, center=True).quantile(
        0.5,
        interpolation='nearest',
    )
    df['ebc_cf_75'] = df.ebc_cf_stable.rolling('15D', min_periods=min_period_thresh, center=True).quantile(
        0.75,
        interpolation='nearest',
    )

    df['LEcorr25'] = df.ebc_cf_25 * df['LE' + _f_]
    df['LEcorr50'] = df.ebc_cf_50 * df['LE' + _f_]
    df['LEcorr75'] = df.ebc_cf_75 * df['LE' + _f_]
    df['LEcorr_ann'] = df.ebc_cf_all * df['LE' + _f_]

    le_lims = [-100, 800]
    for col in ['LEcorr_ann', 'LEcorr25', 'LEcorr50', 'LEcorr75']:
        df.loc[(df[col] >= le_lims[1]) | (df[col] <= le_lims[0]), col] = np.nan

    cf_lims = [0.5, 2]
    for col in ['ebc_cf_all', 'ebc_cf_25', 'ebc_cf_50', 'ebc_cf_75']:
        df.loc[(df[col] >= cf_lims[1]) | (df[col] <= cf_lims[0]), col] = np.nan

    df['Hcorr25'] = df.ebc_cf_25 * df['H' + _f_]
    df['Hcorr50'] = df.ebc_cf_50 * df['H' + _f_]
    df['Hcorr75'] = df.ebc_cf_75 * df['H' + _f_]
    df['Hcorr_ann'] = df.ebc_cf_all * df['H' + _f_]

    out_vars = [
        'LEcorr_ann',
        'LEcorr25',
        'LEcorr50',
        'LEcorr75',
        'ebc_cf',
        'Hcorr_ann',
        'Hcorr25',
        'Hcorr50',
        'Hcorr75',
    ]
    df_out = df[out_vars]

    cr = 1.0 / np.round(np.nanmean(df.ebc_cf_stable.values), 5)
    cf = np.round(np.nanmean(df.ebc_cf_stable.values), 5)
    if verbose:
        print(f'\n\tmean stable correction factor\n\t{cr} closure\n\t{cf} correction factor\n')
        print(f'\tclosure at site when filtered for stable conditions is:\t{cr}')

    return df_out


def force_close_br_daily(in_df, filtered=True):
    """
    Performs forced closure using a daily Bowen ratio approach.
    """
    _f_ = '_filt' if filtered else ''
    df = in_df.copy()
    vars_to_use = ['LE' + _f_, 'H' + _f_, 'NETRAD' + _f_, 'G' + _f_]
    df = df[vars_to_use].astype(float).copy()

    if int(df['G' + _f_].count()) == 0 or df['G' + _f_].count() / len(df.index) > 0.3:
        df['_RadFlux_'] = df['NETRAD' + _f_]
        df['no_G_flag'] = 1
    else:
        df['_RadFlux_'] = df['NETRAD' + _f_] - df['G' + _f_]
        df['no_G_flag'] = 0

    min_period_thresh = int(12 * 3)
    df['cf'] = df['_RadFlux_'] / (df['LE' + _f_] + df['H' + _f_])
    df['cf_1day'] = df.cf.rolling('3D', min_periods=min_period_thresh, center=True).median()

    df['LEcorr_br'] = df['cf_1day'] * df['LE' + _f_]
    df.loc[(df.LEcorr_br >= 1200) | (df.LEcorr_br <= -150), 'LEcorr_br'] = np.nan

    df['Hcorr_br'] = df['cf_1day'] * df['H' + _f_]
    df.loc[(df.Hcorr_br >= 1200) | (df.Hcorr_br <= -150), 'Hcorr_br'] = np.nan

    out_vars = ['LEcorr_br', 'Hcorr_br']
    return df[out_vars]


def read_amflx_data(filename, site_meta_fname, filtered=True, gapfill_interp=True, verbose=True):
    """
    Reads, cleans, and processes an AmeriFlux data file.

    This is a comprehensive function that handles multiple steps:
    1. Reads the file and sets the time index.
    2. Identifies and extracts key variables (e.g., LE, H, NETRAD).
    3. Applies QAQC filters like spike removal and quantile filtering.
    4. Performs energy balance closure corrections.
    5. Converts time to UTC and solar time.

    Args:
        filename (str): Path to the AmeriFlux data file.
        site_meta_fname (str): Path to the site metadata file.
        filtered (bool): If True, applies QAQC filtering.
        gapfill_interp (bool): If True, interpolates small data gaps.
        verbose (bool): If True, prints status messages.

    Returns:
        pd.DataFrame: The fully processed and cleaned DataFrame.
    """
    site = filename.split('/')[-1].split('_')[1]
    if verbose:
        print(f'starting to process & clean:\t{site}')

    df_amf = pd.read_csv(filename, skiprows=2, header=0)
    df_amf['local_time'] = pd.to_datetime(df_amf['TIMESTAMP_END'], format='%Y%m%d%H%M')
    df_amf.set_index(['local_time'], inplace=True)
    if verbose:
        print('\tfile read and time set to local')

    df_amf = df_amf[df_amf.index >= '2018-10-01']
    df_amf.replace(-9999, np.nan, inplace=True)

    g_exists = False
    try:
        df_amf['G'] = calc_G(df_amf)
        g_exists = True
    except (KeyError, IndexError):
        print('\tno ground heat flux\nassigning 0 to G for energy balance closure')
        df_amf['G'] = 0
        df_amf['G_filt'] = 0

    if len([c for c in df_amf.columns if c.startswith('NETRAD')]) >= 1:
        df_amf['NETRAD'] = calc_NETRAD(df_amf)
    elif site == 'US-MMS':
        df_amf['NETRAD'] = df_amf['SW_IN_1_1_1'] - df_amf['SW_OUT_1_1_1'] + df_amf['LW_IN_1_1_1'] - df_amf['LW_OUT_1_1_1']
    else:
        df_amf['NETRAD'] = np.nan

    if len([c for c in df_amf.columns if c.startswith('LE')]) >= 1:
        df_amf['LE'] = calc_LE(df_amf)
    else:
        df_amf['LE'] = np.nan

    if len([c for c in df_amf.columns if c.startswith('H')]) >= 1:
        df_amf['H'] = calc_H(df_amf)
    else:
        df_amf['H'] = np.nan

    if verbose:
        print('\tchecked for energy balance variables')

    if len([c for c in df_amf.columns if c.startswith('SWC')]) >= 1:
        df_amf['SM_surf'] = calc_SWC(df_amf)
        df_amf['SM_rz'] = calc_all_SWC(df_amf)
    else:
        df_amf['SM_surf'], df_amf['SM_rz'] = np.nan, np.nan

    if len([c for c in df_amf.columns if c.startswith('RH')]) >= 1:
        df_amf['RH'] = calc_RH(df_amf)
    else:
        df_amf['RH'] = np.nan

    if len([c for c in df_amf.columns if c.startswith('TA')]) >= 1:
        df_amf['AirTempC'] = calc_AirTemp(df_amf)
    else:
        df_amf['AirTempC'] = np.nan

    if len([c for c in df_amf.columns if c.startswith('SW_IN')]) >= 1:
        df_amf['SW_IN'] = calc_SWin(df_amf)
    else:
        df_amf['SW_IN'] = np.nan

    if verbose:
        print('\tchecked for ancillary variables')

    if filtered:
        for var in ['LE', 'H', 'NETRAD']:
            df_amf = remove_spikes(df_amf, varnames=[var])
            df_amf = rolling_quantile_filter(df_amf, f'{var}_filt')

        if g_exists:
            df_amf = rolling_quantile_filter(df_amf, 'G')
            df_amf = remove_spikes(df_amf, varnames=['G'])

    if gapfill_interp:
        for var in ['LE_filt', 'H_filt', 'G_filt', 'NETRAD_filt']:
            if var in df_amf.columns:
                df_amf[var].interpolate('linear', limit=8, inplace=True)

    df_amf = filter_based_on_threshs(df_amf, filtered=True)
    df_corr_flux = force_close_fluxnet(df_amf, filtered=True)
    df_corr_br_daily = force_close_br_daily(df_amf, filtered=True)
    df_out = pd.concat([df_amf, df_corr_flux, df_corr_br_daily], axis=1)

    df_out['LE_std'] = df_out.LE.rolling(4, min_periods=3).std()
    df_out['LE_2hr_med'] = df_out.LE.rolling(4, min_periods=3).median()
    df_out['LE_2hr_avg'] = df_out.LE.rolling(4, min_periods=3).mean()

    print('\tmeta data read to access utc offset')
    offset = get_utc_hr_offset(site_meta_fname)
    out_times = change_to_utc(df_out.index, offset)
    df_out['time_utc'] = out_times

    site_long = get_lon(site_meta_fname)
    df_out['solar_time'] = utc_to_solar(df_out.time_utc, site_long)
    df_out['solar_hour'] = df_out['solar_time'].dt.hour
    df_out.set_index(['time_utc'], inplace=True)
    df_out['local_time'] = pd.to_datetime(df_out['TIMESTAMP_END'], format='%Y%m%d%H%M')

    return df_out


# --- OTHER UTILITIES ---
def LE_2_ETmm(LE_Wm2, freq='day'):
    """
    Converts Latent Energy (LE) flux to Evapotranspiration (ET).

    Args:
        LE_Wm2 (np.ndarray): Latent energy flux in W/m^2.
        freq (str): The time frequency ('30 min' or 'day').

    Returns:
        np.ndarray: Evapotranspiration in mm.
    """
    lambda_e = 2.460 * 10**6
    roe_w = 1000
    m_2_mm = 1000

    if freq == '30 min':
        sec_conv = 60 * 30
    elif freq == 'day':
        sec_conv = 60 * 30 * 48
    else:
        raise ValueError("Invalid frequency. Choose '30 min' or 'day'.")

    mask = ~np.isnan(LE_Wm2)
    ET_mm = np.empty(LE_Wm2.shape)
    ET_mm[:] = np.nan
    ET_mm[mask] = LE_Wm2[mask] * (m_2_mm * sec_conv) / (lambda_e * roe_w)
    return ET_mm


def assign_time(in_df, time_col='time_UTC'):
    """
    Converts a specified time column to a datetime index.

    Args:
        in_df (pd.DataFrame): DataFrame with a time column.
        time_col (str): The name of the time column to use.

    Returns:
        pd.DataFrame: A new DataFrame with the time column set as the index.
    """
    df_test_var = in_df.copy()
    df_test_var['time'] = pd.to_datetime(df_test_var[time_col])
    df_test_var.set_index('time', inplace=True)
    df_test_var.drop(time_col, axis=1, inplace=True)
    return df_test_var.copy()