"""
This module contains various error and statistical functions for model
evaluation, particularly for comparing simulated data with observations.

The functions cover a range of metrics, including R-squared, Kendall's Tau,
linear regression, bias, RMSE, and mean absolute bias. It also includes
utility functions for data filtering and generating summary statistics tables.
"""
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import scipy.stats
from scipy import stats


def filter_nan(s, o):
    """
    Removes data from simulated and observed arrays wherever the observed data
    contains NaN.

    This ensures that all functions operate on valid, comparable data points.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        tuple: A tuple containing the filtered simulated and observed arrays.
    """
    s = np.array(s)
    o = np.array(o)
    mask = ~np.isnan(o)
    return s[mask], o[mask]


def R2_fun(s, o):
    """
    Calculates R^2 (coefficient of determination) or the correlation
    coefficient squared.

    It handles cases where the data is constant or contains no valid points
    after filtering NaNs.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        float: The R^2 value, or NaN if calculation is not possible.
    """
    o = np.array(o)
    s = np.array(s)
    if np.all(o == o[0]) or np.all(s == s[0]):
        return np.nan
    valid_mask = ~np.isnan(o) & ~np.isnan(s)
    if np.sum(valid_mask) == 0:
        return np.nan
    _, _, r_value, _, _ = stats.linregress(o[valid_mask], s[valid_mask])
    r2 = r_value**2
    return r2


def KT_fun(s, o):
    """
    Calculates Kendall's Tau correlation coefficient and p-value.

    This non-parametric test measures the strength of dependence between two
    variables.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        tuple: A tuple containing the Kendall's Tau correlation coefficient
               and the p-value.
    """
    s, o = filter_nan(s, o)
    tau, pvalue = scipy.stats.kendalltau(s, o)
    return tau, pvalue


def lin_regress(Y, X):
    """
    Performs a linear regression of Y on X and returns the slope and intercept.

    This function uses numpy's least-squares method.

    Args:
        Y (array-like): The dependent variable.
        X (array-like): The independent variable.

    Returns:
        tuple: A tuple containing the slope and intercept of the regression line.
    """
    x, y = filter_nan(X, Y)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def BIAS_fun(s, o):
    """
    Returns the mean bias of the simulated data in relation to the observations.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        float: The mean bias.
    """
    s, o = filter_nan(s, o)
    dif = s - o
    bias = np.mean(dif)
    return bias


def rmse(s, o):
    """
    Calculates root mean squared error between simulated and observed values.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        float: The RMSE value.
    """
    s, o = filter_nan(s, o)
    return np.sqrt(np.mean((s - o) ** 2))


def ABS_BIAS_fun(s, o):
    """
    Returns the mean absolute difference (bias) between the simulated and
    observed data.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        float: The mean of the absolute difference.
    """
    s, o = filter_nan(s, o)
    dif = np.absolute(s - o)
    abs_bias = np.mean(dif)
    return abs_bias


def get_summary_stats(s, o):
    """
    Returns a list of summary statistics for model evaluation.

    Args:
        s (array-like): Simulated values.
        o (array-like): Observed values.

    Returns:
        list: A list containing [mbe, mae, rmse, r2, kt, slope, intercept].
    """
    s, o = filter_nan(s, o)
    mbe = np.mean(s) - np.mean(o)
    mae = metrics.mean_absolute_error(o, s)
    mse = metrics.mean_squared_error(o, s)
    _rmse = np.sqrt(mse)
    r2 = R2_fun(s, o)
    kt, _ = KT_fun(s, o)
    slope, intercept = lin_regress(s, o)
    return [mbe, mae, _rmse, r2, kt, slope, intercept]


def intersection(lst1, lst2):
    """
    Finds the common elements between two lists.

    Args:
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
        list: A new list containing elements that are in both input lists.
    """
    return [value for value in lst1 if value in lst2]


def create_sum_stats(in_df, LE_var='LEcorr50'):
    """
    Creates a table of statistics for models and ancillary variables.

    This function calculates and populates a pandas DataFrame with key metrics
    like RMSE, MAB, BIAS, R2, Slope, and Intercept for various model outputs
    against observed data.

    Args:
        in_df (pd.DataFrame): A DataFrame containing model and ground
                                observations.
        LE_var (str, optional): The name of the column in in_df to use as the
                                reference LE variable. Defaults to 'LEcorr50'.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    stats_df = pd.DataFrame(
        columns=['VAR', 'RMSE', 'MAB', 'BIAS', 'R2', 'Slope', 'Int'],
    )
    models = ['SM', 'BESS', 'MOD16', 'Rn', 'Rg', 'Ta', 'RH']

    for model in models:
        # Define model-specific columns and observation names
        obs_name_map = {
            'SM': 'SM_surf',
            'Rn': 'NETRAD_filt',
            'Rg': 'SW_IN',
            'Ta': 'AirTempC',
            'RH': 'RH_percentage',
        }
        obs_name = obs_name_map.get(model, LE_var)
        model_col = model + 'inst' if model in ['BESS', 'MOD16'] else model

        # Handle special cases for SM_surf and SM_rz
        if model == 'SM':
            obs_names = ['SM_surf', 'SM_rz']
            for obs_n in obs_names:
                m_rmse = rmse(in_df[model].to_numpy(), in_df[obs_n].to_numpy())
                m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obs_n].to_numpy())
                m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obs_n].to_numpy())
                m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obs_n].to_numpy())
                m_slope, m_int = lin_regress(
                    in_df[model].to_numpy(),
                    in_df[obs_n].to_numpy(),
                )
                stats_df.loc[len(stats_df.index)] = [
                    model + obs_n.split('_')[-1],
                    m_rmse,
                    m_mab,
                    m_bias,
                    m_r2,
                    m_slope,
                    m_int,
                ]
            continue

        # Calculate metrics for other models
        m_rmse = rmse(in_df[model_col].to_numpy(), in_df[obs_name].to_numpy())
        m_mab = ABS_BIAS_fun(in_df[model_col].to_numpy(), in_df[obs_name].to_numpy())
        m_bias = BIAS_fun(in_df[model_col].to_numpy(), in_df[obs_name].to_numpy())
        m_r2 = R2_fun(in_df[model_col].to_numpy(), in_df[obs_name].to_numpy())
        m_slope, m_int = lin_regress(
            in_df[model_col].to_numpy(),
            in_df[obs_name].to_numpy(),
        )
        stats_df.loc[len(stats_df.index)] = [
            model,
            m_rmse,
            m_mab,
            m_bias,
            m_r2,
            m_slope,
            m_int,
        ]

    return stats_df


def create_sum_stats_daily(in_df, LE_var='ETcorr50daily'):
    """
    Creates a table of statistics for daily models.

    Args:
        in_df (pd.DataFrame): DataFrame with model and daily observation data.
        LE_var (str, optional): The reference variable for daily latent heat
                                flux. Defaults to 'ETcorr50daily'.

    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics.
    """
    stats_df = pd.DataFrame(
        columns=['VAR', 'RMSE', 'MAB', 'BIAS', 'R2', 'Slope', 'Int'],
    )
    models = ['ETdaily_L3T_JET', 'ETdaily_L3T_ET_ALEXI']

    for model in models:
        m_rmse = rmse(in_df[model].to_numpy(), in_df[LE_var].to_numpy())
        m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[LE_var].to_numpy())
        m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[LE_var].to_numpy())
        m_r2 = R2_fun(in_df[model].to_numpy(), in_df[LE_var].to_numpy())
        m_slope, m_int = lin_regress(
            in_df[model].to_numpy(),
            in_df[LE_var].to_numpy(),
        )
        stats_df.loc[len(stats_df.index)] = [
            model,
            m_rmse,
            m_mab,
            m_bias,
            m_r2,
            m_slope,
            m_int,
        ]

    return stats_df


def find_ideal(big_df_ss):
    """
    Calculates an 'ideal' latent heat (LE) value for each observation.

    The 'ideal' LE is defined as the value from a set of LE estimates
    (including different correction methods and an energy balance residual)
    that is closest to the 'JET' model's output for that observation. This
    can be used to evaluate the consistency of the 'JET' model.

    Args:
        big_df_ss (pd.DataFrame): DataFrame containing various LE flux
                                    estimates.

    Returns:
        pd.DataFrame: The input DataFrame with a new 'LE_ideal' column.
    """
    big_df_ss['LE_residual'] = (
        big_df_ss['NETRAD_filt'] - big_df_ss['H_filt'] - big_df_ss['G_filt']
    )
    big_df_ss['LE_ideal'] = big_df_ss.apply(
        lambda row: min(
            [
                row[
                    [
                        'LEcorr25',
                        'LEcorr50',
                        'LEcorr75',
                        'LE_filt',
                        'LEcorr_ann',
                        'LE_residual',
                    ]
                ].min(),
                row[
                    [
                        'LEcorr25',
                        'LEcorr50',
                        'LEcorr75',
                        'LE_filt',
                        'LEcorr_ann',
                        'LE_residual',
                    ]
                ].max(),
                (
                    row[
                        [
                            'LEcorr25',
                            'LEcorr50',
                            'LEcorr75',
                            'LE_filt',
                            'LEcorr_ann',
                            'LE_residual',
                        ]
                    ].min()
                    + row[
                        [
                            'LEcorr25',
                            'LEcorr50',
                            'LEcorr75',
                            'LE_filt',
                            'LEcorr_ann',
                            'LE_residual',
                        ]
                    ].max()
                )
                / 2,
            ],
            key=lambda x: abs(x - row['JET']),
        ),
        axis=1,
    )
    return big_df_ss