
# error functions for study
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import scipy.stats
from scipy import stats

def filter_nan(s,o):
    """
    Removes data from simulated and observed arrays wherever the observed data contains NaN.
    This is used by all other functions to avoid producing NaNs.
    """
    s = np.array(s)
    o = np.array(o)
    mask = ~np.isnan(o)
    return s[mask], o[mask]

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

# def R2_fun(s,o):
    """
    
#     o=np.array(o)
#     s=np.array(s)
    
#     if ((o == o[0]).all())|((s == s[0]).all()):
#         r2_o_d_=np.nan
#     else:
#         m_o_d = no_nans(np.array(o),np.array(s))
#         if len(np.array(o)[m_o_d])==0 | len(np.array(s)[m_o_d])==0:
#             r2_o_d_ = np.nan
#         else:
#             stats_o_d = stats.linregress(np.array(o)[m_o_d],np.array(s)[m_o_d])
#             # slope_o_d = stats_o_d[0]; 
#             # int_o_d = stats_o_d[1]; 
#             r2_o_d_ = stats_o_d[2]**2
    
#     return r2_o_d_
def R2_fun(s, o):
    """Calculates R^2 (coefficient of determination) or correlation coefficient squared."""
    o = np.array(o)
    s = np.array(s)
    if np.all(o == o[0]) or np.all(s == s[0]):
        return np.nan
    valid_mask = ~np.isnan(o) & ~np.isnan(s)
    if np.sum(valid_mask) == 0:
        return np.nan
    slope, intercept, r_value, p_value, std_err = stats.linregress(o[valid_mask], s[valid_mask])
    r2 = r_value ** 2
    return r2
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def KT_fun(s,o):
    """
    Calculates Kendall's Tau correlation coefficient and p-value.
    """
    s, o = filter_nan(s, o)
    tau, pvalue = scipy.stats.stats.kendalltau(s, o)
    return tau, pvalue

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def lin_regress(Y, X):
    x, y = filter_nan(X,Y)
    A = np.vstack([x,np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A,y,rcond=-1)[0]
    return slope, intercept

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def BIAS_fun(s, o):
    """
    Returns the mean bias of the simulated data in relation to the observations.

    Parameters:
        s (array-like): simulated values
        o (array-like): observed values

    Returns:
        float: bias
    """
    s, o = filter_nan(s, o)
    dif = s - o
    bias = np.mean(dif)
    return bias
def rmse(s, o):
    """
    Calculates root mean squared error between simulated and observed values.

    Parameters:
        s (array-like): simulated values
        o (array-like): observed values

    Returns:
        float: RMSE
    """
    s, o = filter_nan(s, o)
    return np.sqrt(np.mean((s - o) ** 2))
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def ABS_BIAS_fun(s, o):
    """
    Returns the mean absolute difference (bias) between the simulated data in relation to the observation.

    Parameters:
        s (array-like): simulated values
        o (array-like): observed values

    Returns:
        float: the mean of the absolute difference between simulations and observation
    """
    s,o = filter_nan(s,o)
    dif = np.absolute(s-o)
    abs_bias = np.mean(dif)
    return abs_bias

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def get_summary_stats(s,o):
    """
    Returns summary statistics for model evaluation.

    Returns:
        list: [mbe, mae, rmse, r2, kt, slope, intercept]
    """
    s, o = filter_nan(s, o)
    mbe = np.mean(s) - np.mean(o)
    mae = metrics.mean_absolute_error(o, s)
    mse = metrics.mean_squared_error(o, s)
    rmse = np.sqrt(mse)
    r2 = R2_fun(s, o)
    kt, _ = KT_fun(s, o)
    slope, intercept = lin_regress(s, o)
    return [mbe, mae, rmse, r2, kt, slope, intercept]

def intersection(lst1, lst2):
  lst3 = [value for value in lst1 if value in lst2]
  return lst3

def create_sum_stats(in_df,LE_var='LEcorr50'):
    """
    creates a table of statistics for models and ancillary variables
    params:
        in_df = dataframe including models and ground observations
    import pandas as pd
    stats_df = pd.DataFrame(columns=['VAR','RMSE','MAB','BIAS','R2','Slope','Int'])
    # SM surf
    model = 'SM'
    obsname = 'SM_surf'
    m_rmse = rmse(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_slope, m_int = lin_regress(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model + 'surf', m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # SM rz
    model = 'SM'
    obsname = 'SM_rz'
    m_rmse = rmse(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_slope, m_int = lin_regress(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model + 'rz', m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # BESS
    model = 'BESS'
    m_rmse = rmse(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_bias = BIAS_fun(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_r2 = R2_fun(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_slope, m_int = lin_regress(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model, m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # MOD16
    model = 'MOD16'
    m_rmse = rmse(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_bias = BIAS_fun(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_r2 = R2_fun(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    m_slope, m_int = lin_regress(in_df[model + 'inst'].to_numpy(), in_df[LE_var].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model, m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # Rn
    model = 'Rn'
    obsname = 'NETRAD_filt'
    m_rmse = rmse(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_slope, m_int = lin_regress(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model, m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # Rg
    model = 'Rg'
    obsname = 'SW_IN'
    m_rmse = rmse(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_slope, m_int = lin_regress(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model, m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # Ta
    model = 'Ta'
    obsname = 'AirTempC'
    m_rmse = rmse(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_slope, m_int = lin_regress(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model, m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    # RH
    model = 'RH'
    obsname = 'RH_percentage'
    m_rmse = rmse(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    m_slope, m_int = lin_regress(in_df[model].to_numpy(), in_df[obsname].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model, m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

    return stats_df
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model+'rz',m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  return stats_df

def create_sum_stats_daily(in_df,LE_var='ETcorr50daily'):
    """Creates a table of statistics for models and ancillary variables."""
    stats_df = pd.DataFrame(columns=['VAR','RMSE','MAB','BIAS','R2','Slope','Int'])
    model = 'ETdaily_L3T_JET'
    jet_rmse = rmse(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    jet_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    jet_bias = BIAS_fun(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    jet_r2 = R2_fun(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    jet_slope,jet_int = lin_regress(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model,jet_rmse, jet_mab, jet_bias, jet_r2, jet_slope, jet_int]
    model = 'ETdaily_L3T_ET_ALEXI'
    m_rmse = rmse(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    m_r2 = R2_fun(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[LE_var].to_numpy())
    stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]
    return stats_df

def find_ideal(big_df_ss):
    big_df_ss['LE_residual'] = big_df_ss['NETRAD_filt'] - big_df_ss['H_filt'] - big_df_ss['G_filt']
    big_df_ss['LE_ideal'] = big_df_ss.apply(
        lambda row: min([
            row[['LEcorr25', 'LEcorr50', 'LEcorr75', 'LE_filt', 'LEcorr_ann', 'LE_residual']].min(),
            row[['LEcorr25', 'LEcorr50', 'LEcorr75', 'LE_filt', 'LEcorr_ann', 'LE_residual']].max(),
            (row[['LEcorr25', 'LEcorr50', 'LEcorr75', 'LE_filt', 'LEcorr_ann', 'LE_residual']].min() +
             row[['LEcorr25', 'LEcorr50', 'LEcorr75', 'LE_filt', 'LEcorr_ann', 'LE_residual']].max()) / 2
        ], key=lambda x: abs(x - row['JET'])),
        axis=1
    )
    return big_df_ss