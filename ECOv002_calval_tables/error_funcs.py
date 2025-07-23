# error functions for study
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def filter_nan(s,o):
        """
        this functions removed the data  from simulated and observed data
        whereever the observed data contains nan

        this is used by all other functions, otherwise they will produce nan as
        output
        """
        import numpy as np
        data = np.array([s,o])
        data = np.transpose(data)
        data = data[~np.isnan(data).any(1)]
        return data[:,0],data[:,1]

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def no_nans(A1, A2):
    ''' returns the mask of nans for 2 arrays'''
    import numpy as np
    mask = ~np.isnan(A1) & ~np.isnan(A2)
    return mask

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def rmse(s,o):
        """
        Root Mean Squared Error
        input:
                s: simulated
                o: observed
        output:
                rmses: root mean squared error
        """
        import numpy as np
        s,o = filter_nan(s,o)
        return np.sqrt(np.mean((s-o)**2))

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

# def R2_fun(s,o):
#     """
#     R^2 or Correlation coefficient^0.5
#     input:
#             s: simulated
#             o: observed
#     output:
#             R^2
#     """
#     import numpy as np
#     from scipy import stats
    
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
    """
    R^2 or Correlation coefficient^0.5
    input:
        s: simulated
        o: observed
    output:
        R^2
    """
    import numpy as np
    from scipy import stats
    # Convert inputs to numpy arrays
    o = np.array(o)
    s = np.array(s)
    
    # Check if all values are identical in either array
    if np.all(o == o[0]) or np.all(s == s[0]):
        return np.nan  # R^2 is not defined in this case
    
    # Filter out NaN values
    valid_mask = ~np.isnan(o) & ~np.isnan(s)
    
    if np.sum(valid_mask) == 0:  # Check if there are any valid data points
        return np.nan
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(o[valid_mask], s[valid_mask])
    
    # Calculate R^2
    r2 = r_value ** 2
    return r2
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def KT_fun(s,o):
    import scipy.stats
    """
    Kendalls Tao 
    input:
        s: simulated
        o: observed
    output:
        tau: Kendalls Tao
        p-value
    """
    s,o = filter_nan(s,o)
    tao = scipy.stats.stats.kendalltau(s, o)[0]
    pvalue = scipy.stats.stats.kendalltau(s, o)[1]
    return tao

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def lin_regress(Y, X):
    import numpy as np
    x, y = filter_nan(X,Y)
    A = np.vstack([x,np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A,y,rcond=-1)[0]
    
    return slope, intercept

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def BIAS_fun(s, o):
    '''
    returns the mean bias of the simulated data in relation ot the observations
    
    input:
        s: simulated
        o: observed
    output:
        bias
    '''
    import numpy as np
    s,o = filter_nan(s,o)

    dif = s-o
    bias = np.mean(dif)
    return bias
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

def ABS_BIAS_fun(s, o):
    '''
    returns the mean absolute difference (bias) between the simulated data in relation ot the observtion
    input:
        s: simulated
        o: observed
    output:
        abs_bias: the mean of the absolute difference between simulations and observation
    '''
    import numpy as np
    s,o = filter_nan(s,o)

    dif = np.absolute(s-o)
    abs_bias = np.mean(dif)
    return abs_bias

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
def get_summary_stats(s,o):
  '''
  returns summary statistics:
  * mbe: bias
  * mae: absolute bias
  * rmse: root mean square error
  * r2: correlation coefficient

  inputs:
  - s: simulated data
  - o: in situ observations
  '''
  import sklearn.metrics as metrics
  import numpy as np
  s,o = filter_nan(s,o)
  mbe = np.mean(s)-np.mean(o)
  mae = metrics.mean_absolute_error(o, s)
  mse = metrics.mean_squared_error(o, s)
  rmse = np.sqrt(mse) 
  r2 = R2_fun(s,o)
  kt = KT_fun(s,o)
  slope, intercept = lin_regress(s,o)
  return [mbe, mae, rmse, r2, kt, slope, intercept]

def intersection(lst1, lst2):
  lst3 = [value for value in lst1 if value in lst2]
  return lst3

def create_sum_stats(in_df,LE_var='LEcorr50'):
  '''
  creates a table of statistics for models and ancillary variables
  params:
  in_df = dataframe including models and ground observations

  returns:
  table of statistics
  columns: rmse, mab, bias, r2, slope, intercept
  rows: models or ancillary variables
  '''
  import pandas as pd
  stats_df = pd.DataFrame(columns=['VAR','RMSE','MAB','BIAS','R2','Slope','Int'])
  model = 'ET'
  m_rmse = rmse(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_bias = BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_r2 = R2_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_slope,m_int = lin_regress(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]
  model = 'JET'
  jet_rmse = rmse(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  jet_mab = ABS_BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  jet_bias = BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  jet_r2 = R2_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  jet_slope,jet_int = lin_regress(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,jet_rmse, jet_mab, jet_bias, jet_r2, jet_slope, jet_int]
  model = 'PTJPLSM'
  m_rmse = rmse(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_bias = BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_r2 = R2_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_slope,m_int = lin_regress(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]
  model = 'STIC'
  m_rmse = rmse(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_bias = BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_r2 = R2_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_slope,m_int = lin_regress(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]
  model = 'BESS'
  m_rmse = rmse(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_bias = BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_r2 = R2_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_slope,m_int = lin_regress(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]
  model = 'MOD16'
  m_rmse = rmse(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_bias = BIAS_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_r2 = R2_fun(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  m_slope,m_int = lin_regress(in_df[model+'inst'].to_numpy(),in_df[LE_var].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  model = 'Rn'
  obsname = 'NETRAD_filt'
  m_rmse = rmse(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  model = 'Rg'
  obsname = 'SW_IN'
  m_rmse = rmse(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  model = 'Ta'
  obsname = 'AirTempC'
  m_rmse = rmse(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  model = 'RH'
  obsname = 'RH_percentage'
  m_rmse = rmse(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model,m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  model = 'SM'
  obsname = 'SM_surf'
  m_rmse = rmse(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model+'surf',m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  model = 'SM'
  obsname = 'SM_rz'
  m_rmse = rmse(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_mab = ABS_BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_bias = BIAS_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_r2 = R2_fun(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  m_slope,m_int = lin_regress(in_df[model].to_numpy(),in_df[obsname].to_numpy())
  stats_df.loc[len(stats_df.index)] = [model+'rz',m_rmse, m_mab, m_bias, m_r2, m_slope, m_int]

  return stats_df

def create_sum_stats_daily(in_df,LE_var='ETcorr50daily'):
  '''
  creates a table of statistics for models and ancillary variables
  params:
  in_df = dataframe including models and ground observations

  returns:
  table of statistics
  columns: rmse, mab, bias, r2, slope, intercept
  rows: models or ancillary variables
  '''
  import pandas as pd
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
    big_df_ss['LE_residual']=big_df_ss['NETRAD_filt']-big_df_ss['H_filt']-big_df_ss['G_filt']

    big_df_ss['LE_ideal'] = big_df_ss.apply(
        lambda row: min([row[['LEcorr25', 'LEcorr50', 'LEcorr75','LE_filt','LEcorr_ann','LE_residual']].min(), 
                         row[['LEcorr25', 'LEcorr50', 'LEcorr75','LE_filt','LEcorr_ann','LE_residual']].max(), 
                         (row[['LEcorr25', 'LEcorr50', 'LEcorr75','LE_filt','LEcorr_ann','LE_residual']].min() + 
                          row[['LEcorr25', 'LEcorr50', 'LEcorr75','LE_filt','LEcorr_ann','LE_residual']].max()) / 2], 
                        key=lambda x: abs(x - row['JET'])), 
        axis=1)
    return big_df_ss