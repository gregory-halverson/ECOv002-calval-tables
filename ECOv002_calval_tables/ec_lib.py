import numpy as np
import os
import pandas as pd
from tables import NaturalNameWarning
import warnings
from datetime import timedelta

warnings.filterwarnings(action='ignore', category=NaturalNameWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered') 
pd.options.mode.chained_assignment = None

rel_path = os.getcwd()+'/'
data_path = rel_path+'data/AMF_metadata/'

#------------------------------------------------------------------------#
# filters based on 
# latitude between 53.6N to 53.6S
# Open Access License CC-By-4.0
# end date more recent than 2018
#------------------------------------------------------------------------#

def limit_cols(sites):
  '''
  limit the columns to useful information
  '''
  new_index=[]
  for s in sites.index:
      new_index.append(s.replace('\xa0',""))
  with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      sites['new_index']=new_index
  sites.set_index(sites.new_index,inplace=True)
  out_df = sites[['Name','Lat','Long','Elev_(m)','Clim','Veg','MAT_(°C)','MAP_(mm)']]
  return out_df

#------------------------------------------------------------------------#

def filter_sites(filename=data_path+'ameriflux_meta.csv'):
  ''' 
  inputs
      filename | path to the ameriflux meta filneam 
  returns 
      outlist |   a list of Ameriflux Sites for ECOSTRESS observations 
                  that are filtered by:
                  * latitude between 53.6N to 53.6S
                  * data use policy for CC-By-4.0
                  * end date 'nan' or more recent than 2018
  '''
  lat_filtered_list = []
  table = pd.read_csv(filename)
  table.set_index(table.columns[0],inplace=True)
  lat_f_sites = table[(table[table.columns[2]] > -53.6) & (table[table.columns[2]] < 53.6)]
  out_cols=[]
  for c in lat_f_sites.columns:
      out_cols.append(c.split('\xa0')[0].replace(" ", "_"))
  lat_f_sites.columns = out_cols

  # filter ameriflux_meta_df with non CC sites filtered out
  license_filter = lat_f_sites['Data_Use_Policy1']=='CC-BY-4.0'
  lat_lic_sites = lat_f_sites[license_filter]
  out_df = limit_cols(lat_lic_sites)
#   lat_lic_time_sites = lat_lic_sites[((np.isnan(lat_lic_sites.Site_End))|(lat_lic_sites.Site_End>=2018)) 
#                                       | (np.isnan(lat_lic_sites.BASE_End_)|(lat_lic_sites.BASE_End_>=2018))]
#   out_df = limit_cols(lat_lic_time_sites)
  out_df.index.rename('Sites',inplace=True)

  return out_df

#------------------------------------------------------------------------#

def get_dois(in_path = data_path):
  '''
  return doi to join with the list of sites
  '''
  citation_name = in_path+'ameriflux_citations.csv'
  cite_meta = pd.read_csv(citation_name)
  cite_meta.set_index('site_id',inplace=True)
  return cite_meta[['doi']]

#------------------------------------------------------------------------#
# generates manuscript table from:
#------------------------------------------------------------------------#

def create_table_1(save_to_csv = False, out_dir = ''):
    '''
    generates table 1 of ECOSTRESS Validation Phase II
    '''
    site_df = filter_sites()
    doi_df = get_dois()
    table1 = pd.merge(site_df,doi_df,left_index=True, right_index=True)
    
    if save_to_csv == True:
        table1.to_csv(out_dir + 'table1.csv')
    return table1

#------------------------------------------------------------------------#
# apply utc offsets to time, create new solar time column
#------------------------------------------------------------------------#

def get_utc_hr_offset(site_meta_fname): 
    '''
    return utc offset in hrs to convert time to solar aparent time
    '''
    import numpy as np
    site_meta = pd.read_excel(site_meta_fname)
    
    utc_offset_s = site_meta.DATAVALUE[site_meta['VARIABLE']=='UTC_OFFSET']
    utc_offset_it = iter(np.array(utc_offset_s))
    utc_offset_first = next(utc_offset_it)
    utc_offset = int(float(utc_offset_first))
    print('\tutc offset is:\t'+str(utc_offset))
    return utc_offset

#------------------------------------------------------------------------#

def change_to_utc(times, utc_offset):
    '''
    returns dataframe with time coordinate adjusted to UTC time zone
    Calculate offset by subtracting UTC from local time (most 
    sites in the Americas will have negative offsets and 
    most sites in Africa, Asia, Australia, and Europe will have postive offsets)
    '''
    out_times = times - pd.DateOffset(hours=utc_offset)
    return out_times

def change_to_local(times, utc_offset):
    '''
    returns dataframe with time coordinate adjusted to UTC time zone
    '''
    print('creating local time columns')
    out_times = times + pd.DateOffset(hours=utc_offset)
    return out_times

#------------------------------------------------------------------------#

def get_lon(site_meta_fname):
    '''
    return utc offset in hrs to convert time to solar aparent time
    '''
    site_meta = pd.read_excel(site_meta_fname)
    long = site_meta.DATAVALUE[site_meta['VARIABLE']=='LOCATION_LONG']
    return np.array(long).astype(float)[0]

#------------------------------------------------------------------------#

def longitude_to_offset(longitude_deg):
    from datetime import timedelta

    return timedelta(hours=(np.radians(longitude_deg) / np.pi * 12)) 

#------------------------------------------------------------------------#

def utc_to_solar(datetime_utc, longitude_deg):
    return datetime_utc + longitude_to_offset(longitude_deg)

#------------------------------------------------------------------------#
# combine redundant variables 
#------------------------------------------------------------------------#
def calc_SWin(in_df):
    '''
    returns mean H from observations at tower
    '''
    print('\treading SW_IN')

    final_list = list(in_df.columns[in_df.columns.str.startswith('SW_IN')])
    final_list_filt = [x for x in final_list if not '_F' in x]
    H =in_df[final_list_filt].mean(axis=1).values    

    return H

def calc_H(in_df):
    '''
    returns mean H from observations at tower
    '''
    print('\treading H')
    # PI_list = list(in_df.columns[in_df.columns.str.startswith('H_PI')])

    # if len(PI_list) >= 1:
    #   print('\tH PI list = True')
    #   final_list = list(in_df.columns[in_df.columns.str.startswith('H_PI')])
    #   final_list_filt = [x for x in final_list if not '_F' in x]
    #   H =in_df[final_list_filt].mean(axis=1).values

    # else:
    #   print('\tH PI list = False')
    final_list = list(in_df.columns[in_df.columns.str.startswith('H')])
    final_list_filt = [x for x in final_list if not 'H2O' in x]
    final_list_filt2 = [x for x in final_list_filt if not 'SSITC' in x]
    final_list_filt3 = [x for x in final_list_filt2 if not '_F' in x]
    H =in_df[final_list_filt3].mean(axis=1).values    

    return H

#------------------------------------------------------------------------#

def calc_G(in_df):
    '''
    returns mean G from observations at tower
    '''
    print('\treading G')
    # PI_list = list(in_df.columns[in_df.columns.str.startswith('G_PI')])    
    # if len(PI_list) >= 1:
    #   print('\tG PI list = True')
    #   final_list = list(in_df.columns[in_df.columns.str.startswith('G_PI')])
    #   final_list_filt = [x for x in final_list if not '_F' in x]
    #   G =in_df[final_list_filt].mean(axis=1).values
    # else:
    #   print('\tG PI list = False')
    final_list = list(in_df.columns[in_df.columns.str.startswith('G')])
    final_list_filt = [x for x in final_list if not 'GPP' in x]
    final_list_filt2 = [x for x in final_list if not '_F' in x]
    G =in_df[final_list_filt2].mean(axis=1).values      
    return G

#------------------------------------------------------------------------#

def calc_NETRAD(in_df):
    '''
    returns mean NETRAD from observations at tower
    '''
    print('\treading NETRAD')
    # PI_list = list(in_df.columns[in_df.columns.str.startswith('NETRAD_PI')])

    # if len(PI_list) >= 1:
    #   print('\tNETRAD PI list = True')
    #   final_list = list(in_df.columns[in_df.columns.str.startswith('NETRAD_PI')])
    #   final_list_filt = [x for x in final_list if not '_F' in x]
    #   NETRAD =in_df[final_list_filt].mean(axis=1).values
    # else:
    # print('\tNETRAD PI list = False')
    final_list = list(in_df.columns[in_df.columns.str.startswith('NETRAD')])
    final_list_filt = [x for x in final_list if not '_F' in x]
    NETRAD =in_df[final_list_filt].mean(axis=1).values    
    return NETRAD
    
#------------------------------------------------------------------------#

def calc_LE(in_df):
    '''
    returns mean NETRAD from observations at tower
    '''
    print('\treading LE')
    final_list = list(in_df.columns[in_df.columns.str.startswith('LE')])
    final_list_filt2 = [x for x in final_list if not 'SSITC' in x]
    final_list_filt3 = [x for x in final_list_filt2 if not 'LEAF' in x]
    final_list_filt4 = [x for x in final_list_filt3 if not '_F' in x]
    LE =in_df[final_list_filt4].mean(axis=1).values    
    print(LE)
      
    return LE
    
#------------------------------------------------------------------------#

def calc_SWC(in_df):
    '''
    returns mean surface SWC observations at tower
    this approach gathers all soil moisture observations from the surface
    for description of ameriflux data filtering
    https://ameriflux.lbl.gov/data/aboutdata/data-variables/
    '''
    print('\treading SWC surface')
    final_list = []
    for i in np.arange(1,9):
        try:
            final_list.append(list(in_df.columns[(in_df.columns.str.startswith('SWC_'+str(i)+'_1'))])[0])
        except:
            continue
    final_list_filt2 = [x for x in final_list if not '_PI' in x]

    SWC =in_df[final_list_filt2].mean(axis=1).values
    return SWC

#------------------------------------------------------------------------#

def calc_all_SWC(in_df):
    '''
    returns mean SWC for all observations at tower
    '''
    print('\treading SWC all')
    final_list = in_df.columns[in_df.columns.str.startswith('SWC_')]
    final_list_filt2 = [x for x in final_list if not '_PI' in x]
    SWC = in_df[final_list_filt2].mean(axis=1).values
    return SWC

#------------------------------------------------------------------------#

def calc_RH(in_df):
    '''
    returns mean RH for all observations at tower
    '''
    print('\treading RH')
    final_list = list(in_df.columns[in_df.columns.str.startswith('RH')])
    final_list_filt2 = [x for x in final_list if not '_PI' in x]
    RH =in_df[final_list_filt2].mean(axis=1).values   
    return RH

#------------------------------------------------------------------------#

def calc_AirTemp(in_df):
    '''
    returns mean RH for all observations at tower
    '''
    print('\treading Air Temperature')
    final_list = list(in_df.columns[in_df.columns.str.startswith('TA')])
    final_list_filt2 = [x for x in final_list if not 'TAU' in x]
    final_list_filt3 = [x for x in final_list_filt2 if not '_PI' in x]
    TA =in_df[final_list_filt3].mean(axis=1).values   
    return TA

#------------------------------------------------------------------------#
# assign solar time / adjust for UTC offset
#------------------------------------------------------------------------#

def get_utc_hr_offset(site_meta_fname): 
    '''
    return utc offset in hrs to convert time to solar aparent time
    '''
    import numpy as np
    site_meta = pd.read_excel(site_meta_fname)
    
    utc_offset_s = site_meta.DATAVALUE[site_meta['VARIABLE']=='UTC_OFFSET']
    utc_offset_it = iter(np.array(utc_offset_s))
    utc_offset_first = next(utc_offset_it)
    utc_offset = int(float(utc_offset_first))
    print('\tutc offset is:\t'+str(utc_offset))
    return utc_offset

def change_to_utc(times, utc_offset):
    '''
    returns dataframe with time coordinate adjusted to UTC time zone
    Calculate offset by subtracting UTC from local time (most 
    sites in the Americas will have negative offsets and 
    most sites in Africa, Asia, Australia, and Europe will have postive offsets)
    '''
    out_times = times - pd.DateOffset(hours=utc_offset)
    return out_times

def change_to_local(times, utc_offset):
    '''
    returns dataframe with time coordinate adjusted to UTC time zone
    '''
    print('creating local time columns')
    out_times = times + pd.DateOffset(hours=utc_offset)
    return out_times

def get_lon(site_meta_fname):
    '''
    return utc offset in hrs to convert time to solar aparent time
    '''
    site_meta = pd.read_excel(site_meta_fname)
    long = site_meta.DATAVALUE[site_meta['VARIABLE']=='LOCATION_LONG']
    return np.array(long).astype(float)[0]

def longitude_to_offset(longitude_deg):
    from datetime import timedelta

    return timedelta(hours=(np.radians(longitude_deg) / np.pi * 12)) 

def utc_to_solar(datetime_utc, longitude_deg):
    return datetime_utc + longitude_to_offset(longitude_deg)

#------------------------------------------------------------------------#
# QAQC 
#------------------------------------------------------------------------#
#------------------------------------------------------------------------#
# remove erroneous spikes in observations using median of absolute deviation about median
# remove outliers from low pass filter and stadard deviation
#------------------------------------------------------------------------#
def remove_spikes(in_df, varnames=['LE'], z = 6.5):
    '''
    This function removes spikes or anomalies in data for ameriflux data
    The outlider detection method followsthe median of absolute deviation about the median
    See Papale et al., 2006 | Towards a standardized processing of Net Ecosystem Exchange 
        measured with eddy covariance technique: algorithms and uncertainty estimation
        https://bg.copernicus.org/articles/3/571/2006/
    inputs:
    in_df: dataframe with both LE & NETRAD (e.g. standard Ameriflux Names)
    varnames: List of variables to filter, default parameter set to LE
    z: larger numbers are more conservative and default follows guidance in Papale et al., 2006
    '''
    df_temp=in_df.copy()
    df_day= df_temp[(df_temp.NETRAD > 0)|(df_temp.NETRAD.isnull()) & ((df_temp.index.hour>=7)&(df_temp.index.hour<17))]
    df_night= df_temp[(df_temp.NETRAD <= 0)|(df_temp.NETRAD.isnull()) & ((df_temp.index.hour < 7) | ((df_temp.index.hour >= 17)))]
    
    for var in varnames:
        di_n = df_night[var].diff()-(df_night[var].diff(periods=-1)*-1.0)
        di_d = df_day[var].diff()-(df_day[var].diff(periods=-1)*-1.0)
        md_n = np.nanmedian(di_n)
        md_d = np.nanmedian(di_d)
        mad_n = np.nanmedian(np.abs(di_n-md_n))
        mad_d = np.nanmedian(np.abs(di_d-md_d))

        # mask night data for high and low anomalies and filter for spikes
        mask_nh = di_n < md_n - (z*mad_n/0.6745)
        mask_nl = di_n > md_n + (z*mad_n/0.6745)
        df_night.loc[(mask_nh)|(mask_nl), var] = np.nan
        # df_night[var][mask_nh|mask_nl]=np.nan
        
        # mask daytime data for high and low anomalies and filter for spikes
        mask_dh = di_d < md_d - (z*mad_d/0.6745)
        mask_dl = di_d > md_d + (z*mad_d/0.6745)
        df_day.loc[(mask_dh)|(mask_dl), var] = np.nan
        # df_day[var][mask_dh|mask_dl]=np.nan

    df_out = pd.concat([df_night, df_day],verify_integrity=True).sort_index()
    vnameout = var+'_filt'
    in_df[vnameout]=df_out[var]
    print('\t'+var+'_filt created')
    return in_df     

def rolling_quantile_filter(in_df,_var_ = 'LE'):
    '''
    conservative rolling 15 day quantile filter to remove outlies not detected by the spike removal alogirthm
    2.5 x the inter quartile range is applied to Q1 and Q3 to detect anomalous outliers
    '''
    df=in_df.copy()
    df['IQR']=df[_var_].rolling('15D',min_periods=int(48*5)).quantile(0.75)-df[_var_].rolling('15D',min_periods=int(48*5)).quantile(0.25)
    df['max']=df['IQR']*2.5+df[_var_].rolling('15D',min_periods=int(48*5)).quantile(0.75)
    df['min']=df[_var_].rolling('15D',min_periods=int(48*5)).quantile(0.25)-df['IQR']*2.5

    df.loc[df[_var_]>df['max'], _var_] = np.nan
    df.loc[df[_var_]<df['min'], _var_] = np.nan
    df.drop(['IQR','max','min'],inplace=True,axis=1)

    return df

   
def filter_based_on_threshs(in_df, 
                            LE_threshes = [-150,1200], 
                            H_threshes = [-150,1200], 
                            NETRAD_threshes=[-250,1400], 
                            G_threshes=[-250,500],
                            filtered=True):
    if filtered ==True:
      _f_='_filt'
    else:
      _f_=''

    df_amf = in_df.copy()
    # LE thresholds are in line with the fluxdata_qaqc package
    df_amf.loc[df_amf['LE'+_f_]<LE_threshes[0],'LE'+_f_]=np.nan
    df_amf.loc[df_amf['LE'+_f_]>LE_threshes[1],'LE'+_f_]=np.nan

    # need to find source for filtering thresholds on NETRAD
    df_amf.loc[df_amf['NETRAD'+_f_]<NETRAD_threshes[0],'NETRAD'+_f_]=np.nan
    df_amf.loc[df_amf['NETRAD'+_f_]>NETRAD_threshes[1],'NETRAD'+_f_]=np.nan

    # G_threshes are in line with fluxdata_qaqc package
    # G_threshes are 50% of maximum NETRAD and -200 

    df_amf.loc[df_amf['G'+_f_]<G_threshes[0],'G'+_f_]=np.nan
    df_amf.loc[df_amf['G'+_f_]>G_threshes[1],'G'+_f_]=np.nan

    # H_threshes are in line with fluxdata_qaqc package
    df_amf.loc[df_amf['H'+_f_]<LE_threshes[0],'H'+_f_]=np.nan
    df_amf.loc[df_amf['H'+_f_] > LE_threshes[1], 'H'+_f_]=np.nan

    return df_amf

#------------------------------------------------------------------------#
# Energy Balance Closure 
#------------------------------------------------------------------------#
def force_close_fluxnet(in_df, filtered = False, verbose = True):
    """
    Energy Balance Forced Closure according to Fluxnet
    
    INPUT DATA: insitu_df # pandas dataframe with columns insitu_Rn, insitu_GHF, insitu_LE, & insitu_SHF
    OUTPUT DATA: closure_ratio # tower closure ratio or closure percentage
    Parameters:
    	in_df: Ameriflux data frame read with combined G & filtered (spike removed) for EBC purposes
    Returns:
    	out_df: Dataframe with adjusted LE vars according to fluxnet Method 1.
    """
    # from pandas.core.common import SettingWithCopyWarning
    # warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    if filtered == True:
        _f_ = '_filt'
    else:
        _f_ = ''
    #need to filter data first
    df = in_df.copy()

    # check if mean G, or heat storage measurements exist
    vars_to_use = ['LE'+_f_,'H'+_f_,'NETRAD'+_f_,'G'+_f_]
    df = df[vars_to_use].astype(float).copy() 
    
    # In cases where missing G obs, uses only Rnet
    if int(df['G'+_f_].count()) == 0:   
        df['_RadFlux_'] = df['NETRAD'+_f_]
        df['no_G_flag']=1
        print('\tno valid G data available')
    # In cases where missing more than 30% G obs, uses only Rnet
    elif df['G'+_f_].count() / len(df.index) < 0.3: 
        df['_RadFlux_'] = df['NETRAD'+_f_]
        df['no_G_flag']=1
        print('\tG data available, but less than 30% of record')
    else:
        df['_RadFlux_'] = df['NETRAD'+_f_] - df['G'+_f_]
        df['no_G_flag']=0

    df['ebc_cf'] = (df['_RadFlux_']) /(df['H'+_f_] + df['LE'+_f_])
    Q1 = df['ebc_cf'].quantile(0.25)
    Q3 = df['ebc_cf'].quantile(0.75)
    IQR = Q3 - Q1
    
    # filter values between Q1-1.5IQR and Q3+1.5IQR
    filtered = df.query('(@Q1 - 1.5 * @IQR) <= ebc_cf <= (@Q3 + 1.5 * @IQR)')
    # apply filter
    filtered_mask = filtered.index
    removed_mask = set(df.index) - set(filtered_mask)
    removed_mask = pd.to_datetime(list(removed_mask))
    df.ebc_cf.loc[removed_mask] = np.nan

    if verbose == True:
        print('\tmean correction factor is: '+str(np.round(np.nanmean(df.ebc_cf.values),2)))
        print('\tmedian correction factor is: '+str(np.round(np.nanmean(df.ebc_cf.values),2)))
    if verbose == True:
        print('\tclosure ratio mean is: '+str(1/df.ebc_cf.mean()))
        print('\tclosure ratio median is: '+str(1/df.ebc_cf.median()))
    if verbose == True:        
        print('\tpercent of valid closure crs is: '+str(100*df['ebc_cf'].count()/len(df.index)))
    
    # # This was commented to remove anomalous valuse. Should only impact annual...
    # # removing ebc values greater than (2)x and less than (1/2)x    
    # # change to daily value here 
    # df['ebc_cf'].where(df['ebc_cf']>0.5, np.nan, inplace=True)
    # df['ebc_cf'].where(df['ebc_cf']<2.0, np.nan, inplace=True)

    df['ebc_cf_all']=df.ebc_cf.median()
    
    # copy the ebc unfiltered data to index for night time only
    df['ebc_cf_stable']=df.ebc_cf.copy()

    # isolating times to hours 22 & < 2 and greater than 12 and less than 14
    min_period_thresh = int(48) 
    df.loc[~((df.index.hour > 20) | (df.index.hour <= 3) | ((df.index.hour >10) & (df.index.hour <=14))),'ebc_cf_stable']=np.nan
    df['ebc_cf_25'] = df.ebc_cf_stable.rolling('15D',min_periods=min_period_thresh, center = True).quantile(0.25,interpolation='nearest') 
    df['ebc_cf_50'] = df.ebc_cf_stable.rolling('15D',min_periods=min_period_thresh, center = True).quantile(0.5,interpolation='nearest')       
    df['ebc_cf_75'] = df.ebc_cf_stable.rolling('15D',min_periods=min_period_thresh, center = True).quantile(0.75,interpolation='nearest') 

    df['LEcorr25']= df.ebc_cf_25 * df['LE'+_f_]
    df['LEcorr50']= df.ebc_cf_50 * df['LE'+_f_]
    df['LEcorr75']= df.ebc_cf_75 * df['LE'+_f_]
    df['LEcorr_ann']  = df.ebc_cf_all * df['LE'+_f_]


    df.loc[(df.LEcorr_ann >= 800) | (df.LEcorr_ann <= -100), 'LEcorr_ann'] = np.nan
    df.loc[(df.LEcorr25 >= 800) | (df.LEcorr25 <= -100), 'LEcorr25'] = np.nan
    df.loc[(df.LEcorr50 >= 800) | (df.LEcorr50 <= -100), 'LEcorr50'] = np.nan
    df.loc[(df.LEcorr75 >= 800) | (df.LEcorr75 <= -100), 'LEcorr75'] = np.nan

    #Added based on Volk approach. 
    df.loc[(df.ebc_cf_all >= 2) | (df.ebc_cf_all <= 0.5), 'LEcorr_ann'] = np.nan
    df.loc[(df.ebc_cf_25 >= 2) | (df.ebc_cf_25 <= 0.5), 'LEcorr25'] = np.nan
    df.loc[(df.ebc_cf_50 >= 2) | (df.ebc_cf_50 <= 0.5), 'LEcorr50'] = np.nan
    df.loc[(df.ebc_cf_75 >= 2) | (df.ebc_cf_75 <= 0.5), 'LEcorr75'] = np.nan

    df['Hcorr25']= df.ebc_cf_25 * df['H'+_f_]
    df['Hcorr50']= df.ebc_cf_50 * df['H'+_f_]
    df['Hcorr75']= df.ebc_cf_75 * df['H'+_f_]
    df['Hcorr_ann']  = df.ebc_cf_all * df['H'+_f_]

    out_vars = ['LEcorr_ann','LEcorr25','LEcorr50','LEcorr75','ebc_cf','Hcorr_ann','Hcorr25','Hcorr50','Hcorr75']
    df_out = df[out_vars]
    
    # return the mean value for stable conditions of closure across the year
    cr = 1.0/np.round(np.nanmean(df.ebc_cf_stable.values),5)
    cf = np.round(np.nanmean(df.ebc_cf_stable.values),5);
    print('\n\tmean stable correction factor')
    print('\t'+str(cr)+' closure'+'\n\t'+ str(cf)+' correction factor\n')
    if verbose == True:
        print('\tclosure at site when filtered for stable conditions is:\t'+str(cr))
   
    return df_out

def force_close_br_daily(in_df, filtered=True):
    '''
    forced closure according to bowen ratio approach where daily values are applied
    '''
    if filtered == True:
        _f_ = '_filt'
    else:
        _f_ = ''
    # option to use filtered or unfiltered data
    df = in_df.copy()
    vars_to_use = ['LE'+_f_,'H'+_f_,'NETRAD'+_f_,'G'+_f_]
    df = df[vars_to_use].astype(float).copy()
    # In cases where missing G obs, uses only Rnet
    if int(df['G'+_f_].count()) == 0:   
        df['_RadFlux_'] = df['NETRAD'+_f_]
        df['no_G_flag']=1
        
    # In cases where missing more than 30% G obs, uses only Rnet
    elif df['G'+_f_].count() / len(df.index) > 0.3: 
        df['_RadFlux_'] = df['NETRAD'+_f_]
        df['no_G_flag']=1
        
    else:
        df['_RadFlux_'] = df['NETRAD'+_f_] - df['G'+_f_]
        df['no_G_flag']=0

    min_period_thresh = int(12*3) # 1/4 of data needed from 3 day window
    df['cf'] = df['_RadFlux_']/ (df['LE'+_f_]+ df['H'+_f_])
    df['cf_1day'] = df.cf.rolling('3D',min_periods=min_period_thresh, center = True).median()
    # take the median
    # df['cf_1day'][df['cf_1day']>2.0]=np.nan
    # df['cf_1day'][df['cf_1day']<0.50]=np.nan
    df['LEcorr_br'] = df['cf_1day']*df['LE'+_f_]
    df.loc[(df.LEcorr_br >= 1200) | (df.LEcorr_br <= -150), 'LEcorr_br'] = np.nan

    df['Hcorr_br'] = df['cf_1day']*df['H'+_f_]
    df.loc[(df.Hcorr_br >= 1200) | (df.Hcorr_br <= -150), 'Hcorr_br'] = np.nan

    out_vars = ['LEcorr_br','Hcorr_br']
    return df[out_vars]

#------------------------------------------------------------------------#
# Read AMFLX format and return values
#------------------------------------------------------------------------#
def read_amflx_data(filename, site_meta_fname, filtered = True, gapfill_interp=True, verbose=True):
    site=filename.split('/')[-1].split('_')[1]
    if verbose == True:
      print('starting to process & clean:\t'+site)
    df_amf = pd.read_csv(filename, skiprows=2, header = 0);
    df_amf['local_time'] = pd.to_datetime(df_amf['TIMESTAMP_END'], format='%Y%m%d%H%M');
    df_amf.set_index(['local_time'],inplace=True);
    if verbose == True:
      print('\tfile read and time set to local')
    df_amf= df_amf[df_amf.index >= '2018-10-01'] 
    df_amf[df_amf==-9999]=np.nan;
    outlist =[]
    filt_list = list(['LE','H','NETRAD','SW_IN'])
    try:
        outlist.extend(list(df_amf.columns[df_amf.columns.str.startswith('G')]))
        df_amf['G']=calc_G(df_amf)
        filt_list.extend('G')
        g_exists = True
    except:
        print('\tno ground heat flux\nassigning 0 to G for energy balance closure')
        df_amf['G']=0
        df_amf['G_filt']=0
        g_exists = False

    pass
    if len(list(df_amf.columns[df_amf.columns.str.startswith('NETRAD')]))==0:
        print('\tno NETRAD columns at '+site)
        df_amf['NETRAD']=np.nan

    if len(list(df_amf.columns[df_amf.columns.str.startswith('NETRAD')]))>=1:
        df_amf['NETRAD']=calc_NETRAD(df_amf)
    if site == 'US-MMS': # qc other site net radiation for sum of parts
      df_amf['NETRAD'] = df_amf['SW_IN_1_1_1']-df_amf['SW_OUT_1_1_1']+df_amf['LW_IN_1_1_1']-df_amf['LW_OUT_1_1_1']
    if len(list(df_amf.columns[df_amf.columns.str.startswith('LE')]))==0:
        df_amf['LE']=np.nan
    if len(list(df_amf.columns[df_amf.columns.str.startswith('LE')]))>=1:
        df_amf['LE']=calc_LE(df_amf)
    if len(list(df_amf.columns[df_amf.columns.str.startswith('H')]))==0:
        df_amf['H']=np.nan
    if len(list(df_amf.columns[df_amf.columns.str.startswith('H')]))>=1:
        df_amf['H']=calc_H(df_amf)
    if verbose == True:
      print('\tchecked for energy balance variables')
    if len(list(df_amf.columns[df_amf.columns.str.startswith('SWC')]))==0:
        df_amf['SM_surf']=np.nan
        df_amf['SM_rz']=np.nan
    if len(list(df_amf.columns[df_amf.columns.str.startswith('SWC')]))>=1:
        df_amf['SM_surf']=calc_SWC(df_amf)
        df_amf['SM_rz']=calc_all_SWC(df_amf)
    if len(list(df_amf.columns[df_amf.columns.str.startswith('RH')]))==0:
        df_amf['RH']=np.nan
    if len(list(df_amf.columns[df_amf.columns.str.startswith('RH')]))>=1:
        df_amf['RH']=calc_RH(df_amf)
    if len(list(df_amf.columns[df_amf.columns.str.startswith('TA')]))==0:
        df_amf['AirTempC']=np.nan
    if len(list(df_amf.columns[df_amf.columns.str.startswith('TA')]))>=1:
        df_amf['AirTempC']=calc_AirTemp(df_amf)
    if len(list(df_amf.columns[df_amf.columns.str.startswith('SW_IN')]))==0:
        df_amf['SW_IN']=np.nan
    if len(list(df_amf.columns[df_amf.columns.str.startswith('SW_IN')]))>=1:
        df_amf['SW_IN']=calc_SWin(df_amf)
    try:
      outlist.extend(list(df_amf.columns[df_amf.columns.str.startswith('SW_IN')]))
    except:
      print('\tno shortwave radiation data available')
    pass
    try:
      outlist.extend(list(df_amf.columns[df_amf.columns.str.startswith('SWC')])) 
    except:
      print('\tno soil moisture data available')
    pass
    try:
      outlist.extend(list(df_amf.columns[df_amf.columns.str.startswith('RH')])) 
    except:
      print('\tno relative humidity data available')
    pass
    
    if verbose == True:
      print('\tchecked for ancillary variables')
    
    # local time is used for filtering day/night in spike removal algorithm
    if filtered == True:
      # switched order on 4-12-23
      df_amf = remove_spikes(df_amf,varnames=['LE'])
      df_amf = remove_spikes(df_amf,varnames=['H'])
      df_amf = remove_spikes(df_amf,varnames=['NETRAD'])

      df_amf = rolling_quantile_filter(df_amf, 'LE_filt')
      df_amf = rolling_quantile_filter(df_amf, 'H_filt')
      df_amf = rolling_quantile_filter(df_amf, 'NETRAD_filt')

    if g_exists == True:
      df_amf = rolling_quantile_filter(df_amf, 'G')
      df_amf = remove_spikes(df_amf,varnames=['G'])
    else:
      df_amf = df_amf

    if verbose == True:
      print('\tremoved spikes from energy balance variables')
    if gapfill_interp ==True:
      df_amf.LE_filt.interpolate('linear',limit=8, inplace=True)
      df_amf.H_filt.interpolate('linear',limit=8, inplace=True)
      df_amf.G_filt.interpolate('linear',limit=8, inplace=True)
      df_amf.NETRAD_filt.interpolate('linear',limit=8, inplace=True)
    # add thresholds to remove values outside of observable ranges
    df_amf = filter_based_on_threshs(df_amf, filtered = True)

    # corr_flux is from ONEFLUX
    df_corr_flux= force_close_fluxnet(df_amf,filtered=True)
    # corr_br_daily provides daily closure estimate within reason
    df_corr_br_daily = force_close_br_daily(df_amf,filtered=True)
    # df_out = pd.concat([df_amf, df_corr_ann, df_corr_flux, df_corr_br_daily],axis=1)
    df_out = pd.concat([df_amf, df_corr_flux, df_corr_br_daily],axis=1)

    df_out['LE_std']=df_out.LE.rolling(4,min_periods=3).std()
    df_out['LE_2hr_med']=df_out.LE.rolling(4,min_periods=3).median()
    df_out['LE_2hr_avg']=df_out.LE.rolling(4,min_periods=3).mean()

    # change index to time utc in order to align with ECOSTRESS
    print('\tmeta data read to access utc offset')

    offset               = get_utc_hr_offset(site_meta_fname)
    out_times            =  change_to_utc(df_out.index, offset)
    df_out['time_utc']   = out_times
    # create solar time from longitude
    site_long            = get_lon(site_meta_fname)
    df_out['solar_time'] = utc_to_solar(df_out.time_utc, site_long)
    df_out['solar_hour'] = df_out['solar_time'].dt.hour
    df_out.set_index(['time_utc'],inplace=True);
    # create local time for sanity check on time conversions
    df_out['local_time'] = pd.to_datetime(df_out['TIMESTAMP_END'], format='%Y%m%d%H%M');
    return df_out

#--------------------------------------------------------------------------#
def LE_2_ETmm(LE_Wm2, freq='day'):
  '''
  This tool converts Latent Energy to Evapotranspiration
  INPUT DATA:  LE_2_ET (W/m2)
  OUTPUT DATA: ET_mm (mm/30min)
  '''
  lambda_e = 2.460*10**6       # J kg^-1
  roe_w = 1000                 # kg m^-3
  m_2_mm = 1000                # concert m to mm
  if freq =='30 min':
    s_2_30m = 60*30              # multiply by s in 30 min to get 30 min average mm
    sec_conv = s_2_30m
  if freq =='day':
    s_2_day = 60*30*48
    sec_conv = s_2_day
  mask = ~np.isnan(LE_Wm2)
  ET_mm = np.empty(LE_Wm2.shape)
  ET_mm[:] = np.NAN
  ET_mm[mask] = LE_Wm2[mask]*(m_2_mm*sec_conv)/(lambda_e*roe_w)
  return ET_mm

def assign_time(in_df,time_col='time_UTC'):
    df_test_var = in_df.copy()
    time_col = 'time_UTC'
    df_test_var['time']=pd.to_datetime(df_test_var[time_col])
    df_test_var.set_index('time',inplace=True)
    df_test_var.drop(time_col,axis=1,inplace=True)
    df_out = df_test_var.copy()
    return df_out