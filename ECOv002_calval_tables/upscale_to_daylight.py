import pandas as pd
from daylight_evapotranspiration import daylight_ET_from_instantaneous_LE

def upscale_to_daylight(df: pd.DataFrame, prefix: str = "insitu_") -> pd.DataFrame:
    daylight_results = daylight_ET_from_instantaneous_LE(
        LE_instantaneous_Wm2=df.LEcorr50,
        Rn_instantaneous_Wm2=df.NETRAD_filt,
        G_instantaneous_Wm2=df.G_filt,
        time_UTC=df.time_UTC,
        geometry=df.geometry
    )
    
    daylight_results_prefixed = {f"{prefix}{k}": v for k, v in daylight_results.items()}
    
    for key, value in daylight_results_prefixed.items():
        df[key] = value

    return df