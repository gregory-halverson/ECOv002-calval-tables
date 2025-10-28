import pandas as pd
from daylight_evapotranspiration import daylight_ET_from_instantaneous_LE

def upscale_to_daylight(df: pd.DataFrame, prefix: str = "insitu_") -> pd.DataFrame:
    daylight_results = daylight_ET_from_instantaneous_LE(
        LE_instantaneous_Wm2=df.insitu_LE_Wm2,
        Rn_instantaneous_Wm2=df.insitu_Rn_Wm2,
        G_instantaneous_Wm2=df.insitu_G_Wm2,
        time_UTC=df.time_UTC,
        geometry=df.geometry
    )
    
    daylight_results_prefixed = {f"{prefix}{k}": v for k, v in daylight_results.items()}
    
    for key, value in daylight_results_prefixed.items():
        df[key] = value

    return df