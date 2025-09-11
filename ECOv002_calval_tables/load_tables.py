import os

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

def load_combined_eco_flux_ec_filtered() -> pd.DataFrame:
    """
    Load the filtered eddy covariance (EC) flux dataset used for ECOSTRESS Collection 2 ET product validation.
    This dataset contains site-level, quality-controlled flux measurements that serve as ground truth for evaluating ECOSTRESS evapotranspiration estimates.
    Returns:
        pd.DataFrame: DataFrame of filtered EC flux data for validation analysis.
    """
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'combined_eco_flux_EC_filtered.csv'))


def load_metadata_ebc_filt() -> gpd.GeoDataFrame:
    """
    Load the metadata for the filtered eddy covariance (EC) flux sites used in the ECOSTRESS Collection 2 validation study.
    This table provides site information (location, climate, land cover, etc.) for interpreting and grouping the flux data in the validation analysis.
    Returns:
        pd.DataFrame: DataFrame of site metadata for the filtered EC flux dataset.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'metadata_ebc_filt.csv'))
    
    if 'Lat' not in df.columns or 'Long' not in df.columns:
        raise ValueError("metadata_ebc_filt.csv must contain 'Lat' and 'Long' columns.")
    
    geometry = [Point(xy) for xy in zip(df['Long'], df['Lat'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    return gdf

def load_calval_table() -> gpd.GeoDataFrame:
    """
    Load the combined ECOSTRESS Collection 2 validation table, which includes both the filtered eddy covariance flux data
    and the associated site metadata.
    
    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame of EC flux data and site metadata for validation analysis.
    """
    tower_locations_gdf = load_metadata_ebc_filt()
    tower_data_df = load_combined_eco_flux_ec_filtered()

    # Merge all columns from both tables, matching tower_data_df.ID to tower_locations_gdf["Site ID"]
    merged_df = pd.merge(
        tower_data_df,
        tower_locations_gdf,
        left_on="ID",
        right_on="Site ID",
        how="left",
        suffixes=("", "_meta")
    )

    merged_df["time_UTC"] = merged_df["eco_time_utc"]
    merged_df["ST_K"] = np.array(merged_df.LST)
    merged_df["ST_C"] = merged_df.ST_K - 273.15
    merged_df["Ta_C"] = np.array(merged_df.Ta)
    merged_df["SWin_Wm2"] = np.array(merged_df.Rg)
    merged_df["emissivity"] = np.array(merged_df.EmisWB)

    # Convert merged DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(merged_df, geometry=merged_df["geometry"], crs="EPSG:4326")
    
    return gdf
