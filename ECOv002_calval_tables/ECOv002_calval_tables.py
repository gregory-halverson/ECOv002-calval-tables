
import pandas as pd
import os
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
    tower_IDs = list(tower_locations_gdf["Site ID"])
    tower_names = list(tower_locations_gdf.Name)
    tower_geometries = tower_locations_gdf.geometry
    tower_data_df = load_combined_eco_flux_ec_filtered()

    tower_static_data_gdf = gpd.GeoDataFrame({
        "ID": tower_IDs,
        "name": tower_names,
        "geometry": tower_geometries
    }, crs="EPSG:4326")

    observation_tower_IDs = list(tower_data_df.ID)
    observation_tower_times_UTC = tower_data_df.eco_time_utc

    model_inputs_df = pd.DataFrame({
        "ID": observation_tower_IDs,
        "time_UTC": observation_tower_times_UTC
    })

    merged_df = pd.merge(
        left=model_inputs_df,
        right=tower_static_data_gdf,
        left_on="ID",
        right_on="ID",
        how="left"
    )

    # Convert merged DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(merged_df, geometry=merged_df["geometry"], crs="EPSG:4326")
    # Optionally, drop rows with missing geometry if needed:
    # gdf = gdf[gdf.geometry.notnull()]
    return gdf
    return gdf