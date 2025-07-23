
import pandas as pd
import os

def load_combined_eco_flux_ec_filtered() -> pd.DataFrame:
    """
    Load the filtered eddy covariance (EC) flux dataset used for ECOSTRESS Collection 2 ET product validation.
    This dataset contains site-level, quality-controlled flux measurements that serve as ground truth for evaluating ECOSTRESS evapotranspiration estimates.
    Returns:
        pd.DataFrame: DataFrame of filtered EC flux data for validation analysis.
    """
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'combined_eco_flux_EC_filtered.csv'))


def load_metadata_ebc_filt() -> pd.DataFrame:
    """
    Load the metadata for the filtered eddy covariance (EC) flux sites used in the ECOSTRESS Collection 2 validation study.
    This table provides site information (location, climate, land cover, etc.) for interpreting and grouping the flux data in the validation analysis.
    Returns:
        pd.DataFrame: DataFrame of site metadata for the filtered EC flux dataset.
    """
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'metadata_ebc_filt.csv'))
