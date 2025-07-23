
# ECOSTRESS Collection 2 Calibration/Validation Data Tables (Python Package)

This Python package provides calibration and validation data tables used in the evaluation of the ECOSTRESS Collection 2 Evapotranspiration (ET) data products. The included datasets and loader functions support reproducible research and validation workflows for ET modeling and remote sensing studies.

## Authors

Zoe Pierrat (she/her)  
[zoe.a.pierrat@jpl.nasa.gov](mailto:zoe.a.pierrat@jpl.nasa.gov)  
NASA Jet Propulsion Laboratory 329G

Gregory H. Halverson (they/them)  
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)  
NASA Jet Propulsion Laboratory 329G

## Installation

Install the package directly from PyPI using pip. Note that the package name uses dashes (`-`) for installation:

```bash
pip install ECOv002-calval-tables
```

## Usage


Import the loader functions to access the data tables as pandas DataFrames. Note that the importable package name uses underscores (`_`):

```python
from ECOv002_calval_tables import load_combined_eco_flux_ec_filtered, load_metadata_ebc_filt

df_flux = load_combined_eco_flux_ec_filtered()
df_metadata = load_metadata_ebc_filt()
```

## Data Tables

- `combined_eco_flux_EC_filtered`: Filtered eddy covariance flux data for ECOSTRESS validation.
- `metadata_ebc_filt`: Metadata for the filtered flux sites.

## Citation

If you use these data tables or code in your research, please cite:

> Pierrat, Zoe A.; Purdy, Adam J.; Halverson, Gregory H.; Fisher, Joshua B.; et al. (2024). Evaluation of ECOSTRESS Collection 2 Evapotranspiration Products: Strengths and Uncertainties for Evapotranspiration Modeling. *Water Resources Research*. https://doi.org/10.1029/2024WR039404

BibTeX:

```bibtex
@article{Pierrat2024,
  author       = {Pierrat, Zoe A. and Purdy, Adam J. and Halverson, Gregory H. and Fisher, Joshua B. and et al.},
  title        = {Evaluation of ECOSTRESS Collection 2 Evapotranspiration Products: Strengths and Uncertainties for Evapotranspiration Modeling},
  journal      = {Water Resources Research},
  year         = {2024},
  doi          = {10.1029/2024WR039404}
}
```
