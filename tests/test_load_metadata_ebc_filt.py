import unittest
import geopandas as gpd
from ECOv002_calval_tables import load_metadata_ebc_filt

class TestLoadMetadataEbcFilt(unittest.TestCase):
    def test_returns_geodataframe_with_valid_geometry(self):
        gdf = load_metadata_ebc_filt()
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertIn('geometry', gdf.columns)
        # Check all geometries are valid Points
        self.assertTrue(all(gdf.geometry.geom_type == 'Point'))
        self.assertTrue(gdf.geometry.is_valid.all())

if __name__ == "__main__":
    unittest.main()
