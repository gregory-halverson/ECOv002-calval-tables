import unittest
import geopandas as gpd
from ECOv002_calval_tables import load_calval_table

class TestLoadCalvalTable(unittest.TestCase):
    def test_returns_geodataframe_with_valid_geometry(self):
        gdf = load_calval_table()
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertIn('geometry', gdf.columns)
        self.assertTrue(all(gdf.geometry.geom_type == 'Point'))
        self.assertTrue(gdf.geometry.is_valid.all())

if __name__ == "__main__":
    unittest.main()
