import rasterio
import geopandas as gpd

def load_raster_and_vectors(raster_path, vector_path):
    src = rasterio.open(raster_path)
    arr = src.read()   # shape = (13, H, W)
    meta = src.meta.copy()

    gdf = gpd.read_file(vector_path)
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    return arr, meta, gdf
