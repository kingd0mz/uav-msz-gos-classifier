import numpy as np
from rasterio.sample import sample

def extract_training_samples(arr, meta, gdf):
    with rasterio.open(meta["name"]) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        samples = np.array(list(sample(src, coords)))

    X = samples
    y = gdf["class_id"].astype(int).values
    return X, y
