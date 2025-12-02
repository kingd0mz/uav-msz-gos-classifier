import numpy as np

def classify_full_raster(arr, clf):
    n_bands, H, W = arr.shape

    # reshape to (pixels, bands)
    X_all = arr.reshape(n_bands, -1).T

    y_all = clf.predict(X_all)
    class_map = y_all.reshape(H, W)

    return class_map
