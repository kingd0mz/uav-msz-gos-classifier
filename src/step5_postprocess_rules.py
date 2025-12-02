import numpy as np

def apply_ndvi_rules(arr, rf_map):
    # NDVI band = band 11 = zero index 10
    ndvi = arr[10, :, :]

    # start with RF output
    out = rf_map.copy()

    # vegetation mask
    veg_mask = (rf_map == 1)

    # apply NDVI thresholds
    low_mask  = veg_mask & (ndvi <= 0.3)
    med_mask  = veg_mask & (ndvi > 0.3) & (ndvi <= 0.5)
    high_mask = veg_mask & (ndvi > 0.5)

    out[low_mask]  = 1  # low vegetation
    out[med_mask]  = 2  # medium vegetation
    out[high_mask] = 3  # high vegetation

    # non-vegetation classes keep their IDs
    # water = 4, road = 5, others = 6

    return out
