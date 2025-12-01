import numpy as np

def apply_ndvi_rules(arr, class_map):
    ndvi = arr[10, :, :]  # band 11 (zero-indexed)
    out = class_map.copy()

    out[(ndvi <= 0.3)] = 1
    out[(ndvi > 0.3) & (ndvi <= 0.5)] = 2
    out[(ndvi > 0.5)] = 3

    return out
