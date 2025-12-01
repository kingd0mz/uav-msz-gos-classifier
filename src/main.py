import os
from src.step1_load_data import load_raster_and_vectors
from src.step2_extract_samples import extract_training_samples
from src.step3_train_classifier import train_classifier
from src.step4_classify_raster import classify_full_raster
from src.step5_postprocess_rules import apply_ndvi_rules
from src.utils import write_geotiff

RAW_RASTER = "data/raw/uav_13band.tif"
TRAIN_VECTOR = "data/training/training_points.gpkg"
OUT_PATH = "data/outputs/classified_map.tif"

def main():
    print("STEP 1: Load raster + training data")
    arr, meta, gdf = load_raster_and_vectors(RAW_RASTER, TRAIN_VECTOR)

    print("STEP 2: Extract training samples")
    X, y = extract_training_samples(arr, meta, gdf)

    print("STEP 3: Train classifier")
    clf = train_classifier(X, y)

    print("STEP 4: Classify full raster")
    class_map = classify_full_raster(arr, clf)

    print("STEP 5: Apply NDVI rules")
    class_corrected = apply_ndvi_rules(arr, class_map)

    print("Saving output GeoTIFF...")
    write_geotiff(OUT_PATH, class_corrected, meta)

    print("Done! Output saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
