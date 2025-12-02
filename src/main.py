from step1_load_data import load_raster_and_vectors
from step2_extract_samples import extract_training_samples
from step3_train_classifier import train_classifier
from step4_classify_raster import classify_full_raster
from step5_postprocess_rules import apply_ndvi_rules
from utils import write_geotiff

RAW_RASTER = "data/raw/uav_13band.tif"
TRAIN_VECTOR = "data/training/training_points.gpkg"
OUT_PATH = "data/outputs/classified_map.tif"

def main():
    print("STEP 1: Loading raster and training data...")
    arr, meta, gdf = load_raster_and_vectors(RAW_RASTER, TRAIN_VECTOR)

    print("STEP 2: Extracting training samples...")
    X, y = extract_training_samples(arr, meta, gdf)

    print("STEP 3: Training classifier (Random Forest)...")
    clf = train_classifier(X, y)

    print("STEP 4: Classifying full raster...")
    rf_map = classify_full_raster(arr, clf)

    print("STEP 5: Applying NDVI-based vegetation splitting...")
    final_map = apply_ndvi_rules(arr, rf_map)

    print("Saving final classified map...")
    write_geotiff(OUT_PATH, final_map, meta)

    print("Done! Saved to:", OUT_PATH)

if __name__ == "__main__":
    main()
