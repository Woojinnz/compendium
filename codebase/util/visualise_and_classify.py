import joblib

from codebase.util.visualise_coordinates import *

def classify_with_comprehensive_features(df, model, scaler):
    features, startidx, endidx = extract_comprehensive_features(df)

    X = scaler.transform(pd.DataFrame([features]))
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return pred, prob, features, startidx, endidx


# Load the model from model training
model = joblib.load("../model_training/model_pos.pkl")
scaler = joblib.load("../model_training/scaler_pos.pkl")

# Modify this to open another file.
file_path = "../model_training/training_data/falls/legacy/1_position_data.csv"

# Load data
pos_data = get_file_as_dataframe(file_path)
pos_data = pos_data[(pos_data['x'] != 0) & (pos_data['y'] != 0) & (pos_data['z'] != 0)]

processed_data = process_coordinates(
    pos_data,
    jump_threshold=0.14,
    sg_window=30,
    sg_polyorder=2,
    downsample_fs=30,  # Downsample from 60Hz to 30Hz
    original_fs=60
)

pred, prob, features, start, end = classify_with_comprehensive_features(processed_data, model, scaler)

print(features)

if pred:
    print(f"  Classification: FALL with probability {prob:.3f}")
else:
    print(f"  Classification: NON-FALL with probability {prob:.3f}")

# Plot results
plot_3d_trajectory(pos_data, "Original")
plot_3d_trajectory(processed_data, "Smoothed")
plot_processed_coordinates(processed_data, start, end)