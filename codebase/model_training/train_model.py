# train_fall_model.py
from pathlib import Path
import re, pandas as pd, numpy as np, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from codebase.util.jump_filter import remove_jumps
from codebase.util.savitzky_golay import apply_savgol_filter


def get_file_as_dataframe(file):
    # ignore the first line if it is a header
    content = file.read_text()
    if content.startswith('ts'):
        content = content.split('\n', 1)[1]
    data = [line.split(',') for line in content.strip().split('\n')]
    df = pd.DataFrame(data, columns=['ts', 'x', 'y', 'z'])
    df = df.apply(pd.to_numeric, errors='coerce')

    # Sort by timestamp
    df = df.sort_values('ts').reset_index(drop=True)
    # remove all rows with 0 in any of the columns
    df = df[(df['x'] != 0) & (df['y'] != 0) & (df['z'] != 0)]
    return df


def gather(path, label):
    out = []
    search_queue = []
    search_queue.append(path)
    while search_queue:
        for f in search_queue.pop(0).iterdir():
            if f.is_file() and f.suffix == '.csv':
                df = get_file_as_dataframe(f)
                processed_df = process_coordinates(
                    df,
                    jump_threshold=0.14,
                    sg_window=15,
                    sg_polyorder=2,
                    downsample_fs=30,  # Downsample from 60Hz to 30Hz
                    original_fs=60
                )
                processed_df['label'] = label  # Fall label
                processed_df['file'] = f.resolve()
                out.append(processed_df)
            elif f.is_dir():
                search_queue.append(f)
    return pd.concat(out, ignore_index=True)


def downsample(df, original_freq=60, target_freq=30):
    factor = original_freq // target_freq
    return df.groupby(df.index // factor).mean(numeric_only=True).reset_index(drop=True)


def process_coordinates(df, jump_threshold=0.14, sg_window=30, sg_polyorder=2, downsample_fs=None, original_fs=60):
    """
    Process coordinates with jump removal, optional downsampling, and Savitzky-Golay smoothing
    Order: Remove jumps -> Downsample -> Savitzky-Golay smoothing -> Calculate velocity
    """
    processed_df = df.copy()

    file_name = processed_df['file'].loc[0] if 'file' in processed_df.columns else None

    # Step 1: normalise timestamps
    processed_df['ts'] = processed_df['ts'] - processed_df['ts'].min()
    processed_df = processed_df.reset_index(drop=True)

    # Step 2: Remove jumps
    for coord in ['x', 'y', 'z']:
        # Remove jumps
        cleaned_data, jump_mask, z_scores = remove_jumps(processed_df[coord], threshold=jump_threshold, window_size=5)

        # Store cleaned data and jump info
        processed_df[coord] = cleaned_data

    # Step 3: Downsample if requested
    if downsample_fs is not None and downsample_fs < original_fs:
        downsampled_df = downsample(processed_df, original_freq=original_fs, target_freq=downsample_fs)
    else:
        downsampled_df = processed_df

    # Step 4: Apply Savitzky-Golay smoothing to downsampled data
    smoothed_data = apply_savgol_filter(downsampled_df, window_length=sg_window, polyorder=sg_polyorder, freq=30,
                                        trim=True)

    if file_name is not None:
        smoothed_data['file'] = file_name

    return smoothed_data


def find_optimal_fall_window(array, threshold_percent=0.95):
    n = len(array)

    # Calculate future minimums and maximum decreases
    future_mins = np.zeros(n)
    future_mins[-1] = array[-1]
    for i in range(n - 2, -1, -1):
        future_mins[i] = min(array[i], future_mins[i + 1])

    max_decreases = array - future_mins
    fall_start_idx = np.argmax(max_decreases)

    start_threshold = threshold_percent * max_decreases[fall_start_idx]
    for j in range(n-3, fall_start_idx-1, -1):
        if max_decreases[j] >= start_threshold:
            fall_start_idx = j
            break

    fall_end_idx = np.argmin(array[fall_start_idx:]) + fall_start_idx
    end_threshold = threshold_percent * array[fall_end_idx]

    for k in range(fall_start_idx, n):
        if array[k] <= end_threshold:
            fall_end_idx = k
            break

    fall_height_decrease = array[fall_start_idx] - array[fall_end_idx]

    return fall_start_idx, fall_end_idx, fall_height_decrease


def extract_comprehensive_features(group):
    """
    Extract comprehensive fall features using kinematic analysis
    """
    # Maximum height decrease in the entire sequence
    # find max decrease (point - lowest point after it)
    array = group['z'].to_numpy()

    fall_start_idx, fall_end_idx, fall_height_decrease = find_optimal_fall_window(
        array, threshold_percent=0.9
    )

    fall_duration = group['ts'].iloc[fall_end_idx] - group['ts'].iloc[fall_start_idx]
    fall_horizontal_distance = calculate_total_horizontal_movement(group[fall_start_idx:fall_end_idx + 1])
    fall_avg_horizontal_speed = fall_horizontal_distance / fall_duration if fall_duration > 0 else 0
    fall_avg_vertical_speed = fall_height_decrease / fall_duration if fall_duration > 0 else 0

    z_std = group['z'].std()

    #height_5th = group['z'].quantile(0.05)
    #height_10th = group['z'].quantile(0.10)

    #height_95th = group['z'].quantile(0.95)
    #height_90th = group['z'].quantile(0.90)

    #height_change = height_95th - height_5th
   # height_change_90_10 = height_90th - height_10th
    pre = group['z'].iloc[:max(1, fall_start_idx - 1)]
    post = group['z'].iloc[fall_end_idx + 1:]
    # use top 5% to avoid outliers
    height_change = pre.quantile(0.95) - post.quantile(0.95)

    # Combine all features
    features = {
        'z_std': z_std,
        'fall_duration': fall_duration,
        'fall_height_decrease': fall_height_decrease,
        'fall_horizontal_distance': fall_horizontal_distance,
        'fall_avg_horizontal_speed': fall_avg_horizontal_speed,
        'fall_avg_vertical_speed': fall_avg_vertical_speed,
        'height_change': height_change,
    }

    return pd.Series(features)


def calculate_total_horizontal_movement(pos_df):
    """Calculate total horizontal movement throughout the sequence"""
    dx = np.diff(pos_df['x'])
    dy = np.diff(pos_df['y'])
    total_movement = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    return total_movement


def plot_3d_trajectory(df, file_id, title_suffix=""):
    """Plot 3D trajectory of coordinates"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    scatter = ax.scatter(df['x'], df['y'], df['z'],
                         c=df['ts'], cmap='viridis', alpha=0.7, s=20)

    # Connect points with lines to show path
    ax.plot(df['x'], df['y'], df['z'], 'gray', alpha=0.3, linewidth=1)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'File {file_id}: 3D Trajectory {title_suffix}')

    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Timestamp')

    # Set equal aspect ratio for better visualization
    max_range = np.array([df['x'].max() - df['x'].min(),
                          df['y'].max() - df['y'].min(),
                          df['z'].max() - df['z'].min()]).max() / 2.0

    mid_x = (df['x'].max() + df['x'].min()) * 0.5
    mid_y = (df['y'].max() + df['y'].min()) * 0.5
    mid_z = (df['z'].max() + df['z'].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


# Gather data
falls = gather(Path("training_data/falls"), 1)
adls = gather(Path("training_data/adl"), 0)

# Create the combined dataframes
df_all = pd.concat([falls, adls])

# Extract comprehensive features and labels
print("Extracting comprehensive features...")
Xy = df_all.groupby('file').apply(extract_comprehensive_features)
y = df_all.groupby('file')['label'].first()
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None):
    # Print feature statistics by class
    print("\nFeature means by class:")
    print(Xy.groupby(y).mean())
    print("\nFeature medians by class:")
    print(Xy.groupby(y).median())

Xy['label'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    Xy.drop(columns='label'), Xy['label'], test_size=0.2, random_state=42, stratify=Xy['label'])

print(f"\nTraining set class distribution: {y_train.value_counts()}")
print(f"Test set class distribution: {y_test.value_counts()}")

# Scale the features
scaler = StandardScaler().fit(X_train)
X_train_sc, X_test_sc = scaler.transform(X_train), scaler.transform(X_test)

# Train the model with comprehensive features
model = XGBClassifier(
    scale_pos_weight=1.5,
    eval_metric='aucpr',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
).fit(X_train_sc, y_train)

# Evaluate the model
y_pred = model.predict(X_test_sc)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Save model and scaler
joblib.dump(model, "model_pos.pkl")
joblib.dump(scaler, "scaler_pos.pkl")
print("\nSaved model_pos.pkl and scaler_pos.pkl")

# Optional: Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()