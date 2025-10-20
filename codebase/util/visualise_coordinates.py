import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from codebase.util.jump_filter import remove_jumps
from codebase.util.savitzky_golay import apply_savgol_filter


def get_file_as_dataframe(file):
    # ignore the first line if it is a header
    with open(file, 'r') as f:
        content = f.read()
    if content.startswith('ts'):
        content = content.split('\n', 1)[1]
    data = [line.split(',') for line in content.strip().split('\n')]
    df = pd.DataFrame(data, columns=['ts', 'x', 'y', 'z'])
    df = df.apply(pd.to_numeric, errors='coerce')

    # Sort by timestamp
    df = df.sort_values('ts').reset_index(drop=True)

    return df


def downsample(df, original_freq=60, target_freq=30):
    factor = original_freq // target_freq
    return df.groupby(df.index // factor).mean(numeric_only=True).reset_index(drop=True)


def plot_3d_trajectory(df, file_id, title_suffix=""):
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


def plot_processed_coordinates(processed_df, start, end):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    colors = ['red', 'green', 'blue']

    for i, coord in enumerate(['x', 'y', 'z']):
        ax = axes[i]

        # Plot original data (if available in processed_df)
        if coord in processed_df.columns:
            ax.plot(processed_df['ts'], processed_df[f'{coord}_orig'],
                    label='Original', color=colors[i], alpha=0.7, linewidth=1)

        # Plot smoothed data
        ax.plot(processed_df['ts'], processed_df[f'{coord}'],
                label='Smoothed (Jump Removal + Savitzky-Golay)', color='black', linewidth=2)
        # Highlight fall segment
        ax.axvspan(processed_df['ts'].iloc[start], processed_df['ts'].iloc[end], color='yellow', alpha=0.3, label='Detected Fall Segment')

        ax.set_ylabel(f'{coord.upper()} Position')
        ax.set_title(f'{coord.upper()} Coordinate Processing')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[2].set_xlabel('Timestamp')
    plt.tight_layout()
    plt.show()


def plot_final_smoothed_coordinates(processed_df, file_id):
    plt.figure(figsize=(12, 8))

    plt.plot(processed_df['ts'], processed_df['x'],
             label='X Smoothed', color='red', linewidth=2)
    plt.plot(processed_df['ts'], processed_df['y'],
             label='Y Smoothed', color='green', linewidth=2)
    plt.plot(processed_df['ts'], processed_df['z'],
             label='Z Smoothed', color='blue', linewidth=2)

    plt.xlabel('Timestamp')
    plt.ylabel('Smoothed Position Coordinates')
    plt.title(f'File {file_id}: Final Smoothed Coordinates (After Jump Removal + Savitzky-Golay)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Modify functions below to change data processing and feature extraction pipeline.
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

    return pd.Series(features), fall_start_idx, fall_end_idx


def calculate_total_horizontal_movement(pos_df):
    """Calculate total horizontal movement throughout the sequence"""
    dx = np.diff(pos_df['x'])
    dy = np.diff(pos_df['y'])
    total_movement = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    return total_movement


def process_coordinates(df, jump_threshold=0.14, sg_window=30, sg_polyorder=2, downsample_fs=None, original_fs=60):
    """
    Process coordinates with jump removal, optional downsampling, and Savitzky-Golay smoothing
    Order: Remove jumps -> Downsample -> Savitzky-Golay smoothing -> Calculate velocity
    """
    processed_df = df.copy()

    # Step 1: scale timestamps
    original_timestamps = processed_df['ts'].values - processed_df['ts'].min()
    processed_df['ts'] = original_timestamps

    # Step 2: Remove jumps
    for coord in ['x', 'y', 'z']:
        # Remove jumps
        cleaned_data, jump_mask, z_scores = remove_jumps(processed_df[coord], threshold=jump_threshold, window_size=5)

        # Store cleaned data and jump info
        processed_df[f'{coord}_cleaned'] = cleaned_data

    # Create dataframe with cleaned data (after jump removal)
    cleaned_df = processed_df[['ts', 'x_cleaned', 'y_cleaned', 'z_cleaned']].copy()
    cleaned_df.columns = ['ts', 'x', 'y', 'z']

    # Step 3: Downsample if requested
    if downsample_fs is not None and downsample_fs < original_fs:
        downsampled_df = downsample(cleaned_df, original_freq=original_fs, target_freq=downsample_fs)
    else:
        downsampled_df = cleaned_df

    # Step 4: Apply Savitzky-Golay smoothing to downsampled data
    smoothed_data = apply_savgol_filter(downsampled_df, window_length=sg_window, polyorder=sg_polyorder, freq=30,
                                        trim=True)
    # save old data as x_orig etc
    for coord in ['x', 'y', 'z']:
        smoothed_data[f'{coord}_orig'] = downsampled_df[coord]

    return smoothed_data


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

features, start, end = extract_comprehensive_features(processed_data)

plot_3d_trajectory(pos_data, "Original")
plot_3d_trajectory(processed_data, "Smoothed")
plot_processed_coordinates(processed_data, start, end)

print(features)