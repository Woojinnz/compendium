import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import joblib

from fall.utils.jump_filter import remove_jumps
from fall.utils.savitzky_golay import apply_savgol_filter


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

    return df


def downsample(df, original_freq=60, target_freq=30):
    factor = original_freq // target_freq
    return df.groupby(df.index // factor).mean(numeric_only=True).reset_index(drop=True)


def process_coordinates(df, jump_threshold=0.14, sg_window=30, sg_polyorder=2, downsample_fs=None, original_fs=60):
    """
    Process coordinates with jump removal, optional downsampling, and Savitzky-Golay smoothing
    Order: Remove jumps -> Downsample -> Savitzky-Golay smoothing -> Calculate velocity
    """
    processed_df = df.copy()

    # Calculate original time vector (normalized to start at 0)
    original_timestamps = processed_df['ts'].values - processed_df['ts'].min()
    processed_df['ts'] = original_timestamps

    # Step 1: normalise timestamps
    processed_df['ts'] = processed_df['ts'] - processed_df['ts'].min()
    processed_df = processed_df.reset_index(drop=True)

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


def plot_processed_coordinates(processed_df, file_id):
    """Plot original, cleaned, and smoothed coordinates with jump markers"""
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
                label='Smoothed (Savitzky-Golay)', color='black', linewidth=2)

        ax.set_ylabel(f'{coord.upper()} Position')
        ax.set_title(f'File {file_id}: {coord.upper()} Coordinate Processing')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[2].set_xlabel('Timestamp')
    plt.tight_layout()
    plt.show()


def plot_final_smoothed_coordinates(processed_df, file_id):
    """Plot only the final smoothed coordinates"""
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

fall_dataframes = []
adl_dataframes = []
# get all files in training_data folder
falls_dir = Path("training_data/falls")
adl_dir = Path("training_data/adl")
fall_search_queue = []
adl_search_queue = []
fall_search_queue.append(falls_dir)
adl_search_queue.append(adl_dir)
while fall_search_queue:
    for f in fall_search_queue.pop(0).iterdir():
        if f.is_file() and f.suffix == '.csv':
            fall_dataframes.append(get_file_as_dataframe(f))
        elif f.is_dir():
            fall_search_queue.append(f)
while adl_search_queue:
    for f in adl_search_queue.pop(0).iterdir():
        if f.is_file() and f.suffix == '.csv':
            adl_dataframes.append(get_file_as_dataframe(f))
        elif f.is_dir():
            adl_search_queue.append(f)

# Load data
pos_data = adl_dataframes[1]
pos_data = pos_data[(pos_data['x'] != 0) & (pos_data['y'] != 0) & (pos_data['z'] != 0)]

# Process data with velocity calculation
processed_data = process_coordinates(
    pos_data,
    jump_threshold=0.14,
    sg_window=30,
    sg_polyorder=2,
    downsample_fs=30,  # Downsample from 60Hz to 30Hz
    original_fs=60
)

plot_3d_trajectory(processed_data, "1", "Smoothed")
plot_final_smoothed_coordinates(processed_data, "1")