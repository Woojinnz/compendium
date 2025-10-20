import numpy as np

def detect_jumps(data, threshold, window_size):
    """
    Detect jumps in data using statistical outlier detection
    """
    # Calculate moving statistics
    rolling_mean = data.rolling(window=window_size, center=True).mean()
    rolling_std = data.rolling(window=window_size, center=True).std()

    # Detect outliers (potential jumps)
    z_scores = np.abs((data - rolling_mean) / rolling_std)
    jump_mask = z_scores > threshold

    return jump_mask, z_scores

def remove_jumps(data, threshold, window_size):
    """
    Remove jumps from data by interpolating over detected jump regions
    """
    jump_mask, z_scores = detect_jumps(data, threshold, window_size)

    # Create a copy of the data
    cleaned_data = data.copy()

    # Interpolate over jump regions
    if jump_mask.any():
        # Get indices where jumps occur
        jump_indices = np.where(jump_mask)[0]

        # For each jump, interpolate using surrounding good points
        for idx in jump_indices:
            # Find nearest non-jump points
            left_idx = idx - 1
            while left_idx >= 0 and jump_mask.iloc[left_idx]:
                left_idx -= 1

            right_idx = idx + 1
            while right_idx < len(data) and jump_mask.iloc[right_idx]:
                right_idx += 1

            # If we have valid boundaries, interpolate
            if left_idx >= 0 and right_idx < len(data):
                # Quadratic interpolation using three points if possible
                if left_idx - 1 >= 0 and right_idx + 1 < len(data):
                    x_vals = data.iloc[left_idx - 1:right_idx + 2]
                    t_vals = np.arange(left_idx - 1, right_idx + 2)
                    coeffs = np.polyfit(t_vals, x_vals, 2)
                    cleaned_data.iloc[idx] = np.polyval(coeffs, idx)
                else:
                    # Fallback to linear if not enough points
                    x_left, x_right = data.iloc[left_idx], data.iloc[right_idx]
                    t_left, t_right = left_idx, right_idx
                    cleaned_data.iloc[idx] = x_left + (x_right - x_left) * (idx - t_left) / (t_right - t_left)

    return cleaned_data, jump_mask, z_scores