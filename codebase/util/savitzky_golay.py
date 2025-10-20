import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

DEFAULT_WINDOW = 5  # default window size

def apply_savgol_filter(df, window_length: int = None, polyorder=3, freq=60, trim=False, convert_g=False, deriv=0) -> pd.DataFrame:
    """
    Highly optimized version using pure NumPy operations
    """
    # Validate input and extract numpy arrays
    required_cols = ['x', 'y', 'z']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Convert all coordinates to a single numpy array at once
    positions = df[required_cols].values  # Shape: (n_samples, 3)
    n_samples = len(positions)

    # Auto-calculate window length
    if window_length is None:
        window_length = min(21, max(DEFAULT_WINDOW, int(0.1 * n_samples)))
        if window_length % 2 == 0:
            window_length += 1

    dt = 1.0 / freq
    derivatives = savgol_filter(positions, window_length, polyorder, deriv=deriv, delta=dt, axis=0)

    if convert_g:
        derivatives /= 9.81

    magnitude = np.linalg.norm(derivatives, axis=1)

    result_df = pd.DataFrame(derivatives, columns=required_cols, index=df.index)
    result_df['mag'] = magnitude
    result_df['ts'] = df['ts'].values

    # Trim edges
    if trim:
        trim_size = window_length // 2
        if trim_size > 0 and len(result_df) > 2 * trim_size:
            result_df = result_df.iloc[trim_size:-trim_size]
    else:
        # replace with edge value
        trim_size = window_length // 2
        for col in required_cols + ['mag']:
            result_df.loc[:trim_size, col] = result_df[col].iloc[trim_size]
            result_df.loc[-trim_size:, col] = result_df[col].iloc[-trim_size - 1]

    return result_df