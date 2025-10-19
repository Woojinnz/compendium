import json, redis, pandas as pd, joblib
import threading
import time as thread_time
from collections import deque
import numpy as np

from app.notification.service import NotificationService
from app.services.incident_service import IncidentService
from app.services.fall.jump_filter import remove_jumps
from app.services.fall.savitzky_golay import apply_savgol_filter

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


def classify_buffer(df, model, scaler):
    """
    Classify using comprehensive kinematic features instead of just acceleration magnitude
    """
    features = extract_comprehensive_features(df)

    X = scaler.transform(pd.DataFrame([features]))
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return pred, prob


model  = joblib.load("app/services/fall/model_pos.pkl")
scaler = joblib.load("app/services/fall/scaler_pos.pkl")

REDIS_DB  = 2
r = redis.Redis(host="localhost", port=6379, db=REDIS_DB)

sender = NotificationService()

MIN_SAMPLES = 420
SLEEP       = 4.0  # seconds
FALL_LOCKOUT = 30  # seconds
WINDOW_SIZE = 420 # 1 min at 60 hz
seen_ready = {}
tag_locked = {}
tag_lock_queue = []
buffers = {}

def process_pos_data(position_list, threshold=0.4) -> tuple[pd.DataFrame, bool]:
    data = pd.DataFrame(position_list)
    is_fall = False

    # Process data with velocity calculation
    data = process_coordinates(
        data,
        jump_threshold=0.14,
        sg_window=30,
        sg_polyorder=2,
        downsample_fs=30,  # Downsample from 60Hz to 30Hz
        original_fs=60
    )

    peak_idx = data['z'].idxmax()
    post = data.loc[peak_idx + 1:, 'z']
    if len(post) > 0:
        peak_z = float(data['z'][peak_idx])
        avg_after_peak = np.percentile(post.to_numpy(),25)
        if peak_z - avg_after_peak >= threshold:
            is_fall = True

    return (data, is_fall)


def downsample(df, original_freq=60, target_freq=30):
    factor = original_freq // target_freq
    return df.groupby(df.index // factor).mean(numeric_only=True).reset_index(drop=True)



def process_coordinates(df, jump_threshold=0.14, sg_window=30, sg_polyorder=2, downsample_fs=None, original_fs=60):
    """
    Process coordinates with jump removal, optional downsampling, and Savitzky-Golay smoothing
    Order: Remove jumps -> Downsample -> Savitzky-Golay smoothing -> Calculate velocity
    """

    # Step 1: normalise timestamps
    df['ts'] = df['ts'] - df['ts'].min()
    df = df.reset_index(drop=True)


    # Step 2: Remove jumps
    for coord in ['x', 'y', 'z']:
        # Remove jumps
        remove_jumps(df[coord], threshold=jump_threshold, window_size=5)

    # Step 3: Downsample if requested
    if downsample_fs is not None and downsample_fs < original_fs:
         df = downsample(df, original_freq=original_fs, target_freq=downsample_fs)

    # Step 4: Apply Savitzky-Golay smoothing to downsampled data
    apply_savgol_filter(df, window_length = sg_window, polyorder = sg_polyorder, freq = 30, trim = True)

    return df


def start():


    while True:
        # Handle lockout expiration
        current_time = thread_time.time()
        while tag_lock_queue and current_time > tag_lock_queue[0][1]:
            tag, _ = tag_lock_queue.pop(0)
            tag_locked[tag] = False

        # Process each tag
        for key in r.keys():
            tag = key.decode("utf-8")

            # Skip if locked
            if tag_locked.get(tag, False):
                continue

            # Get only NEW data since last processing
            raw_positions = r.lrange(tag, 0, -1)
            # delete after reading
            r.delete(tag)

            if raw_positions:
                position_list = [json.loads(x.decode('utf-8')) for x in raw_positions]
                # Combine with previous buffer or create new
                if tag in buffers:
                    buffers[tag].extend(position_list)
                    # Keep only last WINDOW_SIZE samples
                else:
                    buffers[tag] = deque(position_list, maxlen=WINDOW_SIZE)

                # Check if we have enough samples
                if len(buffers[tag]) >= MIN_SAMPLES and not seen_ready.get(tag, False):
                    print(f"🟢 {tag} buffer now has ≥{MIN_SAMPLES} samples - classifier active.")
                    seen_ready[tag] = True

                # Run classification if ready
                if seen_ready.get(tag, False) and len(buffers[tag]) >= MIN_SAMPLES:
                    # convert  epoch time to human readable
                    time = thread_time.strftime("%Y-%m-%d %H:%M:%S", thread_time.localtime(buffers[tag][-1]['ts']))
                    processed_data, fall_threshold_passed = process_pos_data(buffers[tag], threshold = 0.4)
                    if fall_threshold_passed:
                        pred, prob = classify_buffer(processed_data, model, scaler)
                        if pred:
                            print(f"🧮 {tag} - FALL DETECTED! prob={prob:.3f}")
                            # pd.DataFrame(buffers[tag])[['ts', 'x', 'y', 'z']].to_csv('fall.csv', index=False)
                            description = f"Tag: {tag}, has fallen"
                            incident_data = IncidentService.fetch_incident_data_from_tag(tag, description)

                            if incident_data:
                                sender.send_notification("fall_notification", {
                                    "tag": tag,
                                    "danger": True,
                                    "time": time
                                })
                                IncidentService.create_incident(incident_data)

                            _reset_after_fall(tag, current_time)
                        else:
                            print('thresh passed but no fall')
                    else:
                        print('no fall')

        # sleep until current_time + SLEEP or 0s if already past
        thread_time.sleep(max(0.0, current_time + SLEEP - thread_time.time()))

def _reset_after_fall(tag: str, now: float):
    try:
        r.delete(tag)
    except Exception:
        pass

    if tag in buffers:
        buffers[tag].clear()
    seen_ready[tag] = False  

    tag_locked[tag] = True
    tag_lock_queue.append((tag, now + FALL_LOCKOUT))


def start_fall_monitor():
    thread = threading.Thread(target=start)
    thread.daemon = True
    thread.start()