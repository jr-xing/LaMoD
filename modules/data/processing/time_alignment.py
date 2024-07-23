import numpy as np

def find_closest_subtimestep(ori_timestamps, query_time, n_interval_steps=10, tolerance=1e-6):
    """
    Determine the closest sub-timestep to a given query time within a range defined by two consecutive timestamps.
    If the query time is very close to an existing timestamp, consider it as exactly on that timestamp.

    Parameters:
    ori_timestamps (array_like): An array or list of timestamp floats, possibly unsorted.
    query_time (float): The specific time at which to find the closest sub-timestep.
    n_interval_steps (int, optional): Number of evenly distributed sub-timesteps between each pair of neighboring timestamps. Default is 10.
    tolerance (float, optional): The threshold to consider query time as exactly on an existing timestamp. Default is 1e-6.

    Returns:
    tuple: A tuple containing:
        - A tuple (index of the timestamp before the query time, timestamp value),
        - A tuple (index of the sub-timestep closest to the query time, time value of that sub-timestep).

    Raises:
    ValueError: If query time is outside the range of the timestamps.
    """

    # Sort the timestamps
    ori_timestamps = np.sort(ori_timestamps)

    # Edge cases when query_time is outside the range of ori_timestamps
    if query_time < ori_timestamps[0] or query_time > ori_timestamps[-1]:
        raise ValueError("Query time is outside the range of the timestamps.")

    # Check if query_time is very close to an existing timestamp
    closest_idx = np.argmin(np.abs(ori_timestamps - query_time))
    if np.abs(ori_timestamps[closest_idx] - query_time) <= tolerance:
        # If query_time is very close to a timestamp, consider it exactly on that timestamp
        return (closest_idx, ori_timestamps[closest_idx]), (0, ori_timestamps[closest_idx])

    # Finding the index for the timestamp just before the query_time
    idx_before = np.searchsorted(ori_timestamps, query_time) - 1
    idx_after = idx_before + 1

    # Calculate sub-timesteps
    t0, t1 = ori_timestamps[idx_before], ori_timestamps[idx_after]
    sub_timesteps = np.linspace(t0, t1, n_interval_steps + 2)[1:-1]  # Excluding the actual timestamps

    # Find the closest sub-timestep index
    sub_idx = np.argmin(np.abs(sub_timesteps - query_time))
    closest_sub_time = sub_timesteps[sub_idx]

    # Return the results
    return (idx_before, ori_timestamps[idx_before]), (sub_idx, closest_sub_time)

# Example usage:
# ori_timestamps = [3.0, 1.0, 2.5, 2.0, 1.5]
# query_time = 1.75
# print(find_closest_subtimestep(ori_timestamps, query_time, 9))
