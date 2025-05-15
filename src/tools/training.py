from typing import List


def load_motion_data(file_path : str) -> List[List[float]]:
    """
       Load motion data from a CSV file, skipping the header line.

       Each subsequent line is expected to have exactly 6 comma-separated float values,
       representing sensor readings (e.g., accelerometer and gyroscope axes). If untrue,
       the line will be skipped.

       Args:
           file_path (str): Path to the CSV file containing the motion data.

       Returns:
           List[List[float]]: A list of samples, where each sample is a list of 6 floats.
       """

    data : List[List[float]] = []

    with open(file_path, 'r') as file:

        file.readline() # skip header file of csv

        for line in file:

            row = line.strip().split(',')

            if len(row) == 6:

                sample = [
                    float(row[0]), float(row[1]), float(row[2]),
                    float(row[3]), float(row[4]), float(row[5]),
                ]

                data.append(sample)

    return data


def create_windows(data: List[List[float]], sample_rate: int, motion_length: float) -> List[List[List[float]]]:
    """
    Splits the raw motion data into fixed-size windows based on the sample rate and motion length.

    Args:
        data: A list of samples, each sample is a list of floats (e.g., [aX, aY, aZ, gX, gY, gZ]).
        sample_rate: Number of samples per second in the data.
        motion_length: Length of one motion segment in seconds.

    Returns:
        A list of windows, where each window is a list of samples.
    """
    windows = []
    window_size = int(sample_rate * motion_length)
    
    # Slide over the data by window_size chunks (no overlap here)
    for start_idx in range(0, len(data), window_size):
        window = data[start_idx:start_idx + window_size]
        if len(window) == window_size:
            windows.append(window)
            
    return windows