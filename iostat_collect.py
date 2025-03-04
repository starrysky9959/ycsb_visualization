import subprocess
import time
import csv
import matplotlib.pyplot as plt
import datetime
import argparse
import threading

def get_iostat_data(device, duration):
    """
    Retrieves iostat data for a given device and duration using a single iostat command.

    Args:
        device (str): The name of the device to monitor (e.g., 'nvme1n1').
        duration (int): The duration in seconds to collect iostat data.

    Returns:
        list or None: A list of dictionaries, each containing 'timestamp', 'w/s', and 'wkB/s' metrics,
                        or None if data retrieval or parsing fails.
    """
    try:
        # Execute iostat command with extended statistics (-x) for the specified duration.
        command = ["iostat", "-x", device, "1", str(duration)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=duration + 10)  # Add a buffer to the timeout

        if stderr:
            print(f"iostat command error: {stderr.decode()}")
            return None
        # print(stdout)
        lines = stdout.decode().splitlines()
        data_points = []
        # Iterate through each line of iostat output to find the device's data.
        # Skip header lines until the device data starts
        device_data_started = False
        time_counter = 0
        # print("line:")
        for line in lines:
            
            if device in line:
                device_data_started = True
                data = line.split()
                # Assuming 'w/s' is the 7th and 'wkB/s' is the 8th column in 'iostat -x' output.
                # Adjust indices if your 'iostat -x' output format differs.
                try:
                    # print(data)
                    ws = float(data[7])  # w/s metric
                    wkbs = float(data[8]) # wkB/s metric
                    data_point = {"timestamp": time_counter, "w/s": ws, "wkB/s": wkbs}
                    data_points.append(data_point)
                    time_counter += 1
                except (IndexError, ValueError):
                    
                    print("Error parsing iostat output. Check 'iostat -x' output format.")
                    return None
            elif "Device" in line and device not in line:
                device_data_started = False

        if not data_points:
            print(f"No iostat data found for device: {device}.")
            return None

        return data_points

    except FileNotFoundError:
        print("iostat command not found. Ensure 'sysstat' package is installed.")
        return None
    except subprocess.TimeoutExpired:
        print("iostat command timed out during execution.")
        return None


def save_to_csv(data, device, filename=None):
    """
    Saves collected iostat data to a CSV file.

    Args:
        data (list): List of iostat data dictionaries.
        filename (str): Name of the CSV file to save data to (default: 'iostat_data.csv').
    """
    if filename is None:
        filename = f"{device}_iostat_data.csv"
    if not data:
        print("No data available to save to CSV file.")
        return

    keys = data[0].keys() # Extract keys from the first data point for CSV header.
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"iostat data saved to: {filename}")

def generate_line_chart(data, device, filename=None):
    """
    Generates and saves a line chart of iostat metrics over time with relative time and dual Y-axis.

    Args:
        data (list): List of iostat data dictionaries.
        filename (str): Name of the file to save the chart to (default: 'iostat_chart.png').
    """
    if filename is None:
        filename = f"{device}_iostat_chart.png"
    if not data:
        print("No data to generate line chart.")
        return

    timestamps = [point['timestamp'] for point in data]
    ws_values = [point['w/s'] for point in data]
    wkbs_values = [point['wkB/s'] for point in data]

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()  # Create the second y-axis

    line1, = ax1.plot(timestamps, ws_values, label='w/s', marker='o', linestyle='-', color='blue')
    line2, = ax2.plot(timestamps, wkbs_values, label='wkB/s', marker='x', linestyle='--', color='orange')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('w/s', color='blue')
    ax2.set_ylabel('wkB/s', color='orange')
    ax1.set_title(f'device {device}: iostat Metrics (w/s and wkB/s) Over Time')
    ax1.grid(True)

    # Rotate x-axis labels and align to right for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Set y-axis tick parameters for color matching
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left') # Adjust legend location if needed

    plt.savefig(filename)
    # plt.show()
    plt.close()
    print(f"Line chart saved as: {filename}")

def process_device(device_name, duration_seconds, collected_data_queue):
    device_name = device_name.strip()
    print(f"Starting iostat data collection for device {device_name} for {duration_seconds} seconds...")
    collected_data = get_iostat_data(device_name, duration_seconds)

    if collected_data:
        save_to_csv(collected_data, device_name)
        collected_data_queue.put((device_name, collected_data))
    else:
        print(f"Failed to collect iostat data for device {device_name}.")


if __name__ == "__main__":
    # Create an ArgumentParser to handle command-line arguments.
    parser = argparse.ArgumentParser(description="Collects iostat data for a specified device and duration.")
    # Add 'device' argument for the device name.
    parser.add_argument("devices", help="Device names to monitor (e.g., nvme1n1,nvme2n2), separated by commas")
    # Add 'duration' argument for the collection duration in seconds.
    parser.add_argument("duration", type=int, help="Duration in seconds to collect iostat data")

    # Parse command-line arguments.
    args = parser.parse_args()
    device_names = args.devices.split(',') # Device name from command line.
    duration_seconds = args.duration # Duration from command line.

    import queue
    collected_data_queue = queue.Queue()
    threads = []
    for device_name in device_names:
        thread = threading.Thread(target=process_device, args=(device_name, duration_seconds, collected_data_queue))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Generate charts serially after all data is collected
    while not collected_data_queue.empty():
        device_name, collected_data = collected_data_queue.get()
        generate_line_chart(collected_data, device_name)

    print("Script execution completed.")
