import os
import re
import glob
import pandas as pd
from pathlib import Path

#!/usr/bin/env python3


def extract_metrics(file_path):
    """Extract throughput and latency metrics from YCSB output file."""
    metrics = {}

    try:
        with open(file_path, "r") as f:
            for line in f:
                # Skip lines with [CLEANUP]
                if "[CLEANUP]" in line:
                    continue

                # Extract overall throughput
                if "Throughput" in line and not metrics.get("Throughput"):
                    match = re.search(r"Throughput\(ops/sec\), ([0-9.]+)", line)
                    if match:
                        metrics["Throughput"] = float(match.group(1))

                # Extract average latency
                if "AverageLatency" in line and not metrics.get("AverageLatency"):
                    match = re.search(r"AverageLatency\(us\), ([0-9.]+)", line)
                    if match:
                        metrics["AverageLatency"] = float(match.group(1))

                # Extract min latency
                if "MinLatency" in line and not metrics.get("MinLatency"):
                    match = re.search(r"MinLatency\(us\), ([0-9.]+)", line)
                    if match:
                        metrics["MinLatency"] = float(match.group(1))

                # Extract max latency
                if "MaxLatency" in line and not metrics.get("MaxLatency"):
                    match = re.search(r"MaxLatency\(us\), ([0-9.]+)", line)
                    if match:
                        metrics["MaxLatency"] = float(match.group(1))

                # Extract percentile latencies
                percentile_match = re.search(
                    r"(\d+)thPercentileLatency\(us\), ([0-9.]+)", line
                )
                if percentile_match:
                    percentile = percentile_match.group(1)
                    value = float(percentile_match.group(2))
                    metrics[f"P{percentile}Latency"] = value

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return metrics


def process_files(base_dir):
    """Process all YCSB output files and compile results into a DataFrame."""
    results = []

    for path in glob.glob(f"{base_dir}/**/*", recursive=True):
        if os.path.isfile(path):
            workload_type = None
            system_config = None
            threads = None

            # Parse workload type, system config, and thread count from path
            path_parts = Path(path).parts
            if len(path_parts) >= 3:
                workload_type = path_parts[-3]
                system_config = path_parts[-2]
                for part in path_parts:
                    if "threads" in part:
                        match = re.search(r"(\d+)threads", part)
                        if match:
                            threads = match.group(1)

            if workload_type and system_config and threads:
                metrics = extract_metrics(path)
                if metrics:
                    metrics["Workload"] = workload_type
                    metrics["SystemConfig"] = system_config
                    metrics["Threads"] = threads
                    metrics["FilePath"] = path
                    results.append(metrics)

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        return pd.DataFrame()


def extract_and_save_metrics(base_dir="output"):
    """Extract metrics from YCSB output files and save to CSV. Return the DataFrame."""
    results_df = process_files(base_dir)

    if not results_df.empty:
        # Sort by number of threads
        results_df["Threads"] = results_df["Threads"].astype(int)
        results_df = results_df.sort_values(by="Threads")

        # Save to CSV
        results_df.to_csv("ycsb_metrics.csv", index=False)
        print(f"Extracted metrics saved to ycsb_metrics.csv")

        # Display some basic stats
        print("\nSummary:")
        for workload in results_df["Workload"].unique():
            workload_df = results_df[results_df["Workload"] == workload]
            print(f"\n{workload.upper()} Workload:")
            for config in workload_df["SystemConfig"].unique():
                config_df = workload_df[workload_df["SystemConfig"] == config]
                print(f"  {config}:")
                for _, row in config_df.iterrows():
                    print(
                        f"    {row['Threads']} threads: {row['Throughput']:.2f} ops/sec, "
                        f"Avg: {row['AverageLatency']:.2f}us, "
                        f"Min: {row.get('MinLatency', 'N/A')}us, "
                        f"Max: {row.get('MaxLatency', 'N/A')}us, "
                        f"P95: {row.get('P95Latency', 'N/A')}us, "
                        f"P99: {row.get('P99Latency', 'N/A')}us"
                    )

        return results_df
    else:
        print("No data found or extracted.")
        return pd.DataFrame()


if __name__ == "__main__":
    extract_and_save_metrics()
