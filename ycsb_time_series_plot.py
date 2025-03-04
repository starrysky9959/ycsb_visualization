import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_metrics_time_series(file_path):
    """Extract throughput and latency metrics time series from YCSB output file."""
    metrics = {
        'timestamp': [],
        'throughput': [],
        'avg_latency': [],
        'min_latency': [],
        'max_latency': [],
        'p90_latency': [],
        'p99_latency': [],
        'p999_latency': [],
        'p9999_latency': []
    }
    
    # Regular expression pattern to match the metrics line
    pattern = r'(\d+) sec: \d+ operations; (\d+\.\d+) current ops/sec.+\[.+: Count=\d+, Max=(\d+), Min=(\d+), Avg=(\d+\.\d+), 90=(\d+), 99=(\d+), 99.9=(\d+), 99.99=(\d+)\]'
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    metrics['timestamp'].append(int(match.group(1)))
                    metrics['throughput'].append(float(match.group(2)))
                    metrics['max_latency'].append(int(match.group(3)))
                    metrics['min_latency'].append(int(match.group(4)))
                    metrics['avg_latency'].append(float(match.group(5)))
                    metrics['p90_latency'].append(int(match.group(6)))
                    metrics['p99_latency'].append(int(match.group(7)))
                    metrics['p999_latency'].append(int(match.group(8)))
                    metrics['p9999_latency'].append(int(match.group(9)))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return metrics

def plot_metrics(metrics, output_path):
    """Generate plots for throughput and latency metrics."""
    if not metrics['timestamp']:
        print(f"No metrics found for {output_path}")
        return
        
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot throughput
    axs[0].plot(metrics['timestamp'], metrics['throughput'], marker='o', markersize=3, linewidth=1.5)
    axs[0].set_title('Throughput over time')
    axs[0].set_ylabel('Operations per second')
    axs[0].set_ylim(bottom=0)
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot average, min and p90 latency
    axs[1].plot(metrics['timestamp'], metrics['avg_latency'], marker='o', markersize=3, linewidth=1.5, label='Average')
    axs[1].plot(metrics['timestamp'], metrics['min_latency'], marker='.', markersize=2, linewidth=1, label='Min')
    axs[1].plot(metrics['timestamp'], metrics['p90_latency'], marker='.', markersize=2, linewidth=1, label='p90')
    axs[1].set_title('Common latency metrics over time')
    axs[1].set_ylabel('Latency (μs)')
    axs[1].set_ylim(bottom=0)
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot high percentile latency
    axs[2].plot(metrics['timestamp'], metrics['p99_latency'], marker='.', markersize=2, linewidth=1, label='p99')
    axs[2].plot(metrics['timestamp'], metrics['p999_latency'], marker='.', markersize=2, linewidth=1, label='p99.9')
    axs[2].plot(metrics['timestamp'], metrics['p9999_latency'], marker='.', markersize=2, linewidth=1, label='p99.99')
    axs[2].plot(metrics['timestamp'], metrics['max_latency'], marker='.', markersize=2, linewidth=1, label='Max', alpha=0.7)
    axs[2].set_title('High percentile latency over time')
    axs[2].set_ylabel('Latency (μs)')
    axs[2].set_xlabel('Time elapsed (seconds)')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Use logarithmic scale for high percentile latencies which can vary widely
    axs[2].set_yscale('log')
    
    # Set title based on output path
    workload_type = output_path.parent.parent.name  # e.g., "insert", "read", "update"
    system_config = output_path.parent.name  # e.g., "mongo_1node", "eloq_1disk3log"
    threads = re.search(r'(\d+)threads', output_path.name).group(1) if re.search(r'(\d+)threads', output_path.name) else "unknown"
    
    plt.suptitle(f"{workload_type.upper()} Workload - {system_config} with {threads} threads", fontsize=16)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}.png")
    plt.close()

def process_all_log_files(base_dir="output"):
    """Process all YCSB log files in the output directory."""
    # Find all log files
    base_path = Path(base_dir)
    
    # Walk through directories
    for workload_type in base_path.iterdir():
        if not workload_type.is_dir():
            continue
            
        for system_config in workload_type.iterdir():
            if not system_config.is_dir():
                continue
                
            for log_file in system_config.iterdir():
                if log_file.is_file() and not log_file.name.endswith('.png'):
                    print(f"Processing {log_file}")
                    
                    # Extract metrics
                    metrics = extract_metrics_time_series(log_file)
                    
                    # Create plot
                    plot_metrics(metrics, log_file)

if __name__ == "__main__":
    process_all_log_files()