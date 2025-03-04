# YCSB Visualization

A tool for extracting and visualizing metrics from Yahoo! Cloud Serving Benchmark (YCSB) output files. This project helps analyze the performance of different database configurations across various workload types and thread counts.

## Project Overview

This project consists of two main components:

1. **Data Extraction** (`ycsb_extract.py`): Extracts performance metrics from YCSB output files.
2. **Data Visualization** (`ycsb_compare_plot.py`): Creates comparative charts to visualize the performance metrics.

## Directory Structure

The tool expects YCSB output files to be organized in the following directory structure:

```
output/
├── [workload_type]/         # e.g., read, insert, update, scan
    ├── [system_config]/     # e.g., mongo, eloq_1ebs_2log
        ├── [file]           # Contains metrics for a specific thread count
```

Where the filename or directory path contains thread count information (e.g., `32threads`).

## Scripts

### ycsb_extract.py

Extracts performance metrics from YCSB output files and saves them to a CSV file.

**Extracted Metrics:**
- Throughput (operations per second)
- Average latency (μs)
- Minimum latency (μs)
- Maximum latency (μs)
- 95th percentile latency (μs)
- 99th percentile latency (μs)

**Usage:**
```bash
python ycsb_extract.py
```

**Output:**
- `ycsb_metrics.csv`: CSV file containing all extracted metrics
- Terminal summary of found data

### ycsb_compare_plot.py

Creates visualization plots to compare performance across different system configurations.

**Plots Generated:**
1. **Throughput vs. Latency Plot**: Bar chart for throughput and line chart for latency, grouped by thread count.
2. **Latency Details Plot**: Multiple subplots showing different latency metrics (Min, Average, P95, P99, Max).

**Usage:**
```bash
python ycsb_compare_plot.py
```

**Output:**
- `plots/[workload_type]_throughput_vs_latency.png`: Combined throughput and latency chart
- `plots/[workload_type]_latency_details.png`: Detailed latency metrics chart

## Configuration Support

This tool is configured to identify and process the following database configurations:
- MongoDB (`mongo`)
- EloqDoc with different EBS/log configurations:
  - `eloq_1ebs_2log`
  - `eloq_1ebs3log`
  - `eloq_2ebs4log`
  - `eloq_3ebs6log`

## Getting Started

1. Place your YCSB output files in the appropriate directory structure.
2. Run the extraction script to generate the CSV data:
   ```bash
   python ycsb_extract.py
   ```
3. Run the visualization script to generate the plots:
   ```bash
   python ycsb_compare_plot.py
   ```
4. View the generated plots in the `plots/` directory.

## Requirements

- Python 3.x
- pandas
- matplotlib
- numpy
