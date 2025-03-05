import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse

def load_metrics_data(csv_path='ycsb_metrics.csv'):
    """Load the YCSB metrics from CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run ycsb_extract.py first.")
        return None
        
    df = pd.read_csv(csv_path)
    # Ensure threads are integers
    df['Threads'] = df['Threads'].astype(int)
    return df

def sort_configs(configs, order=None):
    """
    Sort configurations in the specified order.
    
    Args:
        configs: List of configuration names to sort
        order: Optional custom order list. If None, uses default order.
    """
    # Define the default order if none provided
    if order is None:
        order = ['mongo', 'eloq_1ebs_2log', 'eloq_1ebs3log', 'eloq_2ebs4log', 'eloq_3ebs6log']
    
    # Sort configs based on the specified order
    sorted_configs = sorted(configs, key=lambda x: next((i for i, item in enumerate(order) if item in x), len(order)))
    return sorted_configs

def plot_throughput_and_latency(df, workload_type, output_dir='plots', config_order=None):
    """
    Create bar + line chart showing throughput and average latency for different configurations.
    
    Args:
        df: DataFrame with YCSB metrics
        workload_type: Type of workload (read, insert, update, scan)
        output_dir: Directory to save the plot
        config_order: Optional custom ordering for configurations
    """
    if df.empty:
        return
        
    # Filter data for the specific workload
    workload_df = df[df['Workload'] == workload_type].copy()
    if workload_df.empty:
        print(f"No data for workload {workload_type}")
        return
    
    # Sort by thread count
    workload_df = workload_df.sort_values(by='Threads')
    
    # Get unique configurations and thread counts
    configs = workload_df['SystemConfig'].unique()
    
    # Sort configurations in the desired order
    configs = sort_configs(configs, config_order)
    
    threads = sorted(workload_df['Threads'].unique())
    
    # Check if we have enough data
    data_counts = {}
    for config in configs:
        config_df = workload_df[workload_df['SystemConfig'] == config]
        data_counts[config] = len(config_df)
    
    if all(count < 2 for count in data_counts.values()):
        print(f"Insufficient data for {workload_type} workload. Skipping.")
        return
    
    # Adjust figure size based on number of configurations
    # Make it wider if we have many configs
    fig_width = min(12 + max(0, len(configs) - 4), 18)
    fig, ax1 = plt.subplots(figsize=(fig_width, 7))
    ax2 = ax1.twinx()
    
    # Define width of bars
    total_width = 0.8
    bar_width = total_width / len(configs)
    
    # Set up positions for bars
    positions = np.arange(len(threads))
    
    # Color maps for different configs - use more distinct colors
    color_map_bars = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))
    color_map_lines = plt.cm.plasma(np.linspace(0.1, 0.9, len(configs)))
    
    # Plot bars (throughput) and lines (latency) for each configuration
    for i, config in enumerate(configs):
        config_df = workload_df[workload_df['SystemConfig'] == config]
        
        # Group by threads and get mean values for numeric columns only
        numeric_columns = ['Throughput', 'AverageLatency', 'MinLatency', 'MaxLatency', 'P95Latency', 'P99Latency']
        grouped = config_df.groupby('Threads')[numeric_columns].mean().reindex(threads)
        
        # Get x positions for this config's bars - center the group of bars at each thread position
        # Calculate position to center the bars at each thread tick
        bar_positions = positions + (i - len(configs)/2 + 0.5) * bar_width
        
        # Plot throughput bars - filter out NaN values
        valid_indices = ~np.isnan(grouped['Throughput'].values)
        valid_positions = bar_positions[valid_indices]
        valid_values = grouped['Throughput'].values[valid_indices]
        
        if len(valid_positions) > 0 and len(valid_values) > 0:
            bars = ax1.bar(valid_positions, valid_values, 
                     width=bar_width, label=f'{config} (Throughput)', 
                     alpha=0.8, color=color_map_bars[i])
        
        # Plot average latency line - filter out NaN values
        valid_indices = ~np.isnan(grouped['AverageLatency'].values)
        valid_pos = positions[valid_indices]
        valid_values = grouped['AverageLatency'].values[valid_indices]
        
        if len(valid_pos) > 0 and len(valid_values) > 0:
            line = ax2.plot(valid_pos, valid_values, 
                      marker='o', linestyle='-', linewidth=2.5,
                      label=f'{config} (Latency)',
                      color=color_map_lines[i], markersize=8)
    
    # Set labels and title
    ax1.set_xlabel('Threads')
    ax1.set_ylabel('Throughput (ops/sec)')
    ax2.set_ylabel('Average Latency (μs)')
    plt.title(f'{workload_type.upper()} Workload - Throughput and Latency')
    
    # Set x-ticks to thread counts
    ax1.set_xticks(positions)
    ax1.set_xticklabels(threads)
    
    # Add grid lines for better readability
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Create separate legends for throughput and latency with better positioning
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Calculate optimal number of columns for legends
    if len(configs) <= 3:
        ncol_tp = len(configs)
        ncol_lat = len(configs)
    else:
        ncol_tp = min(len(configs), 3)
        ncol_lat = min(len(configs), 3)
    
    # Create throughput legend with adjusted position - place it higher
    throughput_legend = ax1.legend(lines1, [l.replace(' (Throughput)', '') for l in labels1], 
                                  loc='upper center', 
                                  bbox_to_anchor=(0.5, -0.15), ncol=ncol_tp, 
                                  fontsize=9, frameon=True, facecolor='white', edgecolor='gray',
                                  title='Throughput')
    
    # Create latency legend with adjusted position - place it lower
    latency_legend = ax2.legend(lines2, [l.replace(' (Latency)', '') for l in labels2], 
                              loc='upper center', 
                              bbox_to_anchor=(0.5, -0.28), ncol=ncol_lat, 
                              fontsize=9, frameon=True, facecolor='white', edgecolor='gray',
                              title='Latency')
    
    # Add throughput legend back to the plot
    ax1.add_artist(throughput_legend)
    
    # Adjust layout based on number of configurations - increase bottom margin
    bottom_margin = 0.32 + 0.02 * max(0, len(configs) - 4)  # Increased to accommodate two vertical legends
    plt.subplots_adjust(bottom=bottom_margin)
    plt.tight_layout(rect=[0, bottom_margin-0.2, 1, 0.95])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f'{output_dir}/{workload_type}_throughput_and_latency.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/{workload_type}_throughput_and_latency.png")
    plt.close()

def plot_latency_details(df, workload_type, output_dir='plots', config_order=None):
    """
    Create subplots showing different latency metrics for different configurations.
    
    Args:
        df: DataFrame with YCSB metrics
        workload_type: Type of workload (read, insert, update, scan) 
        output_dir: Directory to save the plot
        config_order: Optional custom ordering for configurations
    """
    if df.empty:
        return
        
    # Filter data for the specific workload
    workload_df = df[df['Workload'] == workload_type].copy()
    if workload_df.empty:
        print(f"No data for workload {workload_type}")
        return
    
    # Sort by thread count
    workload_df = workload_df.sort_values(by='Threads')
    
    # Get unique configurations and thread counts
    configs = workload_df['SystemConfig'].unique()
    
    # Sort configurations in the desired order
    configs = sort_configs(configs, config_order)
    
    threads = sorted(workload_df['Threads'].unique())
    
    # Check if we have enough data
    data_counts = {}
    for config in configs:
        config_df = workload_df[workload_df['SystemConfig'] == config]
        data_counts[config] = len(config_df)
    
    if all(count < 2 for count in data_counts.values()):
        print(f"Insufficient data for {workload_type} workload. Skipping.")
        return
    
    # Define latency metrics to plot
    latency_metrics = ['MinLatency', 'AverageLatency', 'P95Latency', 'P99Latency', 'MaxLatency']
    
    # Create figure with subplots - reduce vertical spacing
    fig, axes = plt.subplots(len(latency_metrics), 1, figsize=(12, 15), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Color map for different configs - use a more distinctive colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(configs)*2))
    # Only take every other color to increase contrast
    colors = colors[::2][:len(configs)]
    
    # Plot each latency metric
    for i, metric in enumerate(latency_metrics):
        ax = axes[i]
        
        # Plot data for each configuration
        for j, config in enumerate(configs):
            config_df = workload_df[workload_df['SystemConfig'] == config]
            
            # Group by threads and get mean values for numeric columns only
            numeric_columns = ['Throughput', 'AverageLatency', 'MinLatency', 'MaxLatency', 'P95Latency', 'P99Latency']
            if metric in config_df.columns:
                grouped = config_df.groupby('Threads')[numeric_columns].mean().reindex(threads)
                
                # Check if metric exists in the data
                if metric in grouped.columns:
                    # Instead of plotting against thread values directly, plot against indices
                    # This ensures equal spacing regardless of thread value distribution
                    thread_indices = {t: i for i, t in enumerate(threads)}
                    valid_indices = ~np.isnan(grouped[metric].values)
                    valid_threads = np.array([thread_indices[t] for t in np.array(threads)[valid_indices]])
                    valid_values = grouped[metric].values[valid_indices]
                    
                    if len(valid_threads) > 0 and len(valid_values) > 0:
                        # Plot the metric line with more distinctive appearance
                        ax.plot(valid_threads, valid_values, 
                                marker='o', linestyle='-', linewidth=2.0,
                                label=config, color=colors[j], markersize=8)
        
        # Configure the subplot
        ax.set_ylabel(f'{metric} (μs)')
        ax.set_title(f'{metric}')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set x-axis to use discrete thread values with equal spacing
        ax.set_xticks(range(len(threads)))
        ax.set_xticklabels(threads)
        ax.set_xlim(-0.5, len(threads) - 0.5)
        
        # Use log scale for y-axis if the range is large
        if ax.get_ylim()[1] / max(ax.get_ylim()[0], 1) > 100:
            ax.set_yscale('log')
    
    # Create a single legend for the entire figure - use first subplot to avoid duplicates
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Remove any duplicate entries from the legend
    by_label = dict(zip(labels, handles))
    unique_handles = list(by_label.values())
    unique_labels = list(by_label.keys())
    
    # Add the common legend at the bottom of the figure
    fig.legend(unique_handles, unique_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), ncol=min(len(configs), 4),
               fontsize=10, frameon=True, facecolor='white',
               edgecolor='lightgray')
    
    # Adjust the layout to make room for the legend and reduce top spacing
    plt.subplots_adjust(right=0.95, bottom=0.08, top=0.92)
    
    # Set common x-label
    axes[-1].set_xlabel('Threads')
    
    # Set the main title with reduced padding
    fig.suptitle(f'{workload_type.upper()} Workload - Latency Metrics', 
                 fontsize=16, y=0.98)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    plt.savefig(f'{output_dir}/{workload_type}_latency_details.png', dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/{workload_type}_latency_details.png")
    plt.close()

def generate_all_plots(config_order=None):
    """
    Generate all plots for all workload types.
    
    Args:
        config_order: Optional list defining the order of configurations in plots
    """
    # Load the metrics data
    df = load_metrics_data()
    if df is None or df.empty:
        print("No data to plot. Make sure ycsb_metrics.csv exists.")
        return
        
    # Create output directory
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all workload types
    workload_types = df['Workload'].unique()
    
    # Generate plots for each workload type
    for workload in workload_types:
        print(f"\nGenerating plots for {workload} workload...")
        plot_throughput_and_latency(df, workload, output_dir, config_order)
        plot_latency_details(df, workload, output_dir, config_order)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate YCSB visualization plots with custom configuration order')
    parser.add_argument('--order', nargs='+', help='Custom order for configurations (space-separated list)')
    
    args = parser.parse_args()
    
    # Generate plots with the specified order (if provided)
    generate_all_plots(config_order=args.order)
