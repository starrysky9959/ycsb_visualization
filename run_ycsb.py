#!/usr/bin/env python3
# filepath: /home/starrysky/workspace/ycsb_visualization/run_ycsb.py

import argparse
import os
import subprocess
import sys
from pathlib import Path

def setup_parser():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run YCSB benchmarks against MongoDB/Eloquent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('workload', 
                       help='Workload type (e.g., read, insert, update)')
    
    parser.add_argument('engine', 
                       help='Database engine (e.g., mongo, eloq)')
    
    parser.add_argument('threads', type=int,
                       help='Number of threads to use for the benchmark')
    
    parser.add_argument('--load', action='store_true',
                       help='Run the load phase before executing workload')
    
    parser.add_argument('--ycsb-dir', default='./ycsb-mongodb-binding-0.17.0',
                       help='Path to YCSB directory')
    
    parser.add_argument('--workload-dir', default='workload',
                       help='Directory containing workload files')
    
    parser.add_argument('--output-dir', default='output',
                       help='Directory for output files')
    
    return parser

def create_output_directory(workload_type, engine, output_base_dir):
    """Create nested output directory structure."""
    # Create directories like output/insert/mongo_1node/
    output_dir = Path(output_base_dir) / workload_type / f"{engine}_1node"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_mongodb_url(engine):
    """Get the appropriate MongoDB connection URL based on engine."""
    if engine == "wiredtiger":
        print("Enable fdatasync for wiredtiger engine per write operation.")
        return "mongodb://localhost:27017/ycsb?w=1&journal=true&maxPoolSize=1000"
    else:
        print("No extra url config for eloq engine")
        return "mongodb://127.0.0.1:27017/ycsb?maxPoolSize=1000"

def run_ycsb_command(cmd, output_file, ycsb_dir):
    """Run a YCSB command and save output to a file."""
    try:
        print(f"Executing: {' '.join(cmd)}")
        print(f"Output will be saved to: {output_file}")
        
        with open(output_file, 'w') as f:
            process = subprocess.run(cmd, 
                                    stdout=f, 
                                    stderr=subprocess.STDOUT, 
                                    cwd=ycsb_dir, 
                                    check=True,
                                    text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def main():
    """Main function to run YCSB benchmarks."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Validate arguments
    workload_type = args.workload.lower()
    engine = args.engine.lower()
    mongodb_url = get_mongodb_url(engine)
    
    # Create output directory structure
    output_dir = create_output_directory(workload_type, engine, args.output_dir)
    
    # Ensure workload file exists
    workload_file_path = f"{args.workload_dir}/workload_{workload_type}"
    if not os.path.exists(workload_file_path):
        print(f"Error: Workload file not found: {workload_file_path}")
        sys.exit(1)
    
    # Run load phase if requested
    if args.load:
        load_output_file = output_dir / f"load_{workload_type}_{engine}"
        load_cmd = [
            "./bin/ycsb", "load", "mongodb", 
            "-s", 
            "-P", f"../{workload_file_path}",
            "-threads", "1", 
            "-p", f"mongodb.url={mongodb_url}"
        ]
        
        success = run_ycsb_command(load_cmd, load_output_file, args.ycsb_dir)
        if not success:
            print("Load phase failed, exiting.")
            sys.exit(1)
    
    # Use thread count directly from args
    t = args.threads
    
    # Run benchmark with specified thread count
    run_output_file = output_dir / f"{workload_type}_{engine}_{t}threads"
    
    run_cmd = [
        "./bin/ycsb", "run", "mongodb", 
        "-s", 
        "-P", f"../{workload_file_path}",
        "-threads", str(t), 
        "-p", f"mongodb.url={mongodb_url}"
    ]
    
    success = run_ycsb_command(run_cmd, run_output_file, args.ycsb_dir)
    if not success:
        print(f"Benchmark failed for {t} threads.")
        sys.exit(1)

if __name__ == "__main__":
    main()