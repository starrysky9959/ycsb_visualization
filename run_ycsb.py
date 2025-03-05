#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import yaml
import traceback
from pathlib import Path

def setup_parser():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run YCSB benchmarks against MongoDB/Eloquent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Make config the only required parameter
    parser.add_argument('--config', required=True,
                       help='Path to YAML config file containing all parameters')
    
    # Optional parameter to generate a template config
    parser.add_argument('--generate-config', metavar='FILE',
                       help='Generate a template config file at the specified path and exit')
    
    return parser

def generate_template_config(config_path):
    """Generate a template YAML configuration file."""
    template_config = {
        # Database settings
        'server_ip': 'localhost',
        'engine': 'mongo',
        'deployment': '1node',
        
        # Workload settings
        'workload': 'read',
        'threads': 16,
        'load': True,
        
        # Directory settings
        'ycsb_dir': './ycsb-mongodb-binding-0.17.0',
        'workload_dir': 'workload',
        'output_dir': 'output'
    }
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False)
        print(f"Template configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error generating config template: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def load_config_from_yaml(config_path):
    """Load parameters from a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None

def validate_config(config):
    """Validate that all required parameters are present in the config."""
    required_params = ['workload', 'engine', 'threads']
    missing_params = [param for param in required_params if param not in config]
    
    if missing_params:
        print(f"Error: Missing required parameters in config: {', '.join(missing_params)}")
        return False
    
    # Set defaults for optional parameters if not provided
    defaults = {
        'load': False,
        'server_ip': 'localhost',
        'deployment': '1node',
        'ycsb_dir': './ycsb-mongodb-binding-0.17.0',
        'workload_dir': 'workload',
        'output_dir': 'output'
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
            print(f"Using default value for {key}: {value}")
            
    # Expand tilde in paths
    if 'ycsb_dir' in config and '~' in config['ycsb_dir']:
        config['ycsb_dir'] = os.path.expanduser(config['ycsb_dir'])
        print(f"Expanded ycsb_dir path to: {config['ycsb_dir']}")
    
    return True

def create_output_directory(workload_type, engine, output_base_dir, deployment):
    """Create nested output directory structure."""
    # Create directories like output/insert/mongo_deployment/
    output_dir = Path(output_base_dir) / workload_type / f"{engine}_{deployment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_mongodb_url(engine, server_ip):
    """Get the appropriate MongoDB connection URL based on engine."""
    if engine == "mongo":
        print("Enable fdatasync for wiredtiger engine per write operation.")
        return f"mongodb://{server_ip}:27017/ycsb?w=1&journal=true&maxPoolSize=10000"
    else:
        print("No extra url config for eloq engine")
        return f"mongodb://{server_ip}:27017/ycsb?maxPoolSize=10000"

def run_ycsb_command(cmd, output_file, ycsb_dir):
    """
    Run a YCSB command and save output to a file while also displaying on console.
    All output is flushed immediately both to file and terminal.
    """
    try:
        print(f"Executing: {' '.join(cmd)}")
        
        # Start the process but capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=ycsb_dir,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Open output file if specified
        f = None
        if output_file:
            print(f"Output will be saved to: {output_file}")
            f = open(output_file, 'w', buffering=1)  # Line buffered
        
        # Read and process output line by line
        for line in iter(process.stdout.readline, ''):
            # Print to terminal
            sys.stdout.write(line)
            sys.stdout.flush()  # Flush terminal output
            
            # Write to file if specified
            if f:
                f.write(line)
                f.flush()  # Flush file output
        
        # Close file if opened
        if f:
            f.close()
        
        # Wait for process to complete
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Command failed with exit code {return_code}")
            return False
        
        return True
    except Exception as e:
        print(f"Error executing command: {e}")
        print("Full traceback:")
        traceback.print_exc()
        if 'f' in locals() and f:
            f.close()
        return False

def main():
    """Main function to run YCSB benchmarks."""
    try:
        parser = setup_parser()
        args = parser.parse_args()
        
        # Generate template config if requested
        if args.generate_config:
            if generate_template_config(args.generate_config):
                print("Template config generated successfully. Exiting.")
            else:
                print("Failed to generate template config.")
            sys.exit(0)
        
        # Load config from YAML
        config = load_config_from_yaml(args.config)
        if not config:
            print("Failed to load configuration. Exiting.")
            sys.exit(1)
        
        # Validate config
        if not validate_config(config):
            print("Configuration validation failed. Exiting.")
            sys.exit(1)
        
        # Extract parameters from config
        workload_type = config['workload'].lower()
        engine = config['engine'].lower()
        threads = config['threads']
        mongodb_url = get_mongodb_url(engine, config['server_ip'])
        
        # Expand any ~ in directory paths
        ycsb_dir = os.path.expanduser(config['ycsb_dir'])
        workload_dir = config['workload_dir']
        output_dir = config['output_dir']
        
        # Create output directory structure with deployment parameter
        output_dir = create_output_directory(workload_type, engine, output_dir, config['deployment'])
        
        # Ensure workload file exists and get its absolute path
        workload_file_path = f"{config['workload_dir']}/{workload_type}"
        if not os.path.exists(workload_file_path):
            print(f"Error: Workload file not found: {workload_file_path}")
            sys.exit(1)
        
        # Convert to absolute path
        workload_absolute_path = os.path.abspath(workload_file_path)
        
        # Run load phase if requested
        if config['load']:
            # Load phase output goes to console, not to file
            load_cmd = [
                "./bin/ycsb", "load", "mongodb", 
                "-s", 
                "-P", workload_absolute_path,
                "-threads", "1", 
                "-p", f"mongodb.url={mongodb_url}"
            ]
            
            success = run_ycsb_command(load_cmd, None, ycsb_dir)
            if not success:
                print("Load phase failed, exiting.")
                sys.exit(1)
        
        # Run benchmark with specified thread count
        run_output_file = output_dir / f"{workload_type}_{engine}_{threads}threads"
        
        run_cmd = [
            "./bin/ycsb", "run", "mongodb", 
            "-s", 
            "-P", workload_absolute_path,
            "-threads", str(threads), 
            "-p", f"mongodb.url={mongodb_url}"
        ]
        
        success = run_ycsb_command(run_cmd, run_output_file, ycsb_dir)
        if not success:
            print(f"Benchmark failed for {threads} threads.")
            sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()