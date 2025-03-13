import argparse
import serial
import time
import os
import re
import signal
import sys

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit"""
    print("\nRecording stopped by user.")
    if 'ser' in globals() and ser.is_open:
        ser.close()
    sys.exit(0)

def main(com_port, output_file, baud_rate=115200):
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Make sure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define pattern to match numeric data (6 comma-separated floats)
    data_pattern = re.compile(r'^(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)$')
    
    # Pattern to match header line (case insensitive)
    header_pattern = re.compile(r'^a[xyz],a[xyz],a[xyz],g[xyz],g[xyz],g[xyz]$', re.IGNORECASE)
    
    try:
        # Open serial connection
        ser = serial.Serial(com_port, baud_rate, timeout=1)
        print(f"Connected to {com_port} at {baud_rate} baud")
        
        # Wait for Arduino to reset after connection
        time.sleep(2)
        
        # Clear any initial data
        ser.reset_input_buffer()
        
        # Check if file exists and has content
        file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
        
        # Open the file for writing
        with open(output_file, 'a' if file_exists else 'w') as f:
            # Write header if new file
            if not file_exists:
                f.write("aX,aY,aZ,gX,gY,gZ\n")
                print(f"Created file: {output_file}")
            else:
                print(f"Appending to existing file: {output_file}")
            
            data_count = 0
            debug_count = 0
            header_count = 0
            start_time = time.time()
            
            print("Recording data... Press Ctrl+C to stop.")
            
            while True:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8', errors='replace').strip()
                
                if line:
                    # Check if it's a data line (6 comma-separated values)
                    match = data_pattern.match(line)
                    if match:
                        # It's a data line, write to file
                        f.write(line + "\n")
                        f.flush()  # Make sure data is written immediately
                        data_count += 1
                        
                        # Show progress
                        elapsed = time.time() - start_time
                        if data_count % 100 == 0:
                            print(f"Recorded {data_count} data points ({data_count/elapsed:.2f} points/sec), filtered {debug_count} debug messages, {header_count} headers")
                    elif header_pattern.match(line):
                        # It's a header line, ignore
                        header_count += 1
                        if header_count % 10 == 0:
                            print(f"Filtered {header_count} header lines")
                    else:
                        # It's a debug message, ignore
                        debug_count += 1
                        if debug_count % 100 == 0:
                            print(f"Debug message: {line[:50]}{'...' if len(line) > 50 else ''}")
    
    except serial.SerialException as e:
        print(f"Error opening serial port {com_port}: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from Arduino to CSV file")
    parser.add_argument("com_port", help="Serial port (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)")
    parser.add_argument("output_file", help="Path to output CSV file")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    args = parser.parse_args()
    
    sys.exit(main(args.com_port, args.output_file, args.baud))