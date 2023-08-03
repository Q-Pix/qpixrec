#!/usr/bin/env python

#User must load foss/2020b and source qpix-setup before running this script

####GLOBAL DEFINITIONS####
import sys

file_path = sys.argv[1]
root_file = file_path + sys.argv[2]

dfoutput_dir = file_path + "rtd_dataframes/"
functional_form_dir = file_path + "functional_form/"
functional_form_file = functional_form_dir + "functional_form.txt"
t0_hitmaker_dir = file_path + "t0_hitmaker/"

#total events in root file (standard = 100)
total_events = "100"

#number of minimum resets for 'good' event (standard = 8)
num_resets = "8"

# Define bin width to be used for distribution plots and Gaussian fits
binWidth = "10e-06"

##########################

import subprocess
import os
output_dirs = [
    dfoutput_dir,
    functional_form_dir,
    t0_hitmaker_dir
]

for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"The directory '{output_dir}' already exists.")

def execute_script(script_name, *args):
    cmd = ["python", script_name, *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Script '{script_name}' executed successfully!")
        print("Output:")
        print(result.stdout)
    else:
        print(f"An error occurred while running '{script_name}'.")
        print("Error message:")
        print(result.stderr)

if __name__ == "__main__":
    # List of analysis scripts along with there system arguments
    scripts_to_run = [
        ("root_to_pandas.py", root_file, dfoutput_dir, total_events, num_resets),
        ("functional_form.py", dfoutput_dir, functional_form_file, total_events, num_resets), 
        ("t0_hitmaker.py", dfoutput_dir, functional_form_file, t0_hitmaker_dir, total_events, binWidth)
    ]

    for script_info in scripts_to_run:
        script_name, *args = script_info
        print(f"Running script '{script_name}'")
        execute_script(script_name, *args)
