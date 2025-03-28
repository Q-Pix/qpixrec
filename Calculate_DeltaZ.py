#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Calculate_DeltaZ.py
#
# Measures the DeltaZ between measured Z above single-hit pixels and the true Z of hits above the pixel
# * Author: Carter Eikenbary
# * Creation date: 27 March 2025
#
# Usage: python /path/to/Calculate_DeltaZ.py /path/to/qpixrec/output/ event_low event_high
# Notes: HPRC users must load foss/2022b and source qpix-setup before running this script
# -----------------------------------------------------------------------------
import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# Create the argument parser
parser = argparse.ArgumentParser(description="Find DeltaZ between single-hit pixel measured Z and true Z.")
    
# Add positional argument for the root file
parser.add_argument("qpixrec_path", type=str, help="Path to the qpixrec output")
    
# Add optional arguments for event_low and event_high
parser.add_argument("event_low", type=int, help="Lowest event number to process")
parser.add_argument("event_high", type=int, help="Highest event number to process")
    
# Parse arguments
args = parser.parse_args()
    
# Extract arguments
qpixrec_path = args.qpixrec_path
event_low = args.event_low
event_high = args.event_high

input_path = os.path.dirname(qpixrec_path)

def std_exp(mean):
    return expected_const * np.sqrt(mean)

def std_difference(mean, std):
    return (std - std_exp(mean))

def single_cdf(x, a, b, c):
    return a * stats.norm.cdf(x, loc=b, scale=c)

def single_cdf_nostd(x, a, b):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b))

def double_cdf_nostd(x, a, b, c, d):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b)) + c * stats.norm.cdf(x, loc=d, scale=std_exp(d))

def triple_cdf_nostd(x, a, b, c, d, e, f):
    return a * stats.norm.cdf(x, loc=b, scale=std_exp(b)) + c * stats.norm.cdf(x, loc=d, scale=std_exp(d)) + e * stats.norm.cdf(x, loc=f, scale=std_exp(f))

def single_gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma)**2)

def print_progress(current, total, event_num, bar_length=50):
    progress = current / total
    bar = "#" * int(progress * bar_length) + "-" * (bar_length - int(progress * bar_length))
    print(f"\rProgress: |{bar}| {current}/{total}, Event {event_num}", end="")
    

diff_L = 6.8223 #cm**2/s
diff_T = 13.1586 #cm**2/s
elec_vel = 164800 #cm**2/s
expected_const = np.sqrt(2*diff_L/elec_vel**2)

#Read in rtd data
rtd_df = pd.read_pickle(input_path + "/rtd_dataframe/rtd_df.pkl").reset_index(drop = True)
g4_df = pd.read_pickle(input_path + "/rtd_dataframe/g4_df.pkl").reset_index(drop = True)
total_events = int(max(rtd_df.event) + 1)

singlehit_df = pd.read_pickle(input_path + "/t0_hitmaker/singlehit_df.pkl").reset_index(drop = True)
hits_df = pd.read_pickle(input_path + "/t0_hitmaker/hits_df.pkl").reset_index(drop = True)

cdf_t0s = []
cdf_Zs = []
g4_Zs = []

for n in range(total_events):
    cdf_t0s.append(hits_df[hits_df.event == n]['t0'].iloc[0])
    cdf_Zs.append(hits_df[hits_df.event == n]['t0'].iloc[0]*-elec_vel)
    g4_Zs.append(np.median(g4_df[(g4_df.event == n) & (g4_df.ParticleID == 1)]['zi']))
    
cdf_t0s = np.array(cdf_t0s)
cdf_Zs = np.array(cdf_Zs)
g4_Zs = np.array(g4_Zs)

############
############

print(f'range of events [{event_low}, {event_high}]')

singlehit_subdf = singlehit_df[(singlehit_df.event >= event_low) & (singlehit_df.event <= event_high)]

event_arr = []
pixelid_arr = []
num_hits_arr = []
delta_Z_arr= []

for i in range(len(singlehit_subdf)):
    event_num = int(singlehit_subdf.iloc[i].event)
    pixel_num = int(singlehit_subdf.iloc[i].PixelID)
    Z_pixel = singlehit_subdf.iloc[i].Mean * elec_vel

    # Define pixel boundaries
    pixel_data = rtd_df[(rtd_df.event == event_num) & (rtd_df.PixelID == pixel_num)].iloc[0]
    pix_x_low = pixel_data.pixel_x * 0.4 - 0.4
    pix_x_high = pixel_data.pixel_x * 0.4
    pix_y_low = pixel_data.pixel_y * 0.4 - 0.4
    pix_y_high = pixel_data.pixel_y * 0.4
    
    # Sets an extended range to 1/2*diff_T distance
    pixel_range_ext = np.sqrt(1 * diff_T * g4_Zs[event_num] / (elec_vel**3)) * elec_vel

    # Find tracks that pass above the pixel
    hit_over_pixel = g4_df[g4_df.event == event_num]
    valid_tracks = []

    for _, track in hit_over_pixel.iterrows():
        xi, xf = track['xi'], track['xf']
        yi, yf = track['yi'], track['yf']

        # Check for intersection with the pixel boundaries extended by pixel_range_ext in the X-Y plane
        if (min(xi, xf) <= pix_x_high + pixel_range_ext and max(xi, xf) >= pix_x_low - pixel_range_ext and
            min(yi, yf) <= pix_y_high + pixel_range_ext and max(yi, yf) >= pix_y_low - pixel_range_ext):
            valid_tracks.append(track)

    if valid_tracks:
        # Interpolate Z values for intersecting tracks
        Z_interpolated = []
        Eweight = []

        for track in valid_tracks:
            xi, xf = track['xi'], track['xf']
            yi, yf = track['yi'], track['yf']
            zi, zf = track['zi'], track['zf']
            En = track['E']

            track_length = np.sqrt((xf - xi)**2 + (yf - yi)**2)
            pixel_overlap_length = 0

            # Interpolate at the edges of the pixel (X dimension)
            for x in [pix_x_low, pix_x_high]:
                if xi != xf and min(xi, xf) <= x <= max(xi, xf):
                    y_int = yi + (x - xi) * (yf - yi) / (xf - xi)
                    if pix_y_low - pixel_range_ext <= y_int <= pix_y_high + pixel_range_ext:
                        z_int = zi + (x - xi) * (zf - zi) / (xf - xi)
                        pixel_overlap_length += np.abs(x - xi)
                        Z_interpolated.append(z_int)
                        Eweight.append(En)

            # Interpolate at the edges of the pixel (Y dimension)
            for y in [pix_y_low, pix_y_high]:
                if yi != yf and min(yi, yf) <= y <= max(yi, yf):
                    x_int = xi + (y - yi) * (xf - xi) / (yf - yi)
                    if pix_x_low - pixel_range_ext <= x_int <= pix_x_high + pixel_range_ext:
                        z_int = zi + (y - yi) * (zf - zi) / (yf - yi)
                        pixel_overlap_length += np.abs(y - yi)
                        Z_interpolated.append(z_int)
                        Eweight.append(En)

            # Scale energy weight by the percentage of the track length within the pixel (extended by pixel_range_ext)
            if pixel_overlap_length > 0:
                Eweight[-1] *= pixel_overlap_length / track_length

        # Calculate average Z for tracks above the pixel
        if Z_interpolated:
            Z_interpolated = np.array(Z_interpolated)
            Eweight = np.array(Eweight)
            num_hits = len(Eweight)
            Z_true = np.average(Z_interpolated, weights=Eweight)
            delta_Z = Z_true - Z_pixel
            event_arr.append(event_num)
            pixelid_arr.append(pixel_num)
            num_hits_arr.append(num_hits)
            delta_Z_arr.append(delta_Z)
            

    # Update progress bar
    print_progress(i, len(singlehit_subdf), int(event_num))

print()

event_arr = np.array(event_arr)
pixelid_arr = np.array(pixelid_arr)
num_hits_arr = np.array(num_hits_arr)
delta_Z_arr = np.array(delta_Z_arr)

filename = f'deltaZs_E{event_low}-E{event_high}.pkl'

with open(filename, 'wb') as f:
    pickle.dump(event_arr, f)
    pickle.dump(pixelid_arr, f)
    pickle.dump(num_hits_arr, f)
    pickle.dump(delta_Z_arr, f)
