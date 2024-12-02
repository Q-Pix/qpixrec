#!/usr/bin/env python

# -----------------------------------------------------------------------------
# t0_hitmaker.py
#
# Determines a t0 and reconstructs the Z positions from RMS and Mean of CDF fits
# * Author: Carter Eikenbary
# * Creation date: 2 December 2024
#
# Usage: python /path/to/t0_hitmaker.py /path/to/t0_hitmaker/output/ verbosity
# Notes: HPRC users must load foss/2022b and source qpix-setup before running this script
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy import stats

from scipy.optimize import curve_fit
from scipy.optimize import minimize

import sys

import warnings
warnings.filterwarnings("ignore")

from cdf_definitions import *

dfoutput_dir = sys.argv[1]
t0_hitmaker_dir = sys.argv[2]
reset_threshold = int(sys.argv[3])
reset_min = int(sys.argv[4])
reset_max = int(sys.argv[5])

diff_L = 6.8223 #cm**2/s
elec_vel = 164800 #cm**2/s
expected_const = np.sqrt(2*diff_L/elec_vel**2)
energy_per_hit = (23.6*reset_threshold)*(1e-6)

##########################
rtd_df = pd.read_pickle(dfoutput_dir + "rtd_df.pkl").reset_index(drop = True)
total_events = int(max(rtd_df.event) + 1)

rtd_t0candidate_df = rtd_df[(rtd_df.nResets >= reset_min) & (rtd_df.nResets <= reset_max)] 
rtd_allpix_df =  rtd_df[(rtd_df.nResets >= 2)]

t0_df = pd.DataFrame()

singlehit_df = pd.DataFrame()
doublehit_df = pd.DataFrame()
triplehit_df = pd.DataFrame()
unfitpix_df = pd.DataFrame()

##########################
for n in range(total_events):
    print("//////////////////////////")
    print("Event =", n)

    rtd_t0candidate_eventdf = rtd_t0candidate_df[(rtd_t0candidate_df.event == n)]
    
    singlecdf_noshift_results = process_singlecdf(rtd_t0candidate_eventdf)
    singlecdf_diff = singlecdf_noshift_results['Diff']

    singlecdf_diff_cut = singlecdf_diff[singlecdf_diff < (np.median(singlecdf_diff) + (np.median(singlecdf_diff) - np.min(singlecdf_diff)))]
    singlecdf_diff_cut = singlecdf_diff_cut[singlecdf_diff_cut < (np.median(singlecdf_diff_cut) + (np.median(singlecdf_diff_cut) - np.min(singlecdf_diff_cut)))]
    print("well-measured pixels =", len(singlecdf_diff_cut))
    if len(singlecdf_diff_cut) == 0:
        print("Skipping Event, no well-measured pixels")
        continue
        
    hist, bin_edges = np.histogram(singlecdf_diff_cut, bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    p0 = [max(hist), np.median(bin_centers), np.std(bin_centers)]
    
    if len(singlecdf_diff_cut) > 100:
        try:
            popt, pcov = curve_fit(single_gaussian, bin_centers, hist, p0=p0)
            diff_amp = popt[0]
            diff_mean = popt[1]
            diff_std = popt[2]
            diff_low = diff_mean - 1*diff_std
            diff_high = diff_mean + 1*diff_std
        
        except RuntimeError:
            print("Gaussian Fitting of delta(sigma) distribution failed \n Using NumPy estimations")
            diff_mean = np.median(singlecdf_diff_cut)
            diff_std = np.std(singlecdf_diff_cut)
            diff_low = diff_mean - 1*diff_std
            diff_high = diff_mean + 1*diff_std
    else:
        print("Not enough well-measured pixels for Gaussian Fitting of delta(sigma) distribution \n Using NumPy estimations")
        diff_mean = np.median(singlecdf_diff_cut)
        diff_std = np.std(singlecdf_diff_cut)
        diff_low = diff_mean - 2*diff_std
        diff_high = diff_mean + 2*diff_std
    
    t0_event = singlecdf_noshift_results[(singlecdf_diff > diff_low) & (singlecdf_diff < diff_high)]
    t0_df = t0_df.append(t0_event, ignore_index=True)
    
    meanvaln = t0_event['Mean']
    stdvaln = t0_event['StD']
    
    def objective(t0_shift):
        RMS_Expected = expected_const * np.sqrt(meanvaln - t0_shift)
        difference = (stdvaln - RMS_Expected)
            
        weighted_avg = np.mean(difference)
        return (weighted_avg ** 2)
      
          # Initial guess for t0_shift (for simulation t0=0)
    initial_t0_shift = 0
    
    # Define the optimization tolerance for higher precision
    tolerance = 1e-15    
    # Define bounds for t0_shift
    bounds = [(-0.001, min(meanvaln))]

    # Perform the nonlinear optimization using the "Nelder-Mead" algorithm
    result = minimize(objective, initial_t0_shift, method='Nelder-Mead', bounds=bounds, tol=tolerance)

    # Get the optimal t0_shift value for the current event
    optimal_t0_shift = result.x[0]
    print("t0 = ", optimal_t0_shift) 
    
    singlecdf_shifted_results = process_singlecdf_t0shifted(rtd_t0candidate_eventdf, optimal_t0_shift)
    singlecdf_diff_shifted = singlecdf_shifted_results['Diff']

    singlecdf_diff_shifted_cut =  singlecdf_diff_shifted[ singlecdf_diff_shifted < (np.median(singlecdf_diff_shifted) + (np.median( singlecdf_diff_shifted) - np.min(singlecdf_diff_shifted)))]
    singlecdf_diff_shifted_cut = singlecdf_diff_shifted_cut[singlecdf_diff_shifted_cut < (np.median(singlecdf_diff_shifted_cut) + (np.median(singlecdf_diff_shifted_cut) - np.min(singlecdf_diff_shifted_cut)))]
    hist, bin_edges = np.histogram(singlecdf_diff_shifted_cut, bins=10)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2    
    p0 = [max(hist), np.median(bin_centers), np.std(bin_centers)]
    try:
        popt, pcov = curve_fit(single_gaussian, bin_centers, hist, p0=p0)
        diff_amp = popt[0]
        diff_mean = popt[1]
        diff_std = popt[2]
    except RuntimeError:
        print("Gaussian Fitting of delta(sigma) distribution failed")  
        
    
    rtd_allpix_eventdf = rtd_allpix_df[rtd_allpix_df.event == n]
    rtd_allpix_eventdf['t0'] = optimal_t0_shift
    print("pixels in event = ", len(rtd_df[rtd_df.event ==n]))
    
    singlehit_event = process_singlehit(rtd_allpix_eventdf, optimal_t0_shift)
    singlehit_percentile_high = np.percentile(singlehit_event[singlehit_event.Amp >= 3]['Avg_Residual'], 85)   
    
    singlehit_cut = np.median(singlehit_event[(singlehit_event.Avg_Residual <= singlehit_percentile_high)]['Avg_Residual']) + 5*np.std(singlehit_event[(singlehit_event.Avg_Residual <= singlehit_percentile_high)]['Avg_Residual'], ddof=1)
    
    singlehit_event = singlehit_event[singlehit_event.Avg_Residual < singlehit_cut]   
    
    print("single hit pixels in event = ", len(singlehit_event))
    
    singlehit_event['t0'] = optimal_t0_shift
    singlehit_df = singlehit_df.append(singlehit_event, ignore_index=True)
    
    notsinglehit_event = rtd_allpix_eventdf.merge(
        singlehit_event[['event', 'PixelID']],
        on=['event', 'PixelID'],
        how='left',
        indicator=True
    )

    notsinglehit_event = notsinglehit_event[notsinglehit_event['_merge'] == 'left_only'].drop(columns=['_merge'])

    doublehit_event = process_doublehit(notsinglehit_event[notsinglehit_event.nResets >= 4], optimal_t0_shift)
    doublehit_percentile_high = np.percentile(doublehit_event[(doublehit_event.Amp1 + doublehit_event.Amp2) >= 5]['Avg_Residual'], 85)   
    
    doublehit_cut = np.median(doublehit_event[(doublehit_event.Avg_Residual <= doublehit_percentile_high)]['Avg_Residual']) + 5*np.std(doublehit_event[(doublehit_event.Avg_Residual <= doublehit_percentile_high)]['Avg_Residual'], ddof=1)
    
    doublehit_event = doublehit_event[doublehit_event.Avg_Residual < doublehit_cut]   
    
    print("double hit pixels in event = ", len(doublehit_event))

    doublehit_event['t0'] = optimal_t0_shift
    doublehit_df = doublehit_df.append(doublehit_event, ignore_index=True)
        
    notdoublehit_event = notsinglehit_event.merge(
        doublehit_event[['event', 'PixelID']],
        on=['event', 'PixelID'],
        how='left',
        indicator=True
    )

    notdoublehit_event = notdoublehit_event[notdoublehit_event['_merge'] == 'left_only'].drop(columns=['_merge'])   
    
    triplehit_event = process_triplehit(notdoublehit_event[notdoublehit_event.nResets >= 6], optimal_t0_shift)
    triplehit_percentile_high = np.percentile(triplehit_event[(triplehit_event.Amp1 + triplehit_event.Amp2 + triplehit_event.Amp3) >= 7]['Avg_Residual'], 85)   
    
    triplehit_cut = np.median(triplehit_event[(triplehit_event.Avg_Residual <= triplehit_percentile_high)]['Avg_Residual']) + 5*np.std(triplehit_event[(triplehit_event.Avg_Residual <= triplehit_percentile_high)]['Avg_Residual'], ddof=1)
    
    triplehit_event = triplehit_event[triplehit_event.Avg_Residual < triplehit_cut]  
    
    print("triple hit pixels in event = ", len(triplehit_event))
    
    triplehit_event['t0'] = optimal_t0_shift
    triplehit_df = triplehit_df.append(triplehit_event, ignore_index=True)
    
    unfitpix_event = notdoublehit_event.merge(
        triplehit_event[['event', 'PixelID']],
        on=['event', 'PixelID'],
        how='left',
        indicator=True
    )
    
    unfitpix_event = unfitpix_event[unfitpix_event['_merge'] == 'left_only'].drop(columns=['_merge'])   
    print("unfit pixels = ", len(unfitpix_event))
    print("remaining pixels that could fit quadruple cdf = ", len(unfitpix_event[unfitpix_event.nResets >= 8])) 
    
    unfitpix_event = unfitpix_event.append(rtd_df[(rtd_df.event == n) & (rtd_df.nResets < 2)])   
    unfitpix_event['t0'] = optimal_t0_shift

    unfitpix_df = unfitpix_df.append(unfitpix_event)       
    
#Reformat singlehit_df
singlehit_transformed = singlehit_df[['event', 'PixelID', 'Amp', 'Mean', 't0']]

# Reshape doublehit_df to include only relevant columns, including 't0'
doublehit_reshaped = doublehit_df.melt(
    id_vars=['event', 'PixelID', 't0'],  # Include 't0' in id_vars
    value_vars=['Amp1', 'Mean1', 'Amp2', 'Mean2'],
    var_name='Feature',
    value_name='Value'
)
doublehit_reshaped['Hit'] = doublehit_reshaped['Feature'].str[-1]  # Extract hit number
doublehit_reshaped['Feature'] = doublehit_reshaped['Feature'].str[:-1]  # Remove hit number from feature
doublehit_pivoted = doublehit_reshaped.pivot(
    index=['event', 'PixelID', 'Hit', 't0'], columns='Feature', values='Value'
).reset_index()
doublehit_pivoted = doublehit_pivoted[['event', 'PixelID', 'Amp', 'Mean', 't0']]  # Keep only relevant columns

# Reshape triplehit_df to include relevant columns, including 't0'
triplehit_reshaped = triplehit_df.melt(
    id_vars=['event', 'PixelID', 't0'],  # Include 't0' in id_vars
    value_vars=['Amp1', 'Mean1', 'Amp2', 'Mean2', 'Amp3', 'Mean3'],
    var_name='Feature',
    value_name='Value'
)
triplehit_reshaped['Hit'] = triplehit_reshaped['Feature'].str[-1]  # Extract hit number
triplehit_reshaped['Feature'] = triplehit_reshaped['Feature'].str[:-1]  # Remove hit number from feature
triplehit_pivoted = triplehit_reshaped.pivot(
    index=['event', 'PixelID', 'Hit', 't0'], columns='Feature', values='Value'
).reset_index()
triplehit_pivoted = triplehit_pivoted[['event', 'PixelID', 'Amp', 'Mean', 't0']]  # Keep only relevant columns

# Concatenate all DataFrames
hits_df = pd.concat(
    [singlehit_transformed, doublehit_pivoted, triplehit_pivoted],
    ignore_index=True
)

# Calculate 'Energy' and 'Z'
hits_df['Energy'] = hits_df['Amp'] * energy_per_hit
hits_df['Z'] = hits_df['Mean'] * elec_vel

# Drop 'Amp' and 'Mean' columns if no longer needed
hits_df = hits_df.drop(columns=['Amp'])
hits_df = hits_df.drop(columns=['Mean'])

# Merge with rtd_df to add 'X' and 'Y'
hits_df = pd.merge(hits_df, rtd_df[['event', 'PixelID', 'pixel_x', 'pixel_y']], on=['event', 'PixelID'], how='left')

# Calculate 'X' and 'Y'
hits_df['X'] = (hits_df['pixel_x'] * 4 - 2)/10 
hits_df['Y'] = (hits_df['pixel_y'] * 4 - 2)/10 

# Drop 'pixel_x' and 'pixel_y' columns
hits_df = hits_df.drop(columns=['pixel_x'])
hits_df = hits_df.drop(columns=['pixel_y'])

hits_df.to_pickle(t0_hitmaker_dir + '/hits_df.pkl')
print("Dataframe written to " + t0_hitmaker_dir + 'hits_df.pkl')

unfitpix_df.to_pickle(t0_hitmaker_dir + '/unfitpix_df.pkl')
print("Dataframe written to " + t0_hitmaker_dir + 'unfitpix_df.pkl')

