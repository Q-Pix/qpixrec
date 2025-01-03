#!/usr/bin/env python

# -----------------------------------------------------------------------------
# t0_hitmaker.py
#
# Determines a t0 and reconstructs the Z positions from RMS and Mean of CDF fits
# * Author: Carter Eikenbary
# * Creation date: 2 December 2024
#
# Usage: python /path/to/t0_hitmaker.py /path/to/t0_hitmaker/output/ -threshold # -rmin # -rmax #
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

def process_singlecdf(df):
    singlecdf_event = []
    singlecdf_pixid = []
    singlecdf_amp = []
    singlecdf_mean = []
    singlecdf_std = []
    singlecdf_diff = []

    for i in range(len(df)):
        reset_times = np.array(df['reset_time'].reset_index().iloc[i][1])
        num_resets = len(reset_times)
        reset_count = np.arange(1, num_resets + 1)  
        
        initial_params = [num_resets + 0.1, np.mean(reset_times), np.std(reset_times)]
        
        try:
            cdf_params, cdf_covariance = curve_fit(single_cdf, reset_times, reset_count, p0=initial_params)
            reset_amp = cdf_params[0]
            reset_mean = cdf_params[1]
            reset_std = cdf_params[2]
            reset_diff = std_difference(reset_mean, reset_std)    
            fitted_values = single_cdf(reset_times, *cdf_params)
            residuals = reset_count - fitted_values
            sum_residuals = np.sum(np.abs(residuals))
            sum_residuals_per_reset = sum_residuals/num_resets                
            
            if ((reset_amp > num_resets) and (reset_amp < num_resets + 1) and (sum_residuals_per_reset < 0.03)):
                singlecdf_event.append(n)
                singlecdf_pixid.append(df.iloc[i].PixelID)
                singlecdf_amp.append(reset_amp)
                singlecdf_mean.append(reset_mean)
                singlecdf_std.append(reset_std)
                singlecdf_diff.append(reset_diff)
            
        except RuntimeError:
            continue
            
    data = {
    'event': singlecdf_event,
    'PixelID': singlecdf_pixid,
    'Amp': singlecdf_amp,
    'Mean': singlecdf_mean,
    'StD': singlecdf_std,
    'Diff': singlecdf_diff,
    }

    return pd.DataFrame(data)

def process_singlehit(df, t0):
    singlecdf_event = []
    singlecdf_pixid = []
    singlecdf_amp = []
    singlecdf_mean = []
    singlecdf_std = []
    singlecdf_residual = []

    for i in range(len(df)):
        reset_times = np.array(df['reset_time'].reset_index().iloc[i][1]) - t0
        num_resets = len(reset_times)
        reset_count = np.arange(1, num_resets + 1)  
        
        initial_params = [num_resets + 0.1, np.mean(reset_times)]
        
        try:
            cdf_params, cdf_covariance = curve_fit(single_cdf_nostd, reset_times, reset_count, p0=initial_params)
            reset_amp = cdf_params[0]
            reset_mean = cdf_params[1]
            reset_std = std_exp(reset_mean)
            fitted_values = single_cdf_nostd(reset_times, *cdf_params)
            residuals = reset_count - fitted_values
            sum_residuals = np.sum(np.abs(residuals))
            sum_residuals_per_reset = sum_residuals/num_resets            
            
            if ((reset_amp > num_resets) and (reset_amp < num_resets + 1) and (sum_residuals_per_reset < 0.1)):
                singlecdf_event.append(n)
                singlecdf_pixid.append(df.iloc[i].PixelID)
                singlecdf_amp.append(reset_amp)
                singlecdf_mean.append(reset_mean)
                singlecdf_std.append(reset_std)
                singlecdf_residual.append(sum_residuals_per_reset)
            
        except RuntimeError:
            continue
            
    data = {
    'event': singlecdf_event,
    'PixelID': singlecdf_pixid,
    'Amp': singlecdf_amp,
    'Mean': singlecdf_mean,
    'StD': singlecdf_std,
    'Avg_Residual': singlecdf_residual,
    }

    return pd.DataFrame(data)

def process_doublehit(df, t0):
    doublecdf_event = []
    doublecdf_pixid = []
    doublecdf_amp_1 = []
    doublecdf_mean_1 = []
    doublecdf_std_1 = []
    doublecdf_amp_2 = []
    doublecdf_mean_2 = []
    doublecdf_std_2 = []
    doublecdf_residual = []
    
    for i in range(len(df)):
        reset_times = np.array(df['reset_time'].reset_index().iloc[i][1]) - t0
        num_resets = len(reset_times)
        reset_count = np.arange(1, num_resets + 1)  
        
        initial_params = [num_resets/2 + 0.1, np.mean(reset_times) - std_exp(np.mean(reset_times)), num_resets/2 + 0.1, np.mean(reset_times) + std_exp(np.mean(reset_times))]
        
        try:
            cdf_params, cdf_covariance = curve_fit(double_cdf_nostd, reset_times, reset_count, p0=initial_params)
            reset_amp_1 = cdf_params[0]
            reset_mean_1 = cdf_params[1]
            reset_std_1 = std_exp(reset_mean_1) 
            reset_amp_2 = cdf_params[2]
            reset_mean_2 = cdf_params[3]
            reset_std_2 = std_exp(reset_mean_2)
            fitted_values = double_cdf_nostd(reset_times, *cdf_params)
            residuals = reset_count - fitted_values
            sum_residuals = np.sum(np.abs(residuals))
            sum_residuals_per_reset = sum_residuals/num_resets                       

            if (((reset_amp_1 + reset_amp_2) > num_resets) and ((reset_amp_1 + reset_amp_2) < num_resets + 1) and (sum_residuals_per_reset < 0.1)):
                doublecdf_event.append(n)
                doublecdf_pixid.append(df.iloc[i].PixelID)
                
                doublecdf_amp_1.append(reset_amp_1)
                doublecdf_mean_1.append(reset_mean_1)
                doublecdf_std_1.append(reset_std_1)
                doublecdf_amp_2.append(reset_amp_2)
                doublecdf_mean_2.append(reset_mean_2)
                doublecdf_std_2.append(reset_std_2)
                doublecdf_residual.append(sum_residuals_per_reset)
                
        except RuntimeError:
            continue
            
    data = {
    'event': doublecdf_event,
    'PixelID': doublecdf_pixid,
    'Amp1': doublecdf_amp_1,
    'Mean1': doublecdf_mean_1,
    'StD1': doublecdf_std_1,
    'Amp2': doublecdf_amp_2,
    'Mean2': doublecdf_mean_2,
    'StD2': doublecdf_std_2,
    'Avg_Residual': doublecdf_residual,
    }

    return pd.DataFrame(data)

def process_triplehit(df, t0):
    triplecdf_event = []
    triplecdf_pixid = []
    triplecdf_amp_1 = []
    triplecdf_mean_1 = []
    triplecdf_std_1 = []
    triplecdf_amp_2 = []
    triplecdf_mean_2 = []
    triplecdf_std_2 = []
    triplecdf_amp_3 = []
    triplecdf_mean_3 = []
    triplecdf_std_3 = []
    triplecdf_residual = []
        
    for i in range(len(df)):
        reset_times = np.array(df['reset_time'].reset_index().iloc[i][1]) - t0
        num_resets = len(reset_times)
        reset_count = np.arange(1, num_resets + 1)  
        
        initial_params = [num_resets/3 + 0.1, np.mean(reset_times) - std_exp(np.mean(reset_times)), num_resets/3 + 0.1, np.mean(reset_times) + std_exp(np.mean(reset_times)), num_resets/3 + 0.1, np.mean(reset_times)]
        
        try:
            cdf_params, cdf_covariance = curve_fit(triple_cdf_nostd, reset_times, reset_count, p0=initial_params)
            reset_amp_1 = cdf_params[0]
            reset_mean_1 = cdf_params[1]
            reset_std_1 = std_exp(reset_mean_1)
            reset_diff_1 = std_difference(reset_mean_1, reset_std_1)    
            reset_amp_2 = cdf_params[2]
            reset_mean_2 = cdf_params[3]
            reset_std_2 = std_exp(reset_mean_2)
            reset_diff_2 = std_difference(reset_mean_2, reset_std_2)
            reset_amp_3 = cdf_params[4]
            reset_mean_3 = cdf_params[5]
            reset_std_3 = std_exp(reset_mean_3)
            reset_diff_3 = std_difference(reset_mean_3, reset_std_3)
            fitted_values = triple_cdf_nostd(reset_times, *cdf_params)
            residuals = reset_count - fitted_values
            sum_residuals = np.sum(np.abs(residuals))
            sum_residuals_per_reset = sum_residuals/num_resets     
            
            if (((reset_amp_1 + reset_amp_2 + reset_amp_3) > num_resets) and ((reset_amp_1 + reset_amp_2 + reset_amp_3) < num_resets + 1) and (sum_residuals_per_reset < 0.1)):
                triplecdf_event.append(n)
                triplecdf_pixid.append(df.iloc[i].PixelID)
                
                triplecdf_amp_1.append(reset_amp_1)
                triplecdf_mean_1.append(reset_mean_1)
                triplecdf_std_1.append(reset_std_1)
                triplecdf_amp_2.append(reset_amp_2)
                triplecdf_mean_2.append(reset_mean_2)
                triplecdf_std_2.append(reset_std_2)
                triplecdf_amp_3.append(reset_amp_3)
                triplecdf_mean_3.append(reset_mean_3)
                triplecdf_std_3.append(reset_std_3)
                triplecdf_residual.append(sum_residuals_per_reset)
                
        except RuntimeError:
            continue
            
    data = {
    'event': triplecdf_event,
    'PixelID': triplecdf_pixid,
    'Amp1': triplecdf_amp_1,
    'Mean1': triplecdf_mean_1,
    'StD1': triplecdf_std_1,
    'Amp2': triplecdf_amp_2,
    'Mean2': triplecdf_mean_2,
    'StD2': triplecdf_std_2,
    'Amp3': triplecdf_amp_3,
    'Mean3': triplecdf_mean_3,
    'StD3': triplecdf_std_3,
    'Avg_Residual': triplecdf_residual,
    }

    return pd.DataFrame(data)

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
    
    if rtd_t0candidate_eventdf.empty:
        print(f"Skipping event {n} because no pixels have 3-5 resets.")
        continue
    
    #try a single CDF fit on all pixels
    singlecdf_noshift_results = process_singlecdf(rtd_t0candidate_eventdf)
    singlecdf_diff = singlecdf_noshift_results['Diff']

    #prune the data to remove obvious multi-hit pixels
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
    

    #Shift the reset_times by t0 and look for hits
    rtd_allpix_eventdf = rtd_allpix_df[rtd_allpix_df.event == n]
    rtd_allpix_eventdf['t0'] = optimal_t0_shift
    print("pixels in event = ", len(rtd_df[rtd_df.event ==n]))

    #Create a singlehit dataframe for the pixels that fit well to a single hit  
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
    #Move on to pixels that don't fit well to a single hit
    
    #Create a doublehit dataframe for the pixels that fit well to a double hit
    doublehit_event = process_doublehit(notsinglehit_event[notsinglehit_event.nResets >= 4], optimal_t0_shift)
    
    if not doublehit_event.empty:
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
    else:
        notdoublehit_event = notsinglehit_event   
     #Move on to pixels that don't fit well to a double hit
        
    #Create a triplehit dataframe for the pixels that fit well to a triple hit
    triplehit_event = process_triplehit(notdoublehit_event[notdoublehit_event.nResets >= 6], optimal_t0_shift)
    
    if not triplehit_event.empty:
        triplehit_percentile_high = np.percentile(triplehit_event[(triplehit_event.Amp1 + triplehit_event.Amp2 + triplehit_event.Amp3) >= 7]['Avg_Residual'], 85)   
    
        triplehit_cut = np.median(triplehit_event[(triplehit_event.Avg_Residual <= triplehit_percentile_high)]['Avg_Residual']) + 5*np.std(triplehit_event[(triplehit_event.Avg_Residual <= triplehit_percentile_high)]['Avg_Residual'], ddof=1)
    
        triplehit_event = triplehit_event[triplehit_event.Avg_Residual < triplehit_cut]  
    
        print("triple hit pixels in event = ", len(triplehit_event))
    
        triplehit_event['t0'] = optimal_t0_shift
        triplehit_df = triplehit_df.append(triplehit_event, ignore_index=True)
  
        #Create an unfitpix dataframe for the pixels that did not fit to single, double, or triple hits
        unfitpix_event = notdoublehit_event.merge(
            triplehit_event[['event', 'PixelID']],
            on=['event', 'PixelID'],
            how='left',
            indicator=True
        )
    
        unfitpix_event = unfitpix_event[unfitpix_event['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        unfitpix_event = notdoublehit_event    

    print("unfit pixels = ", len(unfitpix_event))
    print("remaining pixels that could fit quadruple cdf = ", len(unfitpix_event[unfitpix_event.nResets >= 8])) 
    
    unfitpix_event = unfitpix_event.append(rtd_df[(rtd_df.event == n) & (rtd_df.nResets < 2)])   
    unfitpix_event['t0'] = optimal_t0_shift

    unfitpix_df = unfitpix_df.append(unfitpix_event)       

singlehit_df.to_pickle(t0_hitmaker_dir + '/singlehit_df.pkl')
print("List of single-hit pixels written to " + t0_hitmaker_dir + 'singlehit_df.pkl')

doublehit_df.to_pickle(t0_hitmaker_dir + '/doublehit_df.pkl')
print("List of double-hit pixels written to " + t0_hitmaker_dir + 'doublehit_df.pkl')

triplehit_df.to_pickle(t0_hitmaker_dir + '/triplehit_df.pkl')
print("List of triple-hit pixels written to " + t0_hitmaker_dir + 'triplehit_df.pkl')

unfitpix_df.to_pickle(t0_hitmaker_dir + '/unfitpix_df.pkl')
print("List of unfit pixels written to " + t0_hitmaker_dir + 'unfitpix_df.pkl')

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
print("List of hits written to " + t0_hitmaker_dir + 'hits_df.pkl')