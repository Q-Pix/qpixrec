#!/usr/bin/env python

# -----------------------------------------------------------------------------
# cdf_definitions.py
#
# Determines requirements for hitmaker and CDF fitting
# * Author: Carter Eikenbary
# * Creation date: 2 December 2024
# -----------------------------------------------------------------------------

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
            
            if ((reset_amp > num_resets) and (reset_amp < num_resets + 1) and (sum_residuals_per_reset < 0.02)):
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

def process_singlecdf_t0shifted(df, t0):
    singlecdf_event = []
    singlecdf_pixid = []
    singlecdf_amp = []
    singlecdf_mean = []
    singlecdf_std = []
    singlecdf_diff = []

    for i in range(len(df)):
        reset_times = np.array(df['reset_time'].reset_index().iloc[i][1]) - t0
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
            fitted_values = single_cdf(reset_times, *cdf_params)
            residuals = reset_count - fitted_values
            sum_residuals = np.sum(np.abs(residuals))
            sum_residuals_per_reset = sum_residuals/num_resets                
            
            if ((reset_amp > num_resets) and (reset_amp < num_resets + 1) and (sum_residuals_per_reset < 0.02)):
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