# ==================================================================================================
# Importing Modules
# ==================================================================================================
import sys
import matplotlib.pyplot as plt
import pandas as pd
#Pandas is useful for importing files and extracting data
import numpy as np
#numpy is useful for numerical calculations 
from math import sqrt
#math is also useful for numerical calculations
import os
import shutil
#OS is optional, but allows for the user to create directories to store generated files/figures
from scipy.optimize import curve_fit
# from scipy.stats import norm
#Scipy's curve_fit is one of the most user-friendly ways to make a best-fit plot
pd.set_option('display.max_rows', 100)
# from matplotlib.collections import LineCollection
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D

# For input file, use:
dfoutput_dir = sys.argv[1]
functional_form_dir = sys.argv[2]
functional_form_file = functional_form_dir + "functional_form.txt"
outlier_file = functional_form_dir + "outlier_df.pkl"
total_events = int(sys.argv[3])
num_resets = int(sys.argv[4])


# Open the resets_output.txt file and read into pandas dataframe
main_df = pd.read_pickle(dfoutput_dir + "main_subdf.pkl").reset_index(drop = True)


# Scale each x_pixel and y_pixel by 4mm to get pixel distance
main_df.pixel_x *= 4
main_df.pixel_y *= 4

# =============================================================================
# INITIAL FIT
# =============================================================================

# Relevant Functions:

def sqrt_func(x_f, a_f):
    return a_f*np.sqrt(abs(x_f))
"""
Returns the value a*sqrt(x) for given x(s). 

    Parameters
    ----------
        x_f : array of float or float
            Array of independent float values.
        a_f : float
            Scaling factor of the sqrt function.

    Returns
    -------
        array of float or float: 
            Value(s) equal to a*sqrt(x) corresponding to the array parameter x.
"""

def gauss_func(x_f, amp_f, mu_f, sigma_f):
    return amp_f*np.exp(-(x_f-mu_f)**2/(2*sigma_f**2))
"""
Returns the dependent value on a Gaussian curve for given x(s). 

    Parameters
    ----------
        x_f : array of float or float
            Array of independent float values.
        amp_f : float
            Amplitude of the Gaussian curve.
        mu_f : float
            Mean value for the Gaussian distribution.
        sigma_f : float
            Standard Deviation for the Gaussian distribution.
"""

def sqrt_fit_func(mean_TOA_f, min_RMS_f):
    t = np.linspace(0, mean_TOA_f.max(), 100)
    poptsqrt, pcov = curve_fit(sqrt_func, mean_TOA_f, min_RMS_f)
    rms_fit = []
    rms_th = []
    for I in range(len(t)):
        rms_fit.append(poptsqrt[0]*np.sqrt(t[I]))
        rms_th.append(sqrt_constant*np.sqrt(t[I]))
#     plt.savefig('Muon_Output/Fitting_Plots/Sqrt_Fit_Plots/sqrt_fit_'+str(i))
    return poptsqrt, pcov, rms_fit, t, rms_th, rms_fit
"""
Returns fitting parameters, dependent on type of fitted line, for a given dataset. 

    Parameters
    ----------
        mean_TOA_f: array of float Independent variables corresponding to the x-array. In this case the mean time of arrival
        min_RMS_f : array of float Dependent variables corresponding to the y-array. In this case the minimum RMS value

    Returns
    -------
        array of float: 
            Array of fitting parameters for each needed parameter in function.
"""

def gauss_fit_func(delta_RMS_f):
    n, bins, patches = plt.hist(delta_RMS_f, bins = 30)   #plots the actual histogram
    binCenter = []
    for I in range(len(bins) - 1):    #Loops through the bins and finds the center of the bins
         binCenter.append((bins[I]+bins[I+1])/2)
    poptgauss1, pcovgauss1 = curve_fit(gauss_func, binCenter, n, p0=(max(n), delta_RMS_f.median(), delta_RMS_f.std())) #gaussian curve fitting
    amp = poptgauss1[0] #saves the amplitude from the fit
    mu = poptgauss1[1]  #saves the mu from the fit
    sigma = poptgauss1[2]  #save the std from the fit
    return amp, mu, sigma
"""
    Takes in a histogram and returns the gaussian fitting parameters for the histogram. 

    Parameters
    ----------
        delta_RMS_f : array of float. The calculated distance of points from the best fit sqrt line

    Returns
    -------
            Fitting parameters used to generate a Gaussian curve fitted to histogram bars.
"""

def Initial_fit_func(meanTOA_f, minRMS_f, dataframe_f):
    poptsqrt1, pcov1, rms_fit1, t1, rms_th1, rms_fit1 = sqrt_fit_func(meanTOA_f, minRMS_f)
    sqrt_fit_history = []
    sqrt_fit_history.append(poptsqrt1)
    dataframe_f["expected_RMS"] = sqrt_func(meanTOA_f, poptsqrt1[0])
    dataframe_f["delta_RMS"] = minRMS_f-dataframe_f.expected_RMS
    amp1, mu1, sigma1 = gauss_fit_func(dataframe_f.delta_RMS)
    return sqrt_fit_history
"""
For the initial data, this function makes it easy to just perform both a square root and Gaussian fit
"""

# Defining and Calculating Constants
e_vel = 164800 #cm**2/s
DiffusionL = 6.8223 #cm**2/s
DiffusionT = 13.1586 #cm**2/s
Life_Time = 0.1 #s
sqrt_constant = sqrt(2*DiffusionL)/e_vel


# Find the indicies for the min RMS for each event

min_RMS_indices = []
for event_num in range(total_events):
    min_RMS_indices.append(main_df[(main_df["event"] == event_num)]["RMS"].idxmin())

fit_history = Initial_fit_func(main_df.iloc[min_RMS_indices]["mean_TOA"], main_df.iloc[min_RMS_indices]["RMS"], main_df)
initial_fit = fit_history[0]

print("Initial Fit Constant Found: " + str(initial_fit))

# =============================================================================
# FINDING MIN RMS VALUES AND TOA SHIFT
# =============================================================================

# Relevant Functions
def ToAShift_Index_func(dataframe_f, event_num_f, functionalForm_f):
    # This function will be used in a for loop for all events.
    # Get subset of dataframe consisting of rows with event = event_num_f
    event_df = dataframe_f[dataframe_f["event"] == event_num_f].copy().reset_index(drop = False)
    
    # Get the smallest StDev in the dataframe and shift it's time such that it resides on functional fit
    minRMS_index = event_df["RMS"].idxmin()
    expected_TOA = (event_df["RMS"].min()/functionalForm_f)**2
    delta_t = event_df.iloc[minRMS_index]["mean_TOA"] - expected_TOA
    
    # Shift all pixels by the calculated delta_t
    event_df["mean_TOA"] = event_df["mean_TOA"] - delta_t
    
    # In very rare case delta_t might originally be 0, still force the loop to occur
    if delta_t == 0:
        delta_t = float('inf')
    
    while delta_t != 0:
        
        # Get the expected RMS for a given time and the corresponding delta_StDev
        event_df["Expected_RMS"] = sqrt_func(event_df["mean_TOA"], functionalForm_f)
        event_df["delta_StDev"] = event_df["RMS"] - event_df["Expected_RMS"]
        
        # Find the smallest delta_StDev and shift it's time such that it resides on functional fit
        min_deltaRMS_index = event_df["delta_StDev"].idxmin()
        expected_TOA = (event_df.iloc[min_deltaRMS_index]["RMS"]/functionalForm_f)**2
        delta_t = event_df.iloc[min_deltaRMS_index]["mean_TOA"] - expected_TOA
        
        # Shift all pixels by the calculated delta_t
        event_df["mean_TOA"] = event_df["mean_TOA"] - delta_t 
        
    pixel_id = event_df.iloc[event_df['delta_StDev'].idxmin()]['index']
    return pixel_id

"""
    Finds the pixel's which have the minimum delta_RMS for each event. In the process, it finds the shifted TOA which is useful for calculating the t0.

    Parameters
    ----------
	dataframe_f: The relevant dataframe for your data
	event_num_f: Number of the event you are looking at (This is used later to loop through)
	functionalForm_f: The square root best fit
"""

# Get new list of indices for smallest delta_StDev
min_deltaRMS_indices = []
for event_num in range(total_events):
    min_deltaRMS_indices.append(ToAShift_Index_func(main_df, event_num, initial_fit))

# Create list of event numbers that were changed after ToA shifting (a new pixel is being fitted with the functional form).
changed =[]
for i in range(0, len(min_RMS_indices)):
    if min_RMS_indices[i] != min_deltaRMS_indices[i]:
        changed.append(i)
changed

# =============================================================================
# REMOVING OUTLIERS AND FITTING
# =============================================================================

# Relevant Functions
def rm_outliers_func(dataframe_f, threshold=9):
    # Identify outliers based on the specified threshold
    mu = dataframe_f[dataframe_f["event_outlier"] == False]["delta_RMS"].mean()
    sigma = dataframe_f[dataframe_f["event_outlier"] == False]["delta_RMS"].std()
    dataframe_f["Chi^2_Value"] = (mu - dataframe_f["delta_RMS"])**2 / sigma**2
    outliers = dataframe_f[(dataframe_f["event_outlier"] == False) & (dataframe_f["Chi^2_Value"] >= threshold)]
    return outliers

def remove_outliers(dataframe_f, outliers):
    # Mark the outliers as removed in the original DataFrame
    dataframe_f.loc[outliers.index, "event_outlier"] = True

def outliers_loop_func(dataframe_f, min_deltaRMS_indices_f, threshold=9):
    # Mark the initial outliers based on deltaRMS values
    dataframe_f["event_outlier"] = True
    dataframe_f.loc[min_deltaRMS_indices_f, "event_outlier"] = False

    # Iteratively identify and remove outliers
    all_outliers = pd.DataFrame(columns=dataframe_f.columns)  # Store all identified outliers
    while True:
        outliers = rm_outliers_func(dataframe_f, threshold=threshold)
        if outliers.empty:
            break
        
        remove_outliers(dataframe_f, outliers)
        all_outliers = pd.concat([all_outliers, outliers])

    # Calculate poptsqrti, pcovi, and other values
    valid_data = dataframe_f[dataframe_f["event_outlier"] == False]
    poptsqrti, pcovi, rms_fiti, ti, rms_thi, rms_fiti = sqrt_fit_func(valid_data.mean_TOA, valid_data.RMS)
    
    return all_outliers, poptsqrti

# Call the function and store the returned values
outliers_df, functional_form = outliers_loop_func(main_df, min_deltaRMS_indices)

def sqrt_fit(x_f, a_f):
    return a_f*(x_f)**(1/2)

# Get the "mean_TOA" values within the specified indices
mean_TOA_values = main_df.iloc[min_deltaRMS_indices]["mean_TOA"].to_numpy()

# Sort the "mean_TOA" values in increasing order
sorted_indices = np.argsort(mean_TOA_values)
sorted_mean_TOA = mean_TOA_values.iloc[sorted_indices]

fit_func = sqrt_fit(sorted_mean_TOA, functional_form[0])

plt.scatter(main_df.iloc[min_deltaRMS_indices]["mean_TOA"], main_df.iloc[min_deltaRMS_indices]["RMS"], marker = '.', color='blue', label = 'Outliers')
plt.scatter(main_df[(main_df["event_outlier"] == False)].mean_TOA, main_df[(main_df["event_outlier"] == False)].RMS , marker = '.', color='orange', label = 'RMS Min')
plt.plot(main_df.iloc[min_deltaRMS_indices]["mean_TOA"], fit_func, '-', label='{:0.5}$*\sqrt{{Mean(ToA)}}$'.format(functional_form[0]))
plt.ylim(0, 1.2e-6)
plt.ylabel("Min(StdDev) [s]")
plt.xlabel("Mean(ToA) [s]")
plt.legend()
if sys.argv[5] == "-v"  or sys.argv[5] == "-verbose":
    plt.savefig(functional_form_dir + 'functional_form_plot.png')
plt.close()
print("Functional Form found: " + str(functional_form))

f1 = open(functional_form_file,"w+")
f1.write(str(functional_form[0]))
f1.close()
print("Fit parameter written to " + functional_form_file)

outliers_df = outliers_df[['event', 'PixelID', 'pixel_x', 'pixel_y', 'Chi^2_Value']]
print("Outlier events written to " + outlier_file)