# matplotlib for plotting
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# numpy to calculate RMS with np.std()
import numpy as np

# pandas used for handling the large amounts of data in dataframes rather than python lists
import pandas as pd

# scipy for curve fitting
from scipy.optimize import curve_fit

# sys for command line arguments
import sys


dfoutput_dir = sys.argv[1]
functional_form_file = sys.argv[2]
t0_hitmaker_dir = sys.argv[3]
total_events = int(sys.argv[4])
binWidth = float(sys.argv[5])

# Read in the dataframes from root_to_pandas.py
main_df = pd.read_pickle(dfoutput_dir + "main_subdf.pkl").reset_index(drop = True)
main_df = main_df.drop(columns=['reset_time', 'TSLR'])

g4_df = pd.read_pickle(dfoutput_dir + "g4_df.pkl")

resets_df = pd.read_pickle(dfoutput_dir + "main_df.pkl")

# Functional Parameter determined from min(StDev) fitting
with open(functional_form_file, 'r') as file:
    # Step 2: Read the content of the file
    content = file.read()

    # Step 3: Convert the read content to a numerical value
    functionalForm = float(content)

def sqrtFunc(x_f, a_f):
    return a_f*(x_f)**(1/2)


def fitLine(function, x_f, y_f):
    params, _ = curve_fit(function, x_f, y_f)
    return params


def gaussFunc(x_f, A_f, x0_f, sigma_f):
    return A_f * np.exp(-(x_f - x0_f) ** 2 / (2 * sigma_f ** 2))


def gaussFitHist(data_f, bins_f):
    counts, bins = np.histogram(data_f, bins = bins_f)
    binCenters = []
    
    for i in range(len(bins) - 1):
        binCenters.append((bins[i]+bins[i+1])/2)

    gaussDataPoints = pd.DataFrame(data={'xvalues': binCenters, 'yvalues': counts
                                        })
                                                   
    gauss_params, _ = curve_fit(gaussFunc, gaussDataPoints["xvalues"], gaussDataPoints["yvalues"],
                                p0 = [gaussDataPoints["yvalues"].max(), data_f.mean(), data_f.std()])
    
    return gauss_params, gaussDataPoints


def ToAShift_Index(dataframe_f, event_num_f, functionalForm_f):
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
        event_df["Expected_RMS"] = sqrtFunc(event_df["mean_TOA"], functionalForm_f)
        event_df["delta_StDev"] = event_df["RMS"] - event_df["Expected_RMS"]
        
        # Find the smallest delta_StDev and shift it's time such that it resides on functional fit
        min_deltaRMS_index = event_df["delta_StDev"].idxmin()
        expected_TOA = (event_df.iloc[min_deltaRMS_index]["RMS"]/functionalForm_f)**2
        delta_t = event_df.iloc[min_deltaRMS_index]["mean_TOA"] - expected_TOA
        
        # Shift all pixels by the calculated delta_t
        event_df["mean_TOA"] = event_df["mean_TOA"] - delta_t 
        
    pixel_id = event_df.iloc[event_df['delta_StDev'].idxmin()]['index']
    
    return pixel_id


def getZValues(dataFrame_f):
    v0 = 1.648 * 10**6 # mm/s
    
    dataFrame_f["Z_Position"] = (dataFrame_f["mean_TOA"] - dataFrame_f["t0"])*v0
    return dataFrame_f["Z_Position"]

# Find the indicies for the min RMS for each event
min_RMS_indices = []
for event_num in range(total_events):
    min_RMS_indices.append(main_df[(main_df["event"] == event_num)]["RMS"].idxmin())
    
# Get initial fit of smallest RMS for each event
initial_fit = fitLine(sqrtFunc, main_df.iloc[min_RMS_indices]['mean_TOA'], main_df.iloc[min_RMS_indices]['RMS'])[0]

# Get new list of indices for smallest delta_StDev
min_deltaRMS_indices = []
for event_num in range(total_events):
    min_deltaRMS_indices.append(ToAShift_Index(main_df, event_num, initial_fit))
    
# Calculate the expected ToF for 100 min_deltaStDev pixels with a given RMS
main_df['Expected_ToF'] = (main_df.iloc[min_deltaRMS_indices]['RMS'] / functionalForm)**2

# Calculate the t0 for each pixel store into a t0 dataframe
t0_df = pd.DataFrame(data = {'event' : range(total_events), 't0' : (main_df.iloc[min_deltaRMS_indices]['mean_TOA'] - \
                                                                    main_df.iloc[min_deltaRMS_indices]['Expected_ToF'])})

# Merge t0 values into main_df
main_df = main_df.merge(t0_df, how = 'right', on = 'event').copy().reset_index(drop = True)

# Plot the t0 distribution

# Get plot limits for binning
plt.hist(main_df.iloc[min_deltaRMS_indices]['t0'])
xmin, xmax = plt.xlim()
plt.close()

# Create plot parameters
plt.figure(figsize = (9,9))
plt.xlabel('t0 (sec)', fontsize = 16)
plt.title('t0 Distribution - All Events', fontsize = 20)

# Plot histogram based on bins defined by plot limits
plt.hist(main_df.iloc[min_deltaRMS_indices]['t0'], bins = np.arange(xmin, xmax, binWidth))

# Get a gaussian fit to the distribution
t0_gauss_params, hist_data = gaussFitHist(main_df.iloc[min_deltaRMS_indices]['t0'], 
                                          np.arange(xmin, xmax, binWidth)) 

# Plot the Gaussian Fit
xs = np.linspace(xmin, xmax, 1000)
ys = gaussFunc(xs, *t0_gauss_params)
plt.plot(xs, ys, '--', color = 'crimson', label = 'Gaussian Fit')

# Plot vertical line at t = 0
x = [0]*2
y = np.linspace(0, hist_data['yvalues'].max(), 2)
plt.plot(x, y, '--', color = 'lime')

# Show and close plot
plt.savefig(t0_hitmaker_dir + '/t0_distribution.png')
plt.close()

# Print distribution gaussian parameters
print('Gaussian Parameters: A: %3.5E, mu: %3.5E, sigma: %3.5E' %(t0_gauss_params[0], t0_gauss_params[1], 
                                                                 t0_gauss_params[2]))

output_str = 'Gaussian Parameters: A: %3.5E, mu: %3.5E, sigma: %3.5E' % (
    t0_gauss_params[0], t0_gauss_params[1], t0_gauss_params[2]
)

# Specify the file path where you want to save the output
output_file_path = t0_hitmaker_dir + 't0_parameters.txt'

# Write the output to the file
with open(output_file_path, 'w') as file:
    file.write(output_str)

print("t0 output written to " + output_file_path)

# Create a full dataframe with ALL pixels and the calculated t0s
full_df = resets_df.merge(t0_df, how = 'left', on = 'event').copy().reset_index(drop=True)
full_df = full_df.drop(columns=['TSLR'])

def stl_vector_to_list(stl_vector):
    return list(stl_vector)

# Convert the 'reset_time' column of the DataFrame into a NumPy array
full_df['reset_time'] = full_df['reset_time'].apply(stl_vector_to_list).apply(np.array)
full_df = full_df.explode('reset_time').reset_index(drop = True)

# Calculate the XYZ positions for each pixel
v0 = 1.648 * 10**6 # mm/s
full_df['x_calc'] = full_df['pixel_x'] * 4 - 2 # Subtract 2mm to get in center of the pixel (pixels are 4mm wide)
full_df['y_calc'] = full_df['pixel_y'] * 4 - 2 # Subtract 2mm to get in center of the pixel (pixels are 4mm wide)
full_df['z_calc'] = v0 * (full_df['reset_time'] - full_df['t0'])

unique_events_df = main_df[['event', 'PixelID', 'Expected_ToF']].drop_duplicates()
full_df = full_df.merge(unique_events_df, how='left', on=['event', 'PixelID'])
full_df = full_df.drop(columns=['nResets','mean_TOA','RMS'])

full_df['minSTD'] = full_df['Expected_ToF'].notna()
full_df['Expected_ToF'] = full_df.groupby('event')['Expected_ToF'].transform(lambda x: x.ffill().bfill())

# Get the index of the 't0' column
t0_index = full_df.columns.get_loc('t0')

# Get the 'Expected_ToF' column and drop it from its current position
expected_toF_column = full_df['Expected_ToF']
full_df.drop(columns=['Expected_ToF'], inplace=True)

# Insert the 'Expected_ToF' column to the left of the 't0' column
full_df.insert(t0_index, 'Expected_ToF', expected_toF_column)

full_df.to_pickle(t0_hitmaker_dir + '/reconstruction_df.pkl')
print("Dataframe written to " + t0_hitmaker_dir + 'reconstruction_df.pkl')
# Add the t0 dataframe as a 't0' column to the main_df by aligning the event numbers
reconstruction_df = main_df.copy().reset_index(drop = True)

# Convert the pixels, with nResets >= reset_num, pixel numbers to a position in mm
reconstruction_df["X_Position"] = reconstruction_df["pixel_x"] * 4 - 2 # Subtract 2mm to get in center of the pixel 
reconstruction_df["Y_Position"] = reconstruction_df["pixel_y"] * 4 - 2 # Subtract 2mm to get in center of the pixel 
reconstruction_df["Z_Position"] = getZValues(reconstruction_df)

# To plot the G4 data quickly, import LineCollection. This avoiding iteratin grows and plotting it that way
from matplotlib.collections import LineCollection

# Plot the YZ data for event = EVENT_NUM

for n in range(total_events):
    # Changes datafram ecolumns to arrays
    yi = pd.DataFrame(data = {"yi" : g4_df[g4_df["event"] == n]["yi"]}).to_numpy()
    zi = pd.DataFrame(data = {"zi" : g4_df[g4_df["event"] == n]["zi"]}).to_numpy()
    yf = pd.DataFrame(data = {"yf" : g4_df[g4_df["event"] == n]["yf"]}).to_numpy()
    zf = pd.DataFrame(data = {"zf" : g4_df[g4_df["event"] == n]["zf"]}).to_numpy()

    # Combines the 1D arrays to one 2D array
    initials = np.column_stack((yi, zi))
    finals = np.column_stack((yf, zf))

    # Combine to two arrays to one array with row = [[yi, zi], [yf, zf]]
    combined = np.zeros((len(initials), 2, 2))
    combined[:, 0] = initials
    combined[:, 1] = finals

    # Plot all the G4 data overlayed with the reconstruction data for all events:
    # Create the plot
    fig, ax = plt.subplots()
    fig.set_figheight(5.4)
    fig.set_figwidth(9)

    # Label the axes for the plot
    ax.set_title('YZ Hit Reconstruction \nEvent %d' %n, fontsize=20)
    ax.set_xlabel('Y Position (cm)', fontsize=16)
    ax.set_ylabel('Z Position (cm)', fontsize=16)
    
    
    # Plot the G4 data via Line Collections
    line_segments = LineCollection(combined, colors = 'blue', label = 'G4 Data', linewidths = 2.0)
    ax.add_collection(line_segments)
    ax.autoscale()
    
    
    # Plot the points from the t0 values
    # Divide by 10 to convert from mm to cm (G4 is in cm)
    ax.scatter(full_df.y_calc[(full_df.event == n)]/10, 
                full_df.z_calc[(full_df.event == n)]/10, s =1, marker = '.', color = 'darkorange', 
                label = 'Reconstructed Data')
    ax.scatter(reconstruction_df.Y_Position[(reconstruction_df.event == n)]/10, 
            reconstruction_df.Z_Position[(reconstruction_df.event == n)]/10, s = 5, color = 'red', 
            label = 'nResets >= 8')
 
    # Create the legend and get the legend object
    legend = plt.legend()

    # Set the marker size for the second legend entry (index 1, since it is zero-based)
    legend.legendHandles[1]._sizes = [10]
                                     
    if sys.argv[6] == "-v"  or sys.argv[6] == "-verbose":
        plt.savefig(t0_hitmaker_dir + 'yz_reconstruction_Event' + str(n) + '.png')
    plt.close()

if sys.argv[6] == "-v"  or sys.argv[6] == "-verbose":   
    print("YZ Reconstruction plot for events saved to " + t0_hitmaker_dir + 'yz_reconstruction_Event#.png')
