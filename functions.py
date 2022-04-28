import math
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import lmfit

##########################################################################################################################
############################################# Define functions here ######################################################
##########################################################################################################################

# Fit functions
def Horizontal(x,b):
    return b

def Linear(x,m,b):
    return m*x+b

def Quadratic(x,a,b,c):
    return a*x**2 + b*x + c

def SqRoot(x,a,b,c):
    return a*np.sqrt(x-b)+c

def Sqrt_Shift(x,a,c):
    return a*np.sqrt(x-c)

def Gaussian(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

def DoubleGaussian(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return A1*np.exp(-(x-mu1)**2/(2*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2*sigma2**2))
    
def zofsigma(sigma):
    return (sigma**2)*(164800**3)/(2*6.8223)

def sigmaofz(z):
    return np.sqrt((2*6.8223*z)/(164800**3))

def sigmaoft(t):
    return np.sqrt(2*6.8223/164800**2)*np.sqrt(t)
    
def exponential(x,a,b,c):
    return a*np.exp(b*x)+c
    
# Other functions
    

def ChiSqr_GaussianFit(event,pixel_x,pixel_y,mu,sigma,A):
    tmp = resets[(resets.event == event) & (resets.pixel_x == pixel_x) & (resets.pixel_y == pixel_y)]
    mean,stdev = tmp.reset_time.mean(),tmp.reset_time.std()
    start,stop = math.floor((mean-3*stdev)*1e+07)/1e+07,math.ceil((mean+3*stdev)*1e+07)/1e+07
    if start > tmp.reset_time.min() or stop < tmp.reset_time.max():
        start,stop = math.floor(tmp.reset_time.min()*1e+07)/1e+07,math.ceil(tmp.reset_time.max()*1e+07)/1e+07
    y,x = np.histogram(tmp.reset_time,bins=np.arange(start,stop,1e-07))
    x = (x[1:]+x[:-1])/2
    chisqr = 0
    for i in range(len(x)):
        exp = Gaussian(x[i],mu,sigma,A)
        obs = y[i]
        chisqr += (abs(exp-obs))**2/(exp)
    return chisqr, chisqr/(len(x)-3)

def Profile_Plot(x,y,nPoints=0,binwidth=0,remove_outliers=False):
    x_data,y_data,y_data_sansoutliers,errs,errs_sansoutliers = [],[],[],[],[]
    if nPoints != 0:
        tmp = np.linspace(np.amin(x),np.amax(x),nPoints+1)
        x_data = (tmp[1:]+tmp[:-1])/2
    else:
        if binwidth != 0:
            tmp = np.arange(np.amin(x),np.amax(x),binwidth)
            x_data = tmp + binwidth/2
        else:
            raise KeyError('Need to specify either nPoints or binwidth!')
    xstep = x_data[2]-x_data[1]
    
    for step in x_data:
        tmpx,tmpy = [],[]
        for idx in np.where((x>step-xstep/2)&(x<step+xstep/2))[0]:
            tmpx.append(x[idx])
            tmpy.append(y[idx])
        done = False
        tmptmpy = tmpy
        while done == False:
            mu = np.mean(tmptmpy)
            sigma = np.std(tmptmpy)
            chis = []
            for i in range(len(tmptmpy)):
                chi_cont = abs(tmptmpy[i]-mu)**2/sigma**2
                chis.append(chi_cont)
            if np.amax(chis) > 4:
                idx = np.where((chis == np.amax(chis)))[0][0]
                tmptmpy = tmptmpy[:idx]+tmptmpy[idx+1:]
                done = False
            else:
                done = True
                
               
        mu,sigma = np.mean(tmpy),np.std(tmpy)
        mu_sans,sigma_sans = np.mean(tmptmpy),np.std(tmptmpy)
        y_data.append(np.mean(tmpy))
        y_data_sansoutliers.append(np.mean(tmptmpy))
        
        if len(tmpy)>1:
            errs.append(np.std(tmpy))
        else:
            errs.append(0)

        if len(tmptmpy)>1:
            errs_sansoutliers.append(np.std(tmptmpy))
        else:
            errs_sansoutliers.append(0)
            
    return x_data,y_data,errs,y_data_sansoutliers,errs_sansoutliers