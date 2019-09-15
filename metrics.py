#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:46:23 2019

@author: Sean Maudsley-Barton

A set of utilities that help to calculate common sway metrics
"""

import numpy as np
import pandas as pd
import os

from scipy import signal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from enum import Enum

#%%
class SOTTrial(Enum):
    '''
    Enumeration for SOT trial types
    '''
    Eyes_Open_Fixed_Surround_and_Support = 1
    Eyes_Closed_Fixed_Support = 2
    Open_Sway_Referenced_Surrond = 3
    Eyes_Open_Sway_Referenced_Support = 4
    Eyes_Closed_Sway_Referenced_Support = 5
    Eyes_Open_Sway_Referenced_Surround_and_Support = 6


class SwayMetric(Enum):
    ''' 
    Enumeration for sway metric 
    '''
    ALL = 0
    RDIST = 1
    RDIST_ML = 2
    RDIST_AP = 3
    MDIST = 4
    MDIST_ML = 5
    MDIST_AP = 6
    TOTEX = 7
    TOTEX_ML = 8
    TOTEX_AP = 9
    MVELO = 10
    MVELO_ML = 11
    MVELO_AP = 12
    MFREQ = 13
    MFREQ_ML = 14
    MFREQ_AP = 15
    AREA_CE = 16
    FRAC_DIM = 17
    ROMBERG_RATIO = 18
    ROMBERG_RATIO_FOAM = 19
    
    
class DeviceType(Enum):
    ''' 
    Enumeration for sway metric 
    '''
    BALANCE_MASTER = 1
    KINECT = 2

class SwayGroup(Enum):
    '''
    Enumeration for sway groups
    '''
    All = 0
    All_healthy = 1
    All_fallers = 2
    
    Young = 3
    Middle = 4
    Old = 5
    
    Faller_by_history = 6
    Faller_by_history_single = 7
    Faller_by_history_muliple = 8
    Faller_by_miniBEStest = 9
    
    Young_vs_old = 10
    Old_vs_all_fallers = 11
    Old_vs_single_fallers = 12
    Young_vs_all_fallers = 13
    Young_and_Middle = 14

#%%
def get_unique_values(full_list):
    '''
    Gets unique values for a list, but preserves order, unlike numpy.unnique
    '''
    
    unnique_list = []
    
    for item in full_list:
        if item not in unnique_list:
            unnique_list.append(item)
            
    return unnique_list


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    '''
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    '''
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)


     # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)
    angle = np.degrees(theta)

    # get Width and height of ellipse
    width, height = 2 * n_std * np.sqrt(eigvals)
    area = (width/2) * (height/2) * np.pi

    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                     angle=angle, facecolor=facecolor, **kwargs)

    #ellipse.set_transform(transf + ax.transData)
    #return ax.add_patch(ellipse), width, height, area, angle #ell_radius_x, ell_radius_y
    return ax.add_patch(ell), width, height, area, angle #ell_radius_x, ell_radius_y
    # render plot with "plt.show()".


def filter_signal(ML_path, AP_path, UD_path=[], N=1, fc=10, fs=30):

    #N = 1 #1 2
    #fc = 10 #10 6  # Cut-off frequency of the filter
    #fs = 30
    #Wn = fc / (fs / 2) # Normalize the frequency
    
    Wn = np.pi * fc / (2 * fs) # Normalize the frequency
    
    b, a = signal.butter(N, Wn, 'low', fs=fs)
    filtered_ML_path = signal.filtfilt(b, a, ML_path)
    filtered_AP_path = signal.filtfilt(b, a, AP_path)
    
    if UD_path != []:
        filtered_UD_path = signal.filtfilt(b, a, UD_path)

    if UD_path != []:
        return filtered_ML_path, filtered_AP_path, filtered_UD_path
    else:
        return filtered_ML_path, filtered_AP_path


def calculate_RD(selected_recording,
               deviceType = DeviceType.KINECT,
               rd_path = ''):
    ''' 
        Calulate resultant distance, 
        that is the distance from each point on the raw CoM path to the 
        mean of the time series ad displays RD path and sway area 
        
        Saves AREA_CE image if rd_parh is filled in
        
    '''

    if deviceType == DeviceType.KINECT:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGz'].values.astype(float)
        ML_path, AP_path = filter_signal(ML_path, AP_path)
        
    elif deviceType == DeviceType.BALANCE_MASTER:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGy'].values.astype(float)

    mean_ML = np.mean(ML_path)
    mean_AP = np.mean(AP_path)

    ML_RD = np.sqrt(np.square(np.subtract(ML_path, mean_ML)))
    AP_RD = np.sqrt(np.square(np.subtract(AP_path, mean_AP)))
    #ML_RD =  np.abs(np.subtract(ML_path, mean_ML))
    #AP_RD =  np.abs(np.subtract(AP_path, mean_AP))
    
    #get hypotinuse of ML_RD and AP_RD
    RD = np.sqrt(np.add(np.square(ML_RD),np.square(AP_RD)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    
    elp, width, height, AREA_CE, _ = confidence_ellipse(ML_path, AP_path, ax, n_std=1.96, edgecolor='red')
    ax.scatter(ML_path, AP_path, s=3)
    
    #area = height * width * np.pi
    AREA_CE = round(AREA_CE, 2)
    #elp_AREA = (elp.height/2) * (elp.width/2) * np.pi
    #ax.set_title(deviceType.name + ' CoM ' + str(round(elp.height, 1)) + ' x ' + str(round(elp.width, 1)) + ' area:' + str(AREA_CE))
    ax.set_title(deviceType.name + ' CoM ' + 'W:' + str(round(width/2, 2)) + ' cm' + ' x ' + 'H:' + str(round(height/2, 2)) + ' cm' + ' Ar:' + str(AREA_CE) + ' cm sq')
    
    if rd_path != '':
        plt.savefig(rd_path)

    plt.show()

    return ML_RD, AP_RD, RD, AREA_CE


def calculate_RD_3D(selected_recording,
                    deviceType = DeviceType.KINECT,):
    ''' Calulate resultant distance '''

    if deviceType == DeviceType.KINECT:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGz'].values.astype(float)
        UD_path = selected_recording['CoGy'].values.astype(float)
        ML_path, AP_path = filter_signal(ML_path, AP_path)
    elif deviceType == DeviceType.BALANCE_MASTER:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGy'].values.astype(float)

    mean_ML = np.mean(ML_path)
    mean_AP = np.mean(AP_path)
    mean_UD = np.mean(UD_path)

    ML_RD = np.sqrt(np.square(np.subtract(ML_path, mean_ML)))
    AP_RD = np.sqrt(np.square(np.subtract(AP_path, mean_AP)))
    UD_RD = np.sqrt(np.square(np.subtract(UD_path, mean_UD)))
    RD = np.sqrt(np.square(np.subtract(ML_path, mean_ML)) + np.square(np.subtract(AP_path, mean_AP)) + np.square(np.subtract(UD_path, mean_UD)))

#    ML_RD =  np.abs(np.subtract(ML_path, mean_ML))
#    AP_RD =  np.abs(np.subtract(AP_path, mean_AP))
#    RD =  np.add(ML_RD, AP_RD)

    return ML_RD, AP_RD, UD_RD, RD


def calulate_TOTEX_from_RD(ML_RD, AP_RD, RD):

    arr_ML_diff = []
    arr_AP_diff = []
    arr_tot_diff = []

    ML_TOTEX = 0
    AP_TOTEX = 0
    TOTEX = 0

    for step in range(1, len(RD)):
        ML_curr = ML_RD[step]
        AP_curr = AP_RD[step]
        RD_curr = RD[step]

        ML_prev = ML_RD[step - 1]
        AP_prev = AP_RD[step - 1]
        RD_prev = RD[step - 1]

        ML_diff = abs(ML_curr - ML_prev)
        AP_diff = abs(AP_curr - AP_prev)

        arr_ML_diff.append(ML_diff)
        arr_AP_diff.append(AP_diff)
        #arr_tot_diff.append(ML_diff + AP_diff)
        
        # get hypotinuse of ML and AP diff
        RD_diff = np.sqrt(np.add(np.square(ML_diff), np.square(AP_diff)))
        arr_tot_diff.append(RD_diff)

    ML_TOTEX = np.sum(arr_ML_diff)
    AP_TOTEX = np.sum(arr_AP_diff)
    TOTEX = np.sum(arr_tot_diff)
    
    N = len(RD)
    d = max(arr_tot_diff)
    FD = np.log(N) / np.log((N * d) / TOTEX)

    return ML_TOTEX, AP_TOTEX, TOTEX, FD


def calculate_sway_from_recording(selected_recording,
                                  selected_recording_name,
                                  pID,
                                  age,
                                  sex,
                                  SOT_trial_type,
                                  tNum,
                                  swayMetric = SwayMetric.ALL,
                                  deviceType = DeviceType.KINECT,
                                  dp = -1,
                                  rd_path = '',
                                  start = 0,
                                  end = 600):
    
    '''
    Calculates sway from Kinect or Balance master recordings
    
    '''

    #filtered_ML_path, filtered_AP_path = filter_signal(selected_recording, deviceType)
    
    cliped_recording = selected_recording[start : end]

    ML_RD, AP_RD, RD, AREA_CE = calculate_RD(cliped_recording, deviceType, rd_path)
    recording_length = len(RD)

    #--mean DIST and rms DIST
    MDIST_ML = np.sum(ML_RD)/len(ML_RD)
    MDIST_AP = np.sum(RD)/len(AP_RD)
    MDIST = np.sum(RD) / recording_length

    RDIST_ML = np.sqrt(np.sum(np.square(ML_RD) / recording_length))
    RDIST_AP = np.sqrt(np.sum(np.square(AP_RD) / recording_length))
    RDIST = np.sqrt(np.sum(np.square(RD) / recording_length))

    #--Total Excursion - TOTEX
    TOTEX_ML, TOTEX_AP, TOTEX, FD = calulate_TOTEX_from_RD(ML_RD, AP_RD, RD)

    #--Mean Velocity - MVELO
    if deviceType == DeviceType.KINECT:
        T = recording_length / 30
    elif  deviceType == DeviceType.BALANCE_MASTER:
        T = recording_length / 100

    MVELO_ML = TOTEX_ML / T
    MVELO_AP = TOTEX_AP / T
    MVELO = TOTEX / T

    #--Mean Fequency - MFREQ
    MFREQ_ML = MVELO_ML / (4*(np.sqrt(2 * MDIST_ML)))
    MFREQ_AP = MVELO_AP / (4*(np.sqrt(2 * MDIST_AP)))
    MFREQ = MVELO / (2 * np.pi * MDIST)

#    ax = plt.scatter(ML_RD, AP_RD, marker='.')
#    plt.title(deviceType.name + ' RD path ' + pID  + ' ' + str(round(T, 1)) + ' sec')
#    plt.show()


    #TOTEX MVELO RDIST
    if swayMetric == SwayMetric.RDIST:
        swayVal = RDIST
    elif swayMetric == SwayMetric.RDIST_ML:
        swayVal = RDIST_ML
    elif swayMetric == SwayMetric.RDIST_AP:
        swayVal = RDIST_AP

    elif swayMetric == SwayMetric.MDIST:
        swayVal = MDIST
    elif swayMetric == SwayMetric.MDIST_ML:
        swayVal = MDIST_ML
    elif swayMetric == SwayMetric.MDIST_AP:
        swayVal = MDIST_AP

    elif swayMetric == SwayMetric.TOTEX:
        swayVal = TOTEX
    elif swayMetric == SwayMetric.TOTEX_ML:
        swayVal = TOTEX_ML
    elif swayMetric == SwayMetric.TOTEX_AP:
        swayVal = TOTEX_AP

    elif swayMetric == SwayMetric.MVELO:
        swayVal = MVELO
    elif swayMetric == SwayMetric.MVELO_ML:
        swayVal = MVELO_ML
    elif swayMetric == SwayMetric.MVELO_AP:
        swayVal = MVELO_AP

    elif swayMetric == SwayMetric.MFREQ:
        swayVal = MFREQ
    elif swayMetric == SwayMetric.MFREQ_ML:
        swayVal = MFREQ_ML
    elif swayMetric == SwayMetric.MVELO_AP:
        swayVal = MFREQ_AP

    elif swayMetric == SwayMetric.AREA_CE:
        swayVal = AREA_CE
        
    elif swayMetric == SwayMetric.FRAC_DIM:
        swayVal = FD

    tmpSway = []

    if dp != -1:
        swayVal = round(swayVal, dp)
    
    if swayMetric == SwayMetric.ALL:
        tmpSway.append([pID, selected_recording_name, tNum, age, sex, swayMetric.name,
                        RDIST_ML, RDIST_AP, RDIST,
                        MDIST_ML, MDIST_AP, MDIST,
                        TOTEX_ML, TOTEX_AP, TOTEX,
                        MVELO_ML, MVELO_AP, MVELO,
                        MFREQ_ML, MFREQ_AP, MFREQ,
                        AREA_CE])
    else:
        tmpSway.append([pID, selected_recording_name, tNum, age, sex, swayMetric.name,
                        swayVal])


    return tmpSway

# In[Balance master Utils]
def load_balance_master_file(rootDir,
                             participantID,
                             age,
                             kinectTrialType,
                             trialNumber = 1,
                             swayMetric = SwayMetric.RDIST):

    root = ''
    dirs = []
    columns = ''
    arrayOfRows = []
    dfFinal = pd.DataFrame(arrayOfRows)

    for root, dirs, _ in os.walk(rootDir):
        break

    dirs.sort()

    selected_trial = ''
    for dirName in dirs:
        if participantID in dirName:
            print(dirName)
            rootFilePath = os.path.join(root, dirName, 'cm')

            trialRoot = ''
            trialfiles = []
            for trialRoot, _, trialfiles in os.walk(rootFilePath):
                break

            found = False
            for trial in trialfiles:
                #if 'SOT' in trial and str('T'+ str(kinectTrialType.value)) in trial:
                if 'SOT' in trial and str('C'+ str(kinectTrialType.value)) in trial and str('T'+ str(trialNumber)) in trial:
                    print('Collating:', trialRoot + '/' + trial)
                    selected_trial = trial
                    trialFilePath = os.path.join(trialRoot, trial)

                    bMTrial = pd.read_csv(trialFilePath
                                          , sep="\t")


                    i = 0

                    for row in bMTrial[25:].iterrows():
                        parsedRow = row[1][0]
                        arrRow = parsedRow.split('\t')
                        strArrRow = str.split(arrRow[0])

                        if i == 0:
                            #columns = strArrRow
                            columns = ['DP', 'LF', 'RR', 'SH', 'LR', 'RF', 'CoFx', 'CoFy', 'CoGx', 'CoGy']
                        else:
                            arrayOfRows.append(strArrRow)

                        i += 1

                    found = True
                    break

            if found:
                dfFinal = pd.DataFrame(arrayOfRows, columns=columns)
            else:
                print('can not find ', trialRoot + '/' + trial)

            #break


    return dfFinal, selected_trial
    