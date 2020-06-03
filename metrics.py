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

import math
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
    

SPINEBASE = 0
SPINEMID = 1
NECK = 2
HEAD = 3
SHOULDERLEFT = 4
ELBOWLEFT = 5
WRISTLEFT = 6
HANDLEFT = 7
SHOULDERRIGHT = 8
ELBOWRIGHT = 9
WRISTRIGHT = 10
HANDRIGHT = 11
HIPLEFT = 12
KNEELEFT = 13
ANKLELEFT = 14
FOOTLEFT = 15
HIPRIGHT = 16
KNEERIGHT = 17
ANKLERIGHT = 18
FOOTRIGHT = 19
SPINESHOULDER = 20
HANDTIPLEFT = 21
THUMBLEFT = 22
HANDTIPRIGHT = 23
THUMBRIGHT = 24

#%%
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_angle_between_two_joints(j1, j2):
    ''' Returns the angle in degrees between vectors 'v1' and 'v2'::
015044
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966 rad / 90 deg
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0 rad / 0.0 deg
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793 rad / 180 deg
    '''
    v1_u = unit_vector(j1)
    v2_u = unit_vector(j2)
    rad_between_two_joints = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    #deg_between_two_joints = (rad_between_two_joints * 180) / np.pi
    deg_between_two_joints = np.degrees(rad_between_two_joints)
    
    return deg_between_two_joints


def normalise_skeleton(skel_frame, spine_base_joint):
    normalised_skel_frame = np.copy(skel_frame)
    x = 0
    y = 1
    z = 2

    for i in range(skel_frame.shape[0]):
        normalised_skel_frame[i][2] =  str(100*(float(skel_frame[i][2]) - spine_base_joint[x]))
        normalised_skel_frame[i][3] =  str(100*(float(skel_frame[i][3]) - spine_base_joint[y]))
        normalised_skel_frame[i][4] =  str(100*(float(skel_frame[i][4]) - spine_base_joint[z]))

    return normalised_skel_frame


def calculate_com(skelFrame):
    _X = 2
    _Y = 3
    _Z = 4

    spine_shoulder = np.stack([float(skelFrame[SPINESHOULDER, _X]), float(skelFrame[SPINESHOULDER, _Y]), float(skelFrame[SPINESHOULDER, _Z])], axis=0)
    spine_base = np.stack([float(skelFrame[SPINEBASE, _X]), float(skelFrame[SPINEBASE, _Y]), float(skelFrame[SPINEBASE, _Z])], axis=0)
    spine_mid = np.stack([float(skelFrame[SPINEMID, _X]), float(skelFrame[SPINEMID, _Y]), float(skelFrame[SPINEMID, _Z])], axis=0)
    hip_left = np.stack([float(skelFrame[HIPLEFT, _X,]), float(skelFrame[HIPLEFT, _Y,]), float(skelFrame[HIPLEFT, _Z])], axis=0)
    hip_right = np.stack([float(skelFrame[HIPRIGHT, _X]), float(skelFrame[HIPRIGHT, _Y]), float(skelFrame[HIPRIGHT, _Z])], axis=0)

    #xMean = np.mean([spine_shoulder[0],hip_left[0],hip_right[0]])
    #yMean = np.mean([spine_shoulder[1],hip_left[1],hip_right[1]])
    #zMean = np.mean([spine_shoulder[2],hip_left[2],hip_right[2]])

    #xMean = np.mean([spine_base[0],hip_left[0],hip_right[0]])
    #yMean = np.mean([spine_base[1],hip_left[1],hip_right[1]])
    #zMean = np.mean([spine_base[2],hip_left[2],hip_right[2]])
    
    xMean = np.mean([spine_mid[0],hip_left[0],hip_right[0]])
    yMean = np.mean([spine_mid[1],hip_left[1],hip_right[1]])
    zMean = np.mean([spine_mid[2],hip_left[2],hip_right[2]])
    
    com = np.stack([xMean,yMean,zMean],axis=0)

    return com.tolist()


def euclidean_distance_between_joints(j1, j2):
    ed = np.sqrt(np.square(j1[0] - j2[0]) + np.square(j1[1] - j2[1]) + np.square(j1[2] - j2[2]))
    #ed = np.sqrt(np.square(j1[0] - j2[0]) + 0 + np.square(j1[2] - j2[2]))
    #ed = np.sqrt(np.square(j1[0] - j2[0]) + np.square(j1[1] - j2[1]) + 0)  
    #ed = np.sqrt(0 + np.square(j1[1] - j2[1]) + np.square(j1[2] - j2[2]))
    
    return ed

def get_joint_XYZ(skel_frame_row):
    x = float(skel_frame_row[2])
    y = float(skel_frame_row[3])
    z = float(skel_frame_row[4])

    return [x, y, z]

def mean_twin_joint_pos(j_l, j_r):
    mean_x = np.mean([j_l[0], j_r[0]])
    mean_y = np.mean([j_l[1], j_r[1]])
    mean_z = np.mean([j_l[2], j_r[2]])

    floor = [mean_x, mean_y, mean_z] # this is the floor if you pass in feat

    return floor


def get_angle_between_three_joints_old(j1, j2, j3):
    
    l1 = euclidean_distance_between_joints(j1, j2)
    l2 = euclidean_distance_between_joints(j2, j3)
    
    rad_between_three_joints = np.arccos(l2/l1)
    deg_between_three_joints = np.degrees(rad_between_three_joints) 
    
    return deg_between_three_joints


def get_angle_between_three_joints_by_matirx(j1, j2, j3):
    
    #l1 = euclidean_distance_between_joints(j1, j2)
    #l2 = euclidean_distance_between_joints(j2, j3)
    
    v_j1_To_j2 = [j1[0] - j2[0], j1[1] - j2[1], j1[2] - j2[2]]
    v_j2_To_j3 = [j2[0] - j3[0], j2[1] - j3[1], j2[2] - j3[2]]
    
    #v_j1_To_j2 = [j1[0] - j2[0], j1[1] - j2[1], 0]
    #v_j2_To_j3 = [j2[0] - j3[0], j2[1] - j3[1], 0]
    
    v_j1_To_j2_u = unit_vector(v_j1_To_j2)
    v_j2_To_j3_u = unit_vector(v_j2_To_j3)
    
    #cross_product = np.cross(v_j1_To_j2_u, v_j2_To_j3_u)
    #cross_productLength = cross_product[2]
    #dot_product = np.dot(v_j1_To_j2_u, v_j2_To_j3_u)
    #rad_between_three_joints = np.arctan2(cross_productLength, dot_product)
    #deg_between_three_joints = np.degrees(rad_between_three_joints)
    
    rad_between_three_joints2 = np.arccos(np.clip(np.dot(v_j1_To_j2_u, v_j2_To_j3_u), -1.0, 1.0))
    deg_between_three_joints2 = 180 - np.degrees(rad_between_three_joints2)
    
    return deg_between_three_joints2


def get_angle_between_three_joints(j1, j2, j3):
    
    #useing cosign rule
    
    a = euclidean_distance_between_joints(j1, j2)
    b = euclidean_distance_between_joints(j2, j3)
    c = euclidean_distance_between_joints(j3, j1)

    angle_C = np.degrees(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))
    
    return angle_C

def get_angle_between_two_vectors(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    rad_angle = np.arccos(dot_product)
    deg_angle = np.degrees(rad_angle)
    
    return deg_angle    

def get_atan_angle(j1, j2, j3):
    ba = j1 - j2
    bc = j3 - j2
    
    ba = np.linalg.norm(ba)
    bc = np.linalg.norm(bc)

    cross_product = np.cross(ba, bc)
    cross_product_length = cross_product
    dot_product = np.cross(ba, bc)
    
    rad_angle = math.atan2(cross_product_length, dot_product)
    deg_angle = np.degrees(rad_angle) 

    return deg_angle


def get_angle(j1, j2, j3):
    ba = j1 - j2
    bc = j3 - j2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    rad_angle = np.arccos(cosine_angle)
    deg_angle = np.degrees(rad_angle) 

    return deg_angle


def get_AP_angle_between_three_joints_matrix(j1, j2, j3):
    
    l1 =  np.sqrt(np.square(j1[2] - j2[2]))
    l2 =  np.sqrt(np.square(j2[2] - j3[2]))
    
    rad_between_three_joints = np.arccos(l2/l1)
    deg_between_three_joints = np.degrees(rad_between_three_joints) 
    
    return deg_between_three_joints

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


def confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', **kwargs):
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
        1.96 SD = 95% confidence ellipse
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
        
    ellipse.set_transform(transf + ax.transData)
    
    scaled_ell_radius_x =  ell_radius_x * scale_x * 2
    scaled_ell_radius_y =  ell_radius_y * scale_y * 2
    
    width = scaled_ell_radius_x * 2
    height = scaled_ell_radius_y * 2
    area = np.pi * scaled_ell_radius_x * scaled_ell_radius_y
    
    # width = ellipse.width
    # height = ellipse.height
    # area = np.pi * (ellipse.width/2) * (ellipse.height/2)
    
    return ax.add_patch(ellipse), width, height, area #, angle #ell_radius_x, ell_radius_y
    

    '''
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
    return ax.add_patch(ell), width, height, area, angle #ell_radius_x, ell_radius_y
    '''

    
    
    # render plot with "plt.show()".


def filter_signal(ML_path, AP_path=[], CC_path=np.array([]), N=2, fc=10, fs=30):
    '''
    N = order of the filer - usually 1, 2 or 4
    fc = Cut-off frequency of the filter - usually 10 or 6 
    fs = 30
    Wn = fc / (fs / 2) # Normalize the frequency
    '''
    
    Wn = np.pi * fc / (2 * fs) # Normalize the frequency
    
    b, a = signal.butter(N, Wn, 'low', fs=fs)
    filtered_ML_path = signal.filtfilt(b, a, ML_path)
    
    if AP_path != []:
        filtered_AP_path = signal.filtfilt(b, a, AP_path)
    
    if CC_path != []:
        filtered_CC_path = signal.filtfilt(b, a, CC_path)

    if CC_path != []:
        return filtered_ML_path, filtered_AP_path, filtered_CC_path
    if AP_path != []:
        return filtered_ML_path, filtered_AP_path
    else:
        return filtered_ML_path

def calculate_RD(selected_recording,
               deviceType = DeviceType.KINECT,
               rd_path = '',
               part_id = '',
               SOT_trial_type = ''):
    ''' 
        Calulate resultant distance, 
        that is the distance from each point on the raw CoM path to the 
        mean of the time series ad displays RD path and sway area 
        
        Saves AREA_CE image if rd_parh is filled in
        
    '''

    if deviceType == DeviceType.KINECT:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGz'].values.astype(float)
        ML_path, AP_path = filter_signal(ML_path, AP_path, fc=8)
        
    elif deviceType == DeviceType.BALANCE_MASTER:
        ML_path = selected_recording['CoGx'].values.astype(float)
        AP_path = selected_recording['CoGy'].values.astype(float)

    mean_ML = np.mean(ML_path)
    mean_AP = np.mean(AP_path)

    #ML =  np.subtract(ML_path, mean_ML)
    #AP =  np.subtract(AP_path, mean_AP)
    ML =  np.abs(np.subtract(ML_path, mean_ML))
    AP =  np.abs(np.subtract(AP_path, mean_AP))
    #ML = np.sqrt(np.square(np.subtract(ML_path, mean_ML)))
    #AP = np.sqrt(np.square(np.subtract(AP_path, mean_AP)))
    
    #get hypotinuse of ML_RD and AP_RD
    RD = np.sqrt(np.add(np.square(ML),np.square(AP)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    
    elp, width, height, AREA_CE = confidence_ellipse(ML_path, AP_path, ax, n_std=1.96, edgecolor='red')
    ax.scatter(ML_path, AP_path, s=3)
    
    #area = height * width * np.pi
    AREA_CE = round(AREA_CE, 2)
    #elp_AREA = (elp.height/2) * (elp.width/2) * np.pi
    #ax.set_title(deviceType.name + ' CoM ' + str(round(elp.height, 1)) + ' x ' + str(round(elp.width, 1)) + ' area:' + str(AREA_CE))
    #ax.set_title(deviceType.name + ' CoM ' + 'W:' + str(round(width/2, 2)) + ' cm' + ' x ' + 'H:' + str(round(height/2, 2)) + ' cm' + ' Ar:' + str(AREA_CE) + ' cm sq')
    
    ax.set_title(str.replace(SOT_trial_type, '-', ' ') + ' ' + part_id + ' CoM 95% CE' + 
                  '\nW:' + str(round(width, 2)) + ' cm' + 
                  ' x ' + 
                  'H:' + str(round(height, 2)) + ' cm' + 
                  ' Ar:' + str(AREA_CE) + ' cm sq')
    
    ax.set_aspect('equal')

    if rd_path != '':
        plt.savefig(rd_path)
    
    plt.show()
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_xlim([-8, 8])
    #ax.set_ylim([-8, 8])
    
    elp, width, height, AREA_CE = confidence_ellipse(ML_path, AP_path, ax, n_std=1.96, edgecolor='red')
    ax.scatter(ML_path, AP_path, s=3)
    
    #area = height * width * np.pi
    AREA_CE = round(AREA_CE, 2)
    #elp_AREA = (elp.height/2) * (elp.width/2) * np.pi
    #ax.set_title(deviceType.name + ' CoM ' + str(round(elp.height, 1)) + ' x ' + str(round(elp.width, 1)) + ' area:' + str(AREA_CE))
    #ax.set_title(deviceType.name + ' CoM ' + 'W:' + str(round(width/2, 2)) + ' cm' + ' x ' + 'H:' + str(round(height/2, 2)) + ' cm' + ' Ar:' + str(AREA_CE) + ' cm sq')
    ax.set_title(str.replace(SOT_trial_type, '-', ' ') + ' ' + part_id + ' CoM 95% CE' + 
                 '\nW:' + str(round(width, 2)) + ' cm' + 
                 ' x ' + 
                 'H:' + str(round(height, 2)) + ' cm' + 
                 ' Ar:' + str(AREA_CE) + ' cm sq')
    
    ax.set_aspect('equal')
    
    if rd_path != '':
        detailed_rd_path = rd_path.replace('confidence_ellipse', 'confidence_ellipse_detailed')
        plt.savefig(detailed_rd_path)

    plt.show()

    return ML, AP, RD, AREA_CE


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


def calculate_angle_change_fequencey():
    return


def calculate_TOTEX(ML, AP):

    arr_ML_diff = []
    arr_AP_diff = []
    arr_tot_diff = []

    ML_TOTEX = 0
    AP_TOTEX = 0
    TOTEX = 0

    for step in range(1, len(AP)):
        ML_curr = ML[step]
        AP_curr = AP[step]
        #RD_curr = RD[step]

        ML_prev = ML[step - 1]
        AP_prev = AP[step - 1]
        #RD_prev = RD[step - 1]

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
    
    N = len(AP)
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
                                  impairment_self = 'healthy',
                                  impairment_confedence = 'healthy',
                                  impairment_clinical = 'healthy',
                                  impairment_stats = 'healthy',
                                  dp = -1,
                                  rd_path = '',
                                  start = 0,
                                  end = 600):
    
    '''
    Calculates sway from Kinect or Balance master recordings
    
    '''

    #filtered_ML_path, filtered_AP_path = filter_signal(selected_recording, deviceType)
    
    cliped_recording = selected_recording[start : end]

    ML, AP, RD, AREA_CE = calculate_RD(cliped_recording, deviceType, rd_path, pID, SOT_trial_type)
    recording_length = len(RD)

    #--mean DIST and rms DIST
    MDIST_ML = np.sum(ML) / recording_length
    MDIST_AP = np.sum(AP) / recording_length
    MDIST = np.sum(RD) / recording_length

    RDIST_ML = np.sqrt(np.sum(np.square(ML) / recording_length))
    RDIST_AP = np.sqrt(np.sum(np.square(AP) / recording_length))
    RDIST = np.sqrt(np.sum(np.square(RD) / recording_length))
    #rms = np.sqrt(np.mean(RD**2))

    #--Total Excursion - TOTEX
    TOTEX_ML, TOTEX_AP, TOTEX, FD = calculate_TOTEX(ML, AP)

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
    
    #MFREQ_1 = TOTEX / (2 * np.pi * MDIST * T)
    #MFREQ_AP_1 = TOTEX_AP / (4*(np.sqrt(2 * MDIST_AP ))  * T)

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
        tmpSway.append([pID, selected_recording_name, tNum, age, sex, 
                        impairment_self, impairment_confedence, 
                        impairment_clinical, impairment_stats,
                        swayMetric.name,
                        RDIST_ML, RDIST_AP, RDIST,
                        MDIST_ML, MDIST_AP, MDIST,
                        TOTEX_ML, TOTEX_AP, TOTEX,
                        MVELO_ML, MVELO_AP, MVELO,
                        MFREQ_ML, MFREQ_AP, MFREQ,
                        AREA_CE])
    else:
        tmpSway.append([pID, selected_recording_name, tNum, age, sex, 
                        impairment_self, impairment_confedence, 
                        impairment_clinical, impairment_stats,
                        swayMetric.name,
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
    

#%%
#A = [3,2,1,3,4,2,1]
#A_mean = np.mean(A)
#A_ml = np.subtract(A, A_mean)
#X = list(range(len(A)))
#Y = np.full(len(A), A_mean)
#
#plt.plot(A)
#plt.plot(A_ml)
#plt.plot(X, Y)
#plt.show()
    
#%%

#from scipy import stats
#
#n_std = 1.96
#x = np.random.randint(1,100,100)
#y = x*3 #np.random.randint(1,100,100)
#
#cov = np.cov(x, y)
#pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
#
#pearsonr, _ = stats.pearsonr(x, y)
#
#ell_radius_x = np.sqrt(1 + pearson)
#ell_radius_y = np.sqrt(1 - pearson)
#
#scale_x = np.sqrt(cov[0, 0]) * n_std
#mean_x = np.mean(x)
#
## calculating the stdandard deviation of y ...
#scale_y = np.sqrt(cov[1, 1]) * n_std
#mean_y = np.mean(y)
#
## Find and sort eigenvalues and eigenvectors into descending order
##eigvals, eigvecs = np.linalg.eigh(cov)
##order = eigvals.argsort()[::-1]
##eigvals, eigvecs = eigvals[order], eigvecs[:, order]
#
## The anti-clockwise angle to rotate our ellipse by
##vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
##theta = np.arctan2(vy, vx)
##angle = np.degrees(theta)
#
## get Width and height of ellipse
##width, height = 2 * n_std * np.sqrt(eigvals)
##area = (width/2) * (height/2) * np.pi
#scaled_ell_radius_x = ell_radius_x * scale_x
#scaled_ell_radius_y = ell_radius_y * scale_y
#
#area = np.pi * scaled_ell_radius_x * scaled_ell_radius_y
