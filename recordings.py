#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:53:26 2019

@author: 55129822
"""
'''
    consider tracked?
'''

import numpy as np
import pandas as pd
import os

from scipy import signal
from scipy import stats
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from mpl_toolkits.mplot3d import Axes3D

import os
import sys
from tqdm import tqdm

from enum import Enum
import re

from mgen import rotation_from_angles
from mgen import rotation_around_x, rotation_around_y, rotation_around_z


sys.path.append(os.path.abspath('../'))

from sway_utils import metrics as sm
from Utils.Utils import walkFileSystem, get_part_dirs


import csv

import transforms3d

from pytransform3d.rotations import *

from squaternion import Quaternion

#%%
class SkeletonJoints(Enum):
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
    COM = 25
    #HEELLEFT = 26
    #HEELRIGHT = 27
    
    
class HierarchicalSkeletonJoints(Enum):
    COM = 25
    
    HEAD = 3
    NECK = 2
    SPINESHOULDER = 20
    SPINEMID = 1
    SPINEBASE = 0
    
    SHOULDERLEFT = 4
    SHOULDERRIGHT = 8
    
    ELBOWLEFT = 5
    ELBOWRIGHT = 9
    
    WRISTLEFT = 6
    WRISTRIGHT = 10
    
    HIPLEFT = 12
    HIPRIGHT = 16
    
    KNEELEFT = 13
    KNEERIGHT = 17
    
    ANKLELEFT = 14
    ANKLERIGHT = 18
    
    FOOTLEFT = 15
    FOOTRIGHT = 19
    
    HANDLEFT = 7
    HANDTIPLEFT = 21
    THUMBLEFT = 22
    
    HANDRIGHT = 11
    HANDTIPRIGHT = 23
    THUMBRIGHT = 24
    
    
class SkeletonJointAngles(Enum):
    BODY_COM_ANGLE = 1
    BODY_LEAN_ANGLE = 2
    KNEELEFT_ANGLE = 3
    KNEERIGHT_ANGLE = 4
    HIPLEFT_ANGLE = 5
    HIPRIGHT_ANGLE = 6
    ELBOWLEFT_ANGLE = 7
    ELBOWLRIGHT_ANGLE = 8
    ARMPITLEFT_ANGLE = 9
    ARMPITROIGHT_ANGLE = 10
    ANKLELEFT_ANGLE = 11
    ANKLELRIGHT_ANGLE = 12
    
class SkeletonIJCP(Enum):
    '''
    https://www.mdpi.com/1424-8220/20/5/1291
    (1) head, (2) neck, 
    (3) shoulder center (SPINESHOULDER), (4) left shoulder, (5) right shoulder, 
    (6) trunk center (SPINEMID), 
    (7) left elbow, (8) right elbow, 
    (9) hip center (SPINEBASE), (10)left hip, (11) right hip, 
    (12) left hand, (13) right hand, 
    (14) left knee, and (15) right knee
    '''
    
    HEAD = 3
    NECK = 2
    
    SPINESHOULDER = 20
    SHOULDERLEFT = 4
    SHOULDERRIGHT = 8
    
    SPINEMID = 1
    
    ELBOWLEFT = 5
    ELBOWRIGHT = 9
    
    SPINEBASE = 0
    HIPLEFT = 12
    HIPRIGHT = 16
    
    HANDLEFT = 7
    HANDRIGHT = 11
    #WRISTLEFT = 6
    #WRISTRIGHT = 10
    
    KNEELEFT = 13
    KNEERIGHT = 17
    
    
    def get_joint_from_skel_frame(skel_frame_row):
        x = float(skel_frame_row[2])
        y = float(skel_frame_row[3])
        z = float(skel_frame_row[4])

        return [x, y, z]

    def get_joint_from_raw_XZY(skel_frame_row):
        x = float(skel_frame_row[0])
        y = float(skel_frame_row[1])
        z = float(skel_frame_row[2])

        return [x, y, z]


class WalkedSkelAngles(Enum):
    '''
    inspired by: 
    A. Vakanski, H. P. Jun, D. Paul, and R. Baker,
    “A data set of human body movements for physical rehabilitation exercises,”
    Data, vol. 3, no. 1, 2018.
    NB no head tip and no tip
    '''
    SPINEBASE_SPINEMID = [0, 1]
    SPINEMID_SPINESHOULDER = [1, 20]
    SPINESHOULDER_NECK = [20, 2]
    NECK_HEAD = [2, 3]
    
    SPINESHOULDER_SHOULDERLEFT = [20, 4]
    SHOULDERLEFT_ELBOWLEFT = [4, 5]
    ELBOWLEFT_WRISTLEFT = [5, 6]
    WRISTLEFT_HANDLEFT = [6, 7]
    
    SPINESHOULDER_SHOULDERRIGHT = [20, 8]
    SHOULDERRIGHT_ELBOWRIGHT = [8, 9]
    ELBOWRIGHT_WRISTRIGHT = [9, 10]
    WRISTRIGHT_HANDRIGHT = [10, 11]
    
    SPINEBASE_HIPLEFT = [0, 12]
    HIPLEFT_KNEELEFT = [12, 13]
    KNEELEFT_ANKLELEFT = [13, 14]
    ANKLELEFT_FOOTLEFT = [14, 15]
    
    SPINEBASE_HIPRIGHT = [0, 16]
    HIPRIGHT_KNEERIGHT = [16, 17]
    KNEERIGHT_ANKLERIGHT = [17, 18]
    ANKLERIGHT_FOOTRIGHT = [18, 19]
    
    
class WalkedSkelAnglesIn3s(Enum):
    '''
    Walk skeleton and get angles
    '''
    #Torso
    SPINEMID_a = [SkeletonJoints.SPINEBASE.value, 
                  SkeletonJoints.SPINEMID.value,
                  SkeletonJoints.SPINESHOULDER.value]
    SPINESHOULDER_a = [SkeletonJoints.SPINEMID.value,
                       SkeletonJoints.SPINESHOULDER.value,
                       SkeletonJoints.NECK.value]
    NECK_a = [SkeletonJoints.SPINESHOULDER.value,
              SkeletonJoints.NECK.value,
              SkeletonJoints.HEAD.value]
    
    #Upper body
    ARMPITLEFT_a = [SkeletonJoints.SPINESHOULDER.value,
                    SkeletonJoints.SHOULDERLEFT.value,
                    SkeletonJoints.ELBOWLEFT.value]
    ELBOWLEFT_a = [SkeletonJoints.SHOULDERLEFT.value,
                   SkeletonJoints.ELBOWLEFT.value,
                   SkeletonJoints.WRISTLEFT.value]
    
    ARMPITRIGHT_a = [SkeletonJoints.SPINESHOULDER.value,
                     SkeletonJoints.SHOULDERRIGHT.value,
                     SkeletonJoints.ELBOWRIGHT.value]
    ELBOWRIGHT_a = [SkeletonJoints.SHOULDERLEFT.value,
                    SkeletonJoints.ELBOWRIGHT.value,
                    SkeletonJoints.WRISTRIGHT.value]

    #Lower body
    HIPLEFT_a = [SkeletonJoints.SPINEBASE.value,
                 SkeletonJoints.HIPLEFT.value,
                 SkeletonJoints.KNEELEFT.value]
    KNEELEFT_a = [SkeletonJoints.HIPLEFT.value,
                  SkeletonJoints.KNEELEFT.value,
                  SkeletonJoints.ANKLELEFT.value]
    # ANKLELEFT_a = [SkeletonJoints.FOOTLEFT.value,
    #               SkeletonJoints.ANKLELEFT.value,
    #               SkeletonJoints.KNEELEFT.value,]
    
    HIPRIGHT_a = [SkeletonJoints.SPINEBASE.value,
                  SkeletonJoints.HIPRIGHT.value,
                  SkeletonJoints.KNEERIGHT.value]
    KNEERIGHT_a = [SkeletonJoints.HIPLEFT.value,
                   SkeletonJoints.KNEERIGHT.value,
                   SkeletonJoints.ANKLERIGHT.value]
    # ANKLELRIGHT_a = [SkeletonJoints.FOOTRIGHT.value,
    #               SkeletonJoints.ANKLERIGHT.value,
    #               SkeletonJoints.KNEERIGHT.value,]
          

X = 0
Y = 1
Z = 2

#%%
class KinectRecording:
    '''
        Class that represents a single kinect recording
    '''
    _root_path = ''
    _skel_root_path = '' 
    _skel_file_root = ''
    
    _dataset_prefix = ''
    _movement = ''
    _part_id = 0
    _labels = []
    _ref_spine_base = [] #spine base of first skel frame
    _ref_skel_frame = [] #whole      of first skel frame
    
    _frame_count = -1
    _load_from_cache = False

    _cached_stacked_raw_XYZ_file_path = ''
    _cached_skeletons_path = ''

    ''''
        Values from files
        skeletons: list of pandas DataFrames, representing 
        text files from kinect: 
            [features, SkeletonJoints] tipically, [6,26]
            fetures : Indes, Tracked, X, Y, Z, px_X, px_Y
    '''
    skeletons = []
    raw_XYZ_values = []
    
    ''''
        stacked XYZ values are flattend versions of the raw and filteterd 
        kinect recordings
        [XYZ, SkeletonJoints, #frames] typically,[3, 26, 600]
    '''
    stacked_raw_XYZ_values = []
    stacked_filtered_XYZ_values = []
    
    ''''
        stacked angle values represent the key angles of the 
        filtered XYZ values 
        [fetures, #frames] typically,[3, 600]
        featurs: COM_angle, knee_angle, hip_angle_side, hip_angle_front,
                lean_angle
    '''
    
    stacked_raw_angle_values = []
    stacked_filtered_angle_vlaues = []
    
    
    ''' 
        get sway metrics from file
    '''
    sway_metrics = []

    
    '''
        calulate and store    
        local inter-joint coordination pattern (IJCP)
    ''' 
    IJCP_Dist = []
    IJCP_Velo = []
    
    
    '''
        Calulate and store smc features
    '''
    
    smc_features = []
    
    '''
        calulate and store Walling Skel Angles
    '''
    walked_skel_angles = []
    
    
    def __init__(self, skel_root_path, dataset_prefix, movement, part_id, labels=[]):
        self._root_path = str.replace(skel_root_path, '/skel', '')
        self._root_path = str.replace(self._root_path, '\skel', '')
        self._dataset_prefix = dataset_prefix
        self._skel_root_path = skel_root_path
        self._movement = movement
        self._part_id = part_id
        self._labels = labels
        
        self.load_skeletons(skel_root_path)
        self.load_sway_metrics()
        
        
    def load_sway_metrics(self):
        sway_metric_path = os.path.join(self._root_path, 'sway_metrics.csv')
        if os.path.exists(sway_metric_path):
            self.sway_metrics = pd.read_csv(sway_metric_path)
        
        
    def save_stacked_raw_XYZ(self):
        if not os.path.exists(self._cached_stacked_raw_XYZ_file_path):
            np.save(self._cached_stacked_raw_XYZ_file_path, self.stacked_raw_XYZ_values)
            
            
    def save_skeletons(self):
        if not os.path.exists(self._cached_skeletons_path):
            #np.save(self._cached_skeletons_path, self.skeletons)
            pd.concat(self.skeletons).to_csv(self._cached_skeletons_path)
        
    
    def load_skeletons(self, skel_root_path):
        #If chahed file exists, load
        cached_stacked_raw_XYZ_file_name = (self._dataset_prefix +
                                            str(self._part_id) + '_' +
                                            'cached_stacked_raw_XYZ_' + 
                                            self._movement + '.npy')
        
        cached_skeletons_file_name = (self._dataset_prefix +
                                            str(self._part_id) + '_' +
                                            'cached_skeletons' + 
                                            self._movement + '.csv') 
                                            
        self._cached_stacked_raw_XYZ_file_path = os.path.join(self._root_path, 
                                                        cached_stacked_raw_XYZ_file_name) 
        
        self._cached_skeletons_path = os.path.join(self._root_path, 
                                                        cached_skeletons_file_name) 
        
        if os.path.exists(self._cached_stacked_raw_XYZ_file_path):
            self.stacked_raw_XYZ_values = np.load(self._cached_stacked_raw_XYZ_file_path)
            self._load_from_cache = True
            print('loading:', self._cached_stacked_raw_XYZ_file_path, '\n')
            
            
        if os.path.exists(self._cached_skeletons_path):
            self.skeletons = pd.read_csv(self._cached_skeletons_path)
            #self._load_from_cache = True
            print('loading:', self._cached_skeletons_path, '\n')    
        
        #else calulate stuff
        else:
            root, dirs, skel_files = walkFileSystem(skel_root_path)
            skel_files.sort()
            
            if len(skel_files) == 0:
                print('Skel files for:', self._cached_stacked_raw_XYZ_file_path)
         
            for skelfile in tqdm(skel_files):
            #for skelfile in skel_files:    
                skel_file_path = os.path.join(root, skelfile)
                _skel_frame, _raw_XYZ = self._load_skel_file(skel_file_path)
                '''skels are now normailesed'''
                
                self.skeletons.append(_skel_frame)
                self.raw_XYZ_values.append(_raw_XYZ)
                
                #dont include first frame, having a skel base of 0,0,0 can cause problems
                if self._frame_count > 0:
                    if len(self.stacked_raw_XYZ_values) == 0:
                        self.stacked_raw_XYZ_values = _raw_XYZ
                    else:
                        self.stacked_raw_XYZ_values = np.dstack([self.stacked_raw_XYZ_values, _raw_XYZ])
                
            #remove first frame, having a skel base of 0,0,0 can cause problems
            #self.stacked_raw_XYZ_values = self.stacked_raw_XYZ_values[:,:,1:]
        
        #Save
        self.save_stacked_raw_XYZ()
        
        self.save_skeletons()
        
        #now calulate features
        self.stacked_filtered_XYZ_values = self.filter_joint_sequences(self.stacked_raw_XYZ_values)
        #self.stacked_filtered_angle_vlaues = np.array(self.stacked_filtered_angle_vlaues)
        
        #self.stacked_raw_angle_values = self.calulate_skeleton_angles_from_stacked_XYZ_values(self.stacked_raw_XYZ_values)
        #self.stacked_filtered_angle_vlaues = self.calulate_skeleton_angles_from_stacked_XYZ_values(self.stacked_filtered_XYZ_values)
        
        self.smc_features = self.get_SMC_features(self.stacked_filtered_XYZ_values)
        
        self.calculate_IJCP()
        
        self.calculate_walked_skel_angles()
        

        
        return
        
        
    def filter_joint_sequences(self, noisy_raw_XYZ, N=2, fc=10, fs=30):        
        stacked_filtered_XYZ_values  = []
        filtered_X = []
        filtered_Z = []
        filtered_Z = []
        
        for joint in SkeletonJoints:
            joint_number = joint.value
            #or 'WRIST' in joint.name
            
            if 'HAND' in joint.name  or 'THUMB' in joint.name or 'FOOT' in joint.name or 'ANKLE' in joint.name:
                _N=2
                _fc=20
                _fs=fs
            else:
                _N=N
                _fc=fc
                _fs=fs
            
            X, Y, Z = sm.filter_signal(noisy_raw_XYZ[0, joint_number, :],
                                       noisy_raw_XYZ[1, joint_number, :],
                                       noisy_raw_XYZ[2, joint_number, :],
                                       N=_N, fc=_fc, fs=_fs)
            
            #X = np.transpose(X)
            #Y = np.transpose(Y)
            #Z = np.transpose(Z)
            
            if len(filtered_X) == 0:
                filtered_X = X
                filtered_Y = Y
                filtered_Z = Z
            else:
                filtered_X = np.dstack([filtered_X, X])
                filtered_Y = np.dstack([filtered_Y, Y])
                filtered_Z = np.dstack([filtered_Z, Z])
            
#            filtered_XYZ = 
#            
#            if stacked_filtered_XYZ_values == []:
#                stacked_filtered_XYZ_values = filtered_XYZ
#            else:
                
        
#       
        filtered_X = np.transpose(filtered_X)
        filtered_Y = np.transpose(filtered_Y)
        filtered_Z = np.transpose(filtered_Z)
        
        stacked_filtered_XYZ_values = np.stack([filtered_X[:,:,0], filtered_Y[:,:,0], filtered_Z[:,:,0]])  
        #stacked_filtered_XYZ_values = np.reshape(stacked_filtered_XYZ_values,[3,26,600])
        
        return stacked_filtered_XYZ_values
        
    def normalise_skeleton(self, skel_frame):
        if len(self._ref_spine_base) == 0:
            self._ref_spine_base =  skel_frame.iloc[SkeletonJoints.SPINEBASE.value][['X', 'Y', 'Z']].tolist()
            #self._ref_spine_base =  skel_frame.iloc[SkeletonJoints.SPINESHOULDER.value][['X', 'Y', 'Z']].tolist()
            self._ref_skel_frame =  skel_frame
            #print('Getting Spinebase')
        
        normalised_skel_frame = pd.DataFrame.copy(skel_frame,deep=False)
        # x = 0
        # y = 1
        # z = 2

        for joint in SkeletonJoints:
            #joint_name = joint.name
            joint_number = joint.value
             
            #normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('X')] = (skel_frame.iloc[joint_number]['X'] - self._ref_skel_frame.iloc[joint_number]['X'])*10
            #normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Y')] = (skel_frame.iloc[joint_number]['Y'] - self._ref_skel_frame.iloc[joint_number]['Y'])*10
            #normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Z')] = (skel_frame.iloc[joint_number]['Z'] - self._ref_skel_frame.iloc[joint_number]['Z'])*10
            
            #if skel_frame.iloc[joint_number]['Tracked'] != 'Tracked':
            #    print(self._frame_count, joint_name, skel_frame.iloc[joint_number]['Tracked'])
                
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('X')] = (skel_frame.iloc[joint_number]['X'] - self._ref_spine_base[X])
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Y')] = (skel_frame.iloc[joint_number]['Y'] - self._ref_spine_base[Y])
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Z')] = (skel_frame.iloc[joint_number]['Z'] - self._ref_spine_base[Z])
        
            if joint_number == 24:
                break
            
        self._frame_count +=1     
        return normalised_skel_frame
    
    
    def _load_skel_file(self, skel_file_path):
        """ debug skelfile """
        #print(skel_file_path)
        
        #replace with array of skeletons
        columns = ['Joint', 'Tracked', 'X', 'Y', 'Z', 'px_X', 'px_Y']
        skel_frame = pd.read_csv(skel_file_path, delimiter=' ', header=None, nrows=25, names=columns, index_col=0)
        
        #tmp_frame = tmp_frame[0:25] # removes traking status form K3Da files
        #skel_frame = np.array(tmp_frame)
        #skel_frame = pd.DataFrame(tmp_frame)
        
        #X = np.array(skel_frame[:,2].tolist())
        #Y = np.array(skel_frame[:,3].tolist())
        #Z = np.array(skel_frame[:,4].tolist())
        ''' Normalise '''
        skel_frame = self.normalise_skeleton(skel_frame)
        
        #Add CoM
        tmp_CoM = self.calulate_CoM_position(skel_frame)
        #tmp_CoM_row = np.transpose(['COM', 'Tracked', tmp_CoM[0], tmp_CoM[1], tmp_CoM[1], 0, 0])
        #tmp_pd = pd.DataFrame(tmp_CoM_row, columns=columns)
        
        tmp_CoM_row = {'Tracked':'Tracked',
                       'X':tmp_CoM[0],
                       'Y':tmp_CoM[1],
                       'Z':tmp_CoM[2],
                       'px_X':0,
                       'px_Y':0}
        
        df_CoM_row = pd.DataFrame(tmp_CoM_row, index=['CoM'])
        
        skel_frame = skel_frame.append(df_CoM_row)
        
        X = skel_frame['X']
        Y = skel_frame['Y']
        Z = skel_frame['Z']
        raw_XYZ = np.stack([X.values, Y.values, Z.values])
        #raw_XYZ = np.transpose(np.stack([X.values, Y.values, Z.values]))
        
        #calculate angles
        #tmp_Angles_row = self.calulate_skeleton_angles(skel_frame)
        
        return skel_frame, raw_XYZ
    
    
    def calulate_CoM_position(self, skel_frame):
        # _X = 2
        # _Y = 3
        # _Z = 4
    
        spine_base = np.stack([skel_frame['X'][SkeletonJoints.SPINEMID.value], skel_frame['Y'][SkeletonJoints.SPINEMID.value], skel_frame['Z'][SkeletonJoints.SPINEMID.value]])
        hip_left = np.stack([skel_frame['X'][SkeletonJoints.HIPLEFT.value], skel_frame['Y'][SkeletonJoints.HIPLEFT.value], skel_frame['Z'][SkeletonJoints.HIPLEFT.value]])
        hip_right = np.stack([skel_frame['X'][SkeletonJoints.HIPRIGHT.value], skel_frame['Y'][SkeletonJoints.HIPRIGHT.value], skel_frame['Z'][SkeletonJoints.HIPRIGHT.value]])
    
        x_mean = np.mean([spine_base[0],hip_left[0],hip_right[0]])
        y_mean = np.mean([spine_base[1],hip_left[1],hip_right[1]])
        z_mean = np.mean([spine_base[2],hip_left[2],hip_right[2]])
    
        CoM = np.stack([x_mean,y_mean,z_mean])
    
        #com = spine_base
    
        return CoM.tolist()
    
    
    def calulate_skeleton_angles_from_stacked_XYZ_values(self, stacked_filtered_XYZ_values):
        angles_list = []
        #X = 0
        #Y = 1
        #Z = 2
        
        tmp_foot_left_joint = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value])])
            
        tmp_foot_left_joint = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value]),
                                        np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value])])
        
        tmp_foot_mean_joint = sm.mean_twin_joint_pos(tmp_foot_left_joint,
                                                     tmp_foot_left_joint)
        
        tmp_ground_plane = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value]),
                                           tmp_foot_mean_joint[1],
                                           np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value])])
        
        for i in np.ndindex(stacked_filtered_XYZ_values.shape[2]):
            
            # tmp_spine_base_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SPINEBASE.value][i],
            #                                     stacked_filtered_XYZ_values[Y][SkeletonJoints.SPINEBASE.value][i],
            #                                     stacked_filtered_XYZ_values[Z][SkeletonJoints.SPINEBASE.value][i]])
        
            tmp_shoulder_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SHOULDERLEFT.value][i],
                                                stacked_filtered_XYZ_values[Y][SkeletonJoints.SHOULDERLEFT.value][i],
                                                stacked_filtered_XYZ_values[Z][SkeletonJoints.SHOULDERLEFT.value][i]])
            
            tmp_shoulder_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SHOULDERRIGHT.value][i],
                                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.SHOULDERRIGHT.value][i],
                                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.SHOULDERRIGHT.value][i]])
            
            # tmp_shoulder_hip_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HIPRIGHT.value][i],
            #                                     stacked_filtered_XYZ_values[Y][SkeletonJoints.SHOULDERLEFT.value][i],
            #                                     stacked_filtered_XYZ_values[Z][SkeletonJoints.SHOULDERLEFT.value][i]])
            
            # tmp_shoulder_hip_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HIPRIGHT.value][i],
            #                                      stacked_filtered_XYZ_values[Y][SkeletonJoints.SHOULDERRIGHT.value][i],
            #                                      stacked_filtered_XYZ_values[Z][SkeletonJoints.SHOULDERRIGHT.value][i]])
            
            tmp_spine_shoulder_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SPINESHOULDER.value][i],
                                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.SPINESHOULDER.value][i],
                                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.SPINESHOULDER.value][i]])
            
            tmp_com_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value][i],
                                      stacked_filtered_XYZ_values[Y][SkeletonJoints.COM.value][i],
                                      stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value][i]])
            
           
            tmp_elbow_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ELBOWLEFT.value][i],
                                             stacked_filtered_XYZ_values[Y][SkeletonJoints.ELBOWLEFT.value][i],
                                             stacked_filtered_XYZ_values[Z][SkeletonJoints.ELBOWLEFT.value][i]])
                    
            tmp_elbow_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ELBOWRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Y][SkeletonJoints.ELBOWRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Z][SkeletonJoints.ELBOWRIGHT.value][i]])
            
            tmp_wrist_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.WRISTLEFT.value][i],
                                             stacked_filtered_XYZ_values[Y][SkeletonJoints.WRISTLEFT.value][i],
                                             stacked_filtered_XYZ_values[Z][SkeletonJoints.WRISTLEFT.value][i]])
                    
            tmp_wrist_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.WRISTRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Y][SkeletonJoints.WRISTRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Z][SkeletonJoints.WRISTRIGHT.value][i]])
            
            
            
            tmp_hip_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HIPLEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.HIPLEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.HIPLEFT.value][i]])
            
            tmp_hip_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HIPRIGHT.value][i],
                                  stacked_filtered_XYZ_values[Y][SkeletonJoints.HIPRIGHT.value][i],
                                  stacked_filtered_XYZ_values[Z][SkeletonJoints.HIPRIGHT.value][i]])
            
            tmp_knee_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.KNEELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.KNEELEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.KNEELEFT.value][i]])
            
            tmp_knee_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.KNEERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.KNEERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.KNEERIGHT.value][i]])
            
            tmp_ankle_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][i]])
            
            tmp_ankle_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][i]])
            
            tmp_heel_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][i]])
            
            tmp_heel_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][i]])
            
            tmp_heel_mean_joint = sm.mean_twin_joint_pos(tmp_heel_left_joint, tmp_heel_right_joint)
            
            tmp_ankle_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLELEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][i]])
            
            tmp_ankle_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLERIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][i]])
            
            
            tmp_foot_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value][i]])
            
            tmp_foot_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][i],
                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][i]])
            
            # tmp_foot_mean_joint = sm.mean_twin_joint_pos(tmp_foot_left_joint,
            #                                               tmp_foot_left_joint)
            
            # tmp_ground_plane = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value][i],
            #                             tmp_foot_mean_joint[1],
            #                             stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value][i]])
            
                                      
            # body_com_angle = sm.get_angle_between_three_joints(tmp_com_joint, 
            #                                                    tmp_foot_mean_joint, 
            #                                                    tmp_ground_plane)                          
                                      
            # body_lean_angle = sm.get_angle_between_three_joints(tmp_spine_shoulder_joint, 
            #                                                    tmp_foot_mean_joint, 
            #                                                    tmp_ground_plane)
            
            body_com_angle = sm.get_angle_between_three_joints(tmp_com_joint, 
                                                               tmp_heel_mean_joint, 
                                                               tmp_ground_plane)                          
                                      
            body_lean_angle = sm.get_angle_between_three_joints(tmp_spine_shoulder_joint, 
                                                               tmp_heel_mean_joint, 
                                                               tmp_ground_plane)
            
            knee_angle_left = sm.get_angle_between_three_joints(tmp_hip_left_joint, 
                                                                tmp_knee_left_joint,
                                                                tmp_ankle_left_joint)
            
            knee_angle_right = sm.get_angle_between_three_joints(tmp_hip_right_joint, 
                                                                 tmp_knee_right_joint,
                                                                 tmp_ankle_right_joint)
            
            hip_angle_left = sm.get_angle_between_three_joints(tmp_shoulder_left_joint, 
                                                               tmp_hip_left_joint,            
                                                               tmp_knee_left_joint)
            
            hip_angle_right = sm.get_angle_between_three_joints(tmp_shoulder_right_joint, 
                                                                tmp_hip_right_joint,            
                                                                tmp_knee_right_joint)
            
            
            
            elbow_angle_left = sm.get_angle_between_three_joints(tmp_shoulder_left_joint,
                                                                 tmp_elbow_left_joint,            
                                                                 tmp_wrist_left_joint)
            
            elbow_angle_right = sm.get_angle_between_three_joints(tmp_shoulder_right_joint,
                                                                  tmp_elbow_right_joint,            
                                                                  tmp_wrist_right_joint)
                        
            
            
            armpit_angle_left = sm.get_angle_between_three_joints(tmp_hip_left_joint,
                                                                  tmp_shoulder_left_joint,            
                                                                  tmp_elbow_left_joint)
            
            armpit_angle_right = sm.get_angle_between_three_joints(tmp_hip_right_joint,
                                                                   tmp_shoulder_right_joint,            
                                                                   tmp_elbow_right_joint)
            
            ankle_angle_left = sm.get_angle_between_three_joints(tmp_foot_left_joint,
                                                                  tmp_heel_left_joint,            
                                                                  tmp_knee_left_joint)
            
            ankle_angle_right = sm.get_angle_between_three_joints(tmp_foot_right_joint,
                                                                  tmp_heel_right_joint,            
                                                                  tmp_knee_right_joint)
            
        
            # knee_angle_left.append(360 - sm.get_angle_between_three_joints(hip_left, knee_left, ankle_left))
            # knee_angle_right.append(360 - sm.get_angle_between_three_joints(hip_right, knee_right, ankle_right))
            # knee_angle_mean.append(360 - sm.get_angle_between_three_joints(hip_mean, knee_mean, ankle_mean))
    
            # hip_angle_left.append(sm.get_angle_between_three_joints(shoulder_left, hip_left, knee_left))
            # hip_angle_right.append(sm.get_angle_between_three_joints(shoulder_right, hip_right, knee_right))
            # hip_angle_mean.append(sm.get_angle_between_three_joints(spine_shoulder, hip_mean, knee_mean))
                  
            '''                    
                BODY_COM_ANGLE = 1
                BODY_LEAN_ANGLE = 2
                KNEELEFT_ANGLE = 3
                KNEERIGHT_ANGLE = 4
                HIPLEFT_ANGLE = 5
                HIPRIGHT_ANGLE = 6
                ELBOWLEFT_ANGLE = 7
                ELBOWLEFT_ANGLE = 8
                ARMPITLEFT_ANGLE = 9
                ARMPITLEFT_ANGLE = 10
                ANKLELEFT_ANGLE = 11
                ANKLELRIGHT_ANGLE = 12
            '''
            
            angles_row = np.hstack([self._part_id,
                                    body_com_angle,
                                    body_lean_angle,
                                    knee_angle_left,
                                    knee_angle_right,
                                    hip_angle_left,
                                    hip_angle_right,
                                    elbow_angle_left,
                                    elbow_angle_right,
                                    armpit_angle_left,
                                    armpit_angle_right,
                                    ankle_angle_left,
                                    ankle_angle_right])
            
            if len(self._labels) != 0: 
                arr_labels = np.array(self._labels)[0]
                self.load_sway_metrics()
                angles_row = np.concatenate((angles_row, np.array(self.sway_metrics.iloc[0,10:]), arr_labels))
                
            
            angles_list.append(angles_row)
        
        return np.array(angles_list)
    
    def calulate_stepped_velocity_distance(self, trial, s=1):
        X = 0
        Y = 1
        Z = 2
        
        X_dist = []
        Y_dist = []
        Z_dist = []
        
        X_velos = []
        Y_velos = []
        Z_velos = []
        
        X_trial = trial[X,:]
        Y_trial = trial[Y,:]
        Z_trial = trial[Z,:]
        
        step = s*30
        
        for i in range(0, trial.shape[1], step):
            X_axis = X_trial[i:i+step]
            Y_axis = Y_trial[i:i+step]
            Z_axis = Z_trial[i:i+step]

            if len(X_axis) == step:
                # X_dist.append(np.sum(X_axis)/step)
                # Y_dist.append(np.sum(Y_axis)/step)
                # Z_dist.append(np.sum(Z_axis)/step)

                X_dist.append(np.mean(X_axis - np.mean(X_trial)))
                Y_dist.append(np.mean(Y_axis - np.mean(Y_trial)))
                Z_dist.append(np.mean(X_axis - np.mean(Z_trial)))
                
                
                X_velos.append((X_axis[len(X_axis)-1] - X_axis[0])/s)
                Y_velos.append((Y_axis[len(Y_axis)-1] - Y_axis[1])/s)
                Z_velos.append((Z_axis[len(Z_axis)-1] - Z_axis[2])/s)        
        
        dist = np.vstack([X_dist, Y_dist, Z_dist])
        velos = np.vstack([X_velos, Y_velos, Z_velos])
        
        return dist, velos
    
    def calculate_IJCP(self, s=1):
        X = 0
        Y = 1
        Z = 2
        
        ICJP = []
        ICJP_list = []
        
        X_dist = []
        Y_dist = []
        Z_dist = []
        
        X_velos = []
        Y_velos = []
        Z_velos = []
        
        for joint in SkeletonIJCP:
            tmp_trial = np.stack([self.stacked_filtered_XYZ_values[X][joint.value],
                                  self.stacked_filtered_XYZ_values[Y][joint.value],
                                  self.stacked_filtered_XYZ_values[Z][joint.value]])
            
            tmp_dist, tmp_velos = self.calulate_stepped_velocity_distance(tmp_trial, s=s)
    
            
            X_dist.append(tmp_dist[X,:])
            Y_dist.append(tmp_dist[Y,:])
            Z_dist.append(tmp_dist[Z,:])
            
            X_velos.append(tmp_velos[X,:])
            Y_velos.append(tmp_velos[Y,:])
            Z_velos.append(tmp_velos[Z,:])
            
        X_ICJP_Dist = np.corrcoef(X_dist)
        Y_ICJP_Dist = np.corrcoef(Y_dist)
        Z_ICJP_Dist= np.corrcoef(Z_dist)
            
        X_ICJP_Velo = np.corrcoef(X_velos)
        Y_ICJP_Velo = np.corrcoef(Y_velos)
        Z_ICJP_Velo = np.corrcoef(Z_velos)

        self.IJCP_Dist = np.stack([X_ICJP_Dist, Y_ICJP_Dist, Z_ICJP_Dist])
        self.IJCP_Velo = np.stack([X_ICJP_Velo, Y_ICJP_Velo, Z_ICJP_Velo])
        
    
    def get_SMC_features(self, stacked_filtered_XYZ_values):
        '''
        Parameters
        ----------
        stacked_filtered_XYZ_values : stacked_filtered_XYZ_values
            DESCRIPTION.

        Returns
        -------
        smc_features_list : list of smc features
            DESCRIPTION.
            
            
        headJoint = getJointXYZ(skelFrame[HEAD])
        neckJoint = getJointXYZ(skelFrame[NECK])
        spineSholderJoint = getJointXYZ(skelFrame[SPINESHOULDER])
        spineMidJoint = getJointXYZ(skelFrame[SPINEMID])
        spineBaseJoint = getJointXYZ(skelFrame[SPINEBASE])
    
        footLeftJoint = getJointXYZ(skelFrame[FOOTLEFT])
        footRightJoint = getJointXYZ(skelFrame[FOOTRIGHT])

        '''        
        
        smc_features_list = []
        X = 0
        Y = 1
        Z = 2
        
        tmp_foot_left_joint = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value][0]),
                                        np.mean(stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][0]),
                                        np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value][0])])
            
        tmp_foot_right_joint = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][0]),
                                        np.mean(stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][0]),
                                        np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][0])])
        
        foot_mean_joint = sm.mean_twin_joint_pos(tmp_foot_left_joint,
                                                 tmp_foot_right_joint)
        
        tmp_ankle_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLELEFT.value][0],
                                         stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLELEFT.value][0],
                                         stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLELEFT.value][0]])
            
        tmp_ankle_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.ANKLERIGHT.value][0],
                                          stacked_filtered_XYZ_values[Y][SkeletonJoints.ANKLERIGHT.value][0],
                                          stacked_filtered_XYZ_values[Z][SkeletonJoints.ANKLERIGHT.value][0]])
    
        ankle_mean_joint = sm.mean_twin_joint_pos(tmp_ankle_left_joint,
                                                 tmp_ankle_right_joint)
    
        
        # ground_plane = np.stack([np.mean(stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value]),
        #                                 tmp_foot_mean_joint[Y],
        #                                 np.mean(stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value])])
        
       
        
        for i in range(stacked_filtered_XYZ_values.shape[2]):
            tmp_head_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.HEAD.value][i],
                                       stacked_filtered_XYZ_values[Y][SkeletonJoints.HEAD.value][i],
                                       stacked_filtered_XYZ_values[Z][SkeletonJoints.HEAD.value][i]])

            tmp_neck_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.NECK.value][i],
                                       stacked_filtered_XYZ_values[Y][SkeletonJoints.NECK.value][i],
                                       stacked_filtered_XYZ_values[Z][SkeletonJoints.NECK.value][i]])

            tmp_spine_shoulder_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SPINESHOULDER.value][i],
                                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.SPINESHOULDER.value][i],
                                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.SPINESHOULDER.value][i]])

            tmp_spine_mid_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SPINEMID.value][i],
                                             stacked_filtered_XYZ_values[Y][SkeletonJoints.SPINEMID.value][i],
                                             stacked_filtered_XYZ_values[Z][SkeletonJoints.SPINEMID.value][i]])
            
            tmp_spine_base_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.SPINEBASE.value][i],
                                              stacked_filtered_XYZ_values[Y][SkeletonJoints.SPINEBASE.value][i],
                                              stacked_filtered_XYZ_values[Z][SkeletonJoints.SPINEBASE.value][i]])
            
            # tmp_foot_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value][i],
            #                                 stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][i],
            #                                 stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value][i]])
            
            # tmp_foot_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][i],
            #                                  stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
            #                                  stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][i]])
            
            tmp_ankle_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTLEFT.value][i],
                                            stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTLEFT.value][i],
                                            stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTLEFT.value][i]])
            
            tmp_ankle_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
                                              stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][i]])
            
            # tmp_hip_left_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][i],
            #                                  stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
            #                                  stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][i]])
            
            # tmp_hip_right_joint = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.FOOTRIGHT.value][i],
            #                                  stacked_filtered_XYZ_values[Y][SkeletonJoints.FOOTRIGHT.value][i],
            #                                  stacked_filtered_XYZ_values[Z][SkeletonJoints.FOOTRIGHT.value][i]])
    
            eulidian_distance_spine_base_and_head = sm.euclidean_distance_between_joints(tmp_spine_base_joint, tmp_head_joint)
        
            euler_angle_Spine_base_and_neck = sm.euler_angle(tmp_spine_base_joint, tmp_neck_joint)
        
            #ground_plane = np.array(tmp_foot_left_joint) + np.array(tmp_foot_right_joint) / 2
            
            # ground_plane = sm.mean_twin_joint_pos(tmp_ankle_left_joint,
            #                                       tmp_ankle_right_joint)
            
            # ground_plane = np.stack([foot_mean_joint[X],
            #                          foot_mean_joint[Y],
            #                          foot_mean_joint[Z]])
            
            ground_plane = np.stack([ankle_mean_joint[X],
                                     ankle_mean_joint[Y],
                                     ankle_mean_joint[Z]])
                            
        
            body_lean_angle = sm.euler_angle(tmp_spine_mid_joint, ground_plane)
            
            com = np.stack([stacked_filtered_XYZ_values[X][SkeletonJoints.COM.value][i],
                            stacked_filtered_XYZ_values[Y][SkeletonJoints.COM.value][i],
                            stacked_filtered_XYZ_values[Z][SkeletonJoints.COM.value][i]])
        
            torso_ML_axis = [tmp_head_joint[0], tmp_head_joint[1],
                           tmp_neck_joint[0], tmp_neck_joint[1],
                           tmp_spine_shoulder_joint[0], tmp_spine_shoulder_joint[1],
                           tmp_spine_mid_joint[0], tmp_spine_mid_joint[1],
                           tmp_spine_base_joint[0], tmp_spine_base_joint[1]]
        
            _smc_features = np.hstack([eulidian_distance_spine_base_and_head, 
                                    euler_angle_Spine_base_and_neck, 
                                    body_lean_angle, 
                                    com, 
                                    torso_ML_axis])
            
            smc_features_list.append(_smc_features)
    
        return smc_features_list
    
    # def calulate_skeleton_angles(self, skel_frame): #not used anymore
    #     angles_list = []
        
    #     tmp_shoulder_left_joint = np.stack([skel_frame['X'][SkeletonJoints.SHOULDERLEFT.value],
    #                                         skel_frame['Y'][SkeletonJoints.SHOULDERLEFT.value],
    #                                         skel_frame['Z'][SkeletonJoints.SHOULDERLEFT.value]])
        
    #     tmp_shoulder_right_joint = np.stack([skel_frame['X'][SkeletonJoints.SHOULDERRIGHT.value],
    #                                          skel_frame['Y'][SkeletonJoints.SHOULDERRIGHT.value],
    #                                          skel_frame['Z'][SkeletonJoints.SHOULDERRIGHT.value]])
        
    #     tmp_spine_shoulder_joint = np.stack([skel_frame['X'][SkeletonJoints.SPINESHOULDER.value],
    #                                          skel_frame['Y'][SkeletonJoints.SPINESHOULDER.value],
    #                                          skel_frame['Z'][SkeletonJoints.SPINESHOULDER.value]])
        
    #     tmp_com_joint = np.stack([skel_frame['X'][SkeletonJoints.COM.value],
    #                                          skel_frame['Y'][SkeletonJoints.COM.value],
    #                                          skel_frame['Z'][SkeletonJoints.COM.value]])
        
            
    #     tmp_hip_left_joint = np.stack([skel_frame['X'][SkeletonJoints.HIPLEFT.value],
    #                          skel_frame['Y'][SkeletonJoints.HIPLEFT.value],
    #                          skel_frame['Z'][SkeletonJoints.HIPLEFT.value]])
        
    #     tmp_hip_right_joint = np.stack([skel_frame['X'][SkeletonJoints.HIPRIGHT.value],
    #                           skel_frame['Y'][SkeletonJoints.HIPRIGHT.value],
    #                           skel_frame['Z'][SkeletonJoints.HIPRIGHT.value]])
        
    #     tmp_knee_left_joint = np.stack([skel_frame['X'][SkeletonJoints.KNEELEFT.value],
    #                          skel_frame['Y'][SkeletonJoints.KNEELEFT.value],
    #                          skel_frame['Z'][SkeletonJoints.KNEELEFT.value]])
        
    #     tmp_knee_right_joint = np.stack([skel_frame['X'][SkeletonJoints.KNEERIGHT.value],
    #                          skel_frame['Y'][SkeletonJoints.KNEERIGHT.value],
    #                          skel_frame['Z'][SkeletonJoints.KNEERIGHT.value]])
        
    #     tmp_ankle_left_joint = np.stack([skel_frame['X'][SkeletonJoints.ANKLELEFT.value],
    #                          skel_frame['Y'][SkeletonJoints.ANKLELEFT.value],
    #                          skel_frame['Z'][SkeletonJoints.ANKLELEFT.value]])
        
    #     tmp_ankle_right_joint = np.stack([skel_frame['X'][SkeletonJoints.ANKLERIGHT.value],
    #                          skel_frame['Y'][SkeletonJoints.ANKLERIGHT.value],
    #                          skel_frame['Z'][SkeletonJoints.ANKLERIGHT.value]])
        
    #     tmp_foot_left_joint = np.stack([skel_frame['X'][SkeletonJoints.FOOTLEFT.value],
    #                          skel_frame['Y'][SkeletonJoints.FOOTLEFT.value],
    #                          skel_frame['Z'][SkeletonJoints.FOOTLEFT.value]])
        
    #     tmp_foot_left_joint = np.stack([skel_frame['X'][SkeletonJoints.FOOTRIGHT.value],
    #                          skel_frame['Y'][SkeletonJoints.FOOTLEFT.value],
    #                          skel_frame['Z'][SkeletonJoints.FOOTLEFT.value]])
        
    #     tmp_foot_mean_joint = sm.mean_twin_joint_pos(tmp_foot_left_joint,
    #                                                  tmp_foot_left_joint)
        
    #     tmp_ground_plane = np.stack([skel_frame['X'][SkeletonJoints.COM.value],
    #                                tmp_foot_mean_joint[1],
    #                                skel_frame['Z'][SkeletonJoints.COM.value]])
        
                                  
    #     body_com_angle = sm.get_angle_between_three_joints(tmp_com_joint, 
    #                                                        tmp_foot_mean_joint, 
    #                                                        tmp_ground_plane)                          
                                  
    #     body_lean_angle = sm.get_angle_between_three_joints(tmp_spine_shoulder_joint, 
    #                                                        tmp_foot_mean_joint, 
    #                                                        tmp_ground_plane)
        
    #     knee_angle_left = sm.get_angle_between_three_joints(tmp_hip_left_joint, 
    #                                                         tmp_knee_left_joint,
    #                                                         tmp_ankle_left_joint)
        
    #     knee_angle_right = sm.get_angle_between_three_joints(tmp_hip_right_joint, 
    #                                                          tmp_knee_right_joint,
    #                                                          tmp_ankle_right_joint)
        
    #     hip_angle_left = sm.get_angle_between_three_joints(tmp_shoulder_left_joint, 
    #                                                        tmp_hip_left_joint,            
    #                                                        tmp_knee_left_joint)
        
    #     hip_angle_right = sm.get_angle_between_three_joints(tmp_shoulder_right_joint, 
    #                                                        tmp_hip_right_joint,            
    #                                                        tmp_knee_right_joint)
        
    
    #     # knee_angle_left.append(360 - sm.get_angle_between_three_joints(hip_left, knee_left, ankle_left))
    #     # knee_angle_right.append(360 - sm.get_angle_between_three_joints(hip_right, knee_right, ankle_right))
    #     # knee_angle_mean.append(360 - sm.get_angle_between_three_joints(hip_mean, knee_mean, ankle_mean))

    #     # hip_angle_left.append(sm.get_angle_between_three_joints(shoulder_left, hip_left, knee_left))
    #     # hip_angle_right.append(sm.get_angle_between_three_joints(shoulder_right, hip_right, knee_right))
    #     # hip_angle_mean.append(sm.get_angle_between_three_joints(spine_shoulder, hip_mean, knee_mean))
              
    #     '''                    
    #         BODY_COM_ANGLE = 0
    #         BODY_LEAN_ANGLE = 1
    #         KNEELEFT_ANGLE = 2
    #         KNEERIGHT_ANGLE = 3
    #         HIPLEFT_ANGLE = 4
    #         HIPRIGHT_ANGLE = 5
    #     '''
        
    #     angles_list = np.hstack([body_com_angle,
    #                             body_lean_angle,
    #                             knee_angle_left,
    #                             knee_angle_right,
    #                             hip_angle_left,
    #                             hip_angle_right])
    
    #     self.stacked_filtered_angle_vlaues.append(angles_list)
        
    #     return 
        
    
    #def calulate_euler_angles:
    #    ''' C# sudo code'''
    #        foreach (Skeleton skeleton in newSkeleton)
    #        {
    #            if (skeleton.TrackingState != SkeletonTrackingState.Tracked)
    #                continue;
    #            int j = 0;                
    #            foreach (BoneOrientation orientation in skeleton.BoneOrientations)
    #            { 
    #                Matrix4 matrix = orientation.HierarchicalRotation.Matrix;
    #        
    #                double y = Math.Asin(matrix.M13);
    #                double x = Math.Atan2(-matrix.M23, matrix.M33);
    #                double z = Math.Atan2(-matrix.M12, matrix.M11);
    #        
    #                rotationMatrix[j, 0] = x * 180 / Math.PI;
    #                rotationMatrix[j, 1] = y * 180 / Math.PI;
    #                rotationMatrix[j, 2] = z * 180 / Math.PI;                   
    #                j++;
    #            }
    #        }
    
    def calculate_walked_skel_angles(self):
        columns = []
        for combination in WalkedSkelAnglesIn3s:
        #for combination in WalkedSkelAngles:
            columns = np.hstack([columns, (combination.name + '_SP')])
            columns = np.hstack([columns, (combination.name + '_FP')])
            columns = np.hstack([columns, (combination.name + '_TP')])

        skel_walked_angels = []
        X = 0
        Y = 1
        Z = 2
        for i in range(self.stacked_filtered_XYZ_values.shape[2]):
            skel_angles_row = []
            for combination in WalkedSkelAnglesIn3s:
            #for combination in WalkedSkelAngles:
                j1  = np.stack([1 * self.stacked_filtered_XYZ_values[X][combination.value[0]][i],
                                1 * self.stacked_filtered_XYZ_values[Y][combination.value[0]][i],
                                1 * self.stacked_filtered_XYZ_values[Z][combination.value[0]][i]])
                
                j2  = np.stack([1 * self.stacked_filtered_XYZ_values[X][combination.value[1]][i],
                                1 * self.stacked_filtered_XYZ_values[Y][combination.value[1]][i],
                                1 * self.stacked_filtered_XYZ_values[Z][combination.value[1]][i]])
                
                j3  = np.stack([1 * self.stacked_filtered_XYZ_values[X][combination.value[2]][i],
                                1 * self.stacked_filtered_XYZ_values[Y][combination.value[2]][i],
                                1 * self.stacked_filtered_XYZ_values[Z][combination.value[2]][i]])
        
                #euler_angle_3d = sm.euler_angle_3d(j1, j2)
                
                
                # r = R.from_euler('xyz', [j1,j2], degrees=True)
                # euler_angle_3d = r.as_euler('xyz', degrees=True)
                
                euler_angle_3d = sm.get_3d_angle_between_three_joints(j1, j2, j3, deg=True)
                # q = Quaternion(0,euler_angle_3d[0],euler_angle_3d[1],euler_angle_3d[2])
                # e = q.to_euler(degrees=True)
                
                # #euler_angle_3d = sm.get_3d_angle_between_two_joints(j1, j2, deg=False)
                
                # #rot_mat = sm.rotation_matrix_from_vectors(j1, j2)
                # rot_mat = rotation_from_angles(euler_angle_3d, 'XYZ')
                # x_rot_mat = rotation_around_x(euler_angle_3d[0])
                # y_rot_mat = rotation_around_y(euler_angle_3d[1])
                # z_rot_mat = rotation_around_z(euler_angle_3d[2])
                
                # rot_mat = x_rot_mat + y_rot_mat + z_rot_mat
                
                # #euler_angle_3d = euler_xyz_from_matrix(rot_mat)
                # #euler_angle_3d = np.rad2deg(euler_angle_3d)
                
                # #tmp = matrix_from(a=euler_angle_3d)
                
                # r = R.from_matrix(rot_mat)
                # euler_angle_3d = r.as_euler('xyz', degrees=True)
                
                # #euler_angle_3d = euler_angle_3d * -1
                
                # euler_angle_3d = sm.get_3d_angle_between_three_joints(j1, j2, j3, deg=True)
                
                if len(skel_angles_row) == 0:
                    skel_angles_row = euler_angle_3d
                else:
                    skel_angles_row = np.hstack([skel_angles_row, euler_angle_3d])
                
            skel_walked_angels.append(skel_angles_row)
            
        pd_skel_walked_angels = pd.DataFrame(skel_walked_angels, columns=columns)
        #skel_walked_angels.append(pd_cur_skel_angles)
                        
        self.walked_skel_angles = pd_skel_walked_angels

# %%        
if __name__ == "__main__": 


    root_path = os.path.abspath('../../')
    #_part_id = '3'
    #_part_id = '9'
    #_part_id = '24'
    #_part_id = '404'
    #_part_id = '303'
    #_part_id = '700'
    _movement = 'Chair-Rise'
    #_movement = 'Quiet-Standing-Eyes-Open'
    #_movement = 'Quiet-Standing-Eyes-Closed'
    #_movement = 'Foam-Quiet-Standing-Eyes-Open'
    #_movement = 'Foam-Quiet-Standing-Eyes-Closed'
    #_movement = 'Semi-Tandem-Balance'
    #_movement = 'Tandem-Balance'
    #_movement = 'Unilateral-Stance-Eyes-Closed'
    #_movement = 'Ramp-Quiet-Standing-Eyes-Closed'
    
    #_skel_root_path = os.path.join(root_path, 'SPPB', _part_id, _part_id +'_'+ _movement, 'skel')
    #os.path.abspath(root_path + '/SPPB/' +  _part_id +'_'+ _movement +'/skel')
 
    headers = ['PART_ID',
               'BODY_COM_ANGLE',
               'BODY_LEAN_ANGLE',
               'KNEELEFT_ANGLE',
               'KNEERIGHT_ANGLE',
               'HIPLEFT_ANGLE',
               'HIPRIGHT_ANGLE',
               'ELBOWLEFT_ANGLE',
               'ELBOWRIGHT_ANGLE',
               'ARMPITLEFT_ANGLE',
               'ARMPITRIGHT_ANGLE',
               'ANKLELEFT_ANGLE',
               'ANKLELRIGHT_ANGLE',
               'RDIST_ML',
               'RDIST_AP',
               'RDIST',	
               'MDIST_ML',
               'MDIST_AP',
               'MDIST',	
               'TOTEX_ML',
               'TOTEX_AP',
               'TOTEX',
               'MVELO_ML',
               'MVELO_AP',	
               'MVELO',
               'MFREQ_ML',
               'MFREQ_AP',
               'MFREQ',	
               'AREA_CE',
               'GROUP_INC',
               'AGE',
               'BMI',
               'HEIGHT',
               'WEIGHT',
               'SEX',
               'impairment_confedence',
               'impairment_self', 
               'impairment_clinical',
               'impairment_stats_MAD_2.0',
               'impairment_stats_SD_1.96',
               'impairment_stats_SD_1.5',
               'impairment_adult',
               'impairment_older']
    
    """ Create Register """
    _dataset_prefix = 'SPPB'
    #_dataset_prefix = 'Crewe'
    #register_file = os.path.abspath('../Kinect-Faller-Detection/Data/RecordingsRegister2.xlsx')
    register_file = os.path.abspath('../Kinect-Faller-Detection/Data/RecordingsRegister3.xlsx')
    register = pd.read_excel(register_file)

    if _dataset_prefix == 'All':
        register = register
    if _dataset_prefix == 'SPPB':
        register = register[register['root_dir'].str.contains('SPPB')]
    elif _dataset_prefix == 'K3Da':
        register = register[register['root_dir'].str.contains('K3Da')]
    elif _dataset_prefix == 'Crewe':
        register = register[register['root_dir'].str.contains('Crewe')]
    elif _dataset_prefix == 'SPPB-Crew':
        register = register[register['root_dir'].str.contains('Crewe|SPPB')==True]
        
    dataset_roots, part_dirs = get_part_dirs(register)
    
    all_Stacked_filtered_angle_vlaues = []
    for _part_id in part_dirs: #tqdm(part_dirs):
        #_part_id = 'SPPB11'
        print('\n', _part_id ,'\n')
        _skel_root_path = os.path.join(root_path, _dataset_prefix, _part_id, _part_id +'_'+ _movement, 'skel')
        #os.path.abspath(root_path + '/SPPB/' + _part_id + '/'+ _part_id +'_'+ _movement +'/skel')
        
        #Check for no skel files
        _, _, skel_files = walkFileSystem(_skel_root_path)
         
        if len(skel_files) != 0:
            #Create Recording opbject
            
            if '0A' in _part_id:
                _tmp_part_id = _part_id.replace('0A', '')    
                _int_part_id = int(re.search(r'\d+', _tmp_part_id).group())
                _int_part_id = _int_part_id * 100
                
            elif 'Y' in _part_id:
                _tmp_part_id = _part_id.replace('Y', '')
                _int_part_id = int(re.search(r'\d+', _tmp_part_id).group())
                _int_part_id = _int_part_id * 100
            
            else:
                _int_part_id = int(re.search(r'\d+', _part_id).group())
            
            
            #create kinectRecording obj
            label_columns = ['group_inc', 'age', 'BMI', 'height', 'weight', 'sex',
                             'impairment_confedence', 'impairment_self', 
                             'impairment_clinical', 'impairment_stats_MAD_2.0',
                             'impairment_stats_SD_1.96', 'impairment_stats_SD_1.5',
                             'impairment_adult', 'impairment_older']
            labels = register[register['part_id'] == _part_id][label_columns]
            kinect_recording = KinectRecording(_skel_root_path, _dataset_prefix, _movement, _int_part_id, labels=labels)
        
            if len(all_Stacked_filtered_angle_vlaues) == 0:
                all_Stacked_filtered_angle_vlaues = kinect_recording.stacked_filtered_angle_vlaues
            else:
                all_Stacked_filtered_angle_vlaues = np.vstack([all_Stacked_filtered_angle_vlaues, kinect_recording.stacked_filtered_angle_vlaues])
    
            #break
    #save csv file of angles
    #np.savetxt(_dataset_prefix + '_' + _movement + '_Angles.csv', all_Stacked_filtered_angle_vlaues, delimiter=',')

    #Save csv with headers
    #_file_name = '_Angles_from_heel_and_Sway_Metrics_inc_groups.csv'
    _file_name = '_Angles_from_heel_and_Sway_Metrics_inc_groups_65_1.csv'
    with open(_dataset_prefix + '_' + _movement + _file_name,
              'wt', newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(i for i in headers)
        for j in all_Stacked_filtered_angle_vlaues:
            writer.writerow(j)

    
#%% 
    sys.exit()
    #my_Stacked_filtered_angle_vlaues = my_KinectRecording.stacked_filtered_angle_vlaues
    
    #pd_all_Stacked_filtered_angle_vlaues
    
    
    
    pd_all_Stacked_filtered_angle_vlaues = pd.DataFrame(all_Stacked_filtered_angle_vlaues, columns=headers)
    #pd_all_Stacked_filtered_angle_vlaues.set_index('PART_ID')
    
    

#%%
    
    #all_Stacked_filtered_angle_vlaues = np.array(np_filtered_angle_vlaues)
    #part_id = 315
    tmp_recording = pd_all_Stacked_filtered_angle_vlaues[pd_all_Stacked_filtered_angle_vlaues['PART_ID'] == _part_id]
    recording = np.array(tmp_recording)

    plt.plot(recording[:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='Torso CoM angle')
    plt.plot(recording[:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='Torso lean angle')
    #plt.ylim(78, 90)
    plt.title(_part_id)
    plt.legend()
    plt.show()
    
    plt.plot(recording[:,SkeletonJointAngles.HIPLEFT_ANGLE.value], label='Hip left angle')
    plt.plot(recording[:,SkeletonJointAngles.HIPRIGHT_ANGLE.value], label='Hip right angle')
    #plt.ylim(78, 90)
    plt.title(_part_id)
    plt.legend()
    plt.show()
    
    plt.plot(recording[:,SkeletonJointAngles.KNEELEFT_ANGLE.value], label='Knee left angle')
    plt.plot(recording[:,SkeletonJointAngles.KNEERIGHT_ANGLE.value], label='Knee right angle')
    plt.legend()
    plt.title(_part_id)
    plt.show()
    
    sys.exit()

#%%
    
    plt.plot(all_Stacked_filtered_angle_vlaues[0,:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='BODY_COM_ANGLE')
    plt.plot(all_Stacked_filtered_angle_vlaues[0,:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='BODY_LEAN_ANGLE')
    plt.ylim(78, 90)
    plt.legend()
    plt.show()
    
    plt.plot(all_Stacked_filtered_angle_vlaues[1,:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='BODY_COM_ANGLE')
    plt.plot(all_Stacked_filtered_angle_vlaues[1,:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='BODY_LEAN_ANGLE')
    plt.ylim(78, 90)
    plt.legend()
    plt.show()
    
    plt.plot(all_Stacked_filtered_angle_vlaues[2,:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='BODY_COM_ANGLE')
    plt.plot(all_Stacked_filtered_angle_vlaues[2,:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='BODY_LEAN_ANGLE')
    plt.ylim(78, 90)
    plt.legend()
    plt.show()
    
    plt.plot(all_Stacked_filtered_angle_vlaues[3,:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='BODY_COM_ANGLE')
    plt.plot(all_Stacked_filtered_angle_vlaues[3,:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='BODY_LEAN_ANGLE')
    plt.ylim(78, 90)
    plt.legend()
    plt.show()
    
    
# %%
    my_stacked_raw_angle_vlaues = my_KinectRecording.stacked_raw_angle_values
    my_stacked_filtered_angle_vlaues = my_KinectRecording.stacked_filtered_angle_vlaues
    
    plt.plot(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='BODY_COM_ANGLE')
    plt.plot(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='BODY_LEAN_ANGLE')
    plt.legend()
    plt.title('Raw')
    plt.show()

    plt.plot(my_stacked_filtered_angle_vlaues[:,SkeletonJointAngles.BODY_COM_ANGLE.value], label='BODY_COM_ANGLE')
    plt.plot(my_stacked_filtered_angle_vlaues[:,SkeletonJointAngles.BODY_LEAN_ANGLE.value], label='BODY_LEAN_ANGLE')
    plt.legend()
    plt.title('Filtered')
    plt.show()
    
    plt.plot(sm.filter_signal(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.BODY_COM_ANGLE.value]), label='BODY_COM_ANGLE')
    plt.plot(sm.filter_signal(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.BODY_LEAN_ANGLE.value]), label='BODY_LEAN_ANGLE')
    plt.legend()
    plt.title('Raw filtered')
    plt.show()
    
# %%
    
    plt.plot(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.HIPLEFT_ANGLE.value], label='HIPLEFT_ANGLE')
    plt.plot(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.HIPRIGHT_ANGLE.value], label='HIPRIGHT_ANGLE')
    plt.legend()
    plt.title('Raw')
    plt.show()
    
    plt.plot(my_stacked_filtered_angle_vlaues[:,SkeletonJointAngles.HIPLEFT_ANGLE.value], label='HIPLEFT_ANGLE')
    plt.plot(my_stacked_filtered_angle_vlaues[:,SkeletonJointAngles.HIPRIGHT_ANGLE.value], label='HIPRIGHT_ANGLE')
    plt.legend()
    plt.title('Filtered')
    plt.show()
    
    plt.plot(sm.filter_signal(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.HIPLEFT_ANGLE.value]), label='HIPLEFT_ANGLE')
    plt.plot(sm.filter_signal(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.HIPRIGHT_ANGLE.value]), label='HIPRIGHT_ANGLE')
    plt.title('Raw filtered')
    plt.legend()
    plt.show()
    
    plt.plot(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.KNEELEFT_ANGLE.value], label='KNEELEFT_ANGLE')
    plt.plot(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.KNEERIGHT_ANGLE.value], label='KNEERIGHT_ANGLE')
    plt.legend()
    plt.title('Raw')
    plt.show()
    
    plt.plot(my_stacked_filtered_angle_vlaues[:,SkeletonJointAngles.KNEELEFT_ANGLE.value], label='KNEELEFT_ANGLE')
    plt.plot(my_stacked_filtered_angle_vlaues[:,SkeletonJointAngles.KNEERIGHT_ANGLE.value], label='KNEERIGHT_ANGLE')
    plt.legend()
    plt.title('Filtered')
    plt.show()
    
    plt.plot(sm.filter_signal(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.KNEELEFT_ANGLE.value]), label='KNEELEFT_ANGLE')
    plt.plot(sm.filter_signal(my_stacked_raw_angle_vlaues[:,SkeletonJointAngles.KNEERIGHT_ANGLE.value]), label='KNEERIGHT_ANGLE')
    plt.legend()
    plt.title('Raw filtered')
    plt.show()
    
    sys.exit()
    
    
#%%    
    for joint in HierarchicalSkeletonJoints:
       #A = my_KinectRecording.skeletons
       #B = my_KinectRecording.raw_XYZ_values
       #E = my_KinectRecording.stacked_raw_XYZ_values
       #F = my_KinectRecording.stacked_filtered_XYZ_values
       #joint = SkeletonJoints.COM
       joint_number = joint.value
       joint_name = joint.name
       
       #joint_number = SkeletonJoints.ANKLELEFT.value
       #joint_number = SkeletonJoints.ANKLERIGHT.value
       #joint_number = SkeletonJoints.HANDLEFT.value
       #joint_number = SkeletonJoints.COM.value
    
       #N=4
       #fc=6
       
       #N=1
       #fc=30
       #X, Y, Z = sm.filter_signal(my_KinectRecording.stacked_raw_XYZ_values[0, joint_number, :],
       #                           my_KinectRecording.stacked_raw_XYZ_values[1, joint_number, :],
       #                           my_KinectRecording.stacked_raw_XYZ_values[2, joint_number, :],
       #                           N=N, fc=fc)
       #plt.plot(my_KinectRecording.stacked_raw_XYZ_values[0, joint_number, :])
       #plt.title(joint_name)
       #plt.show()
       
       Xr = my_KinectRecording.stacked_raw_XYZ_values[0, joint_number, :]*100
       Yr = my_KinectRecording.stacked_raw_XYZ_values[1, joint_number, :]*100
       Zr = my_KinectRecording.stacked_raw_XYZ_values[2, joint_number, :]*100
       #plt.plot(Xr, label='Xr')
       #plt.plot(Yr, label='Yr')
       plt.plot(Zr, label='Zr')
       plt.legend()
       plt.title(my_KinectRecording._part_id + ' ' + _movement + ' ' +  joint_name + ' ' + str(round((np.mean(Zr)),3)))
       #plt.show()
    
       X = my_KinectRecording.stacked_filtered_XYZ_values[0, joint_number, :]*100
       Y = my_KinectRecording.stacked_filtered_XYZ_values[1, joint_number, :]*100
       Z = my_KinectRecording.stacked_filtered_XYZ_values[2, joint_number, :]*100
    
       #plt.plot(X, label='X')
       #plt.plot(Y, label='Y')
       plt.plot(Z, label='Z')
       plt.legend()
       plt.title(my_KinectRecording._part_id + ' ' + _movement + ' ' +  joint_name + ' ' + str(round((np.mean(Z)),3)))
       plt.ylim(Zr.min(), Zr.max())
       plt.show()
       #break
    
#       fig = plt.figure(figsize=(10, 10))
#       ax = fig.add_subplot(111, projection='3d')
#       ax.set_xlabel('X')
#       ax.set_ylabel('Y')
#       ax.set_zlabel('Z')
#       #plt.title(my_KinectRecording._part_id + ' ' + _movement + ' ' +  Joint_name + ' ' + str(round((np.mean(X)*100),5)))
#       
#       plt.scatter(X, Y, Z, color='k', marker='o', label=joint_name)
#       #plt.title(my_KinectRecording._part_id + ' ' + _movement + ' ' +  Joint_name + ' ' + str(round((np.mean(Z)*100),5)))
#       ax.autoscale()
#       #ax.axis('off')
#       ax.view_init(elev=90, azim=-90)
#       #ax.set_xlim(np.min(Xr), np.max(Xr))
#       #ax.set_ylim(np.min(Yr), np.max(Yr))
#       #ax.set_zlim(np.min(Zr), np.max(Zr))
#       #plt.zlim(np.min(Z), np.max(Z))
#       #ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
#       plt.show()
#       #break
#sys.exit()
#%%
    for joint in SkeletonJoints:
        joint_number = joint.value
        joint_name = joint.name
        X = my_KinectRecording.stacked_filtered_XYZ_values[0, joint_number, :]*100
        Y = my_KinectRecording.stacked_filtered_XYZ_values[1, joint_number, :]*100
        Z = my_KinectRecording.stacked_filtered_XYZ_values[2, joint_number, :]*100
        
    #    fig = plt.figure(figsize=(10, 10))
    #    ax = fig.add_subplot(111, projection='3d')
    #    ax.set_xlabel('X')
    #    ax.set_ylabel('Y')
    #    ax.set_zlabel('Z')
    #    ax.view_init(elev=90, azim=-90)
    #
    #    ax.scatter(Xr, Yr, Zr, color='k', marker='o', label=joint_name)
    #    plt.title(my_KinectRecording._part_id + ' ' + _movement + ' ' +  joint_name + ' ' + str(round((np.mean(Z)),3)))
    #    ax.autoscale()
    #    plt.show()
        
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=90, azim=-90)
    
        ax.scatter(X, Y, Z, color='k', marker='o', label=joint_name)
        plt.title(my_KinectRecording._part_id + ' ' + _movement + ' ' +  joint_name + ' ' + str(round((np.mean(Z)),3)))
        ax.autoscale()
        plt.show()   
    
    #%%
    joint = SkeletonJoints.COM
    joint_number = joint.value
    joint_name = joint.name
    Z = my_KinectRecording.stacked_raw_XYZ_values[2, joint_number, :]*100  
     
    from scipy import fftpack
    
    #f = 100
    f_s = 30
    
    #t = np.linspace(0, 20, 30 * f_s, endpoint=False)
    plt.plot(Z)
    plt.title(joint_name)
    plt.show() 
    
    x = fftpack.fft(Z)
    freqs = fftpack.fftfreq(len(x)) * f_s
    
    fig, ax = plt.subplots()
    
    ax.stem(freqs, np.abs(X))
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(-f_s / 2, f_s / 2)
    #ax.set_ylim(-5, 110)
    
    
    
    #%%
    
    Zr = my_KinectRecording.stacked_raw_XYZ_values[2, joint_number, :]*100 
    Z = my_KinectRecording.stacked_filtered_XYZ_values[2, joint_number, :]*100
    b = np.abs(X)
    Zr_ep = round(stats.entropy(Zr, base=2),2)
    Z_ep = round(stats.entropy(Z, base=2),2)
    freqs_ep = round(stats.entropy(b, base=2),2)
    print(joint_name)
    print(Zr_ep, Z_ep, freqs_ep)
    
    #%%
    print(_part_id)
    
    #%%
    #from mayavi import mlab
    #mlab.points3d(X, Y, Z, scale_factor=0.06, color=(0,0,0))
    
    
    #%%
    
    for joint in HierarchicalSkeletonJoints:
        res_x = []
        res_y = []
        res_z = []
        
        fc_x = []
        fc_y = []
        fc_z = []
        
        Hzs = list(range(5,30)) #[5,10,15,20,25,30]
        for Hz in Hzs:
            joint_number = joint.value
            joint_name = joint.name
            Xr = my_KinectRecording.stacked_raw_XYZ_values[0, joint_number, :]*100
            Yr = my_KinectRecording.stacked_raw_XYZ_values[1, joint_number, :]*100
            Zr = my_KinectRecording.stacked_raw_XYZ_values[2, joint_number, :]*100
            
            X, Y, Z = sm.filter_signal(Xr,
                                       Yr,
                                       Zr,
                                       N=2,
                                       fc=Hz)
            
            _res_x = round(np.sqrt(np.sum(np.square(np.subtract(Xr,X)))), 3)
            res_x.append(_res_x)
            fc_x.append(round((0.06 * Hz) - (0.000022 * (Hz**2)) + (5.95 + _res_x), 3))
                    
            _res_y = round(np.sqrt(np.sum(np.square(np.subtract(Yr,Y)))), 3)
            res_y.append(_res_y)
            fc_y.append(round((0.06 * Hz) - (0.000022 * (Hz**2)) + (5.95 + _res_y), 3))
            
            _res_z = round(np.sqrt(np.sum(np.square(np.subtract(Zr,Z)))), 3)
            res_z.append(_res_z)
            fc_z.append(round((0.06 * Hz) - (0.000022 * (Hz**2)) + (5.95 + _res_z), 3))
            
        plt.plot(Hzs, res_x, label='res x', marker='x')
        #plt.title(joint_name)
        #plt.legend()
        #plt.show()
        
        
        
        plt.plot(Hzs, res_y, label='res y', marker='o')
        #plt.title(joint_name)
        #plt.legend()
        #plt.show()
        
        plt.plot(Hzs, res_z, label='res z', marker='s')
        plt.title(joint_name)
        plt.legend()
        plt.xlabel('freqency')
        plt.ylabel('residual (mm)')
        plt.show()
        
        print('X cuttoff freq', round(np.mean(fc_x), 1))
        print('X cuttoff freq', round(np.mean(fc_y), 1))
        print('X cuttoff freq', round(np.mean(fc_z), 1))
        
    #%%
    
    #from __future__ import division
    #import numpy as np
    #import matplotlib.pyplot as plt
    
    #data = np.random.rand(301) - 0.5
    joint_number = HierarchicalSkeletonJoints.COM.value
    
    Xr = my_KinectRecording.stacked_raw_XYZ_values[0, joint_number, :]*100
    Yr = my_KinectRecording.stacked_raw_XYZ_values[1, joint_number, :]*100
    Zr = my_KinectRecording.stacked_raw_XYZ_values[2, joint_number, :]*100
    
    X = my_KinectRecording.stacked_filtered_XYZ_values[0, joint_number, :]*100
    Y = my_KinectRecording.stacked_filtered_XYZ_values[1, joint_number, :]*100
    Z = my_KinectRecording.stacked_filtered_XYZ_values[2, joint_number, :]*100
            
    sig = X
    fs = 30
    from scipy import signal
    freqs, times, spectrogram = signal.spectrogram(sig)
    
    #plt.figure(figsize=(5, 4))
    #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    #plt.title('Spectrogram')
    #plt.ylabel('Frequency band')
    #plt.xlabel('Time window')
    #plt.tight_layout()
    
    freqs, psd = signal.welch(sig, fs, scaling='density')
    
    plt.semilogx(freqs, psd)
    plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.tight_layout()
    plt.show()
    
    print('Max peak feq:', psd.max())
    
    #freqs, psd = signal.welch(sig, fs, scaling='spectrum')
    #
    #plt.semilogx(freqs, psd)
    #plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    #plt.xlabel('Frequency')
    #plt.ylabel('Power')
    #plt.tight_layout()
    #plt.show()
    #
    #print('Max peak feq:', psd.max())
    
    #ps = np.abs(np.fft.fft(data))**2
    #
    #time_step = 1 / 30
    #freqs = np.fft.fftfreq(data.size, time_step)
    #idx = np.argsort(freqs)
    #
    #plt.plot(freqs[idx], ps[idx])    
    
    #%%
    
    f, Pxx_den = signal.welch(sig, fs) #, nperseg=1024)
    plt.semilogy(f, Pxx_den)
    #plt.ylim([0.5e-3, 1])
    plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()
    
    #%%
    f, Pxx_spec = signal.welch(sig, fs, 'flattop', 1024, scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.show()
    print('Max peak feq:', np.sqrt(Pxx_spec.max()))
    
    #%%
    plt.plot(sig)
    plt.show()
    
    plt.psd(sig)
    plt.show()
    
    plt.psd(sig, Fs=fs)
    plt.show()
    
    #%%
    from scipy.fftpack import fft
    
    d = 20
    t = np.arange(0,d,1/fs)
    a = np.sin(2*np.pi*20*t)
    
    a = sig
    plt.plot(a)
    plt.show()
    
    #spectrum
    X_f = fft(a)
    plt.plot(np.abs(X_f))
    plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    plt.show() 
    print('mean feq:', np.mean(np.abs(X_f)))
    
    #frequencys
    n = np.size(a)
    fr = (fs/2) * np.linspace(0,1,round(n/2))
    X_m =  (2/n) * abs(X_f[0 : np.size(fr)])
    
    plt.semilogy(fr, X_m)
    plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    plt.show()
    
    #%%
    # import numpy.fft as fft
    # spectrum = fft.fft(sig)
    # abs_spec = spectrum.abs
    # #You can then plot the magnitudes of the FFT as
    
    # freq = fft.fftfreq(len(abs_spec))
    # plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    # plt.plot(freq)
    # plt.show()
    
    # plt.plot(freq, abs_spec)
    # plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    # #plt.xlim(-0.01, 0.01)
    # plt.show()
    
    # #print('mean:', round(np.mean(freq),3))
    # print('SD:', round(np.std(freq),3))
    
# %%
    
# tmp_skel = kinect_recording.stacked_filtered_XYZ_values
# ''' try without flattening '''
# tmp_skel_2D = tmp_skel.reshape(-1, tmp_skel.shape[2])
# tmp_head = tmp_skel_2D[SkeletonJoints.HEAD.value:SkeletonJoints.HEAD.value+3,:]
# tmp_neck = tmp_skel_2D[SkeletonJoints.NECK.value:SkeletonJoints.NECK.value+3,:]
# tmp_spine_mid = tmp_skel_2D[SkeletonJoints.SPINEMID.value:SkeletonJoints.SPINEMID.value+3,:]
# tmp_com = tmp_skel_2D[SkeletonJoints.COM.value:SkeletonJoints.COM.value+3,:]

# tmp_arr = np.vstack([tmp_head, tmp_neck, tmp_com])
# tmp_arr = np.vstack([tmp_head, tmp_neck, tmp_spine_mid])
# #tmp_arr = np.vstack([tmp_head])

# tmp_cov = np.corrcoef(tmp_arr)

# plt.plot(tmp_arr)
# plt.show()

# plt.matshow(tmp_cov, cmap='jet')
# plt.show()

# plt.matshow(pd.DataFrame(tmp_arr).corr(), cmap='jet')
# plt.show()

#%%
# plt.figure(figsize=(10,10))
# #plt.plot(tmp[1][['KNEELEFT_a_SP', 'KNEELEFT_a_FP', 'KNEELEFT_a_TP']][30:])
# #plt.plot(tmp[1][['KNEERIGHT_a_SP', 'KNEERIGHT_a_FP', 'KNEERIGHT_a_TP']][30:])
# #plt.plot(tmp[1][['KNEELEFT_a_SP', 'KNEERIGHT_a_SP']])
# #plt.plot(tmp[1][['KNEELEFT_a_FP', 'KNEERIGHT_a_FP']])
# #plt.plot(tmp[1][['KNEELEFT_a_TP', 'KNEERIGHT_a_TP']])

# #plt.plot(tmp[3][['ELBOWLEFT_a_SP', 'ELBOWRIGHT_a_SP']])
# #plt.plot(tmp[3][['ELBOWLEFT_a_FP', 'ELBOWRIGHT_a_FP']])
# plt.plot(tmp[3][['ELBOWLEFT_a_TP', 'KNEERIGHT_a_TP']])

# #plt.plot(tmp[1][['KNEERIGHT_a_SP', 'KNEERIGHT_a_FP', 'KNEERIGHT_a_TP']][30:])
# plt.show()
    
# %%
# plt.figure(figsize=(7,5))
# plt.plot(tmp.KNEELEFT_a_SP, label='KNEELEFT_a_TP', color='r')
# plt.plot(tmp.KNEERIGHT_a_SP, label='KNEERIGHT_a_TP', color='r')
# #plt.legend()
# #plt.show() 
# #plt.figure(figsize=(7,5))
# #plt.plot(tmp.HIPLEFT_a_SP, label='KNEELEFT_a_TP', color='b')
# #plt.plot(tmp.HIPRIGHT_a_SP, label='KNEERIGHT_a_TP', color='b')

# #plt.plot(tmp.ELBOWLEFT_a_SP, label='ELBOWLEFT_a_FP')
# #plt.plot(tmp.ELBOWRIGHT_a_SP, label='ELBOWRIGHT_a_FP')

# #plt.plot(tmp.ELBOWLEFT_a_SP)
# #plt.plot(tmp.ELBOWRIGHT_a_SP)
# #plt.plot(tmp.NECK_a_SP)
# #plt.legend()
# #plt.show()   

# #plt.figure(figsize=(7,5))
# plt.plot(tmp.KNEELEFT_a_SP, label='KNEELEFT_a_FP', color='k')
# plt.plot(tmp.KNEERIGHT_a_SP, label='KNEERIGHT_a_FP', color='k')
# plt.legend()
# plt.show() 
    
#plt.plot(tmp.SPINEMID_a_FP)
#plt.show()    

# %%
# import seaborn as sns
# import scipy.stats as stats
# list_X = [np.random.normal(0,1,10), np.random.normal(0,1,100), np.random.normal(0,1,10000)]

# for x in list_X:
#     title = 'mu:' + str(round(np.mean(x),3)) + '\n'
#     title = title +'sd:' + str(round(np.std(x),3))
#     title = title + ' se:' + str(round(stats.sem(x),3))
#     title = title + ' ci:+/-' + str(round(np.std(x) * 1.96))
#     sns.distplot(x)
#     plt.title(title)
#     plt.show()
    
#     print(stats.shapiro(x))