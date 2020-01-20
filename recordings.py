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

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from mpl_toolkits.mplot3d import Axes3D

import sys
from tqdm import tqdm

from enum import Enum

sys.path.insert(0, '../Utils')
sys.path.insert(0, '../')
from sway_utils import metrics as sm
#import skel_utils.metrics as sm
import Utils as ul
import Read_Kinect_Files as rk

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
    
#%%
class KinectRecording:
    _skel_root_path = '' 
    _skel_file_root = ''
    _part_id = 0
    _ref_spine_base = []
    _ref_skel_frame = []
    _frame_count = 0

    skeletons = []
    raw_XYZ_values = []
    stacked_raw_XYZ_values = []
    stacked_raw_XYZ_values_filtered = []
    
    def __init__(self, skel_root_path):
        self.load_skeletons(skel_root_path)
        
    
    
    def load_skeletons(self, skel_root_path):
        #load stuff
        root, dirs, skel_files = ul.walkFileSystem(skel_root_path)
        skel_files.sort()
        
        
        for skelfile in tqdm(skel_files):
            skel_file_path = os.path.join(root, skelfile)
            _skel_frame, _raw_XYZ = self._load_skel_file(skel_file_path)
            
            self.skeletons.append(_skel_frame)
            self.raw_XYZ_values.append(_raw_XYZ)
            if len(self.stacked_raw_XYZ_values) == 0:
                self.stacked_raw_XYZ_values = _raw_XYZ
            else:
                self.stacked_raw_XYZ_values = np.dstack([self.stacked_raw_XYZ_values, _raw_XYZ])
        
        self.stacked_raw_XYZ_values = self.stacked_raw_XYZ_values[:,:,1:]
        self.stacked_raw_XYZ_values_filtered = self.filter_joint_sequences(self.stacked_raw_XYZ_values)
        ''' now normalise '''
        
        i = 1
                
        return
    
    
    def separate_into_joint_squences(self):
        return
        
        
    def filter_joint_sequences(self, noisy_raw_XYZ, N=2, fc=10, fs=30):        
        stacked_raw_XYZ_values_filtered  = []
        filtered_X = []
        filtered_Z = []
        filtered_Z = []
        
        for joint in SkeletonJoints:
            joint_number = joint.value
            #or 'WRIST' in joint.name
            
            if 'HAND' in joint.name  or 'THUMB' in joint.name or 'FOOT' in joint.name or 'ANKLE' in joint.name:
                _N=N
                _fc=N
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
            
            if filtered_X == []:
                filtered_X = X
                filtered_Y = Y
                filtered_Z = Z
            else:
                filtered_X = np.dstack([filtered_X, X])
                filtered_Y = np.dstack([filtered_Y, Y])
                filtered_Z = np.dstack([filtered_Z, Z])
            
#            filtered_XYZ = 
#            
#            if stacked_raw_XYZ_values_filtered == []:
#                stacked_raw_XYZ_values_filtered = filtered_XYZ
#            else:
                
        
#       
        filtered_X = np.transpose(filtered_X)
        filtered_Y = np.transpose(filtered_Y)
        filtered_Z = np.transpose(filtered_Z)
        
        stacked_raw_XYZ_values_filtered = np.stack([filtered_X[:,:,0], filtered_Y[:,:,0], filtered_Z[:,:,0]])  
        #stacked_raw_XYZ_values_filtered = np.reshape(stacked_raw_XYZ_values_filtered,[3,26,600])
        
        return stacked_raw_XYZ_values_filtered
        
    def normalise_skeleton(self, skel_frame):
        if len(self._ref_spine_base) == 0:
            self._ref_spine_base =  skel_frame.iloc[SkeletonJoints.SPINEBASE.value][['X', 'Y', 'Z']].tolist()
            self._ref_skel_frame =  skel_frame
            #print('Getting Spinebase')
        
        normalised_skel_frame = pd.DataFrame.copy(skel_frame,deep=False)
        x = 0
        y = 1
        z = 2

        for joint in SkeletonJoints:
            joint_name = joint.name
            joint_number = joint.value
             
            #normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('X')] = (skel_frame.iloc[joint_number]['X'] - self._ref_skel_frame.iloc[joint_number]['X'])*10
            #normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Y')] = (skel_frame.iloc[joint_number]['Y'] - self._ref_skel_frame.iloc[joint_number]['Y'])*10
            #normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Z')] = (skel_frame.iloc[joint_number]['Z'] - self._ref_skel_frame.iloc[joint_number]['Z'])*10
            
            #if skel_frame.iloc[joint_number]['Tracked'] != 'Tracked':
            #    print(self._frame_count, joint_name, skel_frame.iloc[joint_number]['Tracked'])
                
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('X')] = (skel_frame.iloc[joint_number]['X'] - self._ref_spine_base[x])
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Y')] = (skel_frame.iloc[joint_number]['Y'] - self._ref_spine_base[y])
            normalised_skel_frame.iloc[joint_number, normalised_skel_frame.columns.get_loc('Z')] = (skel_frame.iloc[joint_number]['Z'] - self._ref_spine_base[z])
        
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
        
        return skel_frame, raw_XYZ
    
    def calulate_CoM_position(self, skel_frame):
        _X = 2
        _Y = 3
        _Z = 4
    
        spine_base = np.stack([skel_frame['X'][SkeletonJoints.SPINEMID.value], skel_frame['Y'][SkeletonJoints.SPINEMID.value], skel_frame['Z'][SkeletonJoints.SPINEMID.value]])
        hip_left = np.stack([skel_frame['X'][SkeletonJoints.HIPLEFT.value], skel_frame['Y'][SkeletonJoints.HIPLEFT.value], skel_frame['Z'][SkeletonJoints.HIPLEFT.value]])
        hip_right = np.stack([skel_frame['X'][SkeletonJoints.HIPRIGHT.value], skel_frame['Y'][SkeletonJoints.HIPRIGHT.value], skel_frame['Z'][SkeletonJoints.HIPRIGHT.value]])
    
        x_mean = np.mean([spine_base[0],hip_left[0],hip_right[0]])
        y_mean = np.mean([spine_base[1],hip_left[1],hip_right[1]])
        z_mean = np.mean([spine_base[2],hip_left[2],hip_right[2]])
    
        CoM = np.stack([x_mean,y_mean,z_mean])
    
        #com = spine_base
    
        return CoM.tolist()
    
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
        

if __name__ == "__main__": 
    #_part_id = '3'
    #_part_id = '9'
    #_part_id = '24'
    #_part_id = '404'
    #_part_id = '303'
    _part_id = '700'
    _movement = 'Quiet-Standing-Eyes-Open'
    #_movement = 'Quiet-Standing-Eyes-Closed'
    #_movement = 'Foam-Quiet-Standing-Eyes-Open'
    #_movement = 'Foam-Quiet-Standing-Eyes-Closed'
    #_movement = 'Tandem-Balance'
    #_movement = 'Unilateral-Stance-Eyes-Closed'
    #_movement = 'Ramp-Quiet-Standing-Eyes-Closed'
    _skel_root_path = '/media/55129822/Kinect Database/SPPB/SPPB' + _part_id + '/SPPB'+ _part_id +'_'+ _movement +'/skel'   
    my_KinectRecording = KinectRecording(_skel_root_path)
    my_KinectRecording._part_id = _part_id
    
#%%    
    for joint in HierarchicalSkeletonJoints:
       #A = my_KinectRecording.skeletons
       #B = my_KinectRecording.raw_XYZ_values
       #E = my_KinectRecording.stacked_raw_XYZ_values
       #F = my_KinectRecording.stacked_raw_XYZ_values_filtered
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
    
       X = my_KinectRecording.stacked_raw_XYZ_values_filtered[0, joint_number, :]*100
       Y = my_KinectRecording.stacked_raw_XYZ_values_filtered[1, joint_number, :]*100
       Z = my_KinectRecording.stacked_raw_XYZ_values_filtered[2, joint_number, :]*100
    
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
        X = my_KinectRecording.stacked_raw_XYZ_values_filtered[0, joint_number, :]*100
        Y = my_KinectRecording.stacked_raw_XYZ_values_filtered[1, joint_number, :]*100
        Z = my_KinectRecording.stacked_raw_XYZ_values_filtered[2, joint_number, :]*100
        
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
    Z = my_KinectRecording.stacked_raw_XYZ_values_filtered[2, joint_number, :]*100
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
    
    X = my_KinectRecording.stacked_raw_XYZ_values_filtered[0, joint_number, :]*100
    Y = my_KinectRecording.stacked_raw_XYZ_values_filtered[1, joint_number, :]*100
    Z = my_KinectRecording.stacked_raw_XYZ_values_filtered[2, joint_number, :]*100
            
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
    import numpy.fft as fft
    spectrum = fft.fft(sig)
    abs_spec = spectrum.abs
    #You can then plot the magnitudes of the FFT as
    
    freq = fft.fftfreq(len(abs_spec))
    plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    plt.plot(freq)
    plt.show()
    
    plt.plot(freq, abs_spec)
    plt.title('PSD - ' + _part_id + ' ' + HierarchicalSkeletonJoints.COM.name + ' ' + _movement)
    #plt.xlim(-0.01, 0.01)
    plt.show()
    
    #print('mean:', round(np.mean(freq),3))
    print('SD:', round(np.std(freq),3))
    
    
    
