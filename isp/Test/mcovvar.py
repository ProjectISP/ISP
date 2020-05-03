#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:24:12 2019

@author: robertocabieces
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from os import listdir
from os.path import isfile, join
from scipy.stats import chi2

#i=0
#file='mcovvar.txt'

def normalizeAngle(v):
    """
    Normalize angles to [0..180] degrees
    :param v: angle
    :return: normalized angle
    """
    if v < 0.0:
        v += 180.0
    if v > 180.0:
        v -= 180.0
    return v
def computeOriginErrors(i,file):
    ConfidenceEllipsoid={}
    OriginUncertainty={}
    """
    Given a NLL's event build the Confidence Ellipsoid from Covariance Matrix
    :param evt: NLL's QML Event
    :return: Dictionary containing computed errors
    """

    # WARNING: QuakeML uses meter for origin depth, origin uncertainty and confidence ellipsoid, SC3ML uses kilometers.

    d = {}

    confidenceLevel = 0.90  # Conficence level

    kp1 = np.sqrt(chi2.ppf(confidenceLevel, 1))  # 1D confidence coefficient
    kp2 = np.sqrt(chi2.ppf(confidenceLevel, 2))  # 2D confidence coefficient
    kp3 = np.sqrt(chi2.ppf(confidenceLevel, 3))  # 3D confidence coefficient

    # Covariance matrix is given in the NLL's "STATISTICS" line of *.grid0.loc.hyp file and in the Origin's comments parsed by ObsPy
   # Covariance matrix
    cvm = np.genfromtxt(file,dtype='str')
    # Code adapted from IGN's computation of ConfidenceEllipsoid in "locsat.cpp" program
    cvxx = float(cvm[i][1])
    cvxy = float(cvm[i][3])
    cvxz = float(cvm[i][5])
    cvyy = float(cvm[i][7])
    cvyz = float(cvm[i][9])
    cvzz = float(cvm[i][11])

    nll3d = np.array([[cvxx, cvxy, cvxz],
                      [cvxy, cvyy, cvyz],
                      [cvxz, cvyz, cvzz]
                      ])

    # 1D confidence intervals at confidenceLevel
    errx = kp1 * np.sqrt(cvxx)
    
    d['longitude_errors'] = errx

    erry = kp1 * np.sqrt(cvyy)
    
    d['latitude_errors'] = erry

    errz = kp1 * np.sqrt(cvzz)
    
    d['depth_errors'] = errz
    
    #NLL kp1=1 because is up to 1 sigma 68.3%, LocSAT kp1=2.71 because is up to 90% (one dim) 
    #LocSAT np.sqrt(cvzz)/2.71 = NLL np.sqrt(cvzz)



    # 2D confidence intervals at confidenceLevel
    nll2d = np.array(nll3d[:2, :2])
    eigval2d, eigvec2d = np.linalg.eig(nll2d)  # XY (horizontal) plane

    # indexes are not necessarily ordered. Sort them by eigenvalues
    idx = eigval2d.argsort()
    eigval2d = eigval2d[idx]
    eigvec2d = eigvec2d[:, idx]

    # sminax = kp2 * np.sqrt(eigval2d[0]) * 1.0e3  # QML in meters
    # smajax = kp2 * np.sqrt(eigval2d[1]) * 1.0e3  # QML in meters
    sminax = kp2 * np.sqrt(eigval2d[0])  # SC3ML in kilometers
    smajax = kp2 * np.sqrt(eigval2d[1])  # SC3ML in kilometers
    strike = 90.0 - np.rad2deg(np.arctan(eigvec2d[1, 1] / eigvec2d[0, 1]))  # calculate and refer it to North
    # horizontalUncertainty = np.sqrt((errx ** 2) + (erry ** 2)) * 1.0e3   # QML in meters
    horizontalUncertainty = np.sqrt((errx ** 2) + (erry ** 2))   # SC3ML in kilometers

    # 3D confidence intervals at confidenceLevel
    eigval3d, eigvec3d = np.linalg.eig(nll3d)
    idx = eigval3d.argsort()
    eigval3d = eigval3d[idx]
    eigvec3d = eigvec3d[:, idx]

    # s3dminax = kp3 * np.sqrt(eigval3d[0]) * 1.0e3   # QML in meters
    # s3dintax = kp3 * np.sqrt(eigval3d[1]) * 1.0e3   # QML in meters
    # s3dmaxax = kp3 * np.sqrt(eigval3d[2]) * 1.0e3   # QML in meters
    s3dminax = kp3 * np.sqrt(eigval3d[0])   # SC3ML in kilometers
    s3dintax = kp3 * np.sqrt(eigval3d[1])   # SC3ML in kilometers
    s3dmaxax = kp3 * np.sqrt(eigval3d[2])   # SCEML in kilometers

    majaxplunge = normalizeAngle(
        np.rad2deg(np.arctan(eigvec3d[2, 2] / np.sqrt((eigvec3d[2, 0] ** 2) + (eigvec3d[2, 1] ** 2)))))
    majaxazimuth = normalizeAngle(np.rad2deg(np.arctan(eigvec3d[2, 1] / eigvec3d[2, 0])))
    majaxrotation = normalizeAngle(
        np.rad2deg(np.arctan(eigvec3d[0, 2] / np.sqrt((eigvec3d[0, 0] ** 2) + (eigvec3d[0, 1] ** 2)))))

    # print('2D sminax:\t{}\tsmajax:\t{}\tstrike:\t{}'.format(sminax, smajax, strike))
    # print('3D sminax:\t{}\tsmajax:\t{}\tsintax:\t{}'.format(s3dminax, s3dmaxax, s3dintax))
    # print('   plunge:\t{}\tazim:\t{}\trotat:\t{}'.format(majaxplunge, majaxazimuth, majaxrotation))
    # print('-' * 144)
    ConfidenceEllipsoid['semi_major_axis_length']=s3dmaxax
    ConfidenceEllipsoid['semi_major_axis_length']=s3dminax
    ConfidenceEllipsoid['semi_intermediate_axis_length']=s3dintax
    ConfidenceEllipsoid['major_axis_plung']=majaxplunge
    ConfidenceEllipsoid['mmajor_axis_azimuth']=majaxazimuth
    ConfidenceEllipsoid['major_axis_rotation']=majaxrotation
    
    ##
    OriginUncertainty['horizontal_uncertainty']=horizontalUncertainty
    OriginUncertainty['min_horizontal_uncertainty']=sminax
    OriginUncertainty['max_horizontal_uncertainty']=smajax
    OriginUncertainty['azimuth_max_horizontal_uncertainty']=strike
    #OriginUncertainty['confidence_ellipsoid']=ce
    OriginUncertainty['confidence_level']=confidenceLevel * 100.0
    
    
    

    return d,ConfidenceEllipsoid,OriginUncertainty