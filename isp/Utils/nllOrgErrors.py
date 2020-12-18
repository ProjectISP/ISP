#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from obspy.core.event import QuantityError, ConfidenceEllipsoid, OriginUncertainty

"""
################################################################################
#    Copyright (C) 2018 by IGN Madrid                                          #
#                                                                              #
#    author: J. Barco, J. V. Cantavella                                        #
#    email:  jbarco@fomento.es, jvcantavella@fomento.es                        #
#    last modified: 2018/08/20                                                 #
#                                                                              #
#    This program is free software; you can redistribute it and/or modify      #
#    it under the terms of the GNU General Public License as published by      #
#    the Free Software Foundation; either version 2 of the License, or         #
#    (at your option) any later version.                                       #
#                                                                              #
#    This program is distributed in the hope that it will be useful,           #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of            #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             #
#    GNU General Public License for more details.                              #
#                                                                              #
#    You should have received a copy of the GNU General Public License         #
#    along with this program; if not, write to the                             #
#    Free Software Foundation, Inc.,                                           #
#    59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.                 #
################################################################################

@author: Jaime Barco - jbarco@fomento.es
@date:   2018-08-20T12:00:00.0000Z
"""

__program__ = "nll2sc3"
__author__ = "J. Barco <jbarco@fomento.es>, Juan V. Cantavella <jvcantavella@fomento.es>"
__version__ = "0.0.1"
__date__ = "Aug 2018"
__license__ = "GPL (v2 or later)"
__copyright__ = "(c) Jaime Barco, Juan V. Cantavella - Aug 2018"

# system imports
import numpy as np
from scipy.stats import chi2


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


def computeOriginErrors(org):
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
    comments = org['comments'][0].text
    stats = comments.split('STATISTICS')[-1].split()
    cvm = [float(i) for i in stats[1::2]][3:9]  # Covariance matrix

    # Code adapted from IGN's computation of ConfidenceEllipsoid in "locsat.cpp" program
    cvxx = cvm[0]
    cvxy = cvm[1]
    cvxz = cvm[2]
    cvyy = cvm[3]
    cvyz = cvm[4]
    cvzz = cvm[5]

    nll3d = np.array([[cvxx, cvxy, cvxz],
                      [cvxy, cvyy, cvyz],
                      [cvxz, cvyz, cvzz]
                      ])

    # 1D confidence intervals at confidenceLevel
    errx = kp1 * np.sqrt(cvxx)
    qe = QuantityError(uncertainty=errx, confidence_level=confidenceLevel * 100.0)
    d['longitude_errors'] = qe

    erry = kp1 * np.sqrt(cvyy)
    qe = QuantityError(uncertainty=erry, confidence_level=confidenceLevel * 100.0)
    d['latitude_errors'] = qe

    errz = kp1 * np.sqrt(cvzz)
    qe = QuantityError(uncertainty=errz, confidence_level=confidenceLevel * 100.0)
    d['depth_errors'] = qe
    
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

    ce = ConfidenceEllipsoid(semi_major_axis_length=s3dmaxax,
                             semi_minor_axis_length=s3dminax,
                             semi_intermediate_axis_length=s3dintax,
                             major_axis_plunge=majaxplunge,
                             major_axis_azimuth=majaxazimuth,
                             major_axis_rotation=majaxrotation)

    ou = OriginUncertainty(horizontal_uncertainty=horizontalUncertainty,
                           min_horizontal_uncertainty=sminax,
                           max_horizontal_uncertainty=smajax,
                           azimuth_max_horizontal_uncertainty=strike,
                           confidence_ellipsoid=ce,
                           preferred_description='confidence ellipsoid',
                           confidence_level=confidenceLevel * 100.0)

    d['origin_uncertainty'] = ou

    return d


