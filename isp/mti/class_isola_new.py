#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 12:33:24 2020

@author: robertocabieces
"""
import subprocess
import fractions
import os
import re
import warnings
import obspy
from mpl_toolkits.mplot3d import Axes3D
from obspy.core import read, AttribDict
from obspy import Stream, UTCDateTime, Trace
from obspy.geodetics.base import gps2dist_azimuth
import math
import scipy.interpolate
from pyproj import Geod  # transformation of geodetic coordinates
import shutil
from scipy.io import FortranFile
#from isp.mti.MouseTrap import *
import numpy as np
import multiprocessing as mp
from obspy.imaging.beachball import beach
from obspy.imaging.mopad_wrapper import beach as beach2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib import animation
warnings.filterwarnings("ignore", category=DeprecationWarning)

def rename_keys(somedict, prefix='', suffix=''):
	"""
	Returns a dictionary with keys renamed by adding some prefix and/or suffix
	:param somedict: dictionary, whose keys will be remapped
	:type somedict: dictionary
	:param prefix: new keys starts with
	:type prefix: string, optional
	:param suffix: new keys ends with
	:type suffix: string, optional
	:returns : dictionary with keys renamed
	"""
	return dict(map(lambda key, value: (prefix+str(key)+suffix, value), somedict.items()))

def align_yaxis(ax1, ax2, v1=0, v2=0):
    """
    Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)

def to_percent(y, position):
    """
    Something with tics positioning used by :func:`histogram`
    # Ignore the passed in position. This has the effect of scaling the default tick locations.
    """
    s = "{0:2.0f}".format(100 * y)
    if mpl.rcParams['text.usetex'] is True:
        return s + r' $\%$'
    else:
        return s + ' %'



def histogram(data, outfile=None, bins=100, range=None, xlabel='', multiply=1, reference=None, reference2=None,
              fontsize=None):
    """
    Plots a histogram of a given data.

    :param data: input values
    :type data: array
    :param outfile: filename of the output. If ``None``, plots to the screen.
    :type outfile: string or None, optional
    :param bins: number of bins of the histogram
    :type bins: integer, optional
    :param range: The lower and upper range of the bins. Lower and upper outliers are ignored. If not provided, range is (data.min(), data.max()).
    :type range: tuple of 2 floats, optional
    :param xlabel: x-axis label
    :type xlabel: string, optional
    :param multiply: Normalize the sum of histogram to a given value. If not set, normalize to 1.
    :type multiply: float, optional
    :param reference: plots a line at the given value as a reference
    :type reference: array_like, scalar, or None, optional
    :param reference2: plots a line at the given value as a reference
    :type reference2: array_like, scalar, or None, optional
    :param fontsize: size of the font of tics, labels, etc.
    :type fontsize: scalar, optional

    Uses `matplotlib.pyplot.hist <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist>`_
    """
    if fontsize:
        plt.rcParams.update({'font.size': fontsize})
    weights = np.ones_like(data) / float(len(data)) * multiply
    if type(bins) == tuple:
        try:
            n = 1 + 3.32 * np.log10(len(data))  # Sturgesovo pravidlo
        except:
            n = 10
        if range:
            n *= (range[1] - range[0]) / (max(data) - min(data))
        bins = max(min(int(n), bins[1]), bins[0])
    plt.hist(data, weights=weights, bins=bins, range=range)
    ax = plt.gca()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    if reference != None:
        try:
            iter(reference)
        except:
            reference = (reference,)
        for ref in reference:
            ax.add_artist(Line2D((ref, ref), (0, ymax), color='r', linewidth=5))
    if reference2 != None:
        try:
            iter(reference2)
        except:
            reference2 = (reference2,)
        for ref in reference2:
            ax.add_artist(Line2D((ref, ref), (0, ymax), color=(0., 1., 0.2), linewidth=5, linestyle='--'))
    if range:
        plt.xlim(range[0], range[1])
    if xlabel:
        plt.xlabel(xlabel)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(to_percent))
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 3))
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
    plt.close()


def decompose(mt):
    """
    Decomposition of the moment tensor using eigenvalues and eigenvectors according to paper Vavrycuk, JoSE.

    :param mt: moment tensor in system 'NEZ'
    :type mt: list of 6 floats
    :return: dictionary {'dc_perc': double couple percentage, 'clvd_perc': compensated linear vector dipole percentage, 'iso_perc': isotropic component percentage, 'faultplanes': list of fault planes parameters, 'moment': scalar seismic moment, 'Mw': moment magnitude :math:`M_W`, 's1': strike (fault plane 1), 'd1': dip (fault plane 1), 'r1': slip rake (fault plane 1), 's2': strike (fault plane 2), 'd2': dip (fault plane 2), 'r2': slip rake (fault plane 2)}
    """
    M = np.array([
        [mt[0], mt[3], mt[4]],
        [mt[3], mt[1], mt[5]],
        [mt[4], mt[5], mt[2]]])
    m, v = np.linalg.eig(M)
    idx = m.argsort()[::-1]
    m = m[idx]
    v = v[:, idx]

    iso = 1. / 3. * m.sum()
    clvd = 2. / 3. * (m[0] + m[2] - 2 * m[1])
    dc = 1. / 2. * (m[0] - m[2] - np.abs(m[0] + m[2] - 2 * m[1]))
    moment = np.abs(iso) + np.abs(clvd) + dc
    iso_perc = 100 * iso / moment
    clvd_perc = 100 * clvd / moment
    dc_perc = 100 * dc / moment
    Mw = 2. / 3. * np.log10(moment) - 18.1 / 3.
    p = v[:, 0]
    n = v[:, 1]
    t = v[:, 2]
    c1_4 = 0.5 * np.sqrt(2)
    n1 = (p + t) * c1_4  # normals to fault planes
    n2 = (p - t) * c1_4

    if iso_perc < 99.9:
        s1, d1, r1 = angles(n2, n1)
        s2, d2, r2 = angles(n1, n2)
    else:
        s1, d1, r1 = (None, None, None)
        s2, d2, r2 = (None, None, None)
    return {'dc_perc': dc_perc, 'clvd_perc': clvd_perc, 'iso_perc': iso_perc, 'mom': moment, 'Mw': Mw, 'eigvecs': v,
            'eigvals': m,
            'p': p, 't': t, 'n': n,
            's1': s1, 'd1': d1, 'r1': r1, 's2': s2, 'd2': d2, 'r2': r2,
            'faultplanes': [(s1, d1, r1), (s2, d2, r2)]}


def angles(n1, n2):
    """
    Calculate strike, dip, and rake from normals to the fault planes.

    :param n1, n2: normals to the fault planes
    :type n1, n2: list or 1-D array of 3 floats
    :return: return a tuple of the strike, dip, and rake (one of two possible solutions; the second can be obtained by switching parameters ``n1`` and ``n2``)

    Written according to the fortran program sile4_6acek.for by J. Sileny
    """
    eps = 1e-3
    if n1[2] > 0:
        n2 *= -1
        n1 *= -1
    if -n1[2] < 1:
        dip = np.arccos(-n1[2])
    else:
        dip = 0.
    if abs(abs(n1[2]) - 1) < eps:  # n1[2] close to +-1
        rake = 0.
        strike = np.arctan2(n2[1], n2[0])
        if strike < 0: strike += 2 * np.pi
    else:
        strike = np.arctan2(-n1[0], n1[1])
        if strike < 0: strike += 2 * np.pi
        cf = np.cos(strike)
        sf = np.sin(strike)
        if abs(n1[2]) < eps:
            if abs(strike) < eps:
                rake = np.arctan2(-n2[2], n2[0])
            elif abs(abs(strike) - np.pi / 2) < eps:
                rake = np.arctan2(-n2[2], n2[1])
            else:
                if abs(cf) > abs(sf):
                    rake = np.arctan2(-n2[2], n2[0] / cf)
                else:
                    rake = np.arctan2(-n2[2], n2[1] / sf)
        else:
            rake = np.arctan2((n2[0] * sf - n2[1] * cf) / -n1[2], n2[0] * cf + n2[1] * sf)
    strike, dip, rake = np.rad2deg((strike, dip, rake))
    return strike, dip, rake


def a2mt(a, system='NEZ'):
    """
    Convert the coefficients of elementary seismograms to moment tensor components.

    :param a: coefficients of elementary seismograms
    :type a: list of 6 floats
    :param system: coordinate system: 'NEZ' = coordinate positive to north-east-down in given order, 'USE' = up-south-east
    :type system: string
    :return: list of 6 components of the moment tensor
    """
    mt = [-a[3, 0] + a[5, 0], -a[4, 0] + a[5, 0], a[3, 0] + a[4, 0] + a[5, 0], a[0, 0], a[1, 0],
          -a[2, 0]]  # [M11, M22, M33, M12, M13, M23] in NEZ system
    if system == 'NEZ':
        return mt
    elif system == 'USE':
        return [mt[2], mt[0], mt[1], mt[4], -mt[5], -mt[3]]  # convert to USE system


def Green_wrapper(i, model, x, y, z, npts_exp, elemse_start_origin):
    """
    Evaluate Green's function using code ``Axitra`` (programs ``gr_xyz`` and ``elemse``) in a given grid point.

    :param i: number (identifier) of grid point
    :type i: integer
    :param model: identifier of crust model
    :type model: string
    :param x: source coordinate in N-S direction [m] (positive to the north)
    :type x: float
    :param y: source coordinate in E-W direction [m] (positive to the east)
    :type y: float
    :param z: source depth [m] (positive down)
    :type z: float
    :param npts_exp: the number of samples in the computation is :math:`2^{\mathrm{npts\_exp}}`
    :type npts_exp: integer
    :param elemse_start_origin: time between elementary seismogram start and elementary seismogram origin time
    :type elemse_start_origin: float
    :param logfile: path to text file, where are details about computation logged
    :type logfile: string, optional

    Remark: because of paralelisation, this wrapper cannot be part of class :class:`ISOLA`.
    """
    root_path = os.path.dirname(os.path.abspath(__file__))
    logfile = os.path.join(root_path,'output/log_green.txt')
    
    iter_max = 10
    point_id = str(i).zfill(4)
    if model:
        point_id += '-' + model

    log = open(logfile, 'a')
    #meta = open('green/elemse' + point_id + '.txt', 'w')
    meta = open(root_path+'/green/elemse'+ point_id + '.txt', 'w')
    CWD = os.path.join(root_path, 'green')
    for iter in range(iter_max):
        process = subprocess.Popen(
            ['./gr_xyz', '{0:1.3f}'.format(x / 1e3), '{0:1.3f}'.format(y / 1e3), '{0:1.3f}'.format(z / 1e3),
             point_id,
             model], stdout=subprocess.PIPE, cwd=CWD)  # spustit GR_XYZ
        out, err = process.communicate()
        if not out and not err:
            break
        else:
            if iter == iter_max - 1:
                log.write('grid point {0:3d}, gr_xyz failed {1:2d} times, POINT SKIPPED\n'.format(i, iter))
                return False
    log.write('grid point {0:3d}, {1:2d} calculation(s)\n'.format(i, iter + 1))
    process = subprocess.Popen(['./elemse', str(npts_exp), point_id, "{0:8.3f}".format(elemse_start_origin)],
                               stdout=subprocess.PIPE, cwd=CWD)  # spustit CONSHIFT
    out, err = process.communicate()
    if out or err:
        log.write('grid point {0:3d}: elemse FAILED\n'.format(i, iter))
        return False
    log.close()
    meta.write('{0:1.3f} {1:1.3f} {2:1.3f}'.format(x / 1e3, y / 1e3, z / 1e3))
    meta.close()
    return True


def read_elemse(nr, npts, filename, stations):
    """
    Reads elementary seismograms file generated by code ``Axitra``.

    :param nr: number of receivers contained
    :type nr: integer
    :param npts: number of points of each component
    :type npts: integer
    :param filename: path to the file
    :type filename: string, optional
    :param stations: ``isola.stations`` metadata of inverted stations
    :type stations: list of dictionaries

    :return: elementary seismograms in form of list of lists of streams
    """
    ff = {}
    tr = Trace(data=np.empty(npts))
    tr.stats.npts = npts
    elemse_all = []
    for r in range(nr):
        model = stations[r]['model']
        if model not in ff:
            if model:
                f = filename[0:filename.rfind('.')] + '-' + model + filename[filename.rfind('.'):]
            else:
                f = filename
            ff[model] = FortranFile(f)
        elemse = []
        for j in range(6):
            f = {}
            f['N'] = tr.copy()
            f['E'] = tr.copy()
            f['Z'] = tr.copy()

            for i in range(npts):
                t, N, E, Z = ff[model].read_reals('f')
                f['N'].data[i] = N
                f['E'].data[i] = E
                f['Z'].data[i] = Z
                if i == 0:
                    t1 = t
                elif i == npts - 1:
                    t2 = t
            tl = t2 - t1
            samprate = (npts - 1) / tl
            delta = tl / (npts - 1)
            for comp in ['N', 'E', 'Z']:
                f[comp].stats.channel = comp;
                f[comp].stats.sampling_rate = samprate
                f[comp].stats.delta = delta
            st = Stream(traces=[f['Z'], f['N'], f['E']])
            elemse.append(st)
        elemse_all.append(elemse)
    del ff
    return elemse_all


def my_filter(data, fmin, fmax):
    """
    Filter used for filtering both elementary and observed seismograms
    """
    if fmax:
        data.filter('lowpass', freq=fmax)
    if fmin:
        data.filter('highpass', freq=fmin, corners=2)
        data.filter('highpass', freq=fmin, corners=2)


def invert(point_id, d_shifts, norm_d, Cd_inv, nr, comps, stations, npts_elemse, npts_slice, elemse_start_origin,
           deviatoric=False, decomp=True):
    """
    Solves inverse problem in a single grid point for multiple time shifts.

    :param point_id: grid point id, elementary seismograms are readed from 'green/elemse'+point_id+'.dat'
    :type point_id: string
    :param d_shifts: list of shifted data vectors :math:`d`
    :type d_shifts: list of :class:`~numpy.ndarray`
    :param norm_d: list of norms of vectors :math:`d`
    :type norm_d: list of floats
    :param Cd_inv: inverse of the data covariance matrix :math:`C_D^{-1}` saved block-by-block
    :type Cd_inv: list of :class:`~numpy.ndarray`
    :param nr: number of receivers
    :type nr: integer
    :param comps: number of components used in inversion
    :type comps: integer
    :param stations: 2DO popsat
    :type stations: 2DO popsat
    :param npts_elemse: number of points of elementary seismograms
    :type npts_elemse: integer
    :param npts_slice: number of points of seismograms used in inversion (npts_slice <= npts_elemse)
    :type npts_slice: integer
    :param fmin: lower frequency for filtering elementary seismogram
    :type fmin: float
    :param fmax: higher frequency for filtering elementary seismogram
    :type fmax: float
    :param elemse_start_origin: time between elementary seismogram start and elementary seismogram origin time
    :type elemse_start_origin: float
    :param deviatoric: if ``True``, invert only deviatoric part of moment tensor (5 components), otherwise full moment tensor (6 components)
    :type deviatoric: bool, optional
    :param decomp: if ``True``, decomposes found moment tensor in each grid point
    :type decomp: bool, optional
    :returns: Dictionary {'shift': order of `d_shift` item, 'a': coeficients of the elementary seismograms, 'VR': variance reduction, 'CN' condition number, and moment tensor decomposition (keys described at function :func:`decompose`)}

    It reads elementary seismograms for specified grid point, filter them and creates matrix :math:`G`.
    Calculates :math:`G^T`, :math:`G^T G`, :math:`(G^T G)^{-1}`, and condition number of :math:`G^T G` (using :func:`~np.linalg.cond`)
    Then reads shifted vectors :math:`d` and for each of them calculates :math:`G^T d` and the solution :math:`(G^T G)^{-1} G^T d`. Calculates variance reduction (VR) of the result.

    Finally chooses the time shift where the solution had the best VR and returns its parameters.

    Remark: because of parallelisation, this wrapper cannot be part of class :class:`ISOLA`.
    """

    root_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(root_path,'green/elemse')

    if deviatoric:
        ne = 5
    else:
        ne = 6
    elemse = read_elemse(nr, npts_elemse, filename + point_id + '.dat', stations)  # nacist elemse

    #elemse[0][0].plot() # DEBUG

    # filtrovat elemse
    for r in range(nr):
        for i in range(ne):
            # elemse[r][i].filter('highpass', freq=0.01) # DEBUG - pri instrumentalni korekci to same
            my_filter(elemse[r][i], stations[r]['fmin'], stations[r]['fmax'])
    # elemse[0][0].plot() # DEBUG

    if npts_slice != npts_elemse:
        dt = elemse[0][0][0].stats.delta
        for st6 in elemse:
            for st in st6:
                # st.trim(UTCDateTime(0)+dt*elemse_start_origin, UTCDateTime(0)+dt*npts_slice+dt*elemse_start_origin+1)
                st.trim(UTCDateTime(0) + elemse_start_origin)
        npts = npts_slice
    else:
        npts = npts_elemse
    # elemse[0][0].plot() # DEBUG

    # RESIT OBRACENOU ULOHU
    # pro kazdy bod site a cas zdroje
    #   m = (G^T G)^-1 G^T d
    #     pamatovat si m, misfit, kondicni cislo, prip. singularni cisla

    c = 0
    G = np.empty((comps * npts, ne))
    for r in range(nr):
        for comp in range(3):
            if stations[r][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # this component has flag 'use in inversion'
                weight = stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
                for i in range(npts):
                    for e in range(ne):
                        G[c * npts + i, e] = elemse[r][e][comp].data[i] * weight
                c += 1
    if Cd_inv:
        # evaluate G^T C_D^{-1}
        # G^T C_D^{-1} is in ``GtCd`` saved block-by-block, in ``Gt`` in one ndarray
        idx = 0
        GtCd = []
        # print('\nINVERT')
        for C in Cd_inv:
            size = len(C)
            # print(G.shape, size, idx, G[idx:idx+size, : ].T.shape, C.shape) # DEBUG
            GtCd.append(np.dot(G[idx:idx + size, :].T, C))
            idx += size
        Gt = np.concatenate(GtCd, axis=1)
    else:
        Gt = G.transpose()
    GtG = np.dot(Gt, G)
    CN = np.sqrt(np.linalg.cond(GtG))  # condition number
    GtGinv = np.linalg.inv(GtG)
    det_Ca = np.linalg.det(GtGinv)
    # print('det', det_Ca) # DEBUG

    res = {}
    sum_c = 0
    for shift in range(len(d_shifts)):
        d_shift = d_shifts[shift]
        # d : vector of data shifted
        #   shift>0 means that elemse start `shift` samples after data zero time

        # Gtd
        Gtd = np.dot(Gt, d_shift)

        # result : coeficients of elementary seismograms
        a = np.dot(GtGinv, Gtd)
        # a[0] = 1.; a[1] = 2.; a[2] = 3.; a[3] = 4.; a[4] = 5.; a[5] = 6.
        if deviatoric: a = np.append(a, [[0.]], axis=0)

        if Cd_inv:
            dGm = d_shift - np.dot(G, a[:ne])  # dGm = d_obs - G m
            idx = 0
            dGmCd_blocks = []
            # dCd_blocks = []
            for C in Cd_inv:
                size = len(C)
                dGmCd_blocks.append(np.dot(dGm[idx:idx + size, :].T, C))
                idx += size
            dGmCd = np.concatenate(dGmCd_blocks, axis=1)
            misfit = np.dot(dGmCd, dGm)[0, 0]
        else:
            synt = np.zeros(comps * npts)
            for i in range(ne):
                synt += G[:, i] * a[i]
            misfit = 0
            for i in range(npts * comps):
                misfit += (d_shift[i, 0] - synt[i]) ** 2
        VR = 1 - misfit / norm_d[shift]

        res[shift] = {}
        res[shift]['a'] = a.copy()
        res[shift]['misfit'] = misfit
        res[shift]['VR'] = VR

    shift = max(res, key=lambda s: res[s]['VR'])  # best shift

    r = {}
    r['shift'] = shift
    r['a'] = res[shift]['a'].copy()
    r['VR'] = res[shift]['VR']
    r['misfit'] = res[shift]['misfit']
    r['CN'] = CN
    r['GtGinv'] = GtGinv
    r['det_Ca'] = det_Ca
    r['shifts'] = res
    # r['res'] = res
    if decomp:
        r.update(decompose(a2mt(r['a'])))  # add MT decomposition to dict `r`
    return r


class ISOLA:
    def __init__(self, stream, deltas, location_unc=0, depth_unc=0, time_unc=0, deviatoric=False, step_x=500, step_z=500,
                 max_points=100, threads=2, circle_shape=True, use_precalculated_Green=False, rupture_velocity=1000,
                 s_velocity=3000, decompose=True):

        self.root_path = os.path.dirname(os.path.abspath(__file__))
        logfile = os.path.join(self.root_path,'output','log.txt')
        self.data_raw = stream
        self.data_deltas = deltas

        self.location_unc = location_unc  # m
        self.depth_unc = depth_unc  # m
        self.time_unc = time_unc  # s
        self.deviatoric = deviatoric
        self.step_x = step_x  # m
        self.step_z = step_z  # m
        self.max_points = max_points
        self.threads = threads
        self.circle_shape = circle_shape
        self.use_precalculated_Green = use_precalculated_Green
        self.rupture_velocity = rupture_velocity
        self.s_velocity = s_velocity
        self.decompose = decompose
        self.logfile = open(logfile, 'w', 1)
        self.data = []
        #self.data_raw = []
        # self.data_unfiltered = []
        self.noise = []
        self.Cd_inv = []
        self.Cd = []
        self.LT = []
        self.LT3 = []
        self.Cf = []
        self.fmax = 0.
        #self.data_deltas = []  # list of ``stats.delta`` values of traces in ``self.data`` or ``self.data_raw``
        self.mt_decomp = []
        self.max_VR = ()
        self.logtext = {}
        self.idx_use = {0: 'useZ', 1: 'useN', 2: 'useE'}
        self.idx_weight = {0: 'weightZ', 1: 'weightN', 2: 'weightE'}
        self.movie_writer = 'mencoder'  # None for default
        self.models = {}
        self.log('Inversion of ' + {1: 'deviatoric part of', 0: 'full'}[self.deviatoric] + ' moment tensor (' +
                 {1: '5', 0: '6'}[self.deviatoric] + ' components)')

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        self.logfile.close()
        del self.data
        #del self.data_raw
        # del self.data_unfiltered
        del self.noise
        del self.Cd_inv
        del self.Cd
        del self.LT
        del self.LT3
        del self.Cf

    def lcmm(self, b, *args):
        """
        Returns generelized least common multiple.

        :param b,args: numbers to compute least common multiple of them
        :type b,args: float, which is a multiple of 0.00033333
        :returns: the least multiple of ``a`` and ``b``
        """
        b = 3 / b
        if b - round(b) < 1e6:
            b = round(b)
        for a in args:
            a = 3 / a
            if a - round(a) < 1e6:
                a = round(a)
            b = fractions.gcd(a, b)
        return 3 / b

    def next_power_of_2(self, n):
        """
        Return next power of 2 greater than or equal to ``n``

        :type n: integer
        """
        return 2 ** (n - 1).bit_length()

    def log(self, s, newline=True, printcopy=False):
        """
        Write text into log file

        :param s: Text to write into log
        :type s: string
        :param newline: if is ``True``, add LF symbol (\\\\n) at the end
        :type newline: bool, optional
        :param printcopy: if is ``True`` prints copy of ``s`` also to stdout
        :type printcopy: bool, optional
        """
        self.logfile.write(s)
        if newline:
            self.logfile.write('\n')
        if printcopy:
            print(s)

    def set_event_info(self, lat, lon, depth, mag, t, agency=''):
        """
        Sets event coordinates, magnitude, and time from parameters given to this function

        :param lat: event latitude
        :type lat: float
        :param lon: event longitude
        :type lon: float
        :param depth: event depth in km
        :type lat: float
        :param mag: event moment magnitude
        :type lat: float
        :param t: event origin time
        :type t: :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` or string
        :param agency: agency which provides this location
        :type lat: string, optional
        """
        if type(t) == str:
            t = UTCDateTime(t)
        self.event = {'lat': lat, 'lon': lon, 'depth': float(depth) * 1e3, 'mag': float(mag), 't': t, 'agency': agency}
        self.log(
            '\nHypocenter location:\n  Agency: {agency:s}\n  Origin time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth{d:4.1f} km'.format(
                t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=float(lat), lon=float(lon), d=float(depth), agency=agency))

    def check_a_station_present(self):
        """
        Checks whether at least one station is present, otherwise raises error.

        Called from :func:`load_streams_ArcLink` and :func:`load_files`.
        """
        if not len(self.stations):
            self.log('No station present. Exiting...')
            raise ValueError('No station present.')

    def attach_ISOLA_paz(self, tr, paz_file):
        """
        Attaches to a trace a paz AttribDict containing poles zeros and gain.

        :param tr: Trace
        :type tr: :class:`~obspy.core.trace.Trace`
        :param paz_file: path to pazfile in ISOLA format
        :type paz_file: string
        """
        f = open(paz_file, 'r')
        f.readline()  # comment line: A0
        A0 = float(f.readline())
        f.readline()  # comment line: count-->m/sec
        count2ms = float(f.readline())
        f.readline()  # comment line: zeros
        n_zeros = int(f.readline())
        zeros = []
        for i in range(n_zeros):
            line = f.readline()
            search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
            (r, i) = search.groups()
            zeros.append(complex(float(r), float(i)))
        f.readline()  # comment line: poles
        n_poles = int(f.readline())
        poles = []
        for i in range(n_poles):
            line = f.readline()
            search = re.search('([-0-9.eE+]+)[ 	]+([-0-9.eE+]+)', line)
            try:
                (r, i) = search.groups()
            except:
                print(line)
            poles.append(complex(float(r), float(i)))
        tr.stats.paz = AttribDict({
            'sensitivity': A0,
            'poles': poles,
            'gain': 1. / count2ms,
            'zeros': zeros
        })
        f.close()

    def read_event_info(self, filename):
        """
        Read event coordinates, magnitude, and time from a file in specified format (see below)

        :param filename: path to file
        :type filename: string

        .. rubric:: File format
        .. code-block:: none

            longitude  latitude
            depth
            moment_magnitude
            date [YYYYmmDD] (unused)
            hour
            minute
            second
            agency which provides this location

        .. rubric:: File example::
        .. code-block:: none

            21.9877  38.4045
            6
            4.5
            20120425
            10
            34
            11.59
            UPSL
        """
        inp_event = open(filename, 'r')
        l = inp_event.readlines()
        inp_event.close()
        e = {}
        e['lon'], e['lat'] = l[0].split()
        e['depth'] = float(l[1]) * 1e3
        e['mag'] = float(l[2])
        e['t'] = UTCDateTime(l[3]) + int(l[4]) * 3600 + int(l[5]) * 60 + float(l[6])
        e['agency'] = l[7]
        self.event = e
        self.log(
            '\nHypocenter location:\n  Agency: {agency:s}\n  Origin time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth {d:3.1f} km'.format(
                t=e['t'].strftime('%Y-%m-%d %H:%M:%S'), lat=float(e['lat']), lon=float(e['lon']), d=e['depth'] / 1e3,
                agency=e['agency']))
        self.log('\nEvent info: ' + filename)

    def read_network_coordinates(self, filename, network='', location='', channelcode='LH', write = True):
        """
        Read informations about stations from file in ISOLA format.
        Calculate their distances and azimuthes using WGS84 elipsoid.
        Create data structure ``self.stations``. Sorts it according to station epicentral distance.

        :param filename: path to file with network coordinates
        :type filename: string
        :param network: all station are from specified network
        :type network: string, optional
        :param location: all stations has specified location
        :type location: string, optional
        :param channelcode: component names of all stations start with these letters (if channelcode is `LH`, component names will be `LHZ`, `LHN`, and `LHE`)
        :type channelcode: string, optional
        :param write: if not null, specifies name of created file with carthesian coordinates of stations (writed by :func:`write_stations`); if `Null`, no file is created
        :type write: string, optional or `Null`
        """
        # 2DO: osetreni chyby, pokud neni event['lat'] a ['lon']
        self.logtext['network'] = s = 'Station coordinates: ' + filename
        self.log(s)
        inp = open(filename, 'r')
        lines = inp.readlines()
        inp.close()
        # g = Geod(ellps='WGS84')
        stats = []
        for line in lines:
            if line == '\n':  # skip empty lines
                continue
            # 2DO: souradnice stanic dle UTM
            items = line.split()
            sta, lat, lon = items[0:3]
            if len(items) > 3:
                model = items[3]
            else:
                model = ''
            if model not in self.models:
                self.models[model] = 1
            else:
                self.models[model] += 1
            net = network;
            loc = location;
            ch = channelcode  # default values given by function parameters
            if ":" in sta:
                l = sta.split(':')
                net = l[0];
                sta = l[1]
                if len(l) > 2: loc = l[2]
                if len(l) > 3: ch = l[3]
            stn = {'code': sta, 'lat': lat, 'lon': lon, 'network': net, 'location': loc, 'channelcode': ch,
                   'model': model}

            dist, az, baz = gps2dist_azimuth(float(self.event['lat']), float(self.event['lon']), float(lat), float(lon))
            stn['az'] = az
            stn['dist'] = dist
            stn['useN'] = stn['useE'] = stn['useZ'] = True
            stn['weightN'] = stn['weightE'] = stn['weightZ'] = 1.
            stats.append(stn)
        stats = sorted(stats, key=lambda stn: stn['dist'])  # seradit podle vzdalenosti
        self.stations = stats
        self.create_station_index()
        if write:
            self.write_stations()

    def create_station_index(self):
        """
        Creates ``self.stations_index`` which serves for accesing ``self.stations`` items by the station name.
        It is called from :func:`read_network_coordinates`.
        """
        stats = self.stations
        self.nr = len(stats)
        self.stations_index = {}
        for i in range(self.nr):
            self.stations_index[
                '_'.join([stats[i]['network'], stats[i]['code'], stats[i]['location'], stats[i]['channelcode']])] = \
                stats[i]

    def set_use_components(self, map):

        for i in map:
            station_name = i[0]
            channel = i[1]
            ch = channel[2]
            checked = i[2]
            for j in range(len(self.stations)):
                if self.stations[j]['code'] == station_name:
                    if ch == 'E' or "1" or "X":
                        self.stations[j]['useE'] = checked

                    elif  ch == 'N' or "2" or "Y":
                        self.stations[j]['useN'] = checked

                    elif ch == 'Z':
                        self.stations[j]['useZ'] = checked




    def write_stations(self):
        """
        Write file with carthesian coordinates of stations. The file is necessary for Axitra code.

        This function is usually called from :func:`read_network_coordinates`.

        :param filename: name (with path) to created file
        :type filename: string, optional
        """
        filename = os.path.join(self.root_path, 'green/station.dat')
        for model in self.models:
            if model:
                f = filename[0:filename.rfind('.')] + '-' + model + filename[filename.rfind('.'):]
            else:
                f = filename
            outp = open(f, 'w')
            outp.write(' Station co-ordinates\n x(N>0,km),y(E>0,km),z(km),azim.,dist.,stat.\n')
            for s in self.stations:
                if s['model'] != model:
                    continue
                az = math.radians(s['az'])
                dist = s['dist'] / 1000  # from meter to kilometer
                outp.write('{N:10.4f} {E:10.4f} {z:10.4f} {az:10.4f} {d:10.4f} {code:4s} ?\n'.format(N=math.cos(az) * dist,
                                                                                                     E=math.sin(az) * dist,
                                                                                                     z=0, az=s['az'],
                                                                                                     d=dist,
                                                                                                     code=s['code']))

            outp.close()

    def read_crust(self, source):

        """
        Copy a file or files with crustal model definition to location where code ``Axitra`` expects it

        :param source: path to crust file
        :type source: string
        :param output: path to copy target
        :type output: string, optional
        """
        output = os.path.join(self.root_path, 'green/crustal.dat')
        inputs = []
        for model in self.models:
            if model:
                inp = source[0:source.rfind('.')] + '-' + model + source[source.rfind('.'):]
                outp = output[0:output.rfind('.')] + '-' + model + output[output.rfind('.'):]
            else:
                inp = source
                outp = output
            shutil.copyfile(inp, outp)
            inputs.append(inp)
        self.log('Crustal model(s): ' + ', '.join(inputs))
        self.logtext['crust'] = ', '.join(inputs)

    def detect_mouse(self, mouse_len=2.5 * 60, mouse_onset=1 * 60, fit_t1=-20, fit_t2c=0, fit_t2v=1200, figures=None,
                     figures_mkdir=True):
        """
        Wrapper for :class:`MouseTrap`

        :param mouse_len: synthetic mouse length in second
        :param mouse_onset: the onset of the synthetic mouse is `mouse_onset` seconds after synthetic mouse starttime
        :param fit_t1: mouse fitting starts this amount of seconds after an event origin time (negative value to start fitting before the origin time)
        :param fit_t2c: mouse fitting endtime -- constant term
        :param fit_t2v: mouse fitting endtime -- linear term (see equation below)

        Endtime of fitting is :math:`t_2 = \mathrm{fit\_t2c} + \mathrm{dist} / \mathrm{fit\_t2v}` where :math:`\mathrm{dist}` is station epicentral distance.
        """
        self.log('\nMouse detection:')
        out = ''
        for st0 in self.data_raw:
            st = st0.copy()
            t_start = max(st[0].stats.starttime, st[1].stats.starttime, st[2].stats.starttime)
            t_start_origin = self.event['t'] - t_start
            paz = st[0].stats.paz
            # Functions imported from Mouse Trap
            demean(st, t_start_origin)
            ToDisplacement(st)  # Just integrate st
            # error = PrepareRecord(st, t_start_origin) # demean, integrate, check signal-to-noise ratio
            # if error:
            # print ('    %s' % error)
            # create synthetic m1
            # t_len = min(st[0].stats.endtime, st[1].stats.endtime, st[2].stats.endtime) - t_start (unused)
            dt = st[0].stats.delta
            # m1 is an object of class mouse
            m1 = mouse(fit_time_before=50, fit_time_after=60)
            m1.create(paz, int(mouse_len / dt), dt, mouse_onset)
            # fit waveform by synthetic m1
            sta = st[0].stats.station
            for comp in range(3):
                stats = st[comp].stats
                dist = self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])]['dist']
                try:
                    m1.fit_mouse(st[comp], t_min=t_start_origin + fit_t1,
                                 t_max=t_start_origin + fit_t2c + dist / fit_t2v)
                except:
                    out += '  ' + sta + ' ' + stats.channel + ': MOUSE detecting problem (record too short?), ignoring component in inversion\n'
                    self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
                        'use' + stats.channel[2]] = False
                else:
                    onset, amp, dummy, dummy, fit = m1.params(degrees=True)
                    amp = abs(amp)
                    detected = False
                    if (amp > 50e-8) or (amp > 10e-8 and fit > 0.8) or (amp > 7e-8 and fit > 0.9) or (
                            amp > 5e-9 and fit > 0.94) or (
                            fit > 0.985):  # DEBUGGING: fit > 0.95 in the before-last parentheses?
                        out += '  ' + sta + ' ' + stats.channel + ': MOUSE detected, ignoring component in inversion (time of onset: {o:6.1f} s, amplitude: {a:10.2e} m s^-2, fit: {f:7.2f})\n'.format(
                            o=onset - t_start_origin, a=amp, f=fit)
                        self.stations_index['_'.join([stats.network, sta, stats.location, stats.channel[0:2]])][
                            'use' + stats.channel[2]] = False
                        detected = True
                    if figures:
                        if not os.path.exists(figures) and figures_mkdir:
                            os.mkdir(figures)
                        m1.plot(st[comp], outfile=os.path.join(figures,
                                                               'mouse_' + ('no', 'YES')[detected] + '_' + sta + str(
                                                                   comp) + '.png'), xmin=t_start_origin - 60,
                                xmax=t_start_origin + 240, ylabel='raw displacement [counts]',
                                title="{{net:s}}:{{sta:s}} {{ch:s}}, fit: {fit:4.2f}".format(fit=fit))
        self.logtext['mouse'] = out
        self.log(out, newline=False)

    def correct_data(self):
        """
        Corrects ``self.data_raw`` for the effect of instrument. Poles and zeros must be contained in trace stats.
        """
        for st in self.data_raw:
            st.detrend(type='demean')
            st.filter('highpass', freq=0.01)  # DEBUG
            for tr in st:
                tr.simulate(paz_remove=tr.stats.paz)

        self.data_are_corrected = True

    def set_parameters(self, fmax, fmin=0., wavelengths=5, log=True):
        """
        Sets some technical parameters of the inversion.

        Technically, just runs following functions:
         - :func:`set_frequencies`
         - :func:`set_working_sampling`
         - :func:`set_time_window`
         - :func:`set_Greens_parameters`
         - :func:`set_grid`
         - :func:`set_time_grid`
         - :func:`count_components`

        The parameters are parameters of the same name of these functions.
        """
        self.set_frequencies(fmax, fmin, wavelengths)
        self.set_working_sampling()
        self.set_time_window()
        self.set_Greens_parameters()
        self.set_grid()
        self.set_time_grid()
        self.count_components(log)

    def set_frequencies(self, fmax, fmin=0., wavelengths=5):
        """
        Sets frequency range for each station according its distance.

        :type fmax: float
        :param fmax: minimal inverted frequency for all stations
        :type fmax: float, optional
        :param fmax: maximal inverted frequency for all stations
        :type wavelengths: float, optional
        :param wavelengths: maximal number of wavelengths between the source and the station; if exceeded, automatically decreases ``fmax``

        The maximal frequency for each station is determined according to the following formula:

        :math:`\min ( f_{max} = \mathrm{wavelengths} \cdot \mathrm{self.s\_velocity} / r, \; fmax )`,

        where `r` is the distance the source and the station.
        """
        for stn in self.stations:
            dist = np.sqrt(stn['dist'] ** 2 + self.event['depth'] ** 2)
            stn['fmax'] = min(wavelengths * self.s_velocity / dist, fmax)
            stn['fmin'] = fmin
            self.fmax = max(self.fmax, stn['fmax'])

    def set_working_sampling(self, multiple8=False):
        """
        Determine maximal working sampling as at least 8-multiple of maximal inverted frequency (``self.fmax``). If needed, increases the value to eneables integer decimation factor.

        :param multiple8: if ``True``, force the decimation factor to be such multiple, that decimation can be done with factor 8 (more times, if needed) and finaly with factor <= 8. The reason for this is decimation pre-filter unstability for higher decimation factor (now not needed).
        :type multiple8: bool, optional
        """
        # min_sampling = 4 * self.fmax
        min_sampling = 8 * self.fmax  # teoreticky 4*fmax aby fungovala L2 norma????
        SAMPRATE = 1. / self.lcmm(
            *self.data_deltas)  # kazda stanice muze mit jine vzorkovani, bereme nejvetsiho spolecneho delitele (= 1. / nejmensi spolecny nasobek)
        decimate = SAMPRATE / min_sampling
        if multiple8:
            if decimate > 128:
                decimate = int(decimate / 64) * 64
            elif decimate > 16:
                decimate = int(decimate / 8) * 8
            else:
                decimate = int(decimate)
        else:
            decimate = int(decimate)
        self.max_samprate = SAMPRATE
        # print(min_sampling, SAMPRATE, decimate) # DEBUG
        # print(self.data_deltas) # DEBUG
        self.samprate = SAMPRATE / decimate
        self.logtext['samplings'] = samplings_str = ", ".join(
            ["{0:5.1f} Hz".format(1. / delta) for delta in self.data_deltas])
        self.log(
            '\nSampling frequencies:\n  Data sampling: {0:s}\n  Common sampling: {3:5.1f}\n  Decimate factor: {1:3d} x\n  Sampling used: {2:5.1f} Hz'.format(
                samplings_str, decimate, self.samprate, SAMPRATE))

    def set_time_window(self):
        """
        Determines number of samples for inversion (``self.npts_slice``) and for Green's function calculation (``self.npts_elemse`` and ``self.npts_exp``) from ``self.min_time`` and ``self.max_time``.

        :math:`\mathrm{npts\_slice} \le \mathrm{npts\_elemse} = 2^{\mathrm{npts\_exp}} < 2\cdot\mathrm{npts\_slice}`
        """
        self.min_time(self.stations[0]['dist'])
        self.max_time(self.stations[self.nr - 1]['dist'])
        self.t_min -= 20  # FIXED OPTION
        self.t_min = round(self.t_min * self.samprate) / self.samprate
        if self.t_min > 0:
            self.t_min = 0.
        self.elemse_start_origin = -self.t_min
        self.t_len = self.t_max - self.t_min
        self.npts_slice = int(math.ceil(self.t_max * self.samprate))
        self.npts_elemse = self.next_power_of_2(int(math.ceil(self.t_len * self.samprate)))
        if self.npts_elemse < 64:  # FIXED OPTION
            self.npts_exp = 6
            self.npts_elemse = 64
        else:
            self.npts_exp = int(math.log(self.npts_elemse, 2))

    def set_Greens_parameters(self):
        """
        Sets parameters for Green's function calculation:
         - time window length ``self.tl``
         - number of frequencies ``self.freq``
         - spatial periodicity ``self.xl``

        Writes used parameters to the log file.
        """
        self.tl = self.npts_elemse / self.samprate
        # freq = int(math.ceil(fmax*tl))
        # self.freq = min(int(math.ceil(self.fmax*self.tl))*2, self.npts_elemse/2) # pocitame 2x vic frekvenci, nez pak proleze filtrem, je to pak lepe srovnatelne se signalem, ktery je kauzalne filtrovany
        self.freq = int(self.npts_elemse / 2) + 1
        self.xl = max(np.ceil(self.stations[self.nr - 1]['dist'] / 1000),
                      100) * 1e3 * 20  # `xl` 20x vetsi nez nejvetsi epicentralni vzdalenost, zaokrouhlena nahoru na kilometry, minimalne 2000 km
        self.log(
            "\nGreen's function calculation:\n  npts: {0:4d}\n  tl: {1:4.2f}\n  freq: {2:4d}\n  npts for inversion: {3:4d}".format(
                self.npts_elemse, self.tl, self.freq, self.npts_slice))

    def set_grid(self, min_depth=1000):
        """
        Generates grid ``self.grid`` of points, where the inverse problem will be solved.
        `Rupture length` is estimated as :math:`111 \cdot 10^{M_W}`.
        Horizontal diameter of the grid is determined as ``self.location_unc`` + `rupture_length`.
        Vertical half-size of the grid is ``self.depth_unc`` + `rupture_length`.
        If ``self.circle_shape`` is ``True``, the shape of the grid is cylinder, otherwise it is rectangular box.
        The horizontal grid spacing is defined by ``self.step_x``, the vertical by ``self.step_z``. If it leads to more grid points than ``self.max_points``, the both spacings are increased (both by the same ratio) to approximately fit ``self.max_points``.

        :parameter min_depth: minimal grid point depth in meters
        :type min_depth: float, optional
        """
        step_x = self.step_x;
        step_z = self.step_z;
        max_points = self.max_points
        self.rupture_length = rupture_length = math.sqrt(
            111 * 10 ** self.event['mag'])  # M6 ~ 111 km2, M5 ~ 11 km2 		REFERENCE NEEDED
        self.radius = radius = self.location_unc + rupture_length
        self.depth_min = depth_min = max(min_depth, self.event['depth'] - self.depth_unc - rupture_length)
        self.depth_max = depth_max = self.event['depth'] + self.depth_unc + rupture_length
        n_points = np.pi * (radius / step_x) ** 2 * (depth_max - depth_min) / step_z
        if n_points > max_points:
            step_x *= (n_points / max_points) ** 0.333
            step_z *= (n_points / max_points) ** 0.333
        n_steps = int(radius / step_x)
        n_steps_z = int((self.depth_unc + rupture_length) / step_z)
        depths = []
        for k in range(-n_steps_z, n_steps_z + 1):
            z = self.event['depth'] + k * step_z
            if z >= depth_min and z <= depth_max:
                depths.append(z)
        self.grid = []
        self.steps_x = []
        for i in range(-n_steps, n_steps + 1):
            x = i * step_x
            self.steps_x.append(x)
            for j in range(-n_steps, n_steps + 1):
                y = j * step_x
                if math.sqrt(x ** 2 + y ** 2) > radius and self.circle_shape:
                    continue
                for z in depths:
                    edge = z == depths[0] or z == depths[-1] or (
                            math.sqrt((abs(x) + step_x) ** 2 + y ** 2) > radius or math.sqrt(
                        (abs(y) + step_x) ** 2 + x ** 2) > radius) and self.circle_shape or max(abs(i),
                                                                                                abs(j)) == n_steps
                    self.grid.append({'x': x, 'y': y, 'z': z, 'err': 0, 'edge': edge})
        self.depths = depths
        self.step_x = step_x;
        self.step_z = step_z
        self.log(
            '\nGrid parameters:\n  number of points: {0:4d}\n  horizontal step: {1:5.0f} m\n  vertical step: {2:5.0f} m\n  grid radius: {3:6.3f} km\n  minimal depth: {4:6.3f} km\n  maximal depth: {5:6.3f} km'.format(
                len(self.grid), step_x, step_z, radius / 1e3, depth_min / 1e3, depth_max / 1e3))

    def set_time_grid(self):
        """
        Sets equidistant time grid defined by ``self.shift_min``, ``self.shift_max``, and ``self.shift_step`` (in seconds). The corresponding values ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step`` are (rounded) in samples related to the the highest sampling rate common to all stations.
        """
        self.rupture_length = rupture_length = math.sqrt(111 * 10 ** self.event['mag'])  # M6 ~ 111 km2, M5 ~ 11 km2
        self.shift_max = shift_max = self.time_unc + self.rupture_length / self.rupture_velocity
        self.shift_min = shift_min = -self.shift_max
        self.shift_step = shift_step = 1. / self.fmax * 0.01
        self.SHIFT_min = int(round(shift_min * self.max_samprate))
        self.SHIFT_max = int(round(shift_max * self.max_samprate))
        self.SHIFT_step = max(int(round(shift_step * self.max_samprate)), 1)  # max() to avoid step beeing zero
        self.SHIFT_min = int(
            round(self.SHIFT_min / self.SHIFT_step)) * self.SHIFT_step  # shift the grid to contain zero time shift
        self.log(
            '\nGrid-search over time:\n  min = {sn:5.2f} s ({Sn:3d} samples)\n  max = {sx:5.2f} s ({Sx:3d} samples)\n  step = {step:4.2f} s ({STEP:3d} samples)'.format(
                sn=shift_min, Sn=self.SHIFT_min, sx=shift_max, Sx=self.SHIFT_max, step=shift_step,
                STEP=self.SHIFT_step))

    def count_components(self, log=True):
        """
        Counts number of components, which should be used in inversion (e.g. ``self.stations[n]['useZ'] = True`` for `Z` component).
        This is needed for allocation proper size of matrices used in inversion.

        :param log: if true, write into log table of stations and components with information about component usage and weight
        :type log: bool, optional
        """
        c = 0
        stn = self.stations
        for r in range(self.nr):
            if stn[r]['useZ']: c += 1
            if stn[r]['useN']: c += 1
            if stn[r]['useE']: c += 1
        self.components = c
        print("number of componentes used", c)
        if log:
            out = '\nComponents used in inversion and their weights\nstation     \t   \t Z \t N \t E \tdist\tazimuth\tfmin\tfmax\n            \t   \t   \t   \t   \t(km)\t(deg)\t(Hz)\t(Hz)\n'
            for r in range(self.nr):
                out += '{net:>3s}:{sta:5s} {loc:2s}\t{ch:2s} \t'.format(sta=stn[r]['code'], net=stn[r]['network'],
                                                                        loc=stn[r]['location'],
                                                                        ch=stn[r]['channelcode'])
                for c in range(3):
                    if stn[r][self.idx_use[c]]:
                        out += '{0:3.1f}\t'.format(stn[r][self.idx_weight[c]])
                    else:
                        out += '---\t'
                if stn[r]['dist'] > 2000:
                    out += '{0:4.0f}    '.format(stn[r]['dist'] / 1e3)
                elif stn[r]['dist'] > 200:
                    out += '{0:6.1f}  '.format(stn[r]['dist'] / 1e3)
                else:
                    out += '{0:8.3f}'.format(stn[r]['dist'] / 1e3)
                out += '\t{2:3.0f}\t{0:4.2f}\t{1:4.2f}'.format(stn[r]['fmin'], stn[r]['fmax'], stn[r]['az'])
                out += '\n'
            self.logtext['components'] = out
            self.log(out, newline=False)

    def min_time(self, distance, mag=0, v=8000):
        """
        Defines the beginning of inversion time window in seconds from location origin time. Save it into ``self.t_min`` (now save 0 -- FIXED OPTION)

        :param distance: station distance in meters
        :type distance: float
        :param mag: magnitude (unused)
        :param v: the first inverted wave-group characteristic velocity in m/s
        :type v: float

        Sets ``self.t_min`` as minimal time of interest (in seconds).
        """
        t = distance / v  # FIXED OPTION
        # if t<5:
        # t = 0
        self.t_min = t

    # self.t_min = 0		# FIXED OPTION, because Green's functions with beginning in non-zero time are nou implemented yet

    def max_time(self, distance, mag=0, v=1000):
        """
        Defines the end of inversion time window in seconds from location origin time. Calculates it as :math:`\mathrm{distance} / v`.
        Save it into ``self.t_max``.

        :param distance: station distance in meters
        :type distance: float
        :param mag: magnitude (unused)
        :param v: the last inverted wave-group characteristic velocity in m/s
        :type v: float
        """
        t = distance / v  # FIXED OPTION
        self.t_max = t

    def trim_filter_data(self, noise_slice=True, noise_starttime=None, noise_length=None):
        """
        Filter ``self.data_raw`` using function :func:`prefilter_data` and :func:`my_filter`.
        Decimate ``self.data_raw`` to common sampling rate ``self.max_samprate``.
        Optionally, copy a time window for the noise analysis.
        Copy a slice to ``self.data``.

        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Specify the start time of trimmed data
        :type length: float
        :param length: Length in seconds of trimmed data.
        :type noise_slice: bool, optional
        :param noise_slice: If set to ``True``, copy a time window of the length ``lenght`` for later noise analysis. Copied noise is in ``self.noise``.
        :type noise_starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param noise_starttime: Set the starttime of the noise time window. If ``None``, the time window starts in time ``starttime``-``length`` (in other words, it lies just before trimmed data time window).
        :type noise_length: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param noise_length: Length of the noise time window (in seconds).
        """
        # Copy a slice of ``self.data_raw`` to ``self.data_unfiltered`` (the slice range is the same as for ``self.data``).

        # self.log('\nBandpass filter frequencies:\n  min: {0:4.2f}\n  max: {1:4.2f}'.format(self.fmin, self.fmax))
        # calculate time values for creating slices
        starttime = self.event['t'] + self.shift_min + self.t_min
        length = self.t_max - self.t_min + self.shift_max + 10
        endtime = starttime + length
        # length += 5 # we add 5 s to be sure, that no index will point outside the range (1 s was not enough) # WTF??? proc ne o radek vyse?
        if noise_slice:
            if not noise_length:
                noise_length = length * 4
            if not noise_starttime:
                noise_starttime = starttime - noise_length
                noise_endtime = starttime
            else:
                noise_endtime = noise_starttime + noise_length
            DECIMATE = int(round(self.max_samprate / self.samprate))

        print("Noise Window", noise_starttime, noise_endtime)
        for st in self.data_raw:
            stats = st[0].stats
            stats.location = ''
            fmax = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                'fmax']
            # self.data_unfiltered.append(st.slice(starttime, endtime))
            # self.data_unfiltered[-1].filter('lowpass', freq=fmax*10) 	# FIXED OPTION
            self.data.append(st.copy())
        for st in self.data:
            stats = st[0].stats
            stats.location = ''
            fmin = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                'fmin']
            fmax = self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                'fmax']
            decimate = int(round(st[0].stats.sampling_rate / self.max_samprate))
            if noise_slice:
                self.noise.append(st.slice(noise_starttime, noise_endtime))
                if (len(self.noise[-1]) != 3 or (
                        self.noise[-1][0].stats.endtime - self.noise[-1][0].stats.starttime < length * 1.1)) and \
                        self.stations_index[
                            '_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                            'use' + stats.channel[2]]:
                    self.log('Noise slice too short to generate covariance matrix (station ' + st[
                        0].stats.station + '). Stopping generating noise slices.')
                    noise_slice = False
                    self.noise = []
                elif len(self.noise[-1]):
                    self.my_filter(self.noise[-1], fmin / 2, fmax * 2)
                    self.noise[-1].decimate(int(decimate * DECIMATE / 2),
                                            no_filter=True)  # noise has 2-times higher sampling than data
            self.prefilter_data(st)
            st.decimate(decimate, no_filter=True)
            # my_filter(st, fmin, fmax) # moved to decimate_shift()
            st.trim(starttime, endtime)

    def my_filter(self, data, fmin, fmax):
        """
        Filter used for filtering both elementary and observed seismograms
        """
        if fmax:
            data.filter('lowpass', freq=fmax)
        if fmin:
            data.filter('highpass', freq=fmin, corners=2)
            data.filter('highpass', freq=fmin, corners=2)

    def prefilter_data(self, st):
        """
        Drop frequencies above Green's function computation high limit using :func:`numpy.fft.fft`.

        :param st: stream to be filtered
        :type st: :class:`~obspy.core.stream`
        """
        f = self.freq / self.tl
        for tr in st:
            npts = tr.stats.npts
            NPTS = self.next_power_of_2(npts)
            TR = np.fft.fft(tr.data, NPTS)
            df = tr.stats.sampling_rate / NPTS
            # print (NPTS, df, int(np.ceil(f/df)), f, tr.stats.delta) # DEBUG
            flim = int(np.ceil(f / df))
            # for i in range(flim, NPTS-flim+1):
            # TR[i] = 0+0j
            TR[flim:NPTS - flim + 1] = 0 + 0j
            tr_filt = np.fft.ifft(TR)
            tr.data = np.real(tr_filt[0:npts])

    def decimate_shift(self):
        """
        Generate ``self.data_shifts`` where are multiple copies of ``self.data`` (needed for plotting).
        Decimate ``self.data_shifts`` to sampling rate for inversion ``self.samprate``.
        Generate ``self.d_shifts`` where are multiple vectors :math:`d`, each of them shifted according to ``self.SHIFT_min``, ``self.SHIFT_max``, and ``self.SHIFT_step``
        """
        self.d_shifts = []
        self.data_shifts = []
        self.shifts = []
        starttime = self.event['t']  # + self.t_min
        length = self.t_max - self.t_min
        endtime = starttime + length
        decimate = int(round(self.max_samprate / self.samprate))
        self.decimado = decimate
        for SHIFT in range(self.SHIFT_min, self.SHIFT_max + 1, self.SHIFT_step):
            # data = deepcopy(self.data)
            shift = SHIFT / self.max_samprate
            self.shifts.append(shift)
            data = []
            for st in self.data:
                st2 = st.slice(starttime + shift - self.elemse_start_origin,
                               endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
                st2.trim(starttime + shift - self.elemse_start_origin, endtime + shift + 1, pad=True,
                         fill_value=0.)  # short records are not inverted, but they should by padded because of plotting
                st2.decimate(decimate, no_filter=True)
                stats = st2[0].stats
                stats.location = ''
                fmin = \
                    self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                        'fmin']
                fmax = \
                    self.stations_index['_'.join([stats.network, stats.station, stats.location, stats.channel[0:2]])][
                        'fmax']
                self.my_filter(st2, fmin, fmax)
                st2.trim(starttime + shift,
                         endtime + shift + 1)  # we add 1 s to be sure, that no index will point outside the range
                data.append(st2)
            self.data_shifts.append(data)
            c = 0
            d_shift = np.empty((self.components * self.npts_slice, 1))
            for r in range(self.nr):
                for comp in range(3):
                    if self.stations[r][
                        {0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:  # this component has flag 'use in inversion'
                        weight = self.stations[r][{0: 'weightZ', 1: 'weightN', 2: 'weightE'}[comp]]
                        for i in range(self.npts_slice):
                            try:
                                d_shift[c * self.npts_slice + i] = data[r][comp].data[i] * weight
                            except:
                                self.log(
                                    'Index out of range while generating shifted data vectors. Waveform file probably too short.',
                                    printcopy=True)
                                print('values for debugging: ', r, comp, c, self.npts_slice, i, c * self.npts_slice + i,
                                      len(d_shift), len(data[r][comp].data), SHIFT)
                                raise Exception(
                                    'Index out of range while generating shifted data vectors. Waveform file probably too short.')
                        c += 1
            self.d_shifts.append(d_shift)

    # Second part. Covariance matrix
    def decimate(self, a, n=2):
        """
        Decimates given sequence.

        :param data: data
        :type data: 1-D array
        :param n: decimate factor
        :type n: integer, optional

        Before decimating, filter out frequencies over Nyquist frequency using :func:`numpy.fft.fft`
        """
        npts = len(a)
        # NPTS = npts # next_power_of_2(npts)
        NPTS = npts
        A = np.fft.fft(a, NPTS)
        idx = int(np.round(npts / n / 2))
        A[idx:NPTS - idx + 1] = 0 + 0j
        # for i in range(flim, NPTS-flim+1):
        # TR[i] = 0+0j
        a = np.fft.ifft(A)
        if npts % (2 * n) == 1 or n != 2:  # keep odd length for decimate factor 2
            return a[:npts:n].real
        else:
            return a[1:npts:n].real

    def covariance_matrix(self, crosscovariance=False, save_non_inverted=False, save_covariance_function=False):
        """
        Creates covariance matrix :math:`C_D` from ``self.noise``.

        :type crosscovariance: bool, optional
        :param crosscovariance: Set ``True`` to calculate crosscovariance between components. If ``False``, it assumes that noise at components is not correlated, so non-diagonal blocks are identically zero.
        :type save_non_inverted: bool, optional
        :param save_non_inverted: If ``True``, save also non-inverted matrix, which can be plotted later.
        :type save_covariance_function: bool, optional
        :param save_covariance_function: If ``True``, save also the covariance function matrix, which can be plotted later.
        """
        self.log('\nCreating covariance matrix')
        if not self.noise:
            self.log(
                'No noise slice to generate covariance matrix. Some of records probably too short or noise slices not generated [param noise_slice at func trim_filter_data()]. Exiting...',
                printcopy=True)
            raise ValueError('No noise slice to generate covariance matrix.')
        n = self.npts_slice
        self.Cf = []
        for r in range(len(self.stations)):
            sta = self.stations[r]
            idx = []
            if sta['useZ']: idx.append(0)
            if sta['useN']: idx.append(1)
            if sta['useE']: idx.append(2)
            size = len(idx) * n
            C = np.empty((size, size))
            if save_covariance_function:
                self.Cf.append(np.ndarray(shape=(3, 3), dtype=np.ndarray))
            if not crosscovariance:
                for i in idx:
                    I = idx.index(i) * n
                    for j in idx:
                        if i == j:
                            corr = np.correlate(self.noise[r][i].data, self.noise[r][i].data, 'full') / len(
                                self.noise[r][i].data)
                            corr = self.decimate(corr, 2)  # noise has 2-times higher sampling than data
                            middle = len(corr) / 2
                            if save_covariance_function:
                                self.Cf[-1][i, i] = corr.copy()
                                self.Cf_len = len(corr)
                            for k in range(n):
                                for l in range(k, n):
                                    C[l + I, k + I] = C[k + I, l + I] = corr[middle + k - l]
                        if i != j:
                            J = idx.index(j) * n
                            C[I:I + n, J:J + n] = 0.
            else:
                for i in idx:
                    I = idx.index(i) * n
                    for j in idx:
                        J = idx.index(j) * n
                        if i > j:
                            continue
                        # index,value,corr = xcorr(self.noise[r][i], self.noise[r][j], n, True) # there was some problem with small numbers, solved by tr.data *= 1e20
                        corr = np.correlate(self.noise[r][i].data, self.noise[r][j].data, 'full') / len(
                            self.noise[r][i].data)
                        corr = self.decimate(corr, 2)  # noise has 2-times higher sampling than data
                        middle = int(len(corr) / 2)
                        if save_covariance_function:
                            self.Cf[-1][i, j] = corr.copy()
                            self.Cf_len = len(corr)
                        for k in range(n):
                            if i == j:
                                for l in range(k, n):
                                    C[l + I, k + I] = C[k + I, l + I] = corr[middle + int(k - l)]
                            # C[k+I, l+J] = corr[middle+abs(k-l)]
                            else:
                                for l in range(n):
                                    C[l + J, k + I] = C[k + I, l + J] = corr[middle + int(k - l)]
                                    # C[l+J, k+I] = corr[middle-abs(k-l)]

                # podle me nesmysl, ale fungovalo lepe nez predchozi
            # C[k+I, l+J] = corr[middle+abs(k-l)]
            # C[l+J, k+I] = corr[middle-abs(k-l)]
            # C = np.diag(np.ones(size)*corr[middle]*10) # DEBUG
            for i in idx:  # add to diagonal 1% of its average
                I = idx.index(i) * n
                C[I:I + n, I:I + n] += np.diag(np.zeros(n) + np.average(C[I:I + n, I:I + n].diagonal()) * 0.01)
            if save_non_inverted:
                self.Cd.append(C)
            if crosscovariance and len(C):
                try:
                    C_inv = np.linalg.inv(C)
                    self.LT3.append(np.linalg.cholesky(C_inv).T)
                    self.Cd_inv.append(C_inv)
                except:
                    w, v = np.linalg.eig(C)
                    print('Minimal eigenvalue C[{0:1d}]: {1:6.1e}, clipping'.format(r, min(w)))
                    w = w.real.clip(
                        min=0)  # set non-zero eigenvalues to zero and remove complex part (both are numerical artefacts)
                    # mx = max(w)
                    # w = w.real.clip(min=w*1e-18) # set non-zero eigenvalues to almost-zero and remove complex part (both are numerical artefacts)
                    v = v.real  # remove complex part of eigenvectors
                    C = v.dot(np.diag(w)).dot(v.T)
                    # C = nearcorr(C)
                    C_inv = np.linalg.inv(C)
                    w, v = np.linalg.eig(C_inv)  # DEBUG
                    if min(w) < 0:
                        print('Minimal eigenvalue C_inv: {1:6.1e}, CLIPPING'.format(r, min(w)))
                        w = w.real.clip(min=0)  # DEBUG
                        v = v.real  # DEBUG
                        C_inv = v.dot(np.diag(w)).dot(v.T)
                    self.Cd_inv.append(C_inv)
                    self.LT3.append(np.diag(np.sqrt(w)).dot(v.T))
                    self.LT.append([1, 1, 1])
            elif crosscovariance:  # C is zero-size matrix
                self.Cd_inv.append(C)
                self.LT.append([1, 1, 1])
                self.LT3.append(1)
            else:
                C_inv = np.linalg.inv(C)
                self.Cd_inv.append(C_inv)
                self.LT.append([1, 1, 1])
                for i in idx:
                    I = idx.index(i) * n
                    try:
                        self.LT[-1][i] = np.linalg.cholesky(C_inv[I:I + n, I:I + n]).T
                    except:
                        # w,v = np.linalg.eig(C[I:I+n, I:I+n])
                        # mx = max(w)
                        # print ('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, clipping'.format(r,i,min(w)))
                        # w = w.real.clip(min=0)
                        # v = v.real
                        # C[I:I+n, I:I+n] = v.dot(np.diag(w)).dot(v.T)
                        # C_inv[I:I+n, I:I+n] = np.linalg.inv(C[I:I+n, I:I+n])
                        w, v = np.linalg.eig(C_inv[I:I + n, I:I + n])
                        print('Minimal eigenvalue C[{0:1d}, {1:d}]: {2:6.1e}, CLIPPING'.format(r, i, min(w)))
                        w = w.real.clip(min=0)
                        v = v.real
                        self.LT[-1][i] = np.diag(np.sqrt(w)).dot(v.T)

    def run_inversion(self):
        """
        Runs function :func:`invert` in parallel.

        Module :class:`multiprocessing` does not allow running function of the same class in parallel, so the function :func:`invert` cannot be method of class :class:`ISOLA` and this wrapper is needed.
        """
        grid = self.grid
        todo = []
        for i in range(len(grid)):
            point_id = str(i).zfill(4)
            grid[i]['id'] = point_id
            if not grid[i]['err']:
                todo.append(i)

        # create norm_d[shift]
        norm_d = []
        for shift in range(len(self.d_shifts)):
            d_shift = self.d_shifts[shift]
            if self.Cd_inv:
                idx = 0
                dCd_blocks = []
                for C in self.Cd_inv:
                    size = len(C)
                    dCd_blocks.append(np.dot(d_shift[idx:idx + size, :].T, C))
                    idx += size
                dCd = np.concatenate(dCd_blocks, axis=1)
                norm_d.append(np.dot(dCd, d_shift)[0, 0])
            else:
                norm_d.append(0)
                for i in range(self.npts_slice * self.components):
                    norm_d[-1] += d_shift[i, 0] * d_shift[i, 0]
        # Popi Control
        #print(grid[i]['id'], self.d_shifts, norm_d, self.Cd_inv, self.nr, self.components, self.stations,
        #      self.npts_elemse, self.npts_slice, self.elemse_start_origin, self.deviatoric,decompose)

        if self.threads > 1:  # parallel
            pool = mp.Pool(processes=self.threads)
            results = [pool.apply_async(invert, args=(
                grid[i]['id'], self.d_shifts, norm_d, self.Cd_inv, self.nr, self.components, self.stations,
                self.npts_elemse, self.npts_slice, self.elemse_start_origin, self.deviatoric, decompose)) for i in
                       todo]
            output = [p.get() for p in results]
        else:  # serial
            output = []
            for i in todo:
                res = invert(grid[i]['id'], self.d_shifts, norm_d, self.Cd_inv, self.nr, self.components, self.stations,
                             self.npts_elemse, self.npts_slice, self.elemse_start_origin, self.deviatoric,
                             decompose)
                output.append(res)
        min_misfit = output[0]['misfit']
        for i in todo:
            grid[i].update(output[todo.index(i)])
            grid[i]['shift_idx'] = grid[i]['shift']
            # grid[i]['shift'] = self.shift_min + grid[i]['shift']*self.SHIFT_step/self.max_samprate
            grid[i]['shift'] = self.shifts[grid[i]['shift']]
            min_misfit = min(min_misfit, grid[i]['misfit'])
        self.max_sum_c = self.max_c = self.sum_c = 0
        for i in todo:
            gp = grid[i]
            gp['sum_c'] = 0
            for idx in gp['shifts']:
                GP = gp['shifts'][idx]
                GP['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * (GP['misfit'] - min_misfit))
                gp['sum_c'] += GP['c']
            gp['c'] = gp['shifts'][gp['shift_idx']]['c']
            # gp['c'] = np.sqrt(gp['det_Ca']) * np.exp(-0.5 * gp['misfit']-min_misfit)
            assert (gp['c'] == gp['shifts'][gp['shift_idx']]['c'])
            self.sum_c += gp['sum_c']
            self.max_c = max(self.max_c, gp['c'])
            self.max_sum_c = max(self.max_sum_c, gp['sum_c'])

    # 3 Part. Inversion
    def calculate_or_verify_Green(self):
        """
        If ``self.use_precalculated_Green`` is True, verifies whether the pre-calculated Green's functions were calculated on the same grid and with the same parameters (:func:`verify_Greens_headers` and :func:`verify_Greens_parameters`)
        Otherwise calculates Green's function (:func:`write_Greens_parameters` and :func:`calculate_Green`).

        :return: ``True`` if everything is OK, otherwise ``False``
        """

        if not self.use_precalculated_Green:  # calculate Green's functions in all grid points
            self.write_Greens_parameters()
            self.calculate_Green()
            return True
        else:  # verify whether the pre-calculated Green's functions are calculated on the same grid and with the same parameters
            if not self.verify_Greens_parameters():
                return False
            if not self.verify_Greens_headers():
                return False
        return True

    def write_Greens_parameters(self):
        """
        Writes file grdat.hed - parameters for gr_xyz (Axitra)
        """
        filename = os.path.join(self.root_path, 'green/grdat')
        for model in self.models:
            if model:
                f = filename + '-' + model + '.hed'
            else:
                f = os.path.join(self.root_path,'green/grdat.hed')

            grdat = open(f, 'w')
            grdat.write(
                "&input\nnc=10\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(
                    freq=self.freq, tl=self.tl, nr=self.models[model],
                    xl=self.xl))  # 'nc' is probably ignored in the current version of gr_xyz???
            grdat.close()

    def calculate_Green(self):
        """
        Runs :func:`Green_wrapper` (Green's function calculation) in parallel.
        """
        filename = os.path.join(self.root_path,'output/log_green.txt')
        open(filename, "w").close()  # erase file contents
        # run `gr_xyz` aand `elemse`
        for model in self.models:
            if self.threads > 1:  # parallel
                pool = mp.Pool(processes=self.threads)
                results = [pool.apply_async(Green_wrapper, args=(
                    i, model, self.grid[i]['x'], self.grid[i]['y'], self.grid[i]['z'], self.npts_exp,
                    self.elemse_start_origin)) for i in range(len(self.grid))]
                output = [p.get() for p in results]
                for i in range(len(self.grid)):
                    if output[i] == False:
                        self.grid[i]['err'] = 1
                        self.grid[i]['VR'] = -10
            else:  # serial
                for i in range(len(self.grid)):
                    gp = self.grid[i]
                    Green_wrapper(i, model, gp['x'], gp['y'], gp['z'], self.npts_exp, self.elemse_start_origin)

    def verify_Greens_parameters(self):
        """
        Check whetrer parameters in file grdat.hed (probably used in Green's function calculation) are the same as used now.
        If it agrees, return True, otherwise returns False, print error description, and writes it into log.
        """
        filename = os.path.join(self.root_path,'green/grdat.hed')
        grdat = open(filename, 'r')
        if grdat.read() != "&input\nnc=10\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(
                freq=self.freq, tl=self.tl, nr=self.nr, xl=self.xl):
            desc = 'Pre-calculated Green\'s functions calculated with different parameters (e.g. sampling) than used now, calculate Green\'s functions again. Exiting...'
            self.log(desc)
            print(desc)
            print(
                "&input\nnc=10\nnfreq={freq:d}\ntl={tl:1.2f}\naw=0.5\nnr={nr:d}\nns=1\nxl={xl:1.1f}\nikmax=100000\nuconv=0.1E-06\nfref=1.\n/end\n".format(
                    freq=self.freq, tl=self.tl, nr=self.nr, xl=self.xl))
            return False
        grdat.close()
        return True

    def verify_Greens_headers(self):
        """
        Checked whether elementary-seismogram-metadata files (created when the Green's functions were calculated) agree with curent grid points positions.
        Used to verify whether pre-calculated Green's functions were calculated on the same grid as used now.
        """
        filename = os.path.join(self.root_path, 'green/elemse')
        for g in range(len(self.grid)):
            gp = self.grid[g]
            point_id = str(g).zfill(4)
            meta = open(filename + point_id + '.txt', 'r')
            lines = meta.readlines()
            if len(lines) == 0:
                self.grid[g]['err'] = 1
                self.grid[g]['VR'] = -10
            elif lines[0] != '{0:1.3f} {1:1.3f} {2:1.3f}'.format(gp['x'] / 1e3, gp['y'] / 1e3, gp['z'] / 1e3):
                desc = 'Pre-calculated grid point {0:d} has different coordinates, probably the shape of the grid has changed, calculate Green\'s functions again. Exiting...'.format(
                    g)
                self.log(desc)
                print(desc)
                return False
            meta.close()
        return True

    def read_elemse(self, nr, npts, filename, stations):
        """
        Reads elementary seismograms file generated by code ``Axitra``.

        :param nr: number of receivers contained
        :type nr: integer
        :param npts: number of points of each component
        :type npts: integer
        :param filename: path to the file
        :type filename: string, optional
        :param stations: ``isola.stations`` metadata of inverted stations
        :type stations: list of dictionaries

        :return: elementary seismograms in form of list of lists of streams
        """
        ff = {}
        tr = Trace(data=np.empty(npts))
        tr.stats.npts = npts
        elemse_all = []
        for r in range(nr):
            model = stations[r]['model']
            if model not in ff:
                if model:
                    f = filename[0:filename.rfind('.')] + '-' + model + filename[filename.rfind('.'):]
                else:
                    f = filename
                ff[model] = FortranFile(f)
            elemse = []
            for j in range(6):
                f = {}
                f['N'] = tr.copy()
                f['E'] = tr.copy()
                f['Z'] = tr.copy()

                for i in range(npts):
                    t, N, E, Z = ff[model].read_reals('f')
                    f['N'].data[i] = N
                    f['E'].data[i] = E
                    f['Z'].data[i] = Z
                    if i == 0:
                        t1 = t
                    elif i == npts - 1:
                        t2 = t
                tl = t2 - t1
                samprate = (npts - 1) / tl
                delta = tl / (npts - 1)
                for comp in ['N', 'E', 'Z']:
                    f[comp].stats.channel = comp;
                    f[comp].stats.sampling_rate = samprate
                    f[comp].stats.delta = delta
                st = Stream(traces=[f['Z'], f['N'], f['E']])
                elemse.append(st)
            elemse_all.append(elemse)
        del ff
        return elemse_all

    def find_best_grid_point(self):
        """
        Set ``self.centroid`` to a grid point with higher variance reduction -- the best solution of the inverse problem.
        """
        self.centroid = max(self.grid, key=lambda g: g['VR'])  # best grid point
        x = self.centroid['x'];
        y = self.centroid['y']
        az = np.degrees(np.arctan2(y, x))
        dist = np.sqrt(x ** 2 + y ** 2)
        g = Geod(ellps='WGS84')
        self.centroid['lon'], self.centroid['lat'], baz = g.fwd(self.event['lon'], self.event['lat'], az, dist)

    def print_solution(self):
        """
        Write into log the best solution ``self.centroid``.
        """
        C = self.centroid
        t = self.event['t'] + C['shift']
        self.log(
            '\nCentroid location:\n  Centroid time: {t:s}\n  Lat {lat:8.3f}   Lon {lon:8.3f}   Depth {d:5.1f} km'.format(
                t=t.strftime('%Y-%m-%d %H:%M:%S'), lat=C['lat'], lon=C['lon'], d=C['z'] / 1e3))
        self.log(
            '  ({0:5.0f} m to the north and {1:5.0f} m to the east with respect to epicenter)'.format(C['x'], C['y']))
        if C['edge']:
            self.log('  Warning: the solution lies on the edge of the grid!')
        mt2 = a2mt(C['a'], system='USE')
        c = max(abs(min(mt2)), max(mt2))
        c = 10 ** np.floor(np.log10(c))
        MT2 = mt2 / c
        if C['shift'] >= 0:
            self.log('  time: {0:5.2f} s after origin time\n'.format(C['shift']))
        else:
            self.log('  time: {0:5.2f} s before origin time\n'.format(-C['shift']))
        if C['shift'] in (self.shifts[0], self.shifts[-1]):
            self.log('  Warning: the solution lies on the edge of the time-grid!')
        self.log('  VR: {0:4.0f} %\n  CN: {1:4.0f}'.format(C['VR'] * 100, C['CN']))
        # self.log('  VR: {0:8.4f} %\n  CN: {1:4.0f}'.format(C['VR']*100, C['CN'])) # DEBUG
        self.log(
            '  MT [ Mrr    Mtt    Mpp    Mrt    Mrp    Mtp ]:\n     [{1:5.2f}  {2:5.2f}  {3:5.2f}  {4:5.2f}  {5:5.2f}  {6:5.2f}] * {0:5.0e}'.format(
                c, *MT2))

    def decompose_mopad(self, mt):
        """
        Decomposition of the moment tensor using ``obspy-mopad``.

        :param mt: moment tensor in system 'NEZ'
        :type mt: list of 6 floats
        :return: dictionary {'dc_perc': double couple percentage, 'clvd_perc': compensated linear vector dipole percentage, 'iso_perc': isotropic component percentage, 'faultplanes': list of fault planes parameters, 'moment': scalar seismic moment, 'Mw': moment magnitude :math:`M_W`, 's1': strike (fault plane 1), 'd1': dip (fault plane 1), 'r1': slip rake (fault plane 1), 's2': strike (fault plane 2), 'd2': dip (fault plane 2), 'r2': slip rake (fault plane 2)}
        """
        process = subprocess.Popen(
            ['obspy-mopad', 'decompose', '-t 20', '-c', '--', '{0:f},{1:f},{2:f},{3:f},{4:f},{5:f}'.format(*mt)],
            stdout=subprocess.PIPE)
        out, err = process.communicate()
        out = eval(out)
        f = out[23]
        return {
            'iso_perc': out[5],
            'dc_perc': out[9],
            'clvd_perc': out[15],
            'mom': out[16],
            'Mw': out[17],
            'eigvecs': out[18],
            'eigvals': out[19],
            'p': out[20],
            't': out[22],
            'faultplanes': out[23],
            's1': f[0][0], 'd1': f[0][1], 'r1': f[0][2], 's2': f[1][0], 'd2': f[1][1], 'r2': f[1][2]}

    def print_fault_planes(self, precision='3.0', tool=''):
        """
        Decompose the moment tensor of the best grid point by :func:`decompose` and writes the result to the log.

        :param precision: precision of writing floats, like ``5.1`` for 5 letters width and 1 decimal place (default ``3.0``)
        :type precision: string, optional
        :param tool: tool for the decomposition, `mopad` for :func:`decompose_mopad`, otherwise :func:`decompose` is used
        """
        mt = a2mt(self.centroid['a'])
        if tool == 'mopad':
            self.mt_decomp = self.decompose_mopad(mt)
        else:
            self.mt_decomp = decompose(mt)
        self.log('''\nScalar Moment: M0 = {{mom:5.2e}} Nm (Mw = {{Mw:3.1f}})
    DC component: {{dc_perc:{0:s}f}} %,   CLVD component: {{clvd_perc:{0:s}f}} %,   ISOtropic component: {{iso_perc:{0:s}f}} %
    Fault plane 1: strike = {{s1:{0:s}f}}, dip = {{d1:{0:s}f}}, slip-rake = {{r1:{0:s}f}}
    Fault plane 2: strike = {{s2:{0:s}f}}, dip = {{d2:{0:s}f}}, slip-rake = {{r2:{0:s}f}}'''.format(precision).format(
            **self.mt_decomp))


    #### 4 Section .Plotting staff
    def plot_map_backend(self, x, y, s, CN, MT, color, width, highlight, xmin, xmax, ymin, ymax, xlabel='', ylabel='',
                         title='', beachball_size_c=False, outfile=None):
        """
        The plotting back-end for functions :func:`plot_maps`, :func:`plot_slices` and :func:`plot_maps_sum`. There is no need for calling it directly.
        """
        plt.rcParams.update({'font.size': 16})
        xdiff = xmax - xmin
        ydiff = ymax - ymin
        if xdiff > abs(1.3 * ydiff):
            plt.figure(figsize=(16, abs(ydiff / xdiff) * 14 + 3))
        else:
            plt.figure(figsize=(abs(xdiff / ydiff) * 11 + 2, 14))
        ax = plt.gca()
        # if xmin != ymin or xmax != ymax:
        plt.axis('equal')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax, int(np.sign(ydiff)))
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        if title: plt.title(title)
        Xmin = min(x);
        Xmax = max(x);
        Ymin = min(y);
        Ymax = max(y)
        width_max = max(width)

        for i in range(len(x)):
            if highlight[i]:
                c = plt.Circle((x[i], y[i]), self.step_x / 1e3 * 0.5, color='r')
                c.set_edgecolor('r')
                c.set_linewidth(10)
                c.set_facecolor('none')  # "none" not None
                c.set_alpha(0.7)
                ax.add_artist(c)
            if width[i] > self.step_x * 1e-3 * 0.04:
                try:
                    b = beach(MT[i], xy=(x[i], y[i]), width=(width[i], width[i] * np.sign(ydiff)), linewidth=0.5,
                              facecolor=color[i], zorder=10)
                except:
                    # print('Plotting this moment tensor in a grid point crashed: ', mt2, 'using mopad')
                    try:
                        b = beach2(MT[i], xy=(x[i], y[i]), width=(width[i], width[i] * np.sign(ydiff)), linewidth=0.5,
                                   facecolor=color[i],
                                   zorder=10)  # width: at side views, mirror along horizontal axis to avoid effect of reversed y-axis
                    except:
                        print('Plotting this moment tensor in a grid point crashed: ', MT[i])
                    else:
                        ax.add_collection(b)
                else:
                    ax.add_collection(b)
            elif width[i] > self.step_x * 1e-3 * 0.001:
                b = plt.Circle((x[i], y[i]), width[i] / 2, facecolor=color[i], edgecolor='k', zorder=10, linewidth=0.5)
                ax.add_artist(b)

        if CN and s:
            # Set up a regular grid of interpolation points
            xi = np.linspace(Xmin, Xmax, 400)
            yi = np.linspace(Ymin, Ymax, 400)
            xi, yi = np.meshgrid(xi, yi)

            # Interpolate
            rbf = scipy.interpolate.Rbf(x, y, s, function='linear')
            z1 = rbf(xi, yi)
            rbf = scipy.interpolate.Rbf(x, y, CN, function='linear')
            z2 = rbf(xi, yi)

            shift = plt.imshow(z1, cmap=plt.get_cmap('PRGn'),
                               vmin=self.shift_min, vmax=self.shift_max, origin='lower',
                               extent=[Xmin, Xmax, Ymin, Ymax])
            levels = np.arange(1., 21., 1.)
            CN = plt.contour(z2, levels, cmap=plt.get_cmap('gray'), origin='lower', linewidths=1,
                             extent=[Xmin, Xmax, Ymin, Ymax], zorder=4)
            plt.clabel(CN, inline=1, fmt='%1.0f', fontsize=10)  # levels[1::2]  oznacit kazdou druhou caru
            CB1 = plt.colorbar(shift, shrink=0.5, extend='both', label='shift [s]')
            # CB2 = plt.colorbar(CN, orientation='horizontal', shrink=0.4, label='condition number',
            #                   ticks=[levels[0], levels[-1]])
            l, b, w, h = plt.gca().get_position().bounds
            ll, bb, ww, hh = CB1.ax.get_position().bounds
            # CB1.ax.set_position([ll-0.2*w, bb+0.2*h, ww, hh])
            CB1.ax.set_position([ll, bb + 0.2 * h, ww, hh])
            #ll, bb, ww, hh = CB2.ax.get_position().bounds
            #CB2.ax.set_position([l + 0.58 * w, bb + 0.07 * h, ww, hh])

        # legend beachball's color = DC%
        if self.decompose:
            x = y = xmin * 2
            plt.plot([x], [y], marker='o', markersize=15, color=(1, 0, 0), label='DC 100 %')
            plt.plot([x], [y], marker='o', markersize=15, color=(.5, 0, .5), label='DC 50 %')
            plt.plot([x], [y], marker='o', markersize=15, color=(0, 0, 1), label='DC 0 %')
            mpl.rcParams['legend.handlelength'] = 0
            if CN and s:
                plt.legend(loc='upper left', numpoints=1, bbox_to_anchor=(1, -0.05), fancybox=True)
            else:
                plt.legend(loc='upper right', numpoints=1, bbox_to_anchor=(0.95, -0.05), fancybox=True)

        # legend beachball's area
        if beachball_size_c:  # beachball's area = PDF
            r_max = self.step_x / 1e3 / 2
            r_half = r_max / 1.4142
            text_max = 'maximal PDF'
            text_half = 'half-of-maximum PDF'
            text_area = 'Beachball area ~ PDF'
        else:  # beachball's radius = VR
            VRmax = self.centroid['VR']
            r_max = self.step_x / 1e3 / 2 * VRmax
            r_half = r_max / 2
            text_max = 'VR {0:2.0f} % (maximal)'.format(VRmax * 100)
            text_half = 'VR {0:2.0f} %'.format(VRmax * 100 / 2)
            text_area = 'Beachball radius ~ VR'
        x_symb = [xmin + r_max, xmin][bool(CN and s)]  # min(xmin, -0.8*ydiff)
        x_text = xmin + r_max * 1.8
        y_line = ymin - ydiff * 0.11
        VRlegend = plt.Circle((x_symb, y_line), r_max, facecolor=(1, 0, 0), edgecolor='k', clip_on=False)
        ax.add_artist(VRlegend)
        VRlegendtext = plt.text(x_text, y_line, text_max, verticalalignment='center')
        ax.add_artist(VRlegendtext)
        y_line = ymin - ydiff * 0.20
        VRlegend2 = plt.Circle((x_symb, y_line), r_half, facecolor=(1, 0, 0), edgecolor='k', clip_on=False)
        ax.add_artist(VRlegend2)
        VRlegendtext2 = plt.text(x_text, y_line, text_half, verticalalignment='center')
        ax.add_artist(VRlegendtext2)
        y_line = ymin - ydiff * 0.26
        VRlegendtext3 = plt.text(x_text, y_line, text_area, verticalalignment='center')
        ax.add_artist(VRlegendtext3)

        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        plt.close()

    def plot_maps(self, beachball_size_c=False):
        """
        Plot figures showing how the solution is changing across the grid.

        :param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because one figure is plotted for each depth, inserts an identifier before the last dot (`map.png` -> `map_1000.png`, `map_2000.png`, etc.).
        :type outfile: string, optional
        :param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior probability density function (PPD) instead of the variance reduction VR
        :type beachball_size_c: bool, optional

        Plot top view to the grid at each depth. The solutions in each grid point (for the centroid time with the highest VR) are shown by beachballs. The color of the beachball corresponds to its DC-part. The inverted centroid time is shown by a contour in the background and the condition number by contour lines.
        """
        outfile = os.path.join(self.root_path,'output/map.png')

        r = self.radius * 1e-3 * 1.1  # to km, *1.1
        if beachball_size_c:
            max_width = np.sqrt(self.max_sum_c)
        for z in self.depths:
            # prepare data points
            x = [];
            y = [];
            s = [];
            CN = [];
            MT = [];
            color = [];
            width = [];
            highlight = []
            for gp in self.grid:
                if gp['z'] != z or gp['err']:
                    continue
                x.append(gp['y'] / 1e3);
                y.append(gp['x'] / 1e3);
                s.append(gp['shift']);
                CN.append(gp['CN'])  # NS is x coordinate, so switch it with y to be vertical
                MT.append(a2mt(gp['a'], system='USE'))
                VR = max(gp['VR'], 0)
                if beachball_size_c:
                    width.append(self.step_x / 1e3 * np.sqrt(gp['sum_c']) / max_width)
                else:
                    width.append(self.step_x / 1e3 * VR)
                if self.decompose:
                    dc = float(gp['dc_perc']) / 100
                    color.append((dc, 0, 1 - dc))
                else:
                    color.append('black')
                highlight.append(self.centroid['id'] == gp['id'])
            if outfile:
                k = outfile.rfind(".")
                filename = outfile[:k] + "_{0:0>5.0f}".format(z) + outfile[k:]
            else:
                filename = None
            self.plot_map_backend(x, y, s, CN, MT, color, width, highlight, -r, r, -r, r, xlabel='west - east [km]',
                                  ylabel='south - north [km]', title='depth {0:5.2f} km'.format(z / 1000),
                                  beachball_size_c=beachball_size_c, outfile=filename)


    def plot_slices(self, point=None, beachball_size_c=False):
        """
        Plot vertical slices through the grid of solutions in point `point`.
        If `point` not specified, use the best solution as a point.

        :param outfile: Path to the file where to plot. If ``None``, plot to the screen.
        :type outfile: string, optional
        :param point: `x` and `y` coordinates (with respect to the epicenter) of a grid point where the slices are placed through. If ``None``, uses the coordinates of the inverted centroid.
        :type point: tuple, optional
        :param beachball_size_c: If ``True``, the sizes of the beachballs correspond to the posterior probability density function (PPD) instead of the variance reduction VR
        :type beachball_size_c: bool, optional

        The legend is the same as at :func:`plot_maps`.
        """
        outfile = os.path.join(self.root_path,'output/slice.png')

        if point:
            x0, y0 = point
        else:
            x0 = self.centroid['x'];
            y0 = self.centroid['y']
        depth_min = self.depth_min / 1000;
        depth_max = self.depth_max / 1000
        depth = depth_max - depth_min
        r = self.radius * 1e-3 * 1.1  # to km, *1.1
        if beachball_size_c:
            max_width = np.sqrt(self.max_sum_c)
        for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE'):
            x = [];
            y = [];
            s = [];
            CN = [];
            MT = [];
            color = [];
            width = [];
            highlight = []
            for gp in self.grid:
                if slice == 'N-S':
                    X = -gp['x'];    Z = gp['y'] - y0
                elif slice == 'W-E':
                    X = gp['y'];    Z = gp['x'] - x0
                elif slice == 'NW-SE':
                    X = (gp['y'] - gp['x']) * 1 / np.sqrt(2);    Z = gp['x'] + gp['y'] - y0 - x0
                elif slice == 'SW-NE':
                    X = (gp['y'] + gp['x']) * 1 / np.sqrt(2);    Z = gp['x'] - gp['y'] + y0 - x0
                Y = gp['z']
                if abs(Z) > 0.001 or gp['err']:
                    continue
                x.append(X / 1e3);
                y.append(Y / 1e3);
                s.append(gp['shift']);
                CN.append(gp['CN'])
                MT.append(a2mt(gp['a'], system='USE'))
                VR = max(gp['VR'], 0)
                if beachball_size_c:
                    width.append(self.step_x / 1e3 * np.sqrt(gp['sum_c']) / max_width)
                else:
                    width.append(self.step_x / 1e3 * VR)
                if self.decompose:
                    dc = float(gp['dc_perc']) / 100
                    color.append((dc, 0, 1 - dc))
                else:
                    color.append('black')
                highlight.append(self.centroid['id'] == gp['id'])
            if outfile:
                k = outfile.rfind(".")
                filename = outfile[:k] + '_' + slice + outfile[k:]
            else:
                filename = None
            xlabel = {'N-S': 'north - south', 'W-E': 'west - east', 'NW-SE': 'north-west - south-east',
                      'SW-NE': 'south-west - north-east'}[slice] + ' [km]'
            self.plot_map_backend(x, y, s, CN, MT, color, width, highlight, -r, r, depth_max + depth * 0.05,
                                  depth_min - depth * 0.05, xlabel, 'depth [km]', title='vertical slice',
                                  beachball_size_c=beachball_size_c, outfile=filename)

    def plot_maps_sum(self):
        """
        Plot map and vertical slices through the grid of solutions showing the posterior probability density function (PPD).
        Contrary to :func:`plot_maps` and :func:`plot_slices`, the size of the beachball correspond not only to the PPD of grid-point through which is a slice placed, but to a sum of all grid-points which are before and behind.

        :param outfile: Path to the file where to plot. If ``None``, plot to the screen.
        :type outfile: string, optional

        The legend and properties of the function are similar as at function :func:`plot_maps`.
        """
        outfile = os.path.join(self.root_path,'output/map_sum.png')

        if not self.Cd_inv:
            return False  # if the data covariance matrix is unitary, we have no estimation of data errors, so the PDF has good sense
        r = self.radius * 1e-3 * 1.1  # to km, *1.1
        depth_min = self.depth_min * 1e-3;
        depth_max = self.depth_max * 1e-3
        depth = depth_max - depth_min
        Ymin = depth_max + depth * 0.05
        Ymax = depth_min - depth * 0.05
        # for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE', 'top'):
        for slice in ('N-S', 'W-E', 'top'):
            X = [];
            Y = [];
            s = [];
            CN = [];
            MT = [];
            color = [];
            width = [];
            highlight = []
            g = {}
            max_c = 0
            for gp in self.grid:
                if gp['err'] or gp['sum_c'] <= 0:   continue
                if slice == 'N-S':
                    x = -gp['x']
                elif slice == 'W-E':
                    x = gp['y']
                elif slice == 'NW-SE':
                    x = (gp['y'] - gp['x']) * 1 / np.sqrt(2)
                elif slice == 'SW-NE':
                    x = (gp['y'] + gp['x']) * 1 / np.sqrt(2)
                x *= 1e-3
                y = gp['z'] * 1e-3
                if slice == 'top':
                    x = gp['y'] * 1e-3;
                    y = gp['x'] * 1e-3  # NS is x coordinate, so switch it with y to be vertical
                if not x in g:    g[x] = {}
                if not y in g[x]: g[x][y] = {'c': 0, 'max_c': 0, 'highlight': False}
                g[x][y]['c'] += gp['sum_c']
                if g[x][y]['c'] > max_c:
                    max_c = g[x][y]['c']
                if gp['sum_c'] > g[x][y]['max_c']:
                    g[x][y]['max_c'] = gp['sum_c']
                    g[x][y]['a'] = gp['a']
                    # g[x][y]['CN'] = gp['CN']
                    # g[x][y]['s'] = gp['shift']
                    if self.decompose:
                        g[x][y]['dc'] = gp['dc_perc']
                if self.centroid['id'] == gp['id']:
                    g[x][y]['highlight'] = True
            for x in g:
                for y in g[x]:
                    X.append(x)
                    Y.append(y)
                    # s.append(g[x][y]['s'])
                    # CN.append(g[x][y]['CN'])
                    MT.append(a2mt(g[x][y]['a'], system='USE'))
                    if self.decompose:
                        dc = float(g[x][y]['dc']) * 0.01
                        color.append((dc, 0, 1 - dc))
                    else:
                        color.append('black')
                    highlight.append(g[x][y]['highlight'])
                    width.append(self.step_x * 1e-3 * np.sqrt(g[x][y]['c'] / max_c))
            if outfile:
                k = outfile.rfind(".")
                filename = outfile[:k] + '_' + slice + outfile[k:]
            else:
                filename = None
            xlabel = {'N-S': 'north - south', 'W-E': 'west - east', 'NW-SE': 'north-west - south-east',
                      'SW-NE': 'south-west - north-east', 'top': 'west - east'}[slice] + ' [km]'
            if slice == 'top':
                ymin = -r;
                ymax = r
                ylabel = 'south - north [km]'
                title = 'PDF sum: top view'
            else:
                ymin = Ymin;
                ymax = Ymax
                ylabel = 'depth [km]'
                title = 'PDF sum: side view'
            # self.plot_map_backend(X, Y, s, CN, MT, color, width, highlight, -r, r, ymin, ymax, xlabel, ylabel, title, True, filename)
            self.plot_map_backend(X, Y, None, None, MT, color, width, highlight, -r, r, ymin, ymax, xlabel, ylabel,
                                  title, True, filename)

    def plot_MT(self, facecolor='red'):
        """
        Plot the beachball of the best solution ``self.centroid``.

        :param outfile: path to the file where to plot; if ``None``, plot to the screen
        :type outfile: string, optional
        :param facecolor: color of the colored quadrants/parts of the beachball
        """
        outfile = os.path.join(self.root_path,'output/centroid.png')

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        lw = 2
        plt.xlim(-100 - lw / 2, 100 + lw / 2)
        plt.ylim(-100 - lw / 2, 100 + lw / 2)

        a = self.centroid['a']
        mt2 = a2mt(a, system='USE')
        # beachball(mt2, outfile=outfile)
        full = beach2(mt2, linewidth=lw, facecolor=facecolor, edgecolor='black', zorder=1)
        ax.add_collection(full)
        if self.decompose:
            dc = beach2((self.centroid['s1'], self.centroid['d1'], self.centroid['r1']), nofill=True, linewidth=lw / 2,
                        zorder=2)
            ax.add_collection(dc)
        if outfile:
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.clf()
        plt.close()

    def plot_uncertainty(self, n=200, reference=None, best=True, fontsize=None):
        """
        Generates random realizations based on the calculated solution and its uncertainty and plots these mechanisms and histograms of its parameters.

        :param outfile: Path to the file where to plot. If ``None``, plot to the screen. Because more figures are plotted, inserts an identifier before the last dot (`uncertainty.png` -> `uncertainty_MT.png`, `intertanty_time-shift.png`, etc.).
        :type outfile: string, optional
        :param n: number of realizations
        :type n: integer, optional
        :param reference: plot a given reference solution too; the value should be length 6 array of moment tensor in 'NEZ' coordinates or a moment tensor decomposition produced by :func:`decompose`
        :type reference: array or dictionary
        :param best: show the best solutions together too
        :type best: boolean, optional
        :param fontsize: fontsize for histogram annotations
        :type fontsize: scalar, optional
        """
        outfile = os.path.join(self.root_path, 'output/uncertainty.png')
        # Generate mechanisms
        shift = [];
        depth = [];
        NS = [];
        EW = []
        n_sum = 0
        A = []
        c = self.centroid
        for gp in self.grid:
            if gp['err']:
                continue
            for i in gp['shifts']:
                GP = gp['shifts'][i]
                n_GP = int(round(GP['c'] / self.sum_c * n))
                if n_GP == 0:
                    continue
                n_sum += n_GP
                a = GP['a']
                if self.deviatoric:
                    a = a[:5]
                cov = gp['GtGinv']
                A2 = np.random.multivariate_normal(a.T[0], cov, n_GP)
                for a in A2:
                    a = a[np.newaxis].T
                    if self.deviatoric:
                        a = np.append(a, [[0.]], axis=0)
                    A.append(a)
                shift += [self.shifts[i]] * n_GP
                depth += [gp['z'] / 1e3] * n_GP
                NS += [gp['x'] / 1e3] * n_GP
                EW += [gp['y'] / 1e3] * n_GP
        if n_sum <= 1:
            self.log(
                '\nUncertainty evaluation: nothing plotted. Posterior probability density function too wide or prefered number of mechanism ({0:d}) too low.'.format(
                    n))
            return False
        # Process mechanisms
        dc_perc = [];
        clvd_perc = [];
        iso_perc = [];
        moment = [];
        Mw = [];
        strike = [];
        dip = [];
        rake = []
        for a in A:
            mt = a2mt(a)
            MT = decompose(mt)
            dc_perc.append(MT['dc_perc'])
            clvd_perc.append(MT['clvd_perc'])
            iso_perc.append(MT['iso_perc'])
            moment.append(MT['mom'])
            Mw.append(MT['Mw'])
            strike += [MT['s1'], MT['s2']]
            dip += [MT['d1'], MT['d2']]
            rake += [MT['r1'], MT['r2']]

        # Compute standard deviation
        stdev = {'dc': np.std(dc_perc) / 100, 'clvd': np.std(clvd_perc) / 100, 'iso': np.std(iso_perc) / 100,
                 'Mw': np.std(Mw) / 0.2, 't': np.std(shift), 'x': np.std(NS), 'y': np.std(EW), 'z': np.std(depth)}

        # Plot centroid uncertainty
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        lw = 0.5
        plt.xlim(-100 - lw / 2, 100 + lw / 2)
        plt.ylim(-100 - lw / 2, 100 + lw / 2)
        for a in A:
            mt2 = a2mt(a, system='USE')
            try:
                full = beach2(mt2, linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
                ax.add_collection(full)
            except:
                print('plotting this moment tensor failed: ', mt2)
        if best:
            mt2 = a2mt(c['a'], system='USE')
            full = beach2(mt2, linewidth=lw * 3, nofill=True, edgecolor=(0., 1., 0.2))
            ax.add_collection(full)
        if reference and len(reference) == 6:
            ref = decompose(reference)
            mt2 = (reference[2], reference[0], reference[1], reference[4], -reference[5], -reference[3])
            full = beach2(mt2, linewidth=lw * 3, nofill=True, edgecolor='red')
            ax.add_collection(full)
        elif reference:
            ref = reference
            if 'mom' in ref and not 'Mw' in ref:
                ref['Mw'] = 2. / 3. * np.log10(ref['mom']) - 18.1 / 3.
            elif 'Mw' in ref and not 'mom' in ref:
                ref['mom'] = 10 ** ((ref['Mw'] + 18.1 / 3.) * 1.5)
        else:
            ref = {'dc_perc': None, 'clvd_perc': None, 'iso_perc': None, 'mom': None, 'Mw': None, 's1': 0, 's2': 0,
                   'd1': 0, 'd2': 0, 'r1': 0, 'r2': 0}
        k = outfile.rfind(".")
        s1 = outfile[:k] + '_';
        s2 = outfile[k:]
        if outfile:
            plt.savefig(s1 + 'MT' + s2, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.clf()
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        lw = 0.5
        plt.xlim(-100 - lw / 2, 100 + lw / 2)
        plt.ylim(-100 - lw / 2, 100 + lw / 2)
        for i in range(0, len(strike), 2):
            try:
                dc = beach2((strike[i], dip[i], rake[i]), linewidth=lw, nofill=True, edgecolor='black', alpha=0.1)
                ax.add_collection(dc)
            except:
                print('plotting this moment strike / dip / rake failed: ', (strike[i], dip[i], rake[i]))
        if best and self.decompose:
            dc = beach2((c['s1'], c['d1'], c['r1']), nofill=True, linewidth=lw * 3, edgecolor=(0., 1., 0.2))
            ax.add_collection(dc)
        if reference:
            dc = beach2((ref['s1'], ref['d1'], ref['r1']), linewidth=lw * 3, nofill=True, edgecolor='red')
            ax.add_collection(dc)
        if outfile:
            plt.savefig(s1 + 'MT_DC' + s2, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.clf()
        plt.close()

        # Plot histograms
        histogram(dc_perc, s1 + 'comp-1-DC' + s2, bins=(10, 100), range=(0, 100), xlabel='DC %',
                  reference=ref['dc_perc'], reference2=(None, c['dc_perc'])[best], fontsize=fontsize)
        histogram(clvd_perc, s1 + 'comp-2-CLVD' + s2, bins=(20, 200), range=(-100, 100), xlabel='CLVD %',
                  reference=ref['clvd_perc'], reference2=(None, c['clvd_perc'])[best], fontsize=fontsize)
        if not self.deviatoric:
            histogram(iso_perc, s1 + 'comp-3-ISO' + s2, bins=(20, 200), range=(-100, 100), xlabel='ISO %',
                      reference=ref['iso_perc'], reference2=(None, c['iso_perc'])[best], fontsize=fontsize)
        # histogram(moment,    s1+'mech-0-moment'+s2, bins=20, range=(self.mt_decomp['mom']*0.7,self.mt_decomp['mom']*1.4), xlabel='scalar seismic moment [Nm]', reference=ref['mom'], fontsize=fontsize)
        histogram(moment, s1 + 'mech-0-moment' + s2, bins=20,
                  range=(self.mt_decomp['mom'] * 0.7 / 2, self.mt_decomp['mom'] * 1.4 * 2),
                  xlabel='scalar seismic moment [Nm]', reference=ref['mom'], fontsize=fontsize)
        # histogram(Mw,        s1+'mech-0-Mw'+s2,     bins=20, range=(self.mt_decomp['Mw']-0.1,self.mt_decomp['Mw']+0.1), xlabel='moment magnitude $M_W$', reference=ref['Mw'], fontsize=fontsize)
        histogram(Mw, s1 + 'mech-0-Mw' + s2, bins=20,
                  range=(self.mt_decomp['Mw'] - 0.1 * 3, self.mt_decomp['Mw'] + 0.1 * 3),
                  xlabel='moment magnitude $M_W$', reference=ref['Mw'], reference2=(None, c['Mw'])[best],
                  fontsize=fontsize)
        histogram(strike, s1 + 'mech-1-strike' + s2, bins=72, range=(0, 360), xlabel=u'strike [°]', multiply=2,
                  reference=((ref['s1'], ref['s2']), None)[reference == None],
                  reference2=(None, (c['s1'], c['s2']))[best], fontsize=fontsize)
        histogram(dip, s1 + 'mech-2-dip' + s2, bins=18, range=(0, 90), xlabel=u'dip [°]', multiply=2,
                  reference=((ref['d1'], ref['d2']), None)[reference == None],
                  reference2=(None, (c['d1'], c['d2']))[best], fontsize=fontsize)
        histogram(rake, s1 + 'mech-3-rake' + s2, bins=72, range=(-180, 180), xlabel=u'rake [°]', multiply=2,
                  reference=((ref['r1'], ref['r2']), None)[reference == None],
                  reference2=(None, (c['r1'], c['r2']))[best], fontsize=fontsize)
        if len(self.shifts) > 1:
            shift_step = self.SHIFT_step / self.max_samprate
            histogram(shift, s1 + 'time-shift' + s2, bins=len(self.shifts),
                      range=(self.shifts[0] - shift_step / 2., self.shifts[-1] + shift_step / 2.),
                      xlabel='time shift [s]', reference=[0., None][reference == None],
                      reference2=(None, c['shift'])[best], fontsize=fontsize)
        if len(self.depths) > 1:
            min_depth = (self.depths[0] - self.step_z / 2.) / 1e3
            max_depth = (self.depths[-1] + self.step_z / 2.) / 1e3
            histogram(depth, s1 + 'place-depth' + s2, bins=len(self.depths), range=(min_depth, max_depth),
                      xlabel='centroid depth [km]', reference=[self.event['depth'] / 1e3, None][reference == None],
                      reference2=(None, c['z'] / 1e3)[best], fontsize=fontsize)
        if len(self.grid) > len(self.depths):
            x_lim = (self.steps_x[-1] + self.step_x / 2.) / 1e3
            histogram(NS, s1 + 'place-NS' + s2, bins=len(self.steps_x), range=(-x_lim, x_lim),
                      xlabel=u'← to south : centroid place [km] : to north →', reference=[0., None][reference == None],
                      reference2=(None, c['x'] / 1e3)[best], fontsize=fontsize)
            histogram(EW, s1 + 'place-EW' + s2, bins=len(self.steps_x), range=(-x_lim, x_lim),
                      xlabel=u'← to west : centroid place [km] : to east →', reference=[0., None][reference == None],
                      reference2=(None, c['y'] / 1e3)[best], fontsize=fontsize)

        self.log('\nUncertainty evaluation: plotted {0:d} mechanism of {1:d} requested.'.format(n_sum, n))
        self.log(
            'Standard deviation :: dc: {dc:4.2f}, clvd: {clvd:4.2f}, iso: {iso:4.2f}, Mw: {Mw:4.2f}, t: {t:4.2f}, x: {x:4.2f}, y: {y:4.2f}, z: {z:4.2f}'.format(
                **stdev))
        return stdev

    def plot_MT_uncertainty_centroid(self, n=100):
        """
        Similar as :func:`plot_uncertainty`, but only the best point of the space-time grid is taken into account, so the uncertainties should be Gaussian.
        """

        outfile = os.path.join(self.root_path, 'output/MT_uncertainty_centroid.png')
        a = self.centroid['a']
        if self.deviatoric:
            a = a[:5]
        cov = self.centroid['GtGinv']
        A = np.random.multivariate_normal(a.T[0], cov, n)

        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        plt.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        lw = 0.5
        plt.xlim(-100 - lw / 2, 100 + lw / 2)
        plt.ylim(-100 - lw / 2, 100 + lw / 2)

        for a in A:
            a = a[np.newaxis].T
            if self.deviatoric:
                a = np.append(a, [[0.]], axis=0)
            mt2 = a2mt(a, system='USE')
            # full = beach(mt2, linewidth=lw, nofill=True, edgecolor='black')
            try:
                full = beach2(mt2, linewidth=lw, nofill=True, edgecolor='black')
            except:
                print(a)
                print(mt2)
            ax.add_collection(full)
        if outfile:
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.clf()
        plt.close()

    def plot_seismo_backend_1(self, plot_stations, plot_components, comp_order, crosscomp=False, sharey=True,
                              yticks=True, title_prefix='', xlabel='time [s]', ylabel='velocity [m/s]'):
        """
        The first part of back-end for functions :func:`plot_seismo`,
        :func:`plot_covariance_function`, :func:`plot_noise`, :func:`plot_spectra`.
        There is no need for calling it directly.
        """
        data = self.data_shifts[self.centroid['shift_idx']]

        plt.rcParams.update({'font.size': 22})

        if not plot_stations:
            plot_stations = range(self.nr)
        if plot_components:
            comps = plot_components
        elif comp_order == 'NEZ':
            comps = [1, 2, 0]
        else:
            comps = [0, 1, 2]

        COMPS = (1, 3)[crosscomp]
        f, ax = plt.subplots(len(plot_stations) * COMPS, len(comps), sharex=True, sharey=('row', True)[sharey],
                             figsize=(len(comps) * 6, len(plot_stations) * 2 * COMPS))
        if len(plot_stations) == 1 and len(comps) > 1:  # one row only
            ax = np.reshape(ax, (1, len(comps)))
        elif len(plot_stations) > 1 and len(comps) == 1:  # one column only
            ax = np.reshape(ax, (len(plot_stations), 1))
        elif len(plot_stations) == 1 and len(comps) == 1:  # one cell only
            ax = np.array([[ax]])

        for c in range(len(comps)):
            ax[0, c].set_title(title_prefix + data[0][comps[c]].stats.channel[2])

        for sta in plot_stations:
            r = plot_stations.index(sta)
            ax[r, 0].set_ylabel(
                data[sta][0].stats.station + u"\n{0:1.0f} km, {1:1.0f}°".format(self.stations[sta]['dist'] / 1000,
                                                                                self.stations[sta]['az']), fontsize=16)
            # SYNT = {}
            # comps_used = 0
            for comp in comps:
                c = comps.index(comp)
                for C in range(COMPS):  # if crosscomp==False: C = 0
                    ax[COMPS * r + C, c].set_frame_on(False)
                    ax[COMPS * r + C, c].locator_params(axis='x', nbins=7)
                    ax[COMPS * r + C, c].tick_params(labelsize=16)
                    if c == 0:
                        if yticks:
                            ax[r, c].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                            ax[r, c].get_yaxis().tick_left()
                        else:
                            ax[COMPS * r + C, c].tick_params(axis='y', which='both', left='off', right='off',
                                                             labelleft='off')
                            ax[COMPS * r + C, c].yaxis.offsetText.set_visible(False)
                    else:
                        ax[COMPS * r + C, c].get_yaxis().set_visible(False)
                    if r == len(plot_stations) - 1 and C == COMPS - 1:
                        ax[COMPS * r + C, c].get_xaxis().tick_bottom()
                    else:
                        ax[COMPS * r + C, c].get_xaxis().set_visible(False)
        extra_artists = []
        if xlabel:
            extra_artists.append(f.text(0.5, 0.04 + 0.002 * len(plot_stations), xlabel, ha='center', va='center'))
        if ylabel:
            extra_artists.append(
                f.text(0.04 * (len(comps) - 1) - 0.02, 0.5, ylabel, ha='center', va='center', rotation='vertical'))
        return plot_stations, comps, f, ax, extra_artists

    def plot_seismo_backend_2(self, outfile, plot_stations, comps, ax, yticks=True, extra_artists=None):
        """
        The second part of back-end for functions :func:`plot_seismo`, :func:`plot_covariance_function`, :func:`plot_noise`, :func:`plot_spectra`. There is no need for calling it directly.
        """
        xmin, xmax = ax[0, 0].get_xaxis().get_view_interval()
        ymin, ymax = ax[-1, 0].get_yaxis().get_view_interval()
        if yticks:
            for r in range(len(plot_stations)):
                ymin, ymax = ax[r, 0].get_yaxis().get_view_interval()
                ymax = np.round(ymax, int(-np.floor(np.log10(ymax))))  # round high axis limit to first valid digit
                ax[r, 0].add_artist(Line2D((xmin, xmin), (0, ymax), color='black', linewidth=1))
                ax[r, 0].yaxis.set_ticks((0., ymax))
        for c in range(len(comps)):
            ax[-1, c].add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
        if outfile:
            if extra_artists:
                plt.savefig(outfile, bbox_extra_artists=extra_artists, bbox_inches='tight')
            # plt.savefig(outfile, bbox_extra_artists=(legend,))
            else:
                plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        plt.close('all')

    def plot_seismo(self, file, comp_order='ZNE', cholesky=False, obs_style='k', obs_width=3,
                    synt_style='r', synt_width=2, add_file=None, add_file_style='k:', add_file_width=2, add_file2=None,
                    add_file2_style='b-', add_file2_width=2, plot_stations=None, plot_components=None, sharey=False):
        """
        Plots the fit between observed and simulated seismogram.

        :param outfile: path to file for plot output; if ``None`` plots to the screen
        :type outfile: string, optional
        :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
        :type comp_order: string, optional
        :param cholesky: plots standardized seismogram instead of original ones
        :type cholesky: bool, optional
        :param obs_style: line style for observed data
        :param obs_width: line width for observed data
        :param synt_style: line style for simulated data
        :param synt_width: line width for simulated data
        :param add_file: path to a reference file generated by function :func:`save_seismo`
        :type add_file: string or None, optional
        :param add_file_style: line style for reference data
        :param add_file_width: line width for reference data
        :param add_file2: path to second reference file
        :type add_file2: string or None, optional
        :param add_file2_style: line style for reference data
        :param add_file2_width: line width for reference data
        :param plot_stations: list of stations to plot; if ``None`` plots all stations
        :type plot_stations: list or None, optional
        :param plot_components: list of components to plot; if ``None`` plots all components
        :type plot_components: list or None, optional
        :param sharey: if ``True`` the y-axes for all stations have the same limits, otherwise the limits are chosen automatically for every station
        :type sharey: bool, optional
        """
        outfile = os.path.join(self.root_path,'output', file)
        file_path = os.path.join(self.root_path,'green','elemse')
        if cholesky and not len(self.LT) and not len(self.LT3):
            raise ValueError('Covariance matrix not set. Run "covariance_matrix()" first.')
        data = self.data_shifts[self.centroid['shift_idx']]
        npts = self.npts_slice
        samprate = self.samprate
        elemse = read_elemse(self.nr, self.npts_elemse, file_path + self.centroid['id'] + '.dat',
                             self.stations)  # nacist elemse
        # if not no_filter:
        for r in range(self.nr):
            for e in range(6):
                my_filter(elemse[r][e], self.stations[r]['fmin'], self.stations[r]['fmax'])
                elemse[r][e].trim(UTCDateTime(0) + self.elemse_start_origin)

        plot_stations, comps, f, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order,
                                                                     sharey=(cholesky or sharey),
                                                                     title_prefix=('', 'pseudo ')[
                                                                         cholesky and self.LT3 != []],
                                                                     ylabel=('velocity [m/s]', None)[cholesky])

        t = np.arange(0, (npts - 0.5) / samprate, 1. / samprate)
        if add_file:
            add = np.load(add_file)
        if add_file2:
            add2 = np.load(add_file2)
        d_max = 0
        for sta in plot_stations:
            r = plot_stations.index(sta)
            # if no_filter:
            # SAMPRATE = self.data_unfiltered[sta][0].stats.sampling_rate
            # NPTS = int(npts/samprate * SAMPRATE),
            # SHIFT = int(round(self.centroid['shift']*SAMPRATE))
            # T = np.arange(0, (NPTS-0.5) / SAMPRATE, 1. / SAMPRATE)
            SYNT = {}
            for comp in range(3):
                SYNT[comp] = np.zeros(npts)
                for e in range(6):
                    SYNT[comp] += elemse[sta][e][comp].data[0:npts] * self.centroid['a'][e, 0]
            comps_used = 0
            for comp in comps:
                synt = SYNT[comp]
                # if no_filter:
                # D = np.empty(NPTS)
                # for i in range(NPTS):
                # if i+SHIFT >= 0:
                # D[i] = self.data_unfiltered[sta][comp].data[i+SHIFT]
                # else:
                d = data[sta][comp][0:len(t)]
                if cholesky and self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    if self.LT3:
                        # print(r, comp) # DEBUG
                        d = np.zeros(npts)
                        synt = np.zeros(npts)
                        x1 = -npts
                        for COMP in range(3):
                            if not self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
                                continue
                            x1 += npts;
                            x2 = x1 + npts
                            y1 = comps_used * npts;
                            y2 = y1 + npts
                            # print(self.LT3[sta][y1:y2, x1:x2].shape, data[sta][COMP].data[0:npts].shape) # DEBUG
                            d += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
                            synt += np.dot(self.LT3[sta][y1:y2, x1:x2], SYNT[COMP])
                    else:
                        d = np.dot(self.LT[sta][comp], d)
                        synt = np.dot(self.LT[sta][comp], synt)
                    comps_used += 1
                c = comps.index(comp)
                # if no_filter:
                # ax[r,c].plot(T,D, color='k', linewidth=obs_width)
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[
                    comp]] or not cholesky:  # do not plot seismogram if the component is not used and Cholesky decomposition is plotted
                    l_d, = ax[r, c].plot(t, d, obs_style, linewidth=obs_width)
                    if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                        d_max = max(max(d), -min(d), d_max)
                else:
                    ax[r, c].plot([0], [0], 'w', linewidth=0)
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    l_s, = ax[r, c].plot(t, synt, synt_style, linewidth=synt_width)
                    d_max = max(max(synt), -min(synt), d_max)
                else:
                    if not cholesky:
                        ax[r, c].plot(t, synt, color='gray', linewidth=2)
                if add_file:
                    ax[r, c].plot(t, add[:, 3 * sta + comp], add_file_style, linewidth=add_file_width)
                if add_file2:
                    ax[r, c].plot(t, add2[:, 3 * sta + comp], add_file2_style, linewidth=add_file2_width)
        ax[-1, 0].set_ylim([-d_max, d_max])
        ea.append(f.legend((l_d, l_s), ('inverted data', 'modeled (synt)'), loc='lower center',
                           bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=2, numpoints=1,
                           fontsize='small', fancybox=True, handlelength=3))  # , borderaxespad=0.1
        ea.append(f.text(0.1, 1.06 - 0.004 * len(plot_stations), 'x', color='white', ha='center', va='center'))
        self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, extra_artists=ea)



    ## breack
    def plot_noise(self, comp_order='ZNE', obs_style='k', obs_width=2, plot_stations=None,
                   plot_components=None):
        """
        Plots the noise records from which the covariance matrix is calculated together with the inverted data

        :param outfile: path to file for plot output; if ``None`` plots to the screen
        :type outfile: string, optional
        :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
        :type comp_order: string, optional
        :param obs_style: line style
        :param obs_width: line width
        :param plot_stations: list of stations to plot; if ``None`` plots all stations
        :type plot_stations: list or None, optional
        :param plot_components: list of components to plot; if ``None`` plots all components
        :type plot_components: list or None, optional
        """
        outfile = os.path.join(self.root_path,'output/noise.png')

        samprate = self.samprate

        plot_stations, comps, f, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order)

        t = np.arange(0, (self.npts_slice - 0.5) / samprate, 1. / samprate)
        d_max = 0
        for sta in plot_stations:
            r = plot_stations.index(sta)
            for comp in comps:
                d = self.data_shifts[self.centroid['shift_idx']][sta][comp][0:len(t)]
                c = comps.index(comp)
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    color = obs_style
                    d_max = max(max(d), -min(d), d_max)
                else:
                    color = 'gray'
                ax[r, c].plot(t, d, color, linewidth=obs_width)
                if len(self.noise[sta]) > comp:
                    NPTS = len(self.noise[sta][comp].data)
                    T = np.arange(-NPTS * 1. / samprate, -0.5 / samprate, 1. / samprate)
                    ax[r, c].plot(T, self.noise[sta][comp], color, linewidth=obs_width)
                    if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                        d_max = max(max(self.noise[sta][comp]), -min(self.noise[sta][comp]), d_max)
        ax[-1, 0].set_ylim([-d_max, d_max])
        ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
        for r in range(len(plot_stations)):
            for i in range(len(comps)):
                l4 = ax[r, i].add_patch(mpatches.Rectangle((-NPTS / samprate, -ymax), NPTS / samprate, 2 * ymax,
                                                           color=(1.0, 0.6, 0.4)))  # (x,y), width, height
                l5 = ax[r, i].add_patch(
                    mpatches.Rectangle((0, -ymax), self.npts_slice / samprate, 2 * ymax, color=(0.7, 0.7, 0.7)))
        ea.append(f.legend((l4, l5), ('$C_D$', 'inverted'), 'lower center',
                           bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=2, fontsize='small',
                           fancybox=True, handlelength=3, handleheight=1.2))  # , borderaxespad=0.1
        ea.append(f.text(0.1, 1.06 - 0.004 * len(plot_stations), 'x', color='white', ha='center', va='center'))
        self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, extra_artists=ea)


    def plot_spectra(self, comp_order='ZNE', plot_stations=None, plot_components=None):
        """
        Plots spectra of inverted data, standardized data, and before-event noise together

        :param outfile: path to file for plot output; if ``None`` plots to the screen
        :type outfile: string, optional
        :param comp_order: order of component in the plot, supported are 'ZNE' (default) and 'NEZ'
        :type comp_order: string, optional
        :param plot_stations: list of stations to plot; if ``None`` plots all stations
        :type plot_stations: list or None, optional
        :param plot_components: list of components to plot; if ``None`` plots all components
        :type plot_components: list or None, optional
        """
        outfile = os.path.join(self.root_path,'output/spectra.png')

        if not len(self.LT) and not len(self.LT3):
            raise ValueError('Covariance matrix not set. Run "covariance_matrix()" first.')
        data = self.data_shifts[self.centroid['shift_idx']]
        npts = self.npts_slice
        samprate = self.samprate

        plot_stations, comps, fig, ax, ea = self.plot_seismo_backend_1(plot_stations, plot_components, comp_order,
                                                                       yticks=False, xlabel='frequency [Hz]',
                                                                       ylabel='amplitude spectrum')

        # plt.yscale('log')
        ax3 = np.empty_like(ax)
        fmin = np.zeros_like(ax, dtype=float)
        fmax = np.zeros_like(fmin)
        for i in range(len(plot_stations)):
            for j in range(len(comps)):
                # ax[i,j].set_yscale('log')
                ax3[i, j] = ax[i, j].twinx()
            # ax3[i,j].set_yscale('log')
        ax3[0, 0].get_shared_y_axes().join(*ax3.flatten().tolist())

        dt = 1. / samprate
        DT = 0.5 * dt
        f = np.arange(0, samprate * 0.5 * (1 - 0.5 / npts), samprate / npts)
        D_filt_max = 0
        for sta in plot_stations:
            r = plot_stations.index(sta)
            SYNT = {}
            comps_used = 0
            for comp in comps:
                d = data[sta][comp][0:npts]
                d_filt = d.copy()
                c = comps.index(comp)
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    if self.LT3:
                        d_filt = np.zeros(npts)
                        x1 = -npts
                        for COMP in comps:
                            if not self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[COMP]]:
                                continue
                            x1 += npts;
                            x2 = x1 + npts
                            y1 = comps_used * npts;
                            y2 = y1 + npts
                            d_filt += np.dot(self.LT3[sta][y1:y2, x1:x2], data[sta][COMP].data[0:npts])
                    else:
                        d_filt = np.dot(self.LT[sta][comp], d)
                    comps_used += 1
                    fmin[r, c] = self.stations[sta]['fmin']
                    fmax[r, c] = self.stations[sta]['fmax']
                ax[r, c].tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
                ax[r, c].yaxis.offsetText.set_visible(False)
                ax3[r, c].get_yaxis().set_visible(False)
                if self.stations[sta][{0: 'useZ', 1: 'useN', 2: 'useE'}[comp]]:
                    noise = self.noise[sta][comp]
                    NPTS = len(noise)
                    NOISE = np.sqrt(np.square(np.real(np.fft.fft(noise)) * DT) * npts * dt / (NPTS * DT))
                    f2 = np.arange(0, samprate * 1. * (1 - 0.5 / NPTS), samprate * 2 / NPTS)
                    D = np.absolute(np.real(np.fft.fft(d)) * dt)
                    D_filt = np.absolute(np.real(np.fft.fft(d_filt)) * dt)
                    D_filt_max = max(D_filt_max, max(D_filt))
                    l_d, = ax[r, c].plot(f, D[0:len(f)], 'k', linewidth=2, zorder=2)
                    l_filt, = ax3[r, c].plot(f, D_filt[0:len(f)], 'r', linewidth=1, zorder=3)
                    l_noise, = ax[r, c].plot(f2, NOISE[0:len(f2)], 'gray', linewidth=4, zorder=1)
                else:
                    ax[r, c].plot([0], [0], 'w', linewidth=0)
        # y3min, y3max = ax3[-1,0].get_yaxis().get_view_interval()
        ax3[-1, 0].set_ylim([0, D_filt_max])
        # print (D_filt_max, y3max, y3min)
        align_yaxis(ax[0, 0], ax3[0, 0])
        ax[0, 0].set_xlim(0, self.fmax * 1.5)
        # ax[0,0].set_xscale('log')
        # f.legend((l4, l5), ('$C_D$', 'inverted'), 'upper center', ncol=2, fontsize='small', fancybox=True)
        ea.append(
            fig.legend((l_d, l_filt, l_noise), ('data', 'standardized data (by $C_D$)', 'noise'), loc='lower center',
                       bbox_to_anchor=(0.5, 1. - 0.0066 * len(plot_stations)), ncol=3, numpoints=1, fontsize='small',
                       fancybox=True, handlelength=3))  # , borderaxespad=0.1
        ea.append(fig.text(0.1, 1.06 - 0.004 * len(plot_stations), 'x', color='white', ha='center', va='center'))
        ymin, ymax = ax[r, c].get_yaxis().get_view_interval()
        for r in range(len(plot_stations)):
            for c in range(len(comps)):
                if fmax[r, c]:
                    ax[r, c].add_artist(Line2D((fmin[r, c], fmin[r, c]), (0, ymax), color='g', linewidth=1))
                    ax[r, c].add_artist(Line2D((fmax[r, c], fmax[r, c]), (0, ymax), color='g', linewidth=1))
        self.plot_seismo_backend_2(outfile, plot_stations, comps, ax, yticks=False, extra_artists=ea)


    def plot_stations(self, network=True, location=False, channelcode=False, fontsize=0):

        """
        Plot a map of stations used in the inversion.

        :param outfile: path to file for plot output; if ``None`` plots to the screen
        :type outfile: string, optional
        :param network: include network code into station label
        :type network: bool, optional
        :param location: include location code into station label
        :type location: bool, optional
        :param channelcode: include channel code into station label
        :type channelcode: bool, optional
        :param fontsize: font size for all texts in the plot; if zero, the size is chosen automatically
        :type fontsize: scalar, optional


        The stations are marked according to components used in the inversion.
        """

        outfile = os.path.join(self.root_path,'output/stations.png')

        if fontsize:
            plt.rcParams.update({'font.size': fontsize})
        plt.figure(figsize=(16, 12))
        plt.axis('equal')
        plt.xlabel('west - east [km]')
        plt.ylabel('south - north [km]')
        plt.title('Stations used in the inversion')
        plt.plot(self.centroid['y'] / 1e3, self.centroid['x'] / 1e3, marker='*', markersize=75, color='yellow',
                 label='epicenter', linestyle='None')

        L1 = L2 = L3 = True
        for sta in self.stations:
            az = np.radians(sta['az'])
            dist = sta['dist'] / 1000  # from meter to kilometer
            y = np.cos(az) * dist  # N
            x = np.sin(az) * dist  # E
            label = None
            if sta['useN'] and sta['useE'] and sta['useZ']:
                color = 'red'
                if L1: label = 'all components used'; L1 = False
            elif not sta['useN'] and not sta['useE'] and not sta['useZ']:
                color = 'white'
                if L3: label = 'not used'; L3 = False
            else:
                color = 'gray'
                if L2: label = 'some components used'; L2 = False
            if network and sta['network']:
                l = sta['network'] + ':'
            else:
                l = ''
            l += sta['code']
            if location and sta['location']: l += ':' + sta['location']
            if channelcode: l += ' ' + sta['channelcode']
            # sta['weightN'] = sta['weightE'] = sta['weightZ']
            plt.plot([x], [y], marker='^', markersize=18, color=color, label=label, linestyle='None')
            plt.annotate(l, xy=(x, y), xycoords='data', xytext=(0, -14), textcoords='offset points',
                         horizontalalignment='center', verticalalignment='top', fontsize=14)
        # plt.legend(numpoints=1)
        plt.legend(bbox_to_anchor=(0., -0.15 - fontsize * 0.002, 1., .07), loc='lower left', ncol=4, numpoints=1,
                   mode='expand', fontsize='small', fancybox=True)
        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        plt.close()

    def plot_covariance_matrix(self, normalize=False, cholesky=False, fontsize=60, colorbar=False):
        """
        Plots figure of the data covariance matrix :math:`C_D`.

        :param outfile: path to file for plot output; if ``None`` plots to the screen
        :type outfile: string, optional
        :param normalize: normalize each blok (corresponding to one station) of the :math:`C_D` to the same value
        :type normalize: bool, optional
        :param cholesky: plots Cholesky decomposition of the covariance matrix :math:`L^T` instead of the :math:`C_D`
        :type cholesky: bool, optional
        :param fontsize: font size for all texts in the plot
        :type fontsize: scalar, optional
        :param colorbar: show a legend for the color map
        :type colorbar: bool, optional
        """
        outfile = os.path.join(self.root_path,'output/covariance_matrix.png')
        plt.figure(figsize=(55, 50))
        fig, ax = plt.subplots(1, 1)
        if fontsize:
            plt.rcParams.update({'font.size': fontsize})
        Cd = np.zeros((self.components * self.npts_slice, self.components * self.npts_slice))
        if not len(self.Cd):
            raise ValueError(
                'Covariance matrix not set or not saved. Run "covariance_matrix(save_non_inverted=True)" first.')
        i = 0
        if cholesky and self.LT3:
            matrix = self.LT3
        elif cholesky:
            matrix = [item for sublist in self.LT for item in sublist]
        else:
            matrix = self.Cd
        for C in matrix:
            if type(C) == int:
                continue
            if normalize and len(C):
                mx = max(C.max(), abs(C.min()))
                C *= 1. / mx
            l = len(C)
            Cd[i:i + l, i:i + l] = C
            i += l

        values = []
        labels = []
        i = 0
        n = self.npts_slice
        for stn in self.stations:
            if cholesky and self.LT3:
                j = stn['useZ'] + stn['useN'] + stn['useE']
                if j:
                    values.append(i * n + j * n / 2)
                    labels.append(stn['code'])
                    i += j
            else:
                if stn['useZ']:
                    values.append(i * n + n / 2)
                    labels.append(stn['code'] + ' ' + 'Z')
                    i += 1
                if stn['useN']:
                    values.append(i * n + n / 2)
                    labels.append(stn['code'] + ' ' + 'N')
                    i += 1
                if stn['useE']:
                    values.append(i * n + n / 2)
                    labels.append(stn['code'] + ' ' + 'E')
                    i += 1
        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        mx = max(Cd.max(), abs(Cd.min()))
        cax = plt.matshow(Cd, fignum=1, cmap=plt.get_cmap('seismic'), vmin=-mx, vmax=mx)
        # cb = plt.colorbar(shift, shrink=0.6, extend='both', label='shift [s]')

        if colorbar:
            cbar = plt.colorbar(cax, shrink=0.6, label='correlation [$\mathrm{m}^2\,\mathrm{s}^{-2}$]')
        # cbar = plt.colorbar(cax, ticks=[-mx, 0, mx])

        plt.xticks(values, labels, rotation='vertical')
        plt.yticks(values, labels)
        # Turn off all the ticks
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        if outfile:
            plt.savefig(outfile, bbox_inches='tight')
        else:
            plt.show()
        plt.clf()
        plt.close('all')

    def plot_3D(self):
        """
        Creates an animation with the grid of solutios. The grid points are labeled according to their variance reduction.

        :param outfile: path to file for saving animation
        :type outfile: string
        """
        outfile = os.path.join(self.root_path, 'output/animation.mp4')
        n = len(self.grid)
        x = np.zeros(n);
        y = np.zeros(n);
        z = np.zeros(n);
        VR = np.zeros(n)
        c = np.zeros((n, 3))
        for i in range(len(self.grid)):
            gp = self.grid[i]
            if gp['err']:
                continue
            x[i] = gp['y'] / 1e3
            y[i] = gp['x'] / 1e3  # NS is x coordinate, so switch it with y to be vertical
            z[i] = gp['z'] / 1e3
            vr = max(gp['VR'], 0)
            VR[i] = np.pi * (15 * vr) ** 2
            c[i] = np.array([vr, 0, 1 - vr])
        # if self.decompose:
        # dc = float(gp['dc_perc'])/100
        # c[i,:] = np.array([dc, 0, 1-dc])
        # else:
        # c[i,:] = np.array([0, 0, 0])
        # Create a figure and a 3D Axes
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('west - east [km]')
        ax.set_ylabel('south - north [km]')
        ax.set_zlabel('depth [km]')

        # Create an init function and the animate functions.
        # Both are explained in the tutorial. Since we are changing
        # the the elevation and azimuth and no objects are really
        # changed on the plot we don't have to return anything from
        # the init and animate function. (return value is explained
        # in the tutorial).
        def init():
            ax.scatter(x, y, z, marker='o', s=VR, c=c, alpha=1.)

        def animate(i):
            ax.view_init(elev=10., azim=i)

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)  # Animate
        anim.save(outfile, writer=self.movie_writer, fps=30)  # Save


    def html_log(self, reference=None, h1='ISOLA-ObsPy automated solution', backlink=False, plot_MT=None,
                 plot_uncertainty=None, plot_stations=None, plot_seismo_cova=None, plot_seismo_sharey=None,
                 mouse_figures=None, plot_spectra=None, plot_noise=None, plot_covariance_function=None,
                 plot_covariance_matrix=None, plot_maps=None, plot_slices=None, plot_maps_sum=None):

        outfile = os.path.join(self.root_path, 'output/index.html')
        out = open(outfile, 'w')
        e = self.event
        C = self.centroid
        decomp = self.mt_decomp.copy()
        out.write("""
<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="UTF-8">
  <title>2015-06-20 10:30:22 Sion earthquake solution &ndash; ISOLA-ObsPy</title>
  <link rel="stylesheet" href="../html/style.css" />
  <link rel="stylesheet" href="../html/css/lightbox.min.css">
</head>
<body>
""")
        out.write('<h1>'+h1+'</h1>\n')
        if backlink:
            out.write('<p><a href="../index.html">back to event list</a></p>\n')
        out.write('<dl>  <dt>Method</dt>\n  <dd>Waveform inversion for <strong>' +
			{1:'deviatoric part of', 0:'full'}[self.deviatoric] + 
			'</strong> moment tensor (' + 
			{1:'5', 0:'6'}[self.deviatoric] + 
			' components)<br />\n    ' + 
			{1:'with the <strong>data covariance matrix</strong> based on real noise', 0:'without the covariance matrix'}[bool(self.Cd_inv)] + 
			{1:'<br />\n    with <strong>crosscovariance</strong> between components', 0:''}[bool(self.LT3)] + 
			'.</dd>\n  <dt>Reference</dt>\n  <dd>Vackář, J., Burjánek, J., Gallovič, F., Zahradník, J., & Clinton, J. (2017). Bayesian ISOLA: New tool for automated centroid moment tensor inversion. Geophysical Journal International, 210(2), 693-705., <a href="https://academic.oup.com/gji/article/210/2/693/3747443?login=true">PDF</a></dd>\n</dl>\n\n')
        out.write('''
<h2>Hypocenter location</h2>

<dl>
<dt>Agency</dt>
<dd>{agency:s}</dd>
<dt>Origin time</dt>
<dd>{t:s}</dd>
<dt>Latitude</dt>
<dd>{lat:8.3f}° N</dd>
<dt>Longitude</dt>
<dd>{lon:8.3f}° E</dd>
<dt>Depth</dt>
<dd>{d:3.1f} km</dd>
<dt>Magnitude</dt>
<dd>{m:3.1f}</dd>
</dl>
'''.format(t=e['t'].strftime('%Y-%m-%d %H:%M:%S'), lat=float(e['lat']), lon=float(e['lon']), d=e['depth']/1e3, agency=e['agency'], m=e['mag']))
        out.write('\n\n<h2>Results</h2>\n\n')
        if plot_MT:
            out.write('''
<div class="thumb tright">
  <a href="{0:s}" data-lightbox="MT" data-title="moment tensor best solution">
    <img alt="MT" src="{0:s}" width="199" height="200" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    moment tensor best solution
  </div>
</div>
'''.format(plot_MT))
        if plot_uncertainty:
            k = plot_uncertainty.rfind(".")
            s1 = plot_uncertainty[:k]+'_'; s2 = plot_uncertainty[k:]
            out.write('''
<div class="thumb tright">
  <a href="{MT_full:s}" data-lightbox="MT" data-title="moment tensor uncertainty">
    <img alt="MT" src="{MT_full:s}" width="199" height="200" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    moment tensor uncertainty
  </div>
</div>

<div class="thumb tright">
  <a href="{MT_DC:s}" data-lightbox="MT" data-title="moment tensor DC-part uncertainty">
    <img alt="MT" src="{MT_DC:s}" width="199" height="200" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    DC-part uncertainty
  </div>
</div>
'''.format(MT_full=s1+'MT'+s2, MT_DC=s1+'MT_DC'+s2))
        t = self.event['t'] + C['shift']
        out.write('''
<h3>Centroid location</h3>

<table>
    <tr>
      <th></th>
      <th>absolute</th>
      <th>relative</th>
    </tr>
    <tr>
      <th>Time</th>
      <td>{t:s}</td>
      <td>{shift:5.2f} s {sgn_shift:s} origin time</td>
    </tr>
    <tr>
      <th>Latitude</th>
      <td>{lat:8.3f}° {sgn_lat:s}</td>
      <td>{x:5.0f} m {dir_x:s} of the epicenter</td>
    </tr>
    <tr>
      <th>Longitude</th>
      <td>{lon:8.3f}° {sgn_lon:s}</td>
      <td>{y:5.0f} m {dir_y:s} of the epicenter</td>
    </tr>
    <tr>
      <th>Depth</th>
      <td>{d:5.1f} km</td>
      <td>{dd:5.1f} km {sgn_dd:s} than location</td>
    </tr>
</table>

'''.format(
    t = 	t.strftime('%Y-%m-%d %H:%M:%S'),
    lat = 	abs(C['lat']),
    sgn_lat = {1:'N', 0:'', -1:'S'}[int(np.sign(C['lat']))],
    lon = 	abs(C['lon']),
    sgn_lon = {1:'E', 0:'', -1:'W'}[int(np.sign(C['lon']))],
    d = 	C['z']/1e3,
    x = 	abs(C['x']),
    dir_x = 	{1:'north', 0:'', -1:'south'}[int(np.sign(C['x']))],
    y = 	abs(C['y']),
    dir_y = 	{1:'east', 0:'', -1:'west'}[int(np.sign(C['y']))],
    shift = 	abs(C['shift']),
    sgn_shift={1:'after', 0:'after', -1:'before'}[int(np.sign(C['shift']))],
    dd = 	abs(C['z']-self.event['depth'])/1e3,
    sgn_dd = 	{1:'deeper', 0:'deeper', -1:'shallower'}[int(np.sign(C['z']-self.event['depth']))]
))
        if C['edge']:
            out.write('<p class="warning">Warning: the solution lies on the edge of the grid!</p>')
        if C['shift'] in (self.shifts[0], self.shifts[-1]):
            out.write('<p class="warning">Warning: the solution lies on the edge of the time-grid!</p>')

        mt2 = a2mt(C['a'], system='USE')
        c = max(abs(min(mt2)), max(mt2))
        c = 10**np.floor(np.log10(c))
        MT2 = mt2 / c

        out.write('\n\n<h3>Moment tensor and its quality</h3>\n\n')
        if self.mt_decomp and reference:
            decomp.update(rename_keys(reference, 'ref_'))
            out.write('''
<table>
  <tr><th>&nbsp;</th><th>ISOLA-ObsPy</th><th>SeisComP</th></tr>
  <tr><th colspan="3" class="center">Centroid position</th></tr>
  <tr><th>depth</th>	<td>{depth:3.1f} km</td>	<td>{ref_depth:3.1f} km</td></tr>
  <tr><th colspan="3" class="center">Seismic moment</th></tr>
  <tr><th>scalar seismic moment M<sub>0</sub></th>	<td>{mom:5.2e} Nm</td>	<td></td></tr>
  <tr><th>moment magnitude M<sub>w</sub></th>	<td>{Mw:3.1f}</td>	<td>{ref_Mw:3.1f}</td></tr>
  <tr><th colspan="3" class="center">Moment tensor components</th></tr>
  <tr><th>M<sub>rr</sub></th>			<td>{1:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>&theta;&theta;</sub></th>	<td>{2:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>&#981;&#981;</sub></th>		<td>{3:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>r&theta;</sub></th>		<td>{4:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>r&#981;</sub></th>		<td>{5:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th>M<sub>&theta;&#981;</sub></th>	<td>{6:5.2f} * {0:5.0e}</td>	<td>&nbsp;</td></tr>
  <tr><th colspan="3" class="center">Moment tensor decomposition</th></tr>
  <tr><th>DC component</th>	<td>{dc_perc:3.0f} %</td>	<td>{ref_dc_perc:3.0f} %</td></tr>
  <tr><th>CLVD component</th>	<td>{clvd_perc:3.0f} %</td>	<td>{ref_clvd_perc:3.0f} %</td></tr>
'''.format(c, *mt2, depth=C['z']/1e3, **decomp))
            if not self.deviatoric:
                out.write('''
  <tr><th>isotropic component</th>	<td>{iso_perc:3.0f} %</td>	<td>{ref_iso_perc:3.0f} %</td></tr>
'''.format(**decomp))
            out.write('''
  <tr><th>strike</th>	<td>{s1:3.0f} / {s2:3.0f}</td>	<td>{ref_s1:3.0f} / {ref_s2:3.0f}</td></tr>
  <tr><th>dip</th>  	<td>{d1:3.0f} / {d2:3.0f}</td>	<td>{ref_d1:3.0f} / {ref_d2:3.0f}</td></tr>
  <tr><th>slip-rake</th>	<td>{r1:3.0f} / {r2:3.0f}</td>	<td>{ref_r1:3.0f} / {ref_r2:3.0f}</td></tr>
  <tr><th colspan="3" class="center">Result quality</th></tr>
  <tr><th>condition number</th>	<td>{CN:2.0f}</td>	<td></td></tr>
  <tr><th>variance reduction</th>	<td>{VR:2.0f} %</td>	<td></td></tr>
'''.format(VR=C['VR']*100, CN=C['CN'], **decomp))
        elif self.mt_decomp:
            out.write('''
<table>
  <tr><th colspan="2" class="center">Centroid position</th></tr>
  <tr><th>depth</th>	<td>{depth:3.1f} km</td></tr>
  <tr><th colspan="2" class="center">Seismic moment</th></tr>
  <tr><th>scalar seismic moment M<sub>0</sub></th>	<td>{mom:5.2e} Nm</td></tr>
  <tr><th>moment magnitude M<sub>w</sub></th>	<td>{Mw:3.1f}</td></tr>
  <tr><th colspan="2" class="center">Moment tensor components</th></tr>
  <tr><th>M<sub>rr</sub></th>			<td>{1:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&theta;</sub></th>	<td>{2:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&#981;&#981;</sub></th>		<td>{3:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&theta;</sub></th>		<td>{4:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&#981;</sub></th>		<td>{5:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&#981;</sub></th>	<td>{6:5.2f} * {0:5.0e}</td></tr>
  <tr><th colspan="2" class="center">Moment tensor decomposition</th></tr>
  <tr><th>DC</th>	<td>{dc_perc:3.0f} %</td></tr>
  <tr><th>CLVD</th>	<td>{clvd_perc:3.0f} %</td></tr>
'''.format(c, *mt2, depth=C['z']/1e3, **decomp))
            if not self.deviatoric:
                out.write('''
  <tr><th>ISO</th>	<td>{iso_perc:3.0f} %</td></tr>
'''.format(**decomp))
            out.write('''
  <tr><th>strike</th>	<td>{s1:3.0f} / {s2:3.0f}</td></tr>
  <tr><th>dip</th>  	<td>{d1:3.0f} / {d2:3.0f}</td></tr>
  <tr><th>rake</th>	<td>{r1:3.0f} / {r2:3.0f}</td></tr>
  <tr><th colspan="2" class="center">Quality measures</th></tr>
  <tr><th>condition number</th>	<td>{CN:2.0f}</td></tr>
  <tr><th>variance reduction</th>	<td>{VR:2.0f} %</td></tr>
'''.format(VR=C['VR']*100, CN=C['CN'], **decomp))
        else:
            out.write('''
<table>
  <tr><th colspan="2" class="center">Centroid position</th></tr>
  <tr><th>depth</th>	<td>{depth:3.1f} km</td></tr>
  <tr><th colspan="2" class="center">Moment tensor components</th></tr>
  <tr><th>M<sub>rr</sub></th>			<td>{1:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&theta;</sub></th>	<td>{2:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&#981;&#981;</sub></th>		<td>{3:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&theta;</sub></th>		<td>{4:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>r&#981;</sub></th>		<td>{5:5.2f} * {0:5.0e}</td></tr>
  <tr><th>M<sub>&theta;&#981;</sub></th>	<td>{6:5.2f} * {0:5.0e}</td></tr>
  <tr><th colspan="2" class="center">Result quality</th></tr>
  <tr><th>condition number</th>	<td>{CN:2.0f}</td></tr>
  <tr><th>variance reduction</th>	<td>{VR:2.0f} %</td></tr>
'''.format(c, *mt2, depth=C['z']/1e3, VR=C['VR']*100, CN=C['CN']))
        if self.max_VR:
            out.write('  <tr><th>VR ({2:d} closest components)</th>	<td>{1:2.0f} %</td>{0:s}</tr>'.format(('', '<td></td>')[bool(reference)], self.max_VR[0]*100, self.max_VR[1]))
        if reference and 'kagan' in reference:
            out.write('<tr><th>Kagan angle</th>	<td colspan="2" class="center">{0:3.1f}°</td></tr>\n'.format(reference['kagan']))
        out.write('</table>\n\n')
			
        if plot_uncertainty:
            out.write('''
<h3>Histograms&mdash;uncertainty of MT parameters</h3>

<div class="thumb tleft">
  <a href="{DC:s}" data-lightbox="histogram" data-title="DC-part uncertainty">
    <img alt="" src="{DC:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    DC-part
  </div>
</div>

<div class="thumb tleft">
  <a href="{CLVD:s}" data-lightbox="histogram" data-title="CLVD-part uncertainty">
    <img alt="" src="{CLVD:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    CLVD-part
  </div>
</div>
'''.format(DC=s1+'comp-1-DC'+s2, CLVD=s1+'comp-2-CLVD'+s2))
            if not self.deviatoric:
                out.write('''
<div class="thumb tleft">
  <a href="{ISO:s}" data-lightbox="histogram" data-title="isotropic part uncertainty">
    <img alt="" src="{ISO:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    CLVD-part
  </div>
</div>
'''.format(ISO=s1+'comp-3-ISO'+s2))
            out.write('''
<div class="thumb tleft">
  <a href="{Mw:s}" data-lightbox="histogram" data-title="moment magnitude uncertainty">
    <img alt="" src="{Mw:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    moment magnitude
  </div>
</div>

'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW'+s2, NS=s1+'place-NS'+s2, time=s1+'time-shift'+s2))
            if len(self.shifts) > 1 or len(self.grid) > 1:
                out.write('\n\n<h3>Histograms&mdash;uncertainty of centroid position and time</h3>\n\n')
            if len (self.depths) > 1:
                out.write('''
<div class="thumb tleft">
  <a href="{depth:s}" data-lightbox="histogram" data-title="centroid depth uncertainty">
    <img alt="" src="{depth:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    centroid depth
  </div>
</div>

'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW.png'+s2, NS=s1+'place-NS.png'+s2, time=s1+'time-shift'+s2))
            if len(self.grid) > len(self.depths):
                out.write('''
<div class="thumb tleft">
  <a href="{EW:s}" data-lightbox="histogram" data-title="centroid position east-west uncertainty">
    <img alt="" src="{EW:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    position east-west
  </div>
</div>

<div class="thumb tleft">
  <a href="{NS:s}" data-lightbox="histogram" data-title="centroid position north-south uncertainty">
    <img alt="" src="{NS:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    position north-south
  </div>
</div>

'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW'+s2, NS=s1+'place-NS'+s2, time=s1+'time-shift'+s2))
            if len(self.shifts) > 1:
                out.write('''
<div class="thumb tleft">
  <a href="{time:s}" data-lightbox="histogram" data-title="centroid time uncertainty">
    <img alt="" src="{time:s}" height="80" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    centroid time
  </div>
</div>
'''.format(Mw=s1+'mech-0-Mw'+s2, depth=s1+'place-depth'+s2, EW=s1+'place-EW'+s2, NS=s1+'place-NS'+s2, time=s1+'time-shift'+s2))
        out.write('\n\n<h2>Data used</h2>\n\n')
        if plot_stations:
            out.write('''
<div class="thumb tright">
  <a href="{0:s}" data-lightbox="stations">
    <img alt="" src="{0:s}" width="200" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    stations used
  </div>
</div>
'''.format(plot_stations))
        if 'components' in self.logtext:
            s = self.logtext['components']
            i = s.find('(Hz)\t(Hz)\n')
            s = s[i+10:]
            out.write('\n\n<h3>Components used in inversion and their weights</h3>\n\n<table>\n  <tr><th colspan="2">station</th>	<th colspan="3">component</th>		<th><abbr title="epicentral distance">distance *</abbr></th>	<th>azimuth</th>	<th>fmin</th>	<th>fmax</th></tr>\n  <tr><th>code</th>	<th>channel</th>	<th>Z</th>	<th>N</th>	<th>E</th>	<th>(km)</th>	<th>(deg)</th>	<th>(Hz)</th>	<th>(Hz)</th></tr>\n')
            s = s.replace('\t', '</td>\t<td>').replace('\n', '</td></tr>\n<tr><td>')[:-8]
            s = '<tr><td>' + s + '</table>\n\n'
            out.write(s)
        if 'mouse' in self.logtext:
            out.write('<h3>Mouse detection</h3>\n<p>\n')
            s = self.logtext['mouse']
            lines = s.split('\n')
            if mouse_figures:
                p = re.compile('  ([0-9A-Z]+) +([A-Z]{2})([ZNE]{1}).* (MOUSE detected.*)')
            for line in lines:
                if mouse_figures:
                    m = p.match(line)
                    if m:
                        line = '  <a href="{fig:s}mouse_YES_{0:s}{comp:s}.png" data-lightbox="mouse">\n    {0:s} {1:s}{2:s}</a>: {3:s}'.format(*m.groups(), fig=mouse_figures, comp={'Z':'0', 'N':'1', 'E':'2'}[m.groups()[2]])
                out.write(line+'<br />\n')
        out.write('<h3>Data source</h3>\n<p>\n')
        if 'network' in self.logtext:
            out.write(self.logtext['network'] + '<br />\n')
        if 'data' in self.logtext:
            out.write(self.logtext['data'] + '\n')
        out.write('</p>\n\n')
        if plot_seismo_cova:
            out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="seismo" data-title="waveform fit: filtered by C<sub>D</sub>">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    waveform fit <br />(filtered)
  </div>
</div>
'''.format(plot_seismo_cova))
        if plot_seismo_sharey:
            out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="seismo" data-title="waveform fit: original data">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    waveform fit <br />(non-filtered)
  </div>
</div>
'''.format(plot_seismo_sharey))
        if plot_spectra:
            out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="spectra" data-title="waveform spectra">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    waveform spectra<br />&nbsp;
  </div>
</div>
'''.format(plot_spectra))
        if plot_noise:
            out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="noise" data-title="before-event noise">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    before-event noise<br />&nbsp;
  </div>
</div>
'''.format(plot_noise))
        if plot_covariance_function:
            out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="cova_func" data-title="data covariance function">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    data covariance function<br />&nbsp;
  </div>
</div>
'''.format(plot_covariance_function))
        if plot_covariance_matrix:
            out.write('''
<div class="thumb tleft">
  <a href="{0:s}" data-lightbox="Cd" data-title="data covariance matrix">
    <img alt="" src="{0:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    data covariance matrix<br />&nbsp;
  </div>
</div>
'''.format(plot_covariance_matrix))
        if plot_maps or plot_slices or plot_maps_sum:
            out.write('\n\n\n<h2>Stability and uncertainty of the solution</h2>')
        if plot_maps_sum:
            out.write('\n\n<h3>Posterior probability density function (PPD)</h3>\n\n')
            k = plot_maps_sum.rfind(".")
            s1 = plot_maps_sum[:k] + '_'
            s2 = plot_maps_sum[k:]
            out.write('''
<div class="thumb tleft">
  <a href="{top:s}" data-lightbox="PPD">
    <img alt="" src="{top:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    PPD: top view
  </div>
</div>

<div class="thumb tleft">
  <a href="{NS:s}" data-lightbox="PPD">
    <img alt="" src="{NS:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    PPD: north-south view
  </div>
</div>

<div class="thumb tleft">
  <a href="{WE:s}" data-lightbox="PPD">
    <img alt="" src="{WE:s}" height="150" class="thumbimage" />
  </a>
  <div class="thumbcaption">
    PPD: west-east view
  </div>
</div>
'''.format(top=s1+'top'+s2, NS=s1+'N-S'+s2, WE=s1+'W-E'+s2))
        if plot_maps:
            out.write('\n\n<h3>Stability in space (top view)</h3>\n\n<div class="thumb tleft">\n')
            k = plot_maps.rfind(".")
            for z in self.depths:
                filename = plot_maps[:k] + "_{0:0>5.0f}".format(z) + plot_maps[k:]
                out.write('  <a href="{0:s}" data-lightbox="map">\n    <img alt="" src="{0:s}" height="100" class="thumbimage" />\n  </a>\n'.format(filename))
            out.write('  <div class="thumbcaption">\n    click to compare different depths\n  </div>\n</div>\n')
        if plot_slices:
            k = plot_slices.rfind(".")
            s1 = plot_slices[:k] + '_'
            s2 = plot_slices[k:]
            out.write('\n\n<h3>Stability in space (side view)</h3>\n\n<div class="thumb tleft">\n')
            for slice in ('N-S', 'W-E', 'NW-SE', 'SW-NE'):
                out.write('  <a href="{0:s}" data-lightbox="slice">\n    <img alt="" src="{0:s}" height="150" class="thumbimage" />\n  </a>\n'.format(s1+slice+s2))
        out.write('  <div class="thumbcaption">\n    click to compare different points of view\n  </div>\n</div>\n')
        out.write('''

<h2>Calculation parameters</h2>

<h3>Grid-search over space</h3>
<dl>
  <dt>number of points</dt>
  <dd>{points:4d}</dd>
  <dt>horizontal step</dt>
  <dd>{x:5.0f} m</dd>
  <dt>vertical step</dt>
  <dd>{z:5.0f} m</dd>
  <dt>grid radius</dt>
  <dd>{radius:6.3f} km</dd>
  <dt>minimal depth</dt>
  <dd>{dmin:6.3f} km</dd>
  <dt>maximal depth</dt>
  <dd>{dmax:6.3f} km</dd>
</dl>

<h3>Grid-search over time</h3>
<dl>
  <dt>min</dt>
  <dd>{sn:5.2f} s ({Sn:3d} samples)</dd>
  <dt>max</dt>
  <dd>{sx:5.2f} s ({Sx:3d} samples)</dd>
  <dt>step</dt>
  <dd>{step:4.2f} s ({STEP:3d} samples)</dd>
</dl>

<h3>Green's function calculation</h3>
<dl>
  <dt>Crustal model</dt>
  <dd>{crust:s}</dd>
  <dt>npts</dt>
  <dd>{npts_elemse:4d}</dd>
  <dt>tl</dt>
  <dd>{tl:4.2f}</dd>
  <dt>freq</dt>
  <dd>{freq:4d}</dd>
  <dt>npts for inversion</dt>
  <dd>{npts_slice:4d}</dd>
</dl>

<h3>Sampling frequencies</h3>
<dl>
  <dt>Data sampling</dt>
  <dd>{samplings:s}</dd>
  <dt>Common sampling</dt>
  <dd>{SAMPRATE:5.1f} Hz</dd>
  <dt>Decimate factor</dt>
  <dd>{decimate:3.0f} x</dd>
  <dt>Sampling used</dt>
  <dd>{samprate:5.1f} Hz</dd>
</dl>
'''.format(
    points = len(self.grid),
    x = self.step_x,
    z = self.step_z,
    radius = self.radius/1e3,
    dmin = self.depth_min/1e3,
    dmax = self.depth_max/1e3,
    sn = self.shift_min,
    Sn = self.SHIFT_min,
    sx = self.shift_max,
    Sx = self.SHIFT_max,
    step = self.shift_step,
    STEP = self.SHIFT_step,
    npts_elemse = self.npts_elemse,
    tl = self.tl,
    freq = self.freq,
    npts_slice = self.npts_slice,
    samplings = self.logtext['samplings'],
    decimate = self.max_samprate / self.samprate,
    samprate = self.samprate,
    SAMPRATE = self.max_samprate,
    crust = self.logtext['crust']
))


        out.write("""
<script src="../html/js/lightbox-plus-jquery.min.js"></script>
<script>
lightbox.option({
'resizeDuration': 0
})
</script>
</body>
</html>
""")
        out.close()