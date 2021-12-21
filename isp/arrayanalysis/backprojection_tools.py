from obspy.io.mseed.core import _is_mseed
from scipy import signal
from obspy import Stream, read_inventory, read
from obspy.taup import TauPyModel
from obspy.geodetics.base import locations2degrees
import numpy as np
import os

class backproj:

    def griding(self, coords, sta_coords, dx, dy, depth):

        lon1 = sta_coords['longitude']
        lat1 = sta_coords['latitude']
        xmin = coords[0]
        xmax = coords[1]
        ymin = coords[2]
        ymax = coords[3]

        columns = int((xmax - xmin) /dx)
        rows = int((ymax - ymin) / dy)

        travel_times = np.zeros((rows, columns))
        columns_line = np.linspace(xmin, xmax, columns)
        rows_line = np.linspace(ymin, ymax, rows)

        model = TauPyModel(model="iasp91")

        for i in range(rows):
            for j in range(columns):
                lon2 = columns_line[j]
                lat2 = rows_line[i]

                #distance, az1, az2 = gps2dist_azimuth(lat1, lon1, lat2, lon2, a=6378137.0, f=0.0033528106647474805)
                #distance = kilometers2degrees(distance/1000, radius=6378.137)

                #arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance,
                #                                  phase_list = ["P", "p", "pP", "PP"])
                arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=locations2degrees(lat1, lon1, lat2, lon2),
                                                  phase_list = ["P"])

                arrival = [(tt.time, tt.name) for tt in arrivals]
                #arrivals, phases = zip(*arrival)
                travel_times[i, j] = arrival[0][0]

        #travel_times = np.flipud(travel_times)
        return travel_times

    @classmethod
    def run_back(cls, st, map, dt, ddt, window, multichannel = True, **kwargs):

        stack_process = kwargs.pop('stack')

        coords = map['area_coords']
        dx = map['dx']
        dy = map['dy']
        depth = map['depth']
        xmin = coords[0]
        xmax = coords[1]
        ymin = coords[2]
        ymax = coords[3]
        start = np.max([tr.stats.starttime for tr in st])
        columns = int((xmax - xmin) / dx)

        rows = int((ymax - ymin) / dy)

        n = len(range(0, ddt, dt))
        power_matrix = np.zeros((rows, columns, n))

        for t in range(0, ddt, dt):
            print(t)

            for i in range(rows):

                for j in range(columns):

                    traces = []
                    for tr in st:

                        tr_process = tr.copy()
                        net = tr_process.stats.network
                        sta = tr_process.stats.station
                        chn = tr_process.stats.channel
                        tt = map["nets"][net][sta][chn][0]
                        delay = tt[i,j]
                        begin = start + delay + t
                        end = start + delay + t + window
                        tr_process.trim(starttime = begin, endtime = end)
                        traces.append(tr_process)

                    st_new = Stream(traces)
                    st_process = st_new.copy()
                    st_process.detrend(type='simple')
                    st_process.normalize()

                    # Multichannel
                    if multichannel:
                       _ , times = cls.multichanel(st_process)
                       for corr_ind in range(len(st_process)):
                           st_process[corr_ind].data = np.roll(st_process[corr_ind].data, int(times[corr_ind] * st_process[corr_ind].stats.sampling_rate))

                    if stack_process == "Stack":

                        stack =  cls.stack_seismograms(st_process)
                        pow = np.mean(np.sqrt(stack ** 2))
                        power_matrix[i, j, t] = pow

                    else:

                        power_matrix[i, j, t] = cls.zlcc(st_process)

            # clean memory
                del tr_process
                del st_process
                #del st_process_multichannel
                #del stack
                del st_new

            #k = k + 1

        return power_matrix


    def stack_seismograms(self, st):

        mat = np.zeros([len(st), len(st[0].data)])
        N = len(st)

        for i in range(N):
            mat[i, :] = st[i].data
            #plt.plot(mat[i, :])

        stack = np.mean(mat, axis = 0)

        #plt.plot(stack, color = "black")
        #plt.show()
        return stack

    def get_lags(self, cc):
        """
        Return array with lags
        :param cc: Cross-correlation returned by correlate_maxlag.
        :return: lags
        """
        mid = (len(cc) - 1) / 2
        if len(cc) % 2 == 1:
            mid = int(mid)
        return np.arange(len(cc)) - mid

    def _pad_zeros(self, a, num, num2=None):
        """Pad num zeros at both sides of array a"""
        if num2 is None:
            num2 = num
        hstack = [np.zeros(num, dtype=a.dtype), a, np.zeros(num2, dtype=a.dtype)]
        return np.hstack(hstack)

    def _xcorr_padzeros(self, a, b, shift, method):
        """
        Cross-correlation using SciPy with mode='valid' and precedent zero padding
        """
        if shift is None:
            shift = (len(a) + len(b) - 1) // 2
        dif = len(a) - len(b) - 2 * shift
        if dif > 0:
            b = self._pad_zeros(b, dif // 2)
        else:
            a = self._pad_zeros(a, -dif // 2)
        return signal.correlate(a, b, 'valid', method)


    def _xcorr_slice(self, a, b, shift, method):
        """
        Cross-correlation using SciPy with mode='full' and subsequent slicing
        """
        mid = (len(a) + len(b) - 1) // 2
        if shift is None:
            shift = mid
        if shift > mid:
            # Such a large shift is not possible without zero padding
            return self._xcorr_padzeros(a, b, shift, method)
        cc = signal.correlate(a, b, 'full', method)
        return cc[mid - shift:mid + shift + len(cc) % 2]

    def correlate_maxlag(self, a, b, maxlag, demean=True, normalize='naive',
                         method='auto'):
        """
        Cross-correlation of two signals up to a specified maximal lag.
        This function only allows 'naive' normalization with the overall
        standard deviations. This is a reasonable approximation for signals of
        similar length and a relatively small maxlag parameter.
        :func:`correlate_template` provides correct normalization.
        :param a,b: signals to correlate
        :param int maxlag: Number of samples to shift for cross correlation.
            The cross-correlation will consist of ``2*maxlag+1`` or
            ``2*maxlag`` samples. The sample with zero shift will be in the middle.
        :param bool demean: Demean data beforehand.
        :param normalize: Method for normalization of cross-correlation.
            One of ``'naive'`` or ``None``
            ``'naive'`` normalizes by the overall standard deviation.
            ``None`` does not normalize.
        :param method: correlation method to use.
            See :func:`scipy.signal.correlate`.
        :return: cross-correlation function.
        """
        a = np.asarray(a)
        b = np.asarray(b)
        if demean:
            a = a - np.mean(a)
            b = b - np.mean(b)
        # choose the usually faster xcorr function for each method
        _xcorr = self._xcorr_padzeros if method == 'direct' else self._xcorr_slice
        cc = _xcorr(a, b, maxlag, method)
        if normalize == 'naive':
            norm = (np.sum(a ** 2) * np.sum(b ** 2)) ** 0.5
            if norm <= np.finfo(float).eps:
                # norm is zero
                # => cross-correlation function will have only zeros
                cc[:] = 0
            elif cc.dtype == float:
                cc /= norm
            else:
                cc = cc / norm
        elif normalize is not None:
            raise ValueError("normalize has to be one of (None, 'naive'))")
        return cc

    def zlcc(self, st):
        n = len(st)
        nn = int(0.5*n*(n-1))
        cc = np.zeros(nn)
        k = 0
        for i in range(n):
            for j in range(n):
                if j>i:
                    n1 = np.sqrt(np.sum(st[i].data**2))
                    n2 = np.sqrt(np.sum(st[j].data**2))
                    den = n1*n2
                    cc[k] = (np.sum((st[i].data*st[j].data)))/den

                    k = k + 1
        cc = np.mean(cc)

        return cc

    def multichanel(self, st):
        n = len(st)
        rows = int(0.5*n*(n-1)+1)
        columns = n
        m = np.zeros((rows,columns))
        ccs = np.zeros((rows,1))
        k = 0
        for i in range(0, n, 1):
            for j in range(0, n, 1):

                if i < j:

                    m[k,i] = 1
                    m[k, j] = -1

                    cc = self.correlate_maxlag(st[i], st[j], maxlag=max([len(st[i].data), len(st[i].data)]))
                    values = [np.max(cc),np.min(cc)]
                    values = np.abs(values)

                    if values[0]>values[1]:
                         maximo = np.where(cc == np.max(cc))
                    else:
                         maximo = np.where(cc == np.min(cc))

                    lag_time = ((maximo[0][0])/50)-0.5*(len(cc)/50)

                    ccs[k] = lag_time

                    k = k + 1
        m[rows-1,:] = np.ones(columns)
        times = -1*np.matmul(np.linalg.pinv(m),ccs)
        for i in range(len(st)):

            st[i].stats.starttime = st[i].stats.starttime+times[i][0]

        return st, times


class back_proj_organize:

    def __init__(self, data_path, metadata, area_coords, dx, dy, depth):
        #coords = [xmin, xmax, ymin, ymax]
        self.depth = depth
        self.dy = dy
        self.dx = dx
        self.area_coords = area_coords
        self.data_path = data_path
        self.inventory = metadata

    @classmethod
    def list_directory(self, data_path):
        obsfiles = []
        for top_dir, sub_dir, files in os.walk(data_path):
            for file in files:
                obsfiles.append(os.path.join(top_dir, file))
        obsfiles.sort()
        return obsfiles

    def create_dict(self, **kwargs):

        net_list = kwargs.pop('net_list', "").split(',')
        sta_list = kwargs.pop('sta_list', "").split(',')
        chn_list = kwargs.pop('chn_list', "").split(',')

        obsfiles = self.list_directory(self.data_path)

        data_map = {}

        data_map['area_coords'] = self.area_coords
        data_map['dx'] = self.dx
        data_map['dy'] = self.dy
        data_map['depth'] = self.depth

        data_map['nets'] = {}
        size = 0

        for paths in obsfiles:

            if _is_mseed(paths):

                header = read(paths, headlonly=True)
                net = header[0].stats.network
                network = {net: {}}
                sta = header[0].stats.station
                stations = {sta: {}}
                chn = header[0].stats.channel
                meta = [net, sta, chn]
                id = header[0].id
                print(id)
                ## Filter per nets
                # 1. Check if the net exists, else add
                if net not in data_map['nets']:
                    # 1.1 Check the filter per network

                    if net in net_list:
                        data_map['nets'].update(network)
                    # 1.2 the filter per network is not activated
                    if net_list[0] == "":
                        data_map['nets'].update(network)

                # 2. Check if the station exists, else add
                try:
                    if sta not in data_map['nets'][net]:
                        if sta in sta_list:
                            data_map['nets'][net].update(stations)
                        if sta_list[0] == "":
                            data_map['nets'][net].update(stations)
                except:
                    pass

                # 3. Check if the channels exists, else add

                # 3.1 if already exists just add
                if chn in data_map['nets'][net][sta]:
                    if chn in chn_list:
                        sta_coords = self.inventory.get_coordinates(id)
                        time_mat = self.griding(self.area_coords, sta_coords, self.dx, self.dy, self.depth)
                        data_map['nets'][net][sta][chn].append(time_mat)

                        size = size + 1

                    if chn_list[0] == "":
                        sta_coords = self.inventory.get_coordinates(id)
                        time_mat = self.griding(self.area_coords, sta_coords, self.dx, self.dy, self.depth)
                        data_map['nets'][net][sta][chn].append(time_mat)

                        size = size + 1
                else:
                    # 3.2 if does't exist create a list
                    if chn in chn_list:
                        sta_coords = self.inventory.get_coordinates(id)
                        time_mat = backproj.griding(self.area_coords, sta_coords, self.dx, self.dy, self.depth)
                        data_map['nets'][net][sta][chn] = [time_mat]

                        size = size + 1
                    if chn_list[0] == "":
                        sta_coords = self.inventory.get_coordinates(id)
                        time_mat = backproj.griding(self.area_coords, sta_coords, self.dx, self.dy, self.depth)
                        data_map['nets'][net][sta][chn] = [time_mat]

                        size = size + 1


        return data_map

    @classmethod
    def get_st(self, data_path):
        obsfiles = self.list_directory(data_path)
        traces = []
        for paths in obsfiles:
            if _is_mseed(paths):
                st = read(paths)
                tr = st[0]
                traces.append(tr)

        st = Stream(traces)
        return st