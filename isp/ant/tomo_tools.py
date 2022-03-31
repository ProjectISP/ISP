import itertools
import pickle
import numpy as np
import scipy.interpolate as scint
import scipy.optimize as sciop
import scipy.spatial as scitial
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

class tomotools:

    @classmethod
    def read_dispersion(cls, file, wave_type, dispersion_type):

        disp_data = {}
        dict_ = pickle.load(open(file, "rb"))

        for pair in dict_.keys():
            stnm1, stnm2, wave, type_ = pair.split("_")
            if type_ in dispersion_type and wave in wave_type:
                disp_data.setdefault((stnm1, stnm2), {})
                disp_data[(stnm1, stnm2)]['c(f)'] = [float(x) for x in dict_[pair]['velocity']]
                disp_data[(stnm1, stnm2)]['f'] = [float(x) for x in dict_[pair]['period']]

        return disp_data

    @classmethod
    def get_station_info(cls, station_csv):
        st_info = {}
        with open(station_csv, 'r') as f:
            for line in f.readlines():
                line = [x for x in line.split(' ') if x != '']
                stnm = line[0]
                lat = float(line[2])
                lon = float(line[3])
                st_info[stnm] = [lon, lat]

        stations = sorted([stnm for stnm in st_info.keys()])
        pairs = list(itertools.combinations(stations, 2))

        return st_info, pairs

    @staticmethod
    def geo2km(latMid):

        m_per_deg_lat = (111132.954 - 559.822 * np.cos(2 * latMid) + 1.175 * np.cos(4 * latMid))/1000
        m_per_deg_lon = (111132.954 * np.cos(latMid))/1000

        return m_per_deg_lat, m_per_deg_lon


    @classmethod
    def create_grid(cls, stations_info, gridy=5, gridx=5):

        station_lats = []
        station_lons = []


        for key in stations_info[0].keys():
            station_lons.append(stations_info[0][key][0])
            station_lats.append(stations_info[0][key][1])

        latMid = np.mean(station_lats)
        latMid = np.pi *latMid/180
        km2lat, km2lon = cls.geo2km(latMid)

        minlat = min(station_lats) - 0.10
        maxlat = max(station_lats) + 0.10
        latstep = gridy/km2lat

        minlon = min(station_lons) - 0.10
        maxlon = max(station_lons) + 0.10
        lonstep = gridx/km2lon

        lats = np.arange(minlat - latstep, maxlat + latstep, latstep)
        lons = np.arange(minlon - lonstep, maxlon + lonstep, lonstep)

        nodes = np.zeros((len(lats), len(lons), 2))
        nodes_index = np.zeros((len(lats), len(lons)))

        index = 0
        for i in range(len(lats)):
            for j in range(len(lons)):
                nodes[i,j] = [lons[j], lats[i]]
                nodes_index[i,j] = index
                index = index + 1

        return nodes

    @classmethod
    def compute_distance_matrix(cls, grid):

        flattened_grid = grid.reshape(np.shape(grid)[0]*np.shape(grid)[1], 2)
        distance_matrix = np.zeros((len(flattened_grid), len(flattened_grid)))

        node_pairs = np.array(list(itertools.combinations_with_replacement(flattened_grid, 2)))

        lons_A = np.radians(node_pairs[:,0,0])
        lats_A = np.radians(node_pairs[:,0,1])
        lons_B = np.radians(node_pairs[:,1,0])
        lats_B = np.radians(node_pairs[:,1,1])

        dfi = lats_B - lats_A
        dlambd = lons_B - lons_A

        a = np.sin(dfi/2)**2 + np.cos(lats_A) * np.cos(lats_B) * np.sin(dlambd/2)**2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        upper_triangular_portion_indexes = np.triu_indices(len(flattened_grid))
        distance_matrix[upper_triangular_portion_indexes] = c*6371.0
        distance_matrix = distance_matrix + distance_matrix.T

        return distance_matrix

    @classmethod
    def traveltime_tomography(cls, frequency, station_info, station_pairs, disp_data, grid,
                              distance_matrix, alpha0, beta0, alpha1, path_npts=100, reg_lambda=0.3,
                              density_pixel_size=0.05, checkerboard_test=False):

        period = frequency
        # Check which station pairs have a noise correlation function:
        cf_pairs = [x for x in station_pairs if x in disp_data.keys()]

        phase_velocities = []
        cf_freq_pairs = []
        for pair in cf_pairs: # For each station pair
            # Check if the dispersion curve exists for the selected period
            min_freq = min(disp_data[pair]['f'])
            max_freq = max(disp_data[pair]['f'])

            if frequency < min_freq or frequency > max_freq:
                # Skip to the next iteration
                continue
            else:
                pass

            # Retrieve phase velocity and covariance for the selected period
            try:
                phase_velocities.append(scint.interp1d(disp_data[pair]['f'], disp_data[pair]['c(f)'])(frequency))
            except ValueError:
                continue

            # Store the station pair:
            cf_freq_pairs.append(pair) # This holds the station pairs that both have a correlation function and have a defined value
                                       # for the chosen period

        if len(cf_freq_pairs) == 0:
            return 1

        phase_velocities = np.array(phase_velocities)

        # The reference velocity is used to determine a wavelength. If the distance
        # between two of the stations is less than 1/2 this wavelength, we discard
        # that measurement:
        reference_velocity = np.median(phase_velocities)
        reference_wavelength = period * reference_velocity

        #reference_velocity = 3

        # Compute inter-station paths, observed and reference (respect to a homogeneous
        # model with c = reference_velocity) traveltimes:
        # Compute paths and traveltimes
        path_start_points = []
        path_end_points = []

        for pair in cf_freq_pairs:
            path_start_points.append(station_info[pair[0]])
            path_end_points.append(station_info[pair[1]])

        path_ds, path_great_circle_distances, interstation_paths = cls.compute_paths(path_start_points, path_end_points, path_npts)

        if checkerboard_test:
            f_checkerboard = cls.checkerboard(reference_velocity-reference_velocity*0.1, reference_velocity+reference_velocity*0.1, reference_velocity, 2*reference_wavelength)
            traveltimes = np.zeros(len(interstation_paths))
            for i, path in enumerate(interstation_paths):
                lons = interstation_paths[i,:,0]
                lats = interstation_paths[i,:,1]
                v = f_checkerboard(lons, lats)
                t = np.sum(path_ds[i] * (0.5 / v[:-1] + 0.5 / v[1:]))
                traveltimes[i] = t
        else:
            traveltimes = path_great_circle_distances/phase_velocities

        reference_traveltimes = path_great_circle_distances/reference_velocity

        # Next, we reject measurements which differ from the reference (median) velocity more
        # than 3 standard deviations:
        if not checkerboard_test:
            std = np.std(phase_velocities)
            diff = np.abs(phase_velocities - reference_velocity)
            accept_index = np.where(diff < 3*std)[0]

            phase_velocities = phase_velocities[accept_index]
            interstation_paths = interstation_paths[accept_index,:,:]
            traveltimes = traveltimes[accept_index]
            reference_traveltimes = reference_traveltimes[accept_index]
            path_great_circle_distances = path_great_circle_distances[accept_index]
            path_ds = path_ds[accept_index]

        # Then, reject measurements with interstation distances < wavelength:
        accept_index = np.where(path_great_circle_distances >= (reference_wavelength))[0]

        phase_velocities = phase_velocities[accept_index]
        interstation_paths = interstation_paths[accept_index,:,:]
        traveltimes = traveltimes[accept_index]
        reference_traveltimes = reference_traveltimes[accept_index]
        path_great_circle_distances = path_great_circle_distances[accept_index]
        path_ds = path_ds[accept_index]

        # Compute Delaunay triangles for the inversion grid
        flattened_grid = grid.reshape(np.shape(grid)[0]*np.shape(grid)[1], 2)
        delaunay_triangles = scitial.Delaunay(flattened_grid)

        no_grid_nodes = np.shape(grid)[0] * np.shape(grid)[1]

        # Build matrices
        # Covariance matrix. Since it is a diagonal matrix, the inverse is just
        # taking the inverse values of its elements:
        C_inv = np.matrix(np.zeros((np.shape(interstation_paths)[0], np.shape(interstation_paths)[0])))
        np.fill_diagonal(C_inv, ((np.array(path_great_circle_distances) / np.array(phase_velocities)))**-1)

        # Data kernel G
        G = np.zeros((np.shape(interstation_paths)[0], no_grid_nodes))
        # Convert path coordinates from geographic to cartesian:
        cartesian_interstation_paths = cls.geo2cartesian_paths(interstation_paths)
        # Fetch the indexes of the Delaunay triangles that enclose each point
        # of each path:
        triangles = delaunay_triangles.find_simplex(interstation_paths)
        node_indexes = delaunay_triangles.simplices[triangles]
        # From the inversion grid, get the coordinates of the nodes:
        node_coordinates = flattened_grid[node_indexes]
        # Convert the coordinates of the vertices to cartesian coordinates:
        cartesian_node_coordinates = cls.geo2cartesian_triangles(node_coordinates)
        # Project the points of the paths into the plane of the corresponding Delaunay triangle and
        # compute their barycentric coordinates:
        projected_paths = cls.projection(cartesian_interstation_paths, cartesian_node_coordinates)
        barycentric_path_coords = cls.barycentric_coords(projected_paths, cartesian_node_coordinates)
        # Fill the data kernel
        for i in range(np.shape(barycentric_path_coords)[0]):
            for p in range(np.shape(barycentric_path_coords)[1]):
                jA = node_indexes[i,p,0]
                jB = node_indexes[i,p,1]
                jC = node_indexes[i,p,2]
                G[i,jA] += barycentric_path_coords[i,p,0] / reference_velocity * path_ds[i]
                G[i,jB] += barycentric_path_coords[i,p,1] / reference_velocity * path_ds[i]
                G[i,jC] += barycentric_path_coords[i,p,2] / reference_velocity * path_ds[i]

        #return node_indexes
        # Smoothing kernel and regularization matrix F
        correlation_length = 8#np.sqrt(reference_wavelength)/10
        sk = np.exp(-distance_matrix**2/(2*correlation_length**2))
        sk = sk / (np.sum(sk, axis=-1) - np.diag(sk))
        F = -sk
        F[np.diag_indices_from(F)] = 1

        # Path density and regularization matrix H
        path_density = cls.compute_path_density(grid, interstation_paths, pixel_width=density_pixel_size)
        H = np.zeros(np.shape(F))
        for i, dens in enumerate(path_density):
            H[i,i] = np.exp(-reg_lambda*dens)

        # Constraint matrix Q
        Q = alpha0**2*(np.einsum('ij,kj->ik', F, F, optimize=True)) + beta0**2*(np.einsum('ij,kj->ik', H, H, optimize=True))

        # Inversion
        m_est = np.einsum('ij,j->i', np.einsum('ij,jk->ik', np.einsum('ij,kj->ik', np.linalg.inv(np.einsum('ij,jk->ik', np.einsum('ji,jk->ik', G, C_inv, optimize=True), G, optimize=True) + Q), G, optimize=True),
                                     C_inv, optimize=True), (traveltimes - reference_traveltimes), optimize=True)
        overdamped_velocities = reference_velocity / (1 + m_est)

        # Compute traveltimes for the overdamped model:
        overdamped_traveltimes = []
        for i in range(len(interstation_paths)):
            t = 0
            ds = path_ds[i]
            path = interstation_paths[i]
            for j in range(len(path)):
                iA = node_indexes[i,j,0]
                iB = node_indexes[i,j,1]
                iC = node_indexes[i,j,2]

                cA = barycentric_path_coords[i,j,0]
                cB = barycentric_path_coords[i,j,1]
                cC = barycentric_path_coords[i,j,2]

                ds_velocity = cA*overdamped_velocities[iA] + cB*overdamped_velocities[iB] + cC*overdamped_velocities[iC]

                t = t + ds/ds_velocity

            overdamped_traveltimes.append(t)

        # Compare the observed traveltimes vs the overdamped ones. If the residual
        # for a path is larger than three times the standard deviation, that path
        # is discarded:
        if not checkerboard_test:
            overdamped_residuals =  traveltimes - overdamped_traveltimes
            overdamped_residuals_std = np.std(overdamped_residuals)

            accept_index = np.where(np.abs(overdamped_residuals) < 3*overdamped_residuals_std)[0]

            phase_velocities = phase_velocities[accept_index]
            interstation_paths = interstation_paths[accept_index,:,:]
            traveltimes = traveltimes[accept_index]
            reference_traveltimes = reference_traveltimes[accept_index]
            path_great_circle_distances = path_great_circle_distances[accept_index]
            path_ds = path_ds[accept_index]
            node_indexes = node_indexes[accept_index,:,:]
            barycentric_path_coords = barycentric_path_coords[accept_index,:,:]

        reference_velocity = np.median(phase_velocities)
        reference_wavelength = period * reference_velocity

        # Perform the second inversion:
        # Covariance matrix:
        C_inv = np.matrix(np.zeros((np.shape(interstation_paths)[0], np.shape(interstation_paths)[0])))
        np.fill_diagonal(C_inv, ((np.array(path_great_circle_distances) / np.array(phase_velocities)))**-1)
        # Data kernel G
        G = np.zeros((np.shape(interstation_paths)[0], no_grid_nodes))
        for i in range(np.shape(barycentric_path_coords)[0]):
            for p in range(np.shape(barycentric_path_coords)[1]):
                jA = node_indexes[i,p,0]
                jB = node_indexes[i,p,1]
                jC = node_indexes[i,p,2]
                G[i,jA] += barycentric_path_coords[i,p,0] / reference_velocity * path_ds[i]
                G[i,jB] += barycentric_path_coords[i,p,1] / reference_velocity * path_ds[i]
                G[i,jC] += barycentric_path_coords[i,p,2] / reference_velocity * path_ds[i]

        # Smoothing kernel and regularization matrix F
        correlation_length = 8#np.sqrt(reference_wavelength)/10
        sk = np.exp(-distance_matrix**2/(2*correlation_length**2))
        sk = sk / (np.sum(sk, axis=-1) - np.diag(sk))
        F = -sk
        F[np.diag_indices_from(F)] = 1

        # Path density and regularization matrix H
        path_density = cls.compute_path_density(grid, interstation_paths, pixel_width=density_pixel_size)
        H = np.zeros(np.shape(F))
        for i, dens in enumerate(path_density):
            H[i,i] = np.exp(-reg_lambda*dens)

        # Constraint matrix Q
        Q = alpha1**2*(np.einsum('ij,kj->ik', F, F, optimize=True)) + beta0**2*(np.einsum('ij,kj->ik', H, H, optimize=True))

        # Inversion
        cov_m_est = np.linalg.inv(np.einsum('ij,jk->ik', np.einsum('ji,jk->ik', G, C_inv, optimize=True), G, optimize=True) + Q)
        cov_map = np.reshape(np.diag(cov_m_est), (np.shape(grid)[0], np.shape(grid)[1]))

        # m_est ES LA PERTURBACIÓN!!!!!!!!!!!!!! m_est = (v_ref - v) / v
        m_est = np.einsum('ij,j->i', np.einsum('ij,jk->ik', np.einsum('ij,kj->ik', cov_m_est, G, optimize=True),
                                     C_inv, optimize=True), (traveltimes - reference_traveltimes), optimize=True)
        absolute_velocities = reference_velocity / (1 + m_est)
        relative_velocities = 100 * (absolute_velocities - reference_velocity)/reference_velocity
        #velocities = m_est
        relative_velocity_map = np.reshape(relative_velocities, (np.shape(grid)[0], np.shape(grid)[1]))
        absolute_velocity_map = np.reshape(absolute_velocities, (np.shape(grid)[0], np.shape(grid)[1]))

        # Resolution analysis
        resolution_matrix = np.einsum('ij,jk->ik', np.einsum('ij,jk->ik', np.einsum('ij,kj->ik', cov_m_est, G, optimize=True),
                                     C_inv, optimize=True), G, optimize=True)

        resolution_map = cls.resolution_analysis(resolution_matrix, grid, reference_wavelength)

        # Travel time residuals & model RMS error
        model_traveltimes = []
        for i in range(len(interstation_paths)):
            t = 0
            ds = path_ds[i]
            path = interstation_paths[i]
            for j in range(len(path)):
                iA = node_indexes[i,j,0]
                iB = node_indexes[i,j,1]
                iC = node_indexes[i,j,2]

                cA = barycentric_path_coords[i,j,0]
                cB = barycentric_path_coords[i,j,1]
                cC = barycentric_path_coords[i,j,2]

                ds_velocity = cA*absolute_velocities[iA] + cB*absolute_velocities[iB] + cC*absolute_velocities[iC]

                t = t + ds/ds_velocity

            model_traveltimes.append(t)

        traveltime_residuals = traveltimes - model_traveltimes
        RMS = np.sqrt(np.sum((traveltime_residuals**2))/len(traveltime_residuals))

        result_dict = {'period':period,
                       'paths':interstation_paths,
                       'rejected_paths':len(cf_freq_pairs) - np.shape(interstation_paths)[0],
                       'ref_velocity':reference_velocity,
                       'alpha0':alpha0,
                       'alpha1':alpha1,
                       'beta':beta0,
                       'sigma':correlation_length,
                       'm_opt_relative':relative_velocity_map,
                       'm_opt_absolute':absolute_velocity_map,
                       'grid':grid,
                       'resolution_map':resolution_map,
                       'cov_map':cov_map,
                       'residuals':traveltime_residuals,
                       'rms':RMS}

        return result_dict, reference_wavelength


    ###########################

    @staticmethod
    def compute_paths(start_points, end_points, npts):
        A = np.radians(np.array(start_points))
        B = np.radians(np.array(end_points))

        # Compute great circle distance between all the points in A and B

        A_lons = A[:,0]
        A_lats = A[:,1]
        B_lons = B[:,0]
        B_lats = B[:,1]

        dfi = B_lats - A_lats
        dlambd = B_lons - A_lons

        a = np.sin(dfi/2)**2 + np.cos(A_lats) * np.cos(B_lats) * np.sin(dlambd/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        # Compute intermediate points
        # f is the fraction along the great circle path. f = 0 is A, while f = 1
        # is B.
        fs = np.linspace(0, 1, npts)

        interstation_paths = np.zeros((len(start_points), npts, 2))
        i = 0
        for f in fs:
            d = np.sin((1-f)*c) / np.sin(c)
            e = np.sin(f*c) / np.sin(c)

            x = d * np.cos(A_lats) * np.cos(A_lons) + e * np.cos(B_lats) * np.cos(B_lons)
            y = d * np.cos(A_lats) * np.sin(A_lons) + e * np.cos(B_lats) * np.sin(B_lons)
            z = d * np.sin(A_lats) + e * np.sin(B_lats)

            phi = np.arctan2(z, np.sqrt(x**2 + y**2))
            lambd = np.arctan2(y, x)

            interstation_paths[:,i,0] = np.degrees(lambd)
            interstation_paths[:,i,1] = np.degrees(phi)

            i = i + 1

        # Compute approximate distance between intermediate points in km:
        ds = (c*6371.0)/npts

        return ds, c*6371.0, interstation_paths

    @staticmethod
    def checkerboard(vmin, vmax, vmid, squaresize):
        d2rad = np.pi/180
        midlat = 0.5 * (6.4861 + 14.0497)
        latwidth = squaresize / 6371 / d2rad
        lonwidth = squaresize / (6371.0 * np.cos(midlat * d2rad)) / d2rad

        def basis_func(lons, lats, lon0, lat0):
            x = (lons - lon0) / lonwidth
            y = (lats - lat0) / latwidth
            outside_square = (np.abs(x) >= 0.5) | (np.abs(y) >= 0.5)
            return np.where(outside_square, 0.0, np.cos(np.pi*x) * np.cos(np.pi*y))

        startlon = -67.7513 + lonwidth/2
        stoplon = -61.5392 + lonwidth
        centerlons = list(np.arange(startlon, stoplon, lonwidth))

        startlat = 6.4861 + latwidth/2
        stoplat = 14.0497 + latwidth
        centerlats = list(np.arange(startlat, stoplat, latwidth))

        centerlonlats = list(itertools.product(centerlons, centerlats))

        polarities = [(centerlons.index(lon) + centerlats.index(lat)) % 2 for
                      lon, lat in centerlonlats]

        factors = np.where(np.array(polarities) == 1, vmax - vmid, vmin - vmid)

    @staticmethod
    def compute_path_density(grid, interstation_paths, pixel_width=0.05):
        densities = np.zeros(np.shape(grid)[0] * np.shape(grid)[1])

        min_lats = np.expand_dims(grid[:,:,1].flatten()-pixel_width, axis=-1)
        max_lats = np.expand_dims(grid[:,:,1].flatten()+pixel_width, axis=-1)
        min_lons = np.expand_dims(grid[:,:,0].flatten()-pixel_width, axis=-1)
        max_lons = np.expand_dims(grid[:,:,0].flatten()+pixel_width, axis=-1)

        for path in interstation_paths:
            cross_check = (path[:,0] >= min_lons) & (path[:,0] <= max_lons) & (path[:,1] >= min_lats) & (path[:,1] <= max_lats)
            densities = densities + np.any(cross_check, axis=-1)

        return densities

    @staticmethod
    def resolution_analysis(resolution_matrix, grid, reference_wavelength):

        flattened_grid = grid.reshape(np.shape(grid)[0]*np.shape(grid)[1], 2)

        cone_radiuses = []

        for i, resolution_row in enumerate(resolution_matrix):
            cone_lat, cone_lon = flattened_grid[i]
            node_pairs = np.array([[flattened_grid[i], node] for node in flattened_grid])
            r = tomotools.compute_distances_to_node(node_pairs)

            # fit cone to resolution map
            fitted_cone = sciop.curve_fit(tomotools.cone_equation, r, np.abs(resolution_row),
                                           p0=[reference_wavelength, np.max(resolution_matrix)],maxfev=10000)
            resolution = max(reference_wavelength, fitted_cone[0][0])
            cone_radiuses.append(resolution)

        resolution_map = np.reshape(cone_radiuses, (np.shape(grid)[0], np.shape(grid)[1]))

        return resolution_map

    @staticmethod
    def compute_distances_to_node(array):
        lons_A = np.radians(array[:,0,0])
        lats_A = np.radians(array[:,0,1])
        lons_B = np.radians(array[:,1,0])
        lats_B = np.radians(array[:,1,1])

        dfi = lats_B - lats_A
        dlambd = lons_B - lons_A

        a = np.sin(dfi/2)**2 + np.cos(lats_A) * np.cos(lats_B) * np.sin(dlambd/2)**2

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return c*6371

    @staticmethod
    def cone_equation(r, r0, z0):
        return np.piecewise(r, [r < r0, r >= r0], [lambda r: z0 * (r0 - r)/r0, lambda r: 0.0])


    @staticmethod
    def geo2cartesian_paths(interstation_paths, r=1.0):
        cartesian_interstation_paths = np.zeros((np.shape(interstation_paths)[0], np.shape(interstation_paths)[1], 3))
        # spherical coordinates
        phi = np.array(interstation_paths[:, :, 0]) * np.pi / 180.0
        theta = np.pi / 2.0 - np.array(interstation_paths[:, :, 1]) * np.pi / 180.0
        # cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        cartesian_interstation_paths[:, :, 0] = x
        cartesian_interstation_paths[:, :, 1] = y
        cartesian_interstation_paths[:, :, 2] = z

        return cartesian_interstation_paths


    @staticmethod
    def geo2cartesian_triangles(triangle_coordinates, r=1.0):
        cartesian_triangle_coordinates = np.zeros((np.shape(triangle_coordinates)[0], np.shape(triangle_coordinates)[1],
                                                   np.shape(triangle_coordinates)[2], 3))
        # spherical coordinates
        phi = np.array(triangle_coordinates[:,:,:,0]) * np.pi / 180.0
        theta = np.pi / 2.0 - np.array(triangle_coordinates[:,:,:,1]) * np.pi / 180.0
        # cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        cartesian_triangle_coordinates[:,:,:,0] = x
        cartesian_triangle_coordinates[:,:,:,1] = y
        cartesian_triangle_coordinates[:,:,:,2] = z

        return cartesian_triangle_coordinates

    @staticmethod
    def projection(cartesian_paths, cartesian_triangles):
        AB = cartesian_triangles[:,:,1,:] - cartesian_triangles[:,:,0,:]
        AC = cartesian_triangles[:,:,2,:] - cartesian_triangles[:,:,0,:]
        MA = cartesian_paths - cartesian_triangles[:,:,0,:]
        # unit vector u perpendicular to ABC (u = AB x AC / |AB x AC|)
        u = np.zeros(np.shape(AB))
        # Scalar components of the cross product AB x AC:
        u[:,:,0] = AB[:,:,1]*AC[:,:,2] - AB[:,:,2]*AC[:,:,1]
        u[:,:,1] = AB[:,:,2]*AC[:,:,0] - AB[:,:,0]*AC[:,:,2]
        u[:,:,2] = AB[:,:,0]*AC[:,:,1] - AB[:,:,1]*AC[:,:,0]
        # Norm of cross product |AB x AC|
        norm = np.sqrt(u[:,:,0]**2 + u[:,:,1]**2 + u[:,:,2]**2)
        u = np.einsum('ij,ijk->ijk', 1/norm, u, optimize=True)
        # (MA.u)u = MM' (with M' the projection of M on the plane)
        MA_dot_u = np.sum(MA * u, axis=-1)
        MMp = np.einsum('ij,ijk->ijk', MA_dot_u, u, optimize=True)
        projected_paths = cartesian_paths + MMp

        return projected_paths

    @staticmethod
    def barycentric_coords(projected_paths, cartesian_triangles):
        MA = projected_paths - cartesian_triangles[:, :, 0, :]
        MB = projected_paths - cartesian_triangles[:, :, 1, :]
        MC = projected_paths - cartesian_triangles[:, :, 2, :]

        MBxMC = np.zeros(np.shape(MB))
        # Scalar components of the cross product MB x MC
        MBxMC[:, :, 0] = MB[:, :, 1] * MC[:, :, 2] - MB[:, :, 2] * MC[:, :, 1]
        MBxMC[:, :, 1] = MB[:, :, 2] * MC[:, :, 0] - MB[:, :, 0] * MC[:, :, 2]
        MBxMC[:, :, 2] = MB[:, :, 0] * MC[:, :, 1] - MB[:, :, 1] * MC[:, :, 0]
        MBxMC_norm = np.sqrt(MBxMC[:, :, 0] ** 2 + MBxMC[:, :, 1] ** 2 + MBxMC[:, :, 2] ** 2)
        bcA = MBxMC_norm / 2

        MAxMC = np.zeros(np.shape(MA))
        # Scalar components of the cross product MA x MC
        MAxMC[:, :, 0] = MA[:, :, 1] * MC[:, :, 2] - MA[:, :, 2] * MC[:, :, 1]
        MAxMC[:, :, 1] = MA[:, :, 2] * MC[:, :, 0] - MA[:, :, 0] * MC[:, :, 2]
        MAxMC[:, :, 2] = MA[:, :, 0] * MC[:, :, 1] - MA[:, :, 1] * MC[:, :, 0]
        MAxMC_norm = np.sqrt(MAxMC[:, :, 0] ** 2 + MAxMC[:, :, 1] ** 2 + MAxMC[:, :, 2] ** 2)
        bcB = MAxMC_norm / 2

        MAxMB = np.zeros(np.shape(MA))
        # Scalar components of the cross product MA x MB
        MAxMB[:, :, 0] = MA[:, :, 1] * MB[:, :, 2] - MA[:, :, 2] * MB[:, :, 1]
        MAxMB[:, :, 1] = MA[:, :, 2] * MB[:, :, 0] - MA[:, :, 0] * MB[:, :, 2]
        MAxMB[:, :, 2] = MA[:, :, 0] * MB[:, :, 1] - MA[:, :, 1] * MB[:, :, 0]
        MAxMB_norm = np.sqrt(MAxMB[:, :, 0] ** 2 + MAxMB[:, :, 1] ** 2 + MAxMB[:, :, 2] ** 2)
        bcC = MAxMB_norm / 2

        bc_total = bcA + bcB + bcC

        barycentric_coords = np.zeros(np.shape(projected_paths))
        barycentric_coords[:, :, 0] = bcA / bc_total
        barycentric_coords[:, :, 1] = bcB / bc_total
        barycentric_coords[:, :, 2] = bcC / bc_total

        return barycentric_coords

    @classmethod
    def save_results(cls, result_dict,period, wave_type, dispersion_type):
        plt.ioff()
        if result_dict != 1:
            path = Path("isp/ant/data_tomo/{}/{}/pickles/".format(wave_type, dispersion_type))
            path.mkdir(parents=True, exist_ok=True)
            pickle.dump(result_dict, open("isp/ant/data_tomo//{}/{}/pickles/".format(wave_type, dispersion_type) + 'dispersion_{}_{}_{}s.pickle'.format(wave_type, dispersion_type, period), 'wb'))
            lats = result_dict[0]['grid'][:,0][:,1]
            lons = result_dict[0]['grid'][1,:][:,0]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
            map_ = ax.pcolormesh(lons, lats, result_dict[0]['m_opt_relative'], transform=ccrs.PlateCarree(), cmap="RdBu", vmin=-10, vmax=10)
            ax.coastlines()
            fig.colorbar(map_)
            path = Path("isp/ant/data_tomo/{}/{}/figures/".format(wave_type, dispersion_type))
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig("isp/ant/data_tomo/{}/{}/figures/".format(wave_type, dispersion_type)  + 'dispersion_{}_{}_{}s.png'.format(wave_type, dispersion_type, period), dpi=600)
            plt.close(fig=fig)
        else:
            print("exit status 1")


# if __name__ == '__main__':
#     # Set up inversion
#     # Rayleigh -> "ZZ", Love -> "TT"
#     # phase -> phv, group -> dsp
#     plt.ioff()
#     for wave_type in ["TT", "ZZ"]:
#         for dispersion_type in ["dsp", "phv"]:
#
#             disp_data = read_dispersion("disp_maps.pkl", wave_type=wave_type, dispersion_type=dispersion_type)
#
#             what = []
#             for i in disp_data.keys():
#                 for val in disp_data[i]['f']:
#                     what.append(val)
#
#             pmin = min(what)
#             pmax = max(what)
#
#             station_info, station_pairs = get_station_info()
#             grid = create_grid(station_info, gridy=10, gridx=10, km2lat=111.2, km2lon=100)
#             distance_matrix = compute_distance_matrix(grid)
#             for period in np.arange(pmin, pmax, 0.5):
#                 print("Computing {} s map for {} {}".format(period, wave_type, dispersion_type))
#                 result_dict = traveltime_tomography(period, station_info, station_pairs, disp_data, grid, distance_matrix,
#                                                     600, 10, 100, path_npts=150, reg_lambda=0.1, density_pixel_size=0.05, checkerboard_test=False)
#                 if result_dict != 1:
#                     path = Path("results/{}/{}/pickles/".format(wave_type, dispersion_type))
#                     path.mkdir(parents=True, exist_ok=True)
#                     pickle.dump(result_dict, open("results/{}/{}/pickles/".format(wave_type, dispersion_type) + 'dispersion_{}_{}_{}s.pickle'.format(wave_type, dispersion_type, period), 'wb'))
#                     lats = result_dict[0]['grid'][:,0][:,1]
#                     lons = result_dict[0]['grid'][1,:][:,0]
#                     fig = plt.figure()
#                     ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
#                     map_ = ax.pcolormesh(lons, lats, result_dict[0]['m_opt_relative'], transform=ccrs.PlateCarree(), cmap="RdBu", vmin=-10, vmax=10)
#                     ax.coastlines()
#                     fig.colorbar(map_)
#                     path = Path("results/{}/{}/figures/".format(wave_type, dispersion_type))
#                     path.mkdir(parents=True, exist_ok=True)
#                     fig.savefig("results/{}/{}/figures/".format(wave_type, dispersion_type)  + 'dispersion_{}_{}_{}s.png'.format(wave_type, dispersion_type, period), dpi=600)
#                     plt.close(fig=fig)
#                 else:
#                     print("exit status 1")
