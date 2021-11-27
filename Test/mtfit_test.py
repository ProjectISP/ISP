import gc
import os
import time
import traceback

from MTfit.plot import read, MTplot

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from MTfit.inversion import Inversion, memory_profile_test, _MEMTEST, ForwardTask


class MyInversion(Inversion):

    def __init__(self, data={}, data_file=False, location_pdf_file_path=False, algorithm='iterate',
                 parallel=True, n=0, phy_mem=8, dc=False, **kwargs):

        super().__init__(data=data, data_file=data_file, location_pdf_file_path=location_pdf_file_path,
                         algorithm=algorithm, parallel=parallel, n=n, phy_mem=phy_mem,
                         dc=dc, **kwargs)
        self.result = None

    @memory_profile_test(_MEMTEST)
    def _random_sampling_forward(self, source_type='MT', return_zero=False):
        """
        Monte Carlo random sampling event forward function

        Runs event forward model using Monte Carlo random sampling approach - ie parallel for multiple samples,  single event.

        Args
            source_type:['MT'] 'MT' or 'DC' for fid
            return_zero:[True] Return zero probability samples in task result.

        """
        # Loop over events
        for i, event in enumerate(self.data):
            # Get fid
            fid = self._fid(event, i, source_type, single=True)
            # Set logger
            self._set_logger(fid)
            self._print('\n\nEvent ' + str(i + 1) + '\n--------\n')
            try:
                self._print('UID: ' + str(event['UID']) + '\n')
            except Exception:
                self._print('No UID\n')
            # Recover test
            if self._recover_test(fid):
                continue
            # File sample recover test
            if self._file_sample_test(fid):
                self._print('Continuing from previous sampling')
            self.kwargs['fid'] = fid
            # Check event data
            if not event:
                self._print('No Data')
                continue
            try:
                try:
                    event = self._trim_data(event)
                except ValueError:
                    self._print('No Data')
                    continue
                # Get station angle coefficients and data etc.
                (a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio, amplitude_ratio,
                 percentage_error1_amplitude_ratio,
                 percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability,
                 incorrect_polarity_probability, extension_data) = self._station_angles(event, i)
                # Update sample size
                self._update_samples()
                # Set algorithm
                del self.algorithm
                gc.collect()
                self._set_algorithm(single=True, **self.kwargs)
                self._print('\nInitialisation Complete\n\nBeginning Inversion\n')
                # Run ForwardTasks
                if self.pool:
                    # initialise all tasks
                    for w in range(self.pool.number_workers):
                        MTs, end = self._parse_job_result(False)
                        self.pool.task(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                       amplitude_ratio, percentage_error1_amplitude_ratio,
                                       percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability,
                                       self.location_sample_multipliers,
                                       incorrect_polarity_probability, return_zero, False, True, self.generate_samples,
                                       self.generate_cutoff, self.dc, extension_data)
                elif self._MPI:
                    end = False
                    # Carried out in each worker
                    MTs, ignored_end = self._parse_job_result(False)
                    result = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                         amplitude_ratio, percentage_error1_amplitude_ratio,
                                         percentage_error2_amplitude_ratio, a_polarity_probability,
                                         polarity_probability, self.location_sample_multipliers,
                                         incorrect_polarity_probability, return_zero,
                                         generate_samples=self.generate_samples, cutoff=self.generate_cutoff,
                                         dc=self.dc, extension_data=extension_data)()
                    # Return to initiator algorithm
                else:
                    MTs, end = self._parse_job_result(False)
                # Continue until max samples/time reached (end =  = True)
                while not end:
                    if self.pool:
                        result = self.pool.result()
                        MTs, end = self._parse_job_result(result)
                        self.pool.task(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                       amplitude_ratio, percentage_error1_amplitude_ratio,
                                       percentage_error2_amplitude_ratio, a_polarity_probability, polarity_probability,
                                       self.location_sample_multipliers,
                                       incorrect_polarity_probability, return_zero, False, True, self.generate_samples,
                                       self.generate_cutoff, self.dc, extension_data)
                    elif self._MPI:
                        if not self.mpi_output:
                            # Handle result split and together using gather
                            mts = self.comm.gather(result['moment_tensors'], 0)
                            Ps = self.comm.gather(result['ln_pdf'], 0)
                            Ns = self.comm.gather(result['n'], 0)
                            if self.comm.Get_rank() == 0:
                                end = False
                                for i, mt in enumerate(mts):
                                    result = {'moment_tensors': mts[i], 'ln_pdf': Ps[i], 'n': Ns[i]}
                                    ignoredMTs, end = self._parse_job_result(result)
                                if end:
                                    end = True  # Process all results before ending
                            else:
                                end = None
                            end = self.comm.bcast(end, root=0)  # Broadcast end to all mpis
                            _iteration = self.algorithm.iteration
                            _start_time = False
                            try:
                                _start_time = self.algorithm.start_time
                            except Exception:
                                pass
                            MTs, ignored_end = self._parse_job_result(False)  # Get new random MTs
                            self.algorithm.iteration = _iteration
                            if _start_time:
                                self.algorithm.start_time = _start_time
                        else:
                            # Just run in parallel
                            MTs, end = self._parse_job_result(result)
                            end = self.comm.bcast(end, root=0)
                        result = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                             amplitude_ratio, percentage_error1_amplitude_ratio,
                                             percentage_error2_amplitude_ratio, a_polarity_probability,
                                             polarity_probability, self.location_sample_multipliers,
                                             incorrect_polarity_probability, return_zero,
                                             generate_samples=self.generate_samples, cutoff=self.generate_cutoff,
                                             dc=self.dc,
                                             extension_data=extension_data)()
                    else:
                        result = ForwardTask(MTs, a_polarity, error_polarity, a1_amplitude_ratio, a2_amplitude_ratio,
                                             amplitude_ratio, percentage_error1_amplitude_ratio,
                                             percentage_error2_amplitude_ratio, a_polarity_probability,
                                             polarity_probability, self.location_sample_multipliers,
                                             incorrect_polarity_probability, return_zero,
                                             generate_samples=self.generate_samples, cutoff=self.generate_cutoff,
                                             dc=self.dc,
                                             extension_data=extension_data)()
                        MTs, end = self._parse_job_result(result)
                # Get left over pool results
                if self.pool:
                    results = self.pool.all_results()
                    for result in results:
                        if result:
                            MTs, end = self._parse_job_result(result)
                try:
                    self._print('Inversion completed\n\t' + 'Elapsed time: ' +
                                str(time.time() - self.algorithm.start_time).split('.')[0] + ' seconds\n\t' + str(
                        self.algorithm.pdf_sample.n) +
                                ' samples evaluated\n\t' + str(len(self.algorithm.pdf_sample.nonzero())) +
                                ' non-zero samples\n\t' + '{:f}'.format((float(
                        len(self.algorithm.pdf_sample.nonzero())) / float(self.algorithm.pdf_sample.n)) * 100) + '%')
                except ZeroDivisionError:
                    self._print('Inversion completed\n\t' + 'Elapsed time: ' +
                                str(time.time() - self.algorithm.start_time).split('.')[0] + ' seconds\n\t' + str(
                        self.algorithm.pdf_sample.n) +
                                ' samples evaluated\n\t' + str(
                        len(self.algorithm.pdf_sample.nonzero())) + ' non-zero samples\n\t' + '{:f}'.format(0) + '%')
                self._print('Algorithm max value: ' + self.algorithm.max_value())
                output_format = self.output_format
                if self._MPI and self.mpi_output:
                    # MPI output (hyp format)
                    output_format = 'hyp'
                    results_format = self.results_format
                    self.results_format = 'hyp'
                    normalise = self.normalise
                    self.normalise = False
                # Output results
                if (self._MPI and not self.mpi_output and self.comm.Get_rank() == 0) or (
                        self._MPI and self.mpi_output) or not self._MPI:
                    self.output(event, fid, a_polarity=a_polarity, error_polarity=error_polarity,
                                a1_amplitude_ratio=a1_amplitude_ratio, a2_amplitude_ratio=a2_amplitude_ratio,
                                amplitude_ratio=amplitude_ratio,
                                percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,
                                percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio,
                                output_format=output_format)
                    self.result = results

                if self._MPI and self.mpi_output:
                    # Output pkl file with data
                    self.normalise = normalise
                    self.results_format = results_format
                    fids = self.comm.gather(fid, 0)
                    if isinstance(fids, list):
                        self._print('All moment tensor samples outputted: ' + str(len(fids)) + ' files')
                    else:
                        self._print('Error with fids ' + str(fids) + ' type:' + str(type(fids)))
                    if self.comm.Get_rank() == 0:
                        with open(os.path.splitext(fid)[0] + '.pkl', 'wb') as f:
                            pickle.dump({'fids': fids, 'event_data': event}, f)
            except Exception:
                traceback.print_exc()


data = {}
data["PPolarity"] = {'Error': np.array([[0.05], [0.05], [0.05], [0.01], [0.01], [0.01], [0.01],
                                       [0.01], [0.01], [1.], [0.01], [1.], [1.], [0.01],
                                       [0.05], [0.05]]),
                     'Measured': np.array([[-1], [-1], [1], [-1], [1], [-1], [-1], [-1],
                                          [1], [1], [-1], [-1], [-1], [1], [-1], [1]]),
                     'Stations': {'Azimuth': np.array([[55.9], [76.9], [277.9], [5.4], [224.7],
                                                      [31.9], [47.9], [45.2], [224.6], [122.6],
                                                      [328.4], [45.2], [309.3], [187.7], [16.1],
                                                      [193.4]]),
                                  'Name': ['S0517', 'S0415', 'S0347', 'S0534', 'S0244',
                                           'S0618', 'S0650', 'S0595', 'S0271', 'S0155',
                                           'S0529', 'S0649', 'S0450', 'S0195', 'S0588',
                                           'S0142'],
                                  'TakeOffAngle': np.array([[122.8], [120.8], [152.4], [138.7],
                                                           [149.6], [120.], [107.4], [117.],
                                                           [156.4], [115.3], [133.3], [109.1],
                                                           [139.9], [147.2], [128.7], [137.6]])}
                     }
data['UID'] = '_ppolarity'
# Set inversion parameters
# Use an iteration random sampling algorithm
algorithm = 'iterate'
# Run in parallel if set on command line
parallel = True
# uses a soft memory limit of 1Gb of RAM for estimating the sample sizes
# (This is only a soft limit, so no errors are thrown if the memory usage
#       increases above this)
phy_mem = 1
# Run in double-couple space only
dc = True
# Run for one hundred thousand samples
max_samples = 100000
# Set to only use P Polarity data
inversion_options = 'PPolarity'
# Set the convert flag to convert the output to other source parameterisations
convert = True
# Create the inversion object with the set parameters.
# inversion_object = MyInversion(data, algorithm=algorithm, parallel=parallel,
#                                inversion_options=inversion_options, phy_mem=phy_mem, dc=dc,
#                                max_samples=max_samples, convert=convert)

# Run the forward model
# inversion_object.forward()
# print(inversion_object.result)

DCs, DCstations = read('_ppolarityDC.mat')
print(DCs)
plot = MTplot([np.array([3,3]),DCs],
    stations=DCstations,
    station_distribution=False,
    plot_type='faultplane',fault_plane=False,
    show_mean=False,show_max=True,grid_lines=True,TNP=False,text=False)
plot.show()
