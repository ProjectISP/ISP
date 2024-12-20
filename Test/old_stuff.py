    def _picker_thread(self):
        self.progressbar.reset()
        self.progressbar.setLabelText(" Computing Auto-Picking ")
        with ThreadPoolExecutor(1) as executor:
            f = executor.submit(self._run_picker)
            self.progressbar.exec()
            f.cancel()


    def _run_picker(self):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        # Creates a new file
        #with open(self.path_phases, 'w') as fp:
        #    pass
        if self.st:
            stations = ObspyUtil.get_stations_from_stream(self.st)
            N = len(stations)
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setMaximum', qt.AutoConnection, pyc.Q_ARG(int, N))
            pyc.QMetaObject.invokeMethod(self.progressbar, 'setValue', qt.AutoConnection, pyc.Q_ARG(int, 0))
            base_indexes = [*map(lambda x: 3 * x, [*range(N)])]
            # TODO OPEN MP FOR MULTIPLE STATIONS
            for station, base_index in zip(stations, base_indexes):
                self._process_station(station, base_index)
        else:
            pyc.QMetaObject.invokeMethod(self.progressbar, 'reset', qt.AutoConnection)


    def _process_station(self, station, index):
        st2 = self.st.select(station=station)
        try:
            maxstart = np.max([tr.stats.starttime for tr in st2])
            minend = np.min([tr.stats.endtime for tr in st2])
            st2.trim(maxstart, minend)
            self.cnn.setup_stream(st2)  # set stream to use in prediction.
            self.cnn.predict()
            arrivals = self.cnn.get_arrivals()
            for k, times in arrivals.items():
                for t in times:
                    if k == "p":
                        line = self.canvas.draw_arrow(t.matplotlib_date, index + 2,
                                               "P", color="blue", linestyles='--', picker=True)
                        self.lines.append(line)

                        # TODO needs to be reviewed
                        #with open(self.path_phases, "a+") as f:
                        #    f.write(station + " " + k.upper() + " " + t.strftime(format="%Y-%m-%dT%H:%M:%S.%f") + "\n")

                        self.picked_at[str(line)] = PickerStructure(t, st2[2].stats.station, t.matplotlib_date,
                            0.2, 0, "blue", "P", self.get_file_at_index(index + 2))

                        self.pm.add_data(t, 0.2, 0, st2[2].stats.station, "P", Component=st2[2].stats.channel,
                                         First_Motion="?")
                        self.pm.save()

                    if k == "s":

                        line1s = self.canvas.draw_arrow(t.matplotlib_date, index + 0,
                                               "S", color="purple", linestyles='--', picker=True)
                        line2s = self.canvas.draw_arrow(t.matplotlib_date, index + 1,
                                               "S", color="purple", linestyles='--', picker=True)

                        self.lines.append(line1s)
                        self.lines.append(line2s)

                        # TODO needs to be reviewed
                        #with open(self.path_phases, "a+") as f:
                        #    f.write(station+" "+k.upper()+" "+t.strftime(format="%Y-%m-%dT%H:%M:%S.%f") + "\n")

                        self.picked_at[str(line1s)] = PickerStructure(t, st2[1].stats.station, t.matplotlib_date,
                             0.2, 0, "blue", "S", self.get_file_at_index(index + 0))

                        self.picked_at[str(line2s)] = PickerStructure(t, st2[1].stats.station, t.matplotlib_date,
                             0.2, 0, "blue", "S", self.get_file_at_index(index + 1))

                        self.pm.add_data(t, 0.2, 0, st2[1].stats.station, "S", Component=st2[1].stats.channel,
                                         First_Motion="?")
                        self.pm.save()

        except ValueError as e:
            # TODO: summarize errors and show eventually
            # md = MessageDialog(self)
            # md.set_info_message("Prediction failed for station {}\n{}".format(station,e))
            pass
        pyc.QMetaObject.invokeMethod(self, '_increase_progress', qt.AutoConnection)



    def detect_events(self):

        #to make a detection it is needed to trim the data otherwise,
        # is going to take the starttime and endtime of the file

        params = self.settings_dialog.getParameters()
        threshold = params["ThresholdDetect"]
        coincidences = params["Coincidences"]
        cluster  = params["Cluster"]

        standard_deviations = []
        all_traces = []

        starttime = convert_qdatetime_utcdatetime(self.dateTimeEdit_1)
        endtime = convert_qdatetime_utcdatetime(self.dateTimeEdit_2)


        for tr in self.st:
            cw = ConvolveWaveletScipy(tr)
            if self.trimCB.isChecked():
                cw.setup_wavelet(starttime, endtime, wmin=5, wmax=5, tt=10, fmin=0.5, fmax=10, nf=40, use_rfft=False,
                             decimate=False)
            else:
                cw.setup_wavelet(wmin=5, wmax=5, tt=10, fmin=0.5, fmax=10, nf=40, use_rfft=False,
                                 decimate=False)

            cf = cw.cf_lowpass()
            # Normalize
            #cf = cf / max(cf)
            standard_deviations.append(np.std(cf))

            tr_cf = tr.copy()
            tr_cf.data = cf
            all_traces.append(tr_cf)

        max_threshold = threshold*np.mean(standard_deviations)
        min_threshold = 1*np.mean(standard_deviations)

        self.st = Stream(traces=all_traces)

        trigger = coincidence_trigger(trigger_type=None, thr_on = max_threshold, thr_off = min_threshold,
                                     trigger_off_extension = 0, thr_coincidence_sum = coincidences, stream=self.st,
                                      similarity_threshold = 0.8, details=True)


        for k in range(len(trigger)):
            detection = trigger[k]
            for key in detection:

                if key == 'time':
                    time = detection[key]
                    self.events_times.append(time)
        # calling for 1D clustering more than one detection per earthquake //eps seconds span
        try:
            self.events_times, str_times = MseedUtil.cluster_events(self.events_times, eps=cluster)

            with open(self.path_detection, "w") as fp:
                json.dump(str_times, fp)

            self.plot_seismogram()
            md = MessageDialog(self)
            md.set_info_message("Events Detection done")
        except:
            md = MessageDialog(self)
            md.set_info_message("No Detections")



               def picker_all(self):
        # delta is the trim of the data
        IND_OUTPUT_PATH = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'loc', 'last.hyp')
        LOC_OUTPUT_PATH = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'all_locations')
        params = self.settings_dialog.getParameters()
        delta = params["window pick"]
        transform =params["transform"]
        pick_output_path = PickerManager.get_default_output_path()
        self.nll_manager = NllManager(pick_output_path, self.metadata_path_bind.value)
        st_detect_all = self.st.copy()

        self.detect_events()
        events_path = self.path_detection
        picks_path = os.path.join(ROOT_DIR, 'earthquakeAnalisysis', 'location_output', 'obs', 'output.txt')


        with open(events_path, 'rb') as handle:
            events = json.load(handle)

        for k in range(len(events)):
            start = UTCDateTime(events[k]) - delta
            end = UTCDateTime(events[k]) + delta
            self.st.trim(starttime = start , endtime = end)
            # clean previous picks
            os.remove(picks_path)
            # Picking on detected event
            self._run_picker()
            # Locate based on prevous detected and picked event
            std_out = self.nll_manager.run_nlloc(0, 0, 0, transform = transform)

            # restore it
            self.st = st_detect_all
            # copy output to /location_output/all_locations
            LOC_OUTPUT_PATH_TEMP  = os.path.join(LOC_OUTPUT_PATH,events[k]+".hyp")
            shutil.copyfile(IND_OUTPUT_PATH, LOC_OUTPUT_PATH_TEMP)

            #md = MessageDialog(self)
            #md.set_info_message("Location complete. Check details for earthquake located at "+events[k]
            #                    , std_out)
        md = MessageDialog(self)
        md.set_info_message("Location complete. Check details for earthquake located in "+LOC_OUTPUT_PATH)