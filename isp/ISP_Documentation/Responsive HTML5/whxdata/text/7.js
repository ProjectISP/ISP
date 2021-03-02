rh._.exports({"0":[["Pick Event"]],"1":[["\n  ","\n    ","\n  ","\n  ","\n  "],["\n  ","The module Earthquake analysis allows you to analyze waveforms, polarization analysis of seismograms 3-components, calculate different magnitudes and finally locate an event and estimate the focal mechanism (First Polarity). We will walk through all of the functionality following this scheme:","\n  ","\n  ","\n  ","\n  ","From top to bottom in the left side of the window you will see,","\n  ","\n  ","\n    "," Event Info. This small box let enter information about an event, in this way you can plot the theoretical arrivals (ak-135F) with respect your seismograms.","\n  ","\n  ","\n  ","\n    "," Files. By clicking in this button you will place the path of your files (*miniseed or *sac).","\n  ","\n  ","\n  ","\n    "," Metadata. By clicking in this button you will place the path to you metadata file. The metadata file must contain the information of all of the stations seismograms you want to analyze (metadata must be a *xml or *dlsv).","\n  ","\n  ","\n  ","\n    "," Start Time and End Time. The time boxes can be selected checking “Trim Time”. If you choose this option all the seismograms will be cut in accordance with the selected time window.","\n  ","\n  ","\n  ","\n    ","Stations Info will deploy a table with the fundamental information of the seismograms.","\n  ","\n  ","\n  ","\n    "," Phase box. This combo box allow the selection of specific phases for picking it in the seismograms.","\n  ","\n  ","\n  ","\n    "," Waves box. You can Choose Body, surface, coda or noise. With this selected you can highlight a time window of the seismogram for further analysis.","\n  ","\n  ","\n  ","\n    "," Net, Station and Channel. You can fill this boxes (also using wildcards) to select specific files from your files folder (check “select files”).","\n  ","\n  ","\n  ","\n    "," Locate. After pick some phases in your seismogram you can locate an earthquake and show a location map. Previously you must have calculated travel-times for your velocity model (Further details in Event Location tab. 2.3)","\n  ","\n  ","\n  ","\n    "," Stations Map will show you the location of the stations that corresponds to the seismograms of your folder. The seismograms metadata must match with the metadata information.","\n  ","\n  ","\n  ","\n    "," Rotate. This option will rotate all your station - 3 components (must be named N,E,Z) to the GAC taking the reference the data of event info.","\n  ","\n  ","\n  ","\n    "," Cross. This button will compute the cross-correlation (cc) or the autocorrelation of all processed seismograms with respect the reference “Ref”, the number of the seismogram from top to bottom.","\n  ","\n  ","\n  ","\n    "," Process and Plot. This action will read the seismograms from the selected folder and will carry out the processing from the established “MACRO”. Further details in MACRO section \"Macros\".","\n  ","\n  "," ","\n  ","\n    ","Fig 1. Pick Event framework","\n  ","\n  "," ","\n  ","Additional options from the toolbar are:","\n  "," ","\n  ","\n  ","File","\n  "," ","\n  ","\n  ","File ","à"," New location will clean the picks that are saved automatically for be ready to compute new picking/location.","\n  "," ","\n  ","File ","à"," Write Files will write in the folder you select the processed seismograms.","\n  "," ","\n  ","\n  ","File ","à","Open Settings will open a window with the specific parameters of the subprocess that ","\n  ","you can carry out in this module (sta/lta, wavelet detection, spectrogram, entropy…).","\n  "," ","\n  ","\n  ","Actions","\n  "," ","\n  ","\n  ","Actions ","à"," Macro will deploy a window with all of the processing options. (See section 7). All of the processing options will be applied once you press the button Processing and Plot.","\n  "," ","\n  ","\n  ","Actions ","à"," Magnitude Calculator will open a window in which you can compute different magnitudes. For open this you must have estimated an event and have selected the time-windows in the seismograms that you want the magnitudes be calculated.","\n  "," ","\n  ","\n  ","Actions ","à"," Run Picker will carry out the automatic detection/classification of P- and S- waves from the previously loaded “Neural Network”. You must have three components per stations (named *N,*E,*Z). BE CAREFULL this computationally demanding.","\n  "," ","\n  ","\n  ","Action ","à"," Detect event will associate the automatic picks carried out by the wavelet picker ant will declare an event. The events will be shown in the seismograms window.","\n  "," ","\n  ","\n  ","Action ","à"," Open Earth Model Viewer will open a tool to visualize 3D velocity models. For visualize an Earth Model you need the binary files (*buf and  *hdr) created in the tab Event location. ","\n  "," ","\n  ","\n  ","\n    ","                                               ","Compute","\n  ","\n  "," ","\n  ","\n  ","Compute ","à"," STA/LTA from all processed seismograms and will plot the result together. STA/LTA takes the parameters from Parameters Settings.","\n  "," ","\n  ","\n  ","Compute ","à"," CWT (CF) from all processed seismograms and will plot the result together. CWT takes the parameters from Parameters Settings.","\n  "," ","\n  ","\n  ","Compute ","à"," Envelope from all processed seismograms and will plot the result together. ","\n  "," ","\n  ","\n  ","Compute ","à"," Spectral Entropy from all processed seismograms and will plot the result together. Spectral entropy is estimated in small time windows see Parameters Settings.","\n  "," ","\n  ",".","\n  ","All Seismograms ","à"," Plot together all seismograms.","\n  "," ","\n  ","\n  ","Stack ","à"," Compute (linear, nth-root or PWS) stack of all processed seismograms.","\n  "," ","\n  ","\n  ","\n  ","                                               ","Go","\n  ","\n  "," ","\n  ","Will ship you to the rest of ISP modules (RFs, Array analysis….).","\n  ","\n  ","\n\n","\n  ","\n    ","ISP_Documentation","                                                                                                        ","Page ","1"," of\n      ","1","\n        ","\n    ","\n  ","\n\n"]],"2":[["Pick Event"]],"id":"7"})