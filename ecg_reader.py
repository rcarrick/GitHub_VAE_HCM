import scipy.signal as sig
import numpy as np
from matplotlib import pyplot as plt
import base64
import struct
import wfdb


class EcgClass:
    # catch all ecg class that all file types will be converted to, and will be used for subsequent analyses

    def __init__(self):
        # raw voltage data - time vs leads
        self.voltages = np.array([])
        # frequency of stored data (in Hz)
        self.freq = 0
        # number of included voltage data points along temporal axis
        self.samples = 0
        # total duration of recording in seconds
        self.duration = 0
        # any comments/diagnostic codes
        self.metadata = []
        # number of leads included
        self.num_leads = 0
        # names of the leads
        self.leads = []
        # amplitude scaling
        self.amp_scale = 1
        # locations of QRS peaks
        self.qrs_peaks = np.array([])
        # median beats
        self.median_beat = np.array([])
        # all beats
        self.all_beats = np.array([])
        # number of QRS beats
        self.num_beats = 0
        # median beat duration
        self.samples_median = 0
        # plotting indices
        self.plotting_indices = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
        # filename - just for tracking
        self.filename = ''
        # bad data flag
        self.bad_data = False

    def update_ecg(self, new_duration=10, new_freq=250, median_duration=1.2):
        # this is a standard set of ECG processing commands to unify ECG characteristics

        # crop time of ECG (10s duration)
        if self.duration > new_duration:
            self.crop_duration(new_duration=new_duration)
        if self.duration < new_duration:
            self.bad_data = True

        # Scale the voltages
        self.scale_voltage()

        # Up/downsampling (250 Hz)
        if self.freq != new_freq:
            self.convert_freq(new_freq=new_freq)

        # bandpass filter [0.5, 100] Hz
        self.bandpass_filter(low_pass=100, high_pass=0.5, ripple_db=30, width=1)

        # update the lead names if they are using weird lead names (e.g. as in DICOM files)
        if self.leads[0] != 'I':
            self.correct_lead_names()

        # make sure 12 lead data
        if self.num_leads == 8:
            self.calculate_limb_leads()
        if self.num_leads < 8:
            self.bad_data = True

        # check for NAN voltage data -
        if np.isnan(self.voltages).any():
            self.bad_data = True
            self.bad_data = True

        if self.bad_data == False:
            # find qrs peaks
            self.find_qrs_peaks(low_pass=15.0, high_pass=5.0,filter_order=1, integration_window=150,
                                lockout_window=150, maxqrs_window=150, num_stds=3)
            # this is a double check for very fast rate things - e.g. VT (shouldn't come up frequently)
            if len(self.qrs_peaks) <= 1:
                self.find_qrs_peaks(low_pass=15.0, high_pass=5.0, filter_order=1, integration_window=150,
                                    lockout_window=100, maxqrs_window=100, num_stds=3)
                if len(self.qrs_peaks) <= 1:
                    self.bad_data = True

        if self.bad_data == False:
            # find median beats
            self.find_median_beats(beat_window=median_duration)



    def correct_lead_names(self):

        # this is necessary for DICOM files - but not really any of the other file types

        if len(self.leads) == 0:
            self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2','V3', 'V4', 'V5', 'V6']

        for i in range(self.num_leads):
            if self.leads[i] == 'Lead I (Einthoven)':
                self.leads[i] = 'I'
            elif self.leads[i] == 'Lead II':
                self.leads[i] = 'II'
            elif self.leads[i] == 'Lead III':
                self.leads[i] = 'III'
            elif self.leads[i] == 'Lead aVR':
                self.leads[i] = 'aVR'
            elif self.leads[i] == 'Lead aVL':
                self.leads[i] = 'aVL'
            elif self.leads[i] == 'Lead aVF':
                self.leads[i] = 'aVF'
            elif self.leads[i] == 'Lead V1':
                self.leads[i] = 'V1'
            elif self.leads[i] == 'Lead V2':
                self.leads[i] = 'V2'
            elif self.leads[i] == 'Lead V3':
                self.leads[i] = 'V3'
            elif self.leads[i] == 'Lead V4':
                self.leads[i] = 'V4'
            elif self.leads[i] == 'Lead V5':
                self.leads[i] = 'V5'
            elif self.leads[i] == 'Lead V6':
                self.leads[i] = 'V6'

    def calculate_limb_leads(self):

        temp_voltages = np.zeros((self.samples, 12))

        # migrate the known lead data to the new voltage matrix
        # I, II
        temp_voltages[:, 0:2] = self.voltages[:, 0:2]
        # V1-V6
        temp_voltages[:, 6:] = self.voltages[:, 2:]

        # calculate the missing limb lead data
        # III is II-I
        temp_voltages[:, 2] = temp_voltages[:, 1]-temp_voltages[:, 0]
        # aVR is -(I+II)/2
        temp_voltages[:, 3] = -(temp_voltages[:, 0] + temp_voltages[:, 1])/2
        # aVL is (I-III)/2
        temp_voltages[:, 4] = (temp_voltages[:, 0] - temp_voltages[:, 2])/2
        # aVF is (II+III)/2
        temp_voltages[:, 5] = (temp_voltages[:, 1] + temp_voltages[:, 2])/2

        # fix the lead names
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.voltages = temp_voltages
        self.num_leads = 12

    def convert_freq(self, new_freq=250):
        # up or down sample the ecg to reflect a new sampling frequency
        self.voltages = sig.resample(self.voltages, int(self.duration * new_freq), axis=0)

        # update the number ofr data points and the new sampling frequency
        self.samples = self.duration * new_freq
        self.freq = new_freq

    def crop_duration(self, new_duration=10):
        # crop the duration of the ECG at a new duration (which is in seocnds)

        if self.duration > new_duration:
            # do the actual cropping
            max_time_index = int(self.freq * new_duration)
            self.voltages = self.voltages[0:max_time_index, :]

            self.samples = self.voltages.shape[0]
            self.duration = new_duration
        elif self.duration == new_duration:
            return
        else:
            print('Current duration is already shorter than desired duration')

    def scale_voltage(self):
        # unify the ECG voltages such that 1cm (2 large ECG boxes) is equivalent to 1mV
        self.voltages = self.voltages * self.amp_scale / 1000.

    def bandpass_filter(self, low_pass=100., high_pass=0.5, width=1., ripple_db=30.):

        # perform a bandpass filter using scipy signal package

        # The Nyquist rate of the signal.
        nyq_rate = self.freq / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        width = width / nyq_rate

        # The desired attenuation in the stop band, in dB.
        ripple_db = ripple_db

        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = sig.kaiserord(ripple_db, width)

        # The cutoff frequency of the filter.
        cutoff_hz = np.array([high_pass, low_pass])

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = sig.firwin(N, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero='bandpass')

        # The phase offset of the filtered signal.
        offset = int(0.5 * (N - 1))

        # padd the ecg_signal with zeros due to the phase offset
        ecg_signal = np.concatenate((self.voltages, np.zeros((offset, self.voltages.shape[1]))), axis=0)

        # Use lfilter to filter the ecgsignal with the FIR filter.
        ecg_signal = sig.lfilter(taps, 1.0, ecg_signal, axis=0)

        # return the filtered signal, accounting for the offset
        self.voltages = ecg_signal[offset:]

    def find_qrs_peaks(self, low_pass=15.0, high_pass=5.0, filter_order = 1, integration_window=150,
                       lockout_window=150, maxqrs_window=150, num_stds=3):
        # use the Pan-Tompkins Algorithm to identify QRS peaks
        # low/high pass - filter parameters used in step 1)

        ecg_temp = self.voltages.copy()

        # Step 1)
        # apply narrow bandpass filter using butterworth filter across each lead
        nyquist_freq = 0.5 * self.freq
        low = high_pass / nyquist_freq
        high = low_pass / nyquist_freq
        b, a = sig.butter(filter_order, [low, high], btype="band")
        for lead in range(self.num_leads):
            ecg_temp[:, lead] = sig.lfilter(b, a, ecg_temp[:, lead])

        # Step 2)
        # Derivative - provides QRS slope information.
        ecg_temp = np.diff(ecg_temp, axis=0, append=np.zeros((1, self.num_leads)))

        # Step 3)
        # Squaring - intensifies values received in derivative.
        ecg_temp = ecg_temp ** 2

        # Step 4)
        # moving window for the integration
        integration_window_samples = int(integration_window * self.freq / 1000) # 1000 to convert the 150ms to 0.15s
        convolution_array = np.ones(integration_window_samples)
        # convolve each lead individually to do the integration
        for lead in range(self.num_leads):
            ecg_temp[:, lead] = np.convolve(a=ecg_temp[:, lead], v=convolution_array,mode='same')

        # Step 5)
        # find the peaks (ALL PEAKS ACROSS EACH LEAD)
        all_peaks_ids = np.array([])
        for lead in range(self.num_leads):
            # Find the ECG data that is within 2 stds of the mean
            select_ecg_data = ecg_temp[:, lead] < (np.mean(ecg_temp[:, lead]) + num_stds * np.std(ecg_temp[:, lead]))
            # find the mean of this "non-outlier" data
            select_ecg_data_mean = np.mean(ecg_temp[:, lead][select_ecg_data])
            # pick out the outliers - these should reflect the
            select_ecg_data = (ecg_temp[:, lead] >= select_ecg_data_mean)
            outlier_ids = np.nonzero(select_ecg_data)[0]

            # Find the different times that mark distinct groups of outliers
            outlier_id_time_deriv = np.concatenate(([0], np.diff(outlier_ids)))
            # find the IDs of the IDs that mark the start of a consecutive string (bin) of outliers - provided that this
            # string/bin starts after AT LEAST the lockout/blanking period since the last one
            outlier_id_bin_start_id = np.concatenate(([0], np.nonzero(outlier_id_time_deriv >
                                                                      (lockout_window / 1000 * self.freq))[0]))

            # If we don't identify any bins/strings of outliers this way, move onto the next lead
            if len(outlier_id_bin_start_id) < 2:  #
                continue

            # Assign integer IDs to each of the bins/strings of outliers
            total_bins = 0
            outlier_bins = np.zeros(len(outlier_ids))
            for i, bin_start in enumerate(outlier_id_bin_start_id):
                total_bins += 1
                if i < len(outlier_id_bin_start_id) - 1:
                    outlier_bins[bin_start:outlier_id_bin_start_id[i + 1]] = total_bins
                else:
                    outlier_bins[bin_start:] = total_bins

            # For each of these bins of outliers, we want to find the time index of the peak value
            bin_peak_ids = np.zeros(total_bins)
            for i in range(total_bins):
                ## We look at each given window and find the peak voltage amplitude in the given lead
                current_bin_ids = outlier_ids[outlier_bins == i + 1]
                # choose the peak as the value in the ecg data that has the greatest displacement from the mean value
                bin_peak_ids[i] = current_bin_ids[np.argmax(
                    abs(self.voltages[current_bin_ids, lead]) - np.mean(abs(self.voltages[current_bin_ids, lead])))]
                # Find where the largest peak is in the window
            all_peaks_ids = np.concatenate((all_peaks_ids, bin_peak_ids))

        # Step 6)
        # do a cross check between leads to identify false pos/false neg
        # sort all of the peak ids across leads into ascending order
        all_peaks_ids = np.sort(all_peaks_ids)
        # figure out where there are time jumps
        all_peaks_ids_deriv = np.concatenate(([0], np.diff(all_peaks_ids)))  # Figure out where the new QRS windows are
        # Figure out where the starts of these bins are - does include a lockout/blanking period
        all_peak_bin_start_id = np.concatenate(
            ([0], np.nonzero(all_peaks_ids_deriv > (maxqrs_window / 1000 * self.freq))[0]))
        # Assign integer IDs to each of the bins/strings of QRS peak values
        total_bins = 0
        qrs_bins = np.zeros(len(all_peaks_ids))
        for i, bin_start in enumerate(all_peak_bin_start_id):
            total_bins += 1
            if i < len(all_peak_bin_start_id) - 1:
                qrs_bins[bin_start:all_peak_bin_start_id[i + 1]] = total_bins
            else:
                qrs_bins[bin_start:] = total_bins
        # find the median or mode value of each QRS bin
        qrs_peaks = []
        for i in range(total_bins + 1):
            # after checking that each QRS peak shows up in at least half of the leads
            if np.sum(qrs_bins == i) >= 6:
                # it seems like median beat works a little bit better than mode
                qrs_peaks.append(int(np.median(all_peaks_ids[qrs_bins == i])))

        self.qrs_peaks =  np.array(qrs_peaks)

    def find_median_beats(self, beat_window=1.2, fraction_front=0.4, correlation_window=10):

        # figure out what the Ventricular rate is / RR interval
        rr_interval = (np.median(self.qrs_peaks[1:] - self.qrs_peaks[0:-1]) / self.freq)

        # Check how the RR interval relates to the plotting interval that we are using (if the HR is too fast, we will
        # have to crop the beat)
        if rr_interval > beat_window:
            rr_interval = beat_window
        if rr_interval < beat_window:
            too_short = 1
        else:
            too_short = 0

        # determine the various plotting intervals etc

        # the fraction of the plotting interval before the QRS
        before_duration_true = fraction_front * rr_interval
        # how many samples are in the plotting interval
        total_samples_true = int(np.ceil(rr_interval * self.freq))
        # how many samples are before the QRS peak
        before_samples_true = int(np.ceil(before_duration_true * self.freq))
        # whatever samples aren't before the QRS are after it
        after_samples_true = total_samples_true - before_samples_true
        total_samples_plot = int(np.ceil(beat_window * self.freq))

        # drop beats that are cut off by the edges
        peak_ids = self.qrs_peaks[(self.qrs_peaks - before_samples_true > 0) &
                                  (self.qrs_peaks + after_samples_true < self.samples)]

        # each beat of each lead should be recorded for ultimate cross correlation
        ecg_beats = np.zeros((self.num_leads, total_samples_plot, len(peak_ids)))

        # loop throgh each lead
        for i in np.arange(self.num_leads):
            # loop through each of the peaks that have been previously identified
            for j, peak_id in enumerate(peak_ids):
                # figure out where from the 12 lead this particular lead is going and then put it there
                beat = self.voltages[(peak_id - before_samples_true):(peak_id + after_samples_true), i]
                if too_short:

                    missing = total_samples_plot - total_samples_true
                    missing_before = int(np.ceil(missing * fraction_front))
                    missing_after = missing - missing_before

                    beat = np.concatenate((np.zeros(missing_before), beat, np.zeros(missing_after)), axis=0)
                    ecg_beats[i, :, j] = beat
                else:
                    ecg_beats[i, :, j] = beat

        self.samples_median = ecg_beats.shape[1]

        # correlation between beats
        beat_corrs = np.zeros((len(peak_ids), len(peak_ids)))

        # loop through beat 1
        for beat1 in range(len(peak_ids)):

            # and the other beats previously identified
            for beat2 in range(beat1):

                # correlation array is the correlation between beat 1 and beat 2 across each lead
                correlation_array = np.zeros(self.num_leads)

                for lead in range(self.num_leads):
                    # take the maximum value of the correlation of convolved beat1  vs beat 2;
                    # also taking absolute value to account for the fact that negative signals should still be counted!
                    qrs_1 = standardize_ecg_beat(ecg_beats[lead, :, beat1].copy())
                    qrs_2 = standardize_ecg_beat(ecg_beats[lead, :, beat2].copy())

                    correlation_array[lead] = custom_convolution(qrs_1, qrs_2, padding=correlation_window)
                # use a sum to accumulate the correlations
                beat_corrs[beat1, beat2] = sum(correlation_array)

        # fill in missing correlations
        beat_corrs = beat_corrs + beat_corrs.transpose()

        # take the mean value of correlation
        beat_corrs = np.mean(beat_corrs, axis=0)

        # the median beat is the one with the highest average correlation with all other beats
        median_beat_index = np.argmax([beat_corrs == np.max(beat_corrs)])

        self.qrs_peaks = peak_ids
        self.num_beats = len(peak_ids)

        self.median_beat = np.zeros((self.samples_median, self.num_leads))
        self.all_beats = np.zeros((self.samples_median, self.num_leads, self.num_beats))
        for i in range(self.num_leads):
            self.median_beat[:, i] = ecg_beats[i, :, median_beat_index]

            for j in range(self.num_beats):
                self.all_beats[:, i, j] = ecg_beats[i, :, j]

    def plot_ecg(self, include_peaks=1):

        # Corresponding time series for plotting
        time_series = np.arange(0, self.duration, 1. / self.freq)

        # set up the 12 lead
        fig = plt.figure()
        gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)

        # loop through each lead and plot in the appropriate location in 12 lead
        for i in np.arange(self.num_leads):
            # figure out where from the 12 lead this particular lead is going and then put it there
            plot_index_1 = int(np.remainder(self.plotting_indices[i] - 1, 4.))
            plot_index_2 = int(np.floor((self.plotting_indices[i] - 1) / 4.))
            axs = fig.add_subplot(gs[plot_index_2, plot_index_1])

            # take the chunk of time that is appropriate for which segment of the 12 lead
            # (e.g. I, II, III should have 0-2.5s, V1, V2, V3 should have 5-7.5s).
            if i in [0, 1, 2]:

                data_range = np.arange(0, int(len(time_series) / 4))
                grid_ticks_x = np.arange(0, 2.5, 0.2)
                if len(self.qrs_peaks) != 0:
                    peaks = self.qrs_peaks[(self.qrs_peaks >= 0) & (self.qrs_peaks < self.freq * 2.5)]
            elif i in [3, 4, 5]:
                data_range = np.arange(int(len(time_series) / 4), int(len(time_series) / 2))
                grid_ticks_x = np.arange(2.6, 5., 0.2)
                if len(self.qrs_peaks) != 0:
                    peaks = self.qrs_peaks[
                                (self.qrs_peaks >= self.freq * 2.5) & (self.qrs_peaks < self.freq * 5)] - int(
                        self.freq * 2.5)
            elif i in [6, 7, 8]:
                data_range = np.arange(int(len(time_series) / 2), int(len(time_series) * 3 / 4))
                grid_ticks_x = np.arange(5, 7.5, 0.2)
                if len(self.qrs_peaks) != 0:
                    peaks = self.qrs_peaks[
                                (self.qrs_peaks >= self.freq * 5) & (self.qrs_peaks < self.freq * 7.5)] - int(
                        self.freq * 5)
            elif i in [9, 10, 11]:
                data_range = np.arange(int(len(time_series) * 3 / 4), len(time_series))
                grid_ticks_x = np.arange(7.6, 10., 0.2)
                if len(self.qrs_peaks) != 0:
                    peaks = self.qrs_peaks[(self.qrs_peaks >= self.freq * 7.5)] - int(self.freq * 7.5)
            axs.plot((time_series[data_range]), (self.voltages[min(data_range):(max(data_range) + 1), i]), 'k')
            if len(self.qrs_peaks) != 0:
                if include_peaks:
                    axs.scatter(time_series[data_range][peaks],
                                self.voltages[min(data_range):(max(data_range) + 1), i][peaks], s=20, c='r')
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            axs.set_ylim([-2, 2])
            axs.set_xlim([min(time_series[data_range]), max(time_series[data_range])])
            axs.text(min(time_series[data_range]) + 0.05, 1.6, self.leads[i])
            axs.set_yticks(np.arange(-2, 2, 0.5))
            axs.set_xticks(grid_ticks_x)
            axs.grid(which='both', linestyle='--', alpha=0.25, color='k')
            # axs.axis('off')

        plt.suptitle(self.filename)
        fig.set_size_inches(12, 6)
        plt.tight_layout()

    def plot_median_beat(self, color='k', alpha=1, new_fig=1, old_fig='', plot_all_beats=1):
        # set up the 12 lead
        if new_fig==1:
            fig = plt.figure(figsize=(7, 7))
            gs = fig.add_gridspec(3, 4, hspace=0, wspace=0)
        else:
            fig = old_fig[0]
            gs = old_fig[1]

        for i in np.arange(self.num_leads):
            plot_index_1 = int(np.remainder(self.plotting_indices[i] - 1, 4.))
            plot_index_2 = int(np.floor((self.plotting_indices[i] - 1) / 4.))

            if new_fig:
                axs = fig.add_subplot(gs[plot_index_2, plot_index_1])
            else:

                axs = fig.get_axes()[i]
                plt.sca(axs)
            if plot_all_beats:
                for j in range(self.num_beats):
                    plt.plot(np.arange(0, self.samples_median/self.freq, 1/self.freq), self.all_beats[:, i, j], alpha=0.5, c='r')
            plt.plot(np.arange(0, self.samples_median/self.freq, 1/self.freq), self.median_beat[:, i], c=color, alpha=alpha)
            axs.set_yticks(np.arange(-2, 2, 0.5))
            axs.set_xticks(np.arange(0, self.samples_median/self.freq, 0.2))
            axs.grid(which='both', linestyle='--', alpha=0.25, color='k')
            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
            plt.xlim([0, self.samples_median/self.freq])
            plt.ylim([-2, 2])
            plt.text(0.05, 0.95, self.leads[i], horizontalalignment='left', verticalalignment='top',
                     transform=axs.transAxes)
        plt.suptitle(self.filename)
        plt.tight_layout()
        return fig, gs


def empty_ecg(num_leads=12, samples_median=300, freq=250):

    ecg = EcgClass()
    ecg.num_leads = num_leads
    ecg.samples_median = samples_median
    ecg.freq = freq
    ecg.correct_lead_names()

    return ecg


def extract_ecg_wfdb(filename, duration=10, freq=250, median_duration=1.2):
    # These should be interchangable functions designed for use with different ECG file formats
    # input - filepath to use to load the ecg data; should NOT have the file extension included

    # generate our ECG class
    ecg = EcgClass()

    # load the wfdb data
    record = wfdb.rdrecord(filename)
    header = wfdb.rdheader(filename)

    # pull out the relevant data and parameters
    ecg.voltages = record.p_signal
    ecg.freq = header.fs
    ecg.samples = header.sig_len
    ecg.duration = ecg.samples / ecg.freq
    ecg.metadata = header.comments[2]
    ecg.amp_scale = header.adc_gain
    ecg.num_leads = header.n_sig
    ecg.leads = header.sig_name
    ecg.filename = filename + '.hea'

    # some of the  files do NOT have all the leads with complete voltage data
    if any(np.max(np.abs(ecg.voltages[0:int(ecg.freq * ecg.duration / 2.), :]), axis=0) == 0):
        # if any of the leads have no data for the first half of the recording... you know it must be bad data
        ecg.bad_data = True

    # update the ecg
    ecg.update_ecg(new_duration=duration, new_freq=freq, median_duration=median_duration)

    return ecg


def extract_ecg_xml_muse(filename, duration=10, freq=250, median_duration=1.2):

    ecg = EcgClass()

    import xml.etree.ElementTree as ET

    # load the dcm data
    if filename[-4:] == '.xml':
        tree = ET.parse(filename)
    else:
        tree = ET.parse(filename + '.xml')
    root = tree.getroot()

    lead_counter = 0
    record_trigger = 0

    for item in root.findall('./Diagnosis/'):
        for child in item:
            if child.tag == 'StmtText':
                ecg.metadata.append(child.text)

    for item in root.findall('./Waveform/'):

        # we only want to record 'RHYTHM' type data
        if (item.tag == 'WaveformType') & (item.text == 'Rhythm'):
            record_trigger = 1

        if record_trigger:
            # go through the header and pull out relevant pieces of information
            if item.tag == 'NumberofLeads':
                ecg.num_leads = int(item.text)

            elif item.tag == 'SampleBase':
                ecg.freq = float(item.text)

            elif item.tag == 'LeadData':
                for child in item:

                    if (child.tag == 'LeadSampleCountTotal') & (ecg.samples == 0):
                        ecg.samples = int(child.text)
                        ecg.voltages = np.zeros((ecg.samples, ecg.num_leads))

                    elif child.tag == 'LeadID':
                        ecg.leads.append(child.text)
                    elif (child.tag == 'LeadAmplitudeUnitsPerBit') & (ecg.amp_scale == 1):
                        ecg.amp_scale = float(child.text)
                    elif child.tag == 'WaveFormData':
                        ecg.voltages[:, lead_counter] = convert_base64encoded_to_short(child.text)
                        lead_counter += 1

    ecg.duration = ecg.samples / ecg.freq
    ecg.filename = filename + '.xml'

    # some of the  files do NOT have all the leads with complete voltage data
    if any(np.max(np.abs(ecg.voltages[0:int(ecg.freq * ecg.duration / 2.), :]), axis=0) == 0):
        # if any of the leads have no data for the first half of the recording... you know it must be bad data
        ecg.bad_data = True

    ecg.update_ecg(new_duration=duration, new_freq=freq, median_duration=median_duration)

    return ecg


def standardize_ecg_beat(ecg_beat):

    # apply standardization of an ecg beat for use with convolution/correlation

    ecg_beat = np.abs(ecg_beat)-np.min(np.abs(ecg_beat))
    ecg_beat = ecg_beat / (sum(ecg_beat))

    return ecg_beat


def custom_convolution(qrs1, qrs2, padding=10):

    # a custom convolution function that matches one QRS against another

    qrs2 = np.concatenate((np.zeros(padding), qrs2, np.zeros(padding)))
    if padding == 0:
        correlation = 1-sum(np.abs(qrs1-qrs2))
    else:
        correlation_array = np.zeros(padding*2)
        for i in range(padding*2):
            correlation_array[i] = 1-(sum(np.abs(qrs1-qrs2[i:len(qrs1)+i])))
        correlation = np.max(correlation_array)

    return correlation


def convert_base64encoded_to_short(base64_string):
    # this function takes a string of base64 encoded data, and converts it to an array of 'short' values

    # INPUT
    # base64_string: a string of base64 encoded data

    # OUTPUT
    # short_values: array of short value data points

    # determine how many samples are included in the data series (base64-->byte is a 4 to 3 ratio for length, then there
    # are two bytes per data point)

    byte_string = base64.b64decode(base64_string)
    sample_count = int(len(byte_string)/2)

    # unpack takes a format string as well as a decoded byte series, returns signed shorts
    short_values = np.array(struct.unpack('<'+'h'*sample_count, byte_string))

    return short_values