import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.signal import chirp, spectrogram, find_peaks, firwin, filtfilt, hilbert, resample
from scipy.optimize import curve_fit
import librosa as lbr
import scipy as sp
from numpy.linalg import inv
from scipy import signal
import plotly.graph_objects as go
import numba
from numba import jit, vectorize
import soundfile as sf
import samplerate
from scipy.io import wavfile
from shapely.geometry import Polygon
from shapely import Point
import json
with open('config.json') as json_file:
    config = json.load(json_file)

path_fig = 'result_figs\\'


def create_sinewave(freq, duration, fs, amplitude, offset):
    """
    Creates the sine wave signal as numpy array
    :param freq: frequency of the sine wave
    :param duration: duration of the sine wave in s
    :param fs: sample frequency
    :return: w: the signal in np array
    """
    t = np.linspace(0., int(duration), int(fs * duration))
    w = (amplitude * np.sin(2. * np.pi * freq * t))+offset
    # Set first and last bit to zero (interesting when using a DAQ)
    #w[0]=0
    #w[len(w)-1]=0
    return w

def create_chirp(start_freq, stop_freq, chirp_duration, fs, amplitude, offset):
    """
    Creates the chirp signal as a numpy array
    :param start_freq: start frequency of the chirp
    :param stop_freq: stop frequency of the chirp
    :param chirp_duration: duration of the chirp in s
    :param fs: sample frequency
    :return: w: the chirp signal in np array
    """
    t = np.linspace(0., chirp_duration, int(fs * chirp_duration))
    w = (amplitude * chirp(t, f0=start_freq, f1=stop_freq, t1=chirp_duration, method='linear', phi=270))+offset
    w[0]=0
    w[len(w)-1]=0
    return w

def write_to_WAV(filename, fs, signal):
    """
    Write to a WAV file
    :param filename: the name of the WAV file .wav
    :param fs: sample frequency
    :param signal: signal as numpy array
    """
    write(filename, fs, signal.astype(np.int16))

def get_speed_of_sound(T):
    v = 20*np.sqrt(273+T)
    return v

def create_location_grid(distance_boundary, roomdim, nx, ny, nz):
    """
    Creates a grid of points/coordinates in 3D space
    :param distance_boundary: value difference in m from borders
    :param roomdim: dimensions of the room in matrix
    :param nx: amount of x positions in grid
    :param ny: amount of y positions in grid
    :param nz: amount of z positions in grid
    :return: grid of coordinates
    """
    speaker_x_coords = np.linspace(distance_boundary, roomdim[0] - distance_boundary, nx)
    speaker_y_coords = np.linspace(distance_boundary, roomdim[1] - distance_boundary, ny)
    speaker_z_coords = np.linspace(distance_boundary, roomdim[2] - distance_boundary, nz)
    x_pos, y_pos, z_pos = np.meshgrid(speaker_x_coords, speaker_y_coords, speaker_z_coords)
    x_pos_flatten, y_pos_flatten, z_pos_flatten = x_pos.flatten(), y_pos.flatten(), z_pos.flatten()
    grid = np.array([x_pos_flatten, y_pos_flatten, z_pos_flatten]).T
    return grid


def create_location_grid_non_shoebox(vertices, height, distance_boundary, nx, ny, nz):
    """
    Creates a grid of points/coordinates in 3D space
    :param vertices: the coordinates of the ground plan
    :param height: the height of the room
    :param distance_boundary: at which distance the points need to be at minimum from the walls, ceiling and floor
    :param nx: amount of x positions in grid
    :param ny: amount of y positions in grid
    :param nz: amount of z positions in grid
    :return: grid of coordinates
    """

    polygon = Polygon(vertices)
    inner_polygon = polygon.buffer(-distance_boundary)
    bounding_box = inner_polygon.bounds

    out_count = 0
    all_positions = np.empty((1, 3))

    x_min, y_min, x_max, y_max = bounding_box
    speaker_x_coords = np.linspace(x_min, x_max, nx)
    speaker_y_coords = np.linspace(y_min, y_max, ny)
    speaker_z_coords = np.linspace(distance_boundary, height - distance_boundary, nz)
    x_pos, y_pos, z_pos = np.meshgrid(speaker_x_coords, speaker_y_coords, speaker_z_coords)
    x_pos_flatten, y_pos_flatten, z_pos_flatten = x_pos.flatten(), y_pos.flatten(), z_pos.flatten()
    grid = np.array([x_pos_flatten, y_pos_flatten, z_pos_flatten]).T

    for position in range (0, np.size(grid, axis=0)):
        x, y , z = grid[position][0], grid[position][1], grid[position][2]
        point = Point(x, y)

        # Returns True if the boundary or interior of the object
        # intersect in any way with those of the other.
        if inner_polygon.intersects(point):
            # Point is inside the shape
            # Generate a random Z coördinate
            coord = np.array([x, y, z])
            all_positions = np.vstack((all_positions, coord))

        else:
            # Point is outside the shape, do not add
            out_count += 1

    all_positions = np.delete(all_positions, 0, 0)
    return all_positions, out_count

def create_random_positions_shoebox(room_dim, n_positions, dist_from_planes):
    """
    Create a random point cloud in a shoebox space given the dimensions of the box and amount of points
    :param room: x, y and z dimensions
    :param n_positions: amount of positions to generate
    :param dist_from_planes: minimum distance from walls, ceiling and floor
    :return: train_dev_set: the random positions set
    """
    x_min, x_max = dist_from_planes, room_dim[0]-dist_from_planes
    y_min, y_max = dist_from_planes, room_dim[1]-dist_from_planes
    z_min, z_max = dist_from_planes, room_dim[2]-dist_from_planes

    # Generate random x, y, and z values within the dimensions of the shoebox
    x = [np.random.uniform(x_min, x_max) for _ in range(n_positions)]
    y = [np.random.uniform(y_min, y_max) for _ in range(n_positions)]
    z = [np.random.uniform(z_min, z_max) for _ in range(n_positions)]
    pos = np.array([x, y, z])
    train_dev_set = pos.T
    return train_dev_set

def numpy_to_tuple_list(arr):
    '''
    go from [[,],[,]] to [(,),(,)]
    :param arr: numpy array
    :return: tuple list
    '''
    tuple_list = [(row[0], row[1]) for row in arr.tolist()]
    return tuple_list

def tobinairy(signal):
    """
    Create a binary array from given array
    :param signal: numpy array
    :return: binary array
    """
    signal[signal > 0] = 1
    signal[signal <= 0] = -1
    return signal

def stack(A, B):   #ADDED FOR MIC SPEAKER SWITCHING
    '''
    Stacks 2 arrays even if they are not equal in length, A can be a 2 dim vector (matrix), B has to be a 1-d numpy array
    :param A:
    :param B:
    :return:
    '''
    sizeB = np.size(B, axis=0)

    if A.ndim > 1:
        x_sizeA, y_sizeA = np.size(A, axis=1), np.size(A, axis=0)
    else:
        x_sizeA, y_sizeA = np.size(A), 1

    if x_sizeA > sizeB:
        dif = x_sizeA-sizeB
        zero_array = np.zeros(dif)
        longerB = np.append(B, zero_array)
        stacked = np.vstack((A, longerB))

    elif x_sizeA < sizeB:
        dif = sizeB-x_sizeA
        if A.ndim > 1:
            zero_array = np.zeros([y_sizeA, dif])
        else:
            zero_array = np.zeros(dif)
        longerA = np.hstack((A, zero_array))
        stacked = np.vstack((longerA, B))

    elif x_sizeA == sizeB:
        stacked = np.vstack((A,B))

    return stacked

def create_random_positions_in_random_3D_space(vertices, height, distance_boundary, n_points):
    """
    Generates random positions inside a random room given its vertices
    :param vertices: the coordinates of the ground plan
    :param height: the height of the room
    :param distance_boundary: at which distance the points need to be at minimum from the walls, ceiling and floor
    :param n_points: amount of datapoints to generate
    :return:
    """
    polygon = Polygon(vertices)
    inner_polygon = polygon.buffer(-distance_boundary)
    bounding_box = inner_polygon.bounds

    all_positions = np.empty((1,3))
    out_count = 0
    while np.size(all_positions, axis=0)-1<n_points:
        x_min, y_min, x_max, y_max = bounding_box
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)

        # Returns True if the boundary or interior of the object
        # intersect in any way with those of the other.
        if inner_polygon.intersects(point):
            # Point is inside the shape
            # Generate a random Z coördinate
            z = np.random.uniform(distance_boundary, height-distance_boundary)
            coord = np.array([x, y, z])
            all_positions = np.vstack((all_positions, coord))

        else:
            # Point is outside the shape, do not add
            out_count +=1

    all_positions = np.delete(all_positions, 0, 0)
    return all_positions, out_count

def edges_from_vertices_2D(vertices):
    """
    Creates xy plane edges (2D projection or ground plan of the room)
    :param vertices: the coordinates of the room
    :return: the edges of the room
    """
    edges = []
    for i in range(len(vertices)):
        edges.append((vertices[i], vertices[(i+1) % len(vertices)]))
    return edges


def calc_n_dev_train_set(test_set, train_set_ratio, dev_set_ratio, test_set_ratio):
    """
    Calculates amount of the training and dev set points, considering the given test set,
    :param test_set: The grid with positions for the test set (also for plotting)
    :param train_set_ratio: ratio of amount of samples in training set VS total set
    :param dev_set_ratio: ratio of amount of samples in dev set VS total set
    :param test_set_ratio: ratio of amount of samples in test set VS total set
    :return: a matrix with the training and dev positions
    """
    train_dev_ratio = train_set_ratio+dev_set_ratio
    n_train_dev_points = int(np.size(test_set, axis=0)*(train_dev_ratio/test_set_ratio))

    return n_train_dev_points

    # if shoebox:
    #     train_dev_set = create_random_positions_shoebox(roomdims, n_train_dev_points, dist_from_planes)
    # else:
    #     vertices = numpy_to_tuple_list(roomdims[0])
    #     height = roomdims[1]
    #     train_dev_set = create_random_points_in_random_3D_space(vertices, height, dist_from_planes, n_train_dev_points)
    # return train_dev_set

def plot_spectrogram_wake_up(title, signal, fs, NFFT, noverlap):
    """
    Show spectogram of the audio signal
    :param title: title of the plot
    :param signal: audio signal
    :param fs: sampling frequency
    :param NFFT: NFFT from plt.specgram
    :param noverlap: noverlap from plt.specgram
    """
    plt.specgram(x=signal, Fs=fs, NFFT=NFFT, noverlap=noverlap)
    plt.title(title)
    plt.xlabel('t (sec)')
    plt.ylabel('Frequency (Hz)')
    #plt.grid()
    plt.show()

def plot_spectrogram(title, signal, fs, dur_orig_sig, delay):
    """
    Show spectogram of the audio signal
    :param title: title of the plot
    :param signal: audio signal
    :param fs: sampling frequency
    :param dur_orig_sig: duration of the original transmitted signal
    :param delay: the delay of the simulation
    """
    ff, tt, Sxx = spectrogram(signal, fs=fs, nperseg=256, nfft=576)
    #c=plt.pcolormesh(tt, ff[:145], Sxx[:145], cmap='Dark2_r', shading='auto')
    c=plt.pcolormesh(tt, ff[:145], Sxx[:145], shading='gouraud')
    plt.title(title)
    #plt.colorbar(c)
    plt.xlabel('t (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim([delay, 1.5*dur_orig_sig])
    plt.ylim([22000, 47000])
    plt.grid()
    plt.show()

def plot_audio_signal(title, signal, fs, dur_orig_sig, delay):
    """
    Show the audio signal
    :param title: title of the plot
    :param signal: audio signal
    :param fs: sampling frequency
    :param dur_orig_sig: duration of the original transmitted signal
    :param delay: the delay of the simulation
    """
    # audio: signal, fs: sample freq, m: mic nr, s: speaker nr, delay: delay of simulation
    x_signal = np.arange(0, np.size(signal), 1) / fs
    plt.plot(x_signal, signal)
    plt.title(title)
    plt.ylabel('Sound level')
    plt.xlabel('Time [s]')
    plt.xlim([delay, 4*dur_orig_sig])
    plt.grid()
    plt.show()

def plot_audio_signal_and_envelope(title, signal, envelope, fs, dur_orig_sig, delay):
    """
    Show the audio signal with envelope
    :param title: title of the plot
    :param signal: audio signal
    :param fs: sampling frequency
    :param dur_orig_sig: duration of the original transmitted signal
    :param delay: the delay of the simulation
    """
    # audio: signal, fs: sample freq, m: mic nr, s: speaker nr, delay: delay of simulation
    x_signal = np.arange(0, np.size(signal), 1) / fs
    plt.plot(x_signal, signal, label='Audio signal')
    plt.plot(x_signal, envelope, label='Envelope')
    plt.title(title)
    plt.legend()
    plt.ylabel('Sound level')
    plt.xlabel('Time [s]')
    plt.xlim([delay, 1.5*dur_orig_sig])
    plt.grid()
    plt.show()


def plot_RIR(title, rir, fs):
    """
    Show the RIR
    :param title: title of the plot
    :param rir: RIR
    :param fs: sampling frequency
    """
    max = np.argmax(rir)
    plt.plot(rir)
    plt.grid()
    plt.title(title)
    plt.xlim([max-50, max+50])
    plt.ylabel('RIR')
    plt.xlabel('Time [s]')
    plt.show()

# def plot_RIR(title, rir, fs):
#     """
#     Show the RIR
#     :param title: title of the plot
#     :param rir: RIR
#     :param fs: sampling frequency
#     """
#     plt.plot(np.arange(len(rir)) / fs, rir)
#     plt.grid()
#     plt.title(title)
#     plt.ylabel('RIR')
#     plt.xlabel('Time [s]')
#     plt.show()

def loopRIRplot(path_rirs, position_nr, fs_source):
    """
    Loop over rirs and plot them
    :param path_Rirs: location of rir files
    :param position_nr: position number
    :param fs_source: source sample frequency
    :return: plot of the rirs
    """
    for rx in range(config['n_mics']):
        for sp in range(config['n_speakers']):
            rir = np.load(path_rirs + "rir" + str(sp) + "source" + str(rx) + "mic_position" + str(position_nr) + ".npy")
            plot_RIR(title="RIR at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position " +
                           str(position_nr), rir=rir, fs=fs_source)


def plot_corr(title, corr, x_fit, y_fit):
    """
    Show the correlation graph with fitting at certain place
    :param title: title of the plot
    :param corr: correlation values
    """
    plt.plot(corr)
    plt.plot(x_fit, y_fit)
    plt.grid()
    plt.title(title)
    plt.ylabel('Correlation')
    plt.xlabel('Samples')
    # plt.xlim(71000,74000)
    # plt.ylim(485.,490)
    plt.show()

def plot_corr_easy(title, corr):
    """
    Show the correlation graph without fitting
    :param title: title of the plot
    :param corr: correlation values
    """
    plt.plot(corr)
    plt.grid()
    plt.title(title)
    plt.ylabel('Correlation')
    plt.xlabel('Samples')
    # plt.xlim(43500,43700)
    # plt.ylim(0.99,1.01)
    plt.show()

def plot_corr_LPF(title, corr, y_LPF, index_distance=None):
    """
    Show the pulse compression + LPF curve
    :param title: title of the plot
    :param corr: correlation values
    """
    plt.plot(corr, label='Correlation Values')
    plt.plot(y_LPF, label='LPF Values')
    #plt.vlines(x=index_distance, ymin=0, ymax=1.1, colors='brown', alpha=0.4)
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.ylabel('Correlation')
    plt.xlabel('Samples')
    # plt.xlim(20570,20580)
    # plt.ylim(0.996,1.001)
    plt.show()

def plot_corr_LPF_peaks(title, corr, y_LPF, x_peaks, peaks, contour_heights, index_distance):
    """
    Show the pulse compression + LPF + peaks and their prominences
    :param title: title of the plot
    :param corr: correlation values
    :param y_LPF: LPF values
    :param peaks: y_values_peaks
    :param x_peaks: x_values_peaks
    :return: plot
    """
    plt.plot(corr, label='Correlation Values')
    plt.plot(y_LPF, label='LPF Values')
    plt.plot(x_peaks, peaks , "x")
    plt.vlines(x=x_peaks, ymin=contour_heights, ymax=peaks, colors='red', label='Peak prominences')
    plt.vlines(x=index_distance, ymin=0, ymax=1.1, colors='brown', alpha=0.4, label='Selected peak')
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.ylabel('Correlation')
    plt.xlabel('Samples')
    # plt.xlim(20570,20580)
    # plt.ylim(0.996,1.001)

    plt.show()

def calc_distance_3D(x, y, z, xc, yc, zc):
    """
    Calculates the distance between 2 points in 3D
    :param x: coordinate point 1
    :param y: coordinate point 1
    :param z: coordinate point 1
    :param xc: coordinate point 2
    :param yc: coordinate point 2
    :param zc: coordinate point 2
    :return:
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)

def easy_peak_determination(corr_val):
    """
    Determines the peak value by searching the max value
    :param corr_val: the normalized correlation values
    :param margin: amount of samples needed for fitting date, left and right from maximum value
    :return: index: index of peak value and fitting curve values
    """
    # Determine index of the max value
    max_corr_index = np.argmax(corr_val)
    return max_corr_index

def peak_determination(fit_func, corr_val, margin):
    """
    Determines the peak value by fitting a function over it at a certain place around the maximum
    :param fit_func: used function for fitting
    :param corr_val: the normalized correlation values
    :param margin: amount of samples needed for fitting date, left and right from maximum value
    :return: index: index of peak value and fitting curve values
    """
    # Determine index of the max value
    max_corr_index = np.argmax(corr_val)

    # amount of samples needed for fitting data, left and right from maximum value
    x_vals = np.arange(0, np.size(corr_val), 1)
    selected_x = x_vals[max_corr_index-int(margin):max_corr_index+int(margin)+1]
    selected_y = corr_val[max_corr_index-int(margin):max_corr_index+int(margin)+1]

    # Select all peaks to have fitting data
    peaks_index, _ = find_peaks(x=selected_y)

    # Recalculate peaks to data index
    y_peaks_index = peaks_index + (max_corr_index-int(margin))

    # Select y peak values from data on index values
    y_peaks_values = corr_val[y_peaks_index]

    popt, vars = curve_fit(fit_func, peaks_index, y_peaks_values)

    # Create fit_function values
    func = fit_func(peaks_index, *popt)

    # Search index from (new) maximum
    index_opt_fit = np.argmax(func)
    index_opt_selected_part = peaks_index[index_opt_fit]
    index_opt_general = index_opt_selected_part + (max_corr_index-margin)
    return index_opt_general, y_peaks_index, func

def one_peak_determination_LPF(corr_val, LPF_function):
    """
    Determines the peak value by searching the max value
    :param corr_val: the normalized correlation values
    :param margin: amount of samples needed for fitting date, left and right from maximum value
    :return: index: index of peak value and fitting curve values
    """
    # Select all peaks to use later for mapping from max to most close peak
    peaks_index, _ = find_peaks(x=corr_val)

    # Determine index of the max value from LPF function
    max_corr_index = np.argmax(LPF_function)

    # Find closed match with peak indexes
    eucl_dist = np.abs(peaks_index-max_corr_index)
    index = np.argmin(eucl_dist)
    max_corr_index_mapped = peaks_index[index]

    return max_corr_index_mapped

def idx_peak_determination_PP(corr_val, max_corr_index):
    """
    Determines the peak index value after already selecting the most prominent peak
    :param corr_val: original correlation values
    :param max_corr_index: the selected index value for the peak from LPF curve
    :return: index: index of peak value mapped after maximum of fitting curve value
    """
    # Select all peaks to use later for mapping from max to most close peak
    peaks_index, _ = find_peaks(x=corr_val)

    # Find closed match with peak indexes
    eucl_dist = np.abs(peaks_index-max_corr_index)
    index = np.argmin(eucl_dist)
    max_corr_index_mapped = peaks_index[index]

    return max_corr_index_mapped


def LPF(x, typeF, order, cutoff, fs):
    """
    Function to add LPF
        x: input signal
        typeF: filter type ('bandpass', 'lowpass', 'highpass', 'bandstop')
        order: length/order of the filter
        cutoff: cutoff frequency
        fs: sample frequency
        return xf: the filtered signal
    """

    # Use signal.firwin to generate the filter coefficients
    b = firwin(order, cutoff, pass_zero=typeF, fs=fs)

    # Use signal.filtfilt to filter x
    xf = 2 * filtfilt(b, 1, x)

    # Adjust gain
    xf_gained = (np.max(x)/np.max(xf))*xf

    return xf_gained


def downsample_LPF(signal, n_samples):
    """
    Function to downsample the LPF curve
        LPF: orignal LPF curve
        n_samples: amount of samples for output array
        return signal_down: downsampled LPF curve
    """

    ratio =float(n_samples)/float(np.size(signal))

    if ratio != 0:
        signal_down = samplerate.resample(signal, ratio, "sinc_best")

    else:  # in the sample rates are identical, no up/down sampling is required
        signal_down = signal

    return signal_down


def create_rx_audio_matrix(path_audio, position_nr):
    '''
    create matrix with all the received audio fragments for 1 position
    :param n_mics: amount om microphones
    :param n_speakers: amount of speakers
    :param path_audio: path to audio wav files
    :param position_nr: number of the position
    :return:
    '''
    # get length of mic signal
    sig, fs_mic = lbr.load(path_audio + "Received_signal_of_the_0th_mic_from_0th_source_position_"+str(position_nr)+".wav", sr=None)
    rx_mic_sig = np.empty(np.size(sig))

    for rx in range(config['n_mics']):
        for sp in range(config['n_speakers']):
            rx_mic_path = path_audio + "Received_signal_of_the_"+str(rx)+"th_mic_from_"+str(sp)+"th_source_position_"+str(position_nr)+".wav"
            # Read observed/received signal
            sig_mic, _ = lbr.load(rx_mic_path, sr=None)  # preserve native sampling rate of file via sr = None
            rx_mic_sig = np.vstack((rx_mic_sig, sig_mic))
    rx_mic_sig = np.delete(rx_mic_sig, 0, 0)

    # Plot the received audio signals
    index = 0
    if config['plot_audio']:
        for rx in range(config['n_mics']):
            for sp in range(config['n_speakers']):
                signal = np.array(rx_mic_sig[index,:])
                plot_audio_signal(title="RX signal at the "+str(rx)+"th mic from the "+str(sp)+"th speaker, position "+str(position_nr),
                                  signal=signal, fs=fs_mic, dur_orig_sig=config['chirp_duration'], delay=0)
                index += 1

    return rx_mic_sig

def add_AWGN(rx_mic_sig, snr, position_nr, fs_mic, path_audio_awgn):
    """
    Add AWGN to received microphone signals defined for mic 0 and interpolated to others
    :param rx_mic_sig: matrix with received microphone values
    :param snr: snr value in dB
    :param position_nr: number of observed position
    :param fs_mic: sample frequency at microphone
    :param path_audio_awgn: path to save awgn audio
    :return: matrix with mic signals with AWGN, SNR value  per mic
    """
    # https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
    # SNR = 10 log (RMS_signal^2/RMS_noise^2)
    awgn = create_AWGN(rx_mic_sig[0, :], snr)
    rx_mic_sig = rx_mic_sig + awgn

    # Get back SNR values at every microphone
    SNR_val = getSNR(awgn, rx_mic_sig)

    index = 0

    # plot and save
    for rx in range(config['n_mics']):
        for sp in range(config['n_speakers']):
            signal = np.array(rx_mic_sig[index, :])
            if config['plot_audio']:
                plot_audio_signal(
                    title="RX signal at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position " + str(
                        position_nr) + " with AWGN",
                    signal=signal, fs=fs_mic, dur_orig_sig=config['chirp_duration'], delay=0)
            index += 1
            if config['save_rx_audio_with_noise']:
                wavfile.write(path_audio_awgn + 'Received_audio_with_awgn_mic' + str(rx) + '_speaker_' + str(sp) + 'position' + str(position_nr) + '.wav', rate=fs_mic, data=signal.astype(np.int16))

    return rx_mic_sig, SNR_val

def add_interference(interference_signal, SIR, rx_mic_sig, fs_mic, position_nr, path_audio_sir):
    """
    Add interference signal to audio
    :param interference_signal: the interference signal
    :param SIR: the SIR value in dB
    :param rx_mic_sig: the received microphone signals
    :param fs_mic: sample frequency at microphone
    :param position_nr: position number
    :param path_audio_sir: path to save audio files
    :return: rx_mic_sig: matrix with audio files with SIR
    """
    # safe blank first received to calculate SIR later
    first_blanc = rx_mic_sig[0, :]

    # Read audio file of interference noise
    sound_in, fs_in = lbr.load(interference_signal, sr=None)
    time = lbr.get_duration(y=sound_in, sr=fs_in)
    print("Sample from interference noise file: %.2f Hz" % fs_in)

    # resample
    duration_signal = lbr.get_duration(y=first_blanc, sr=fs_mic)
    interf_n = create_interference_noise(first_blanc, SIR, fs_mic, interference_signal)

    # Add noise to signal
    rx_mic_sig = rx_mic_sig + interf_n

    index = 0

    for rx in range(config['n_mics']):
        for sp in range(config['n_speakers']):
            signal = np.array(rx_mic_sig[index, :])
            if config['plot_audio']:
                plot_audio_signal(
                    title="RX signal at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position " + str(position_nr)
                          + " with AWGN and interference noise",
                    signal=signal, fs=fs_mic, dur_orig_sig=config['chirp_duration'], delay=0)
            index += 1
            wavfile.write(path_audio_sir + 'Received_audio_with_awgn_and_interf_mic' + str(rx) + '_speaker_' + str(
                sp) + 'position' + str(position_nr) + '.wav', rate=fs_mic, data=signal.astype(np.int16))

    return rx_mic_sig


def amplitude_envelope(input_signal, axis=-1):
    """Uses a hilbert transform to determine the envelope.
    :param signal: Signal.
    :param axis: Axis.
    :returns: Amplitude envelope of `signal`.
    .. seealso:: :func:`scipy.signal.hilbert`
    """
    return np.abs(hilbert(input_signal, axis=axis))

def poly_area(x,y):
    """
    Calculates the surface/area enclosed by a set of corner coordinate points, based on the shoelace formula
    :param x: x-coordinates
    :param y: y-coordinates
    :return: area
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def rms(x, axis=0):
    """
    Calculates the rms value of a signal
    :param x: the signal
    :return: the rms value
    """
    return np.sqrt(np.mean(x**2, axis=axis, keepdims=True))


def create_AWGN(signal,SNR):
    """
    Given a signal and desired SNR, this gives the required AWGN what should be added to
    the signal to get the desired SNR.
    :param signal:
    :param SNR:
    :return:
    """
    #RMS value of signal
    RMS_s = rms(signal)

    #Needed RMS values of noise for certain SNR value
    RMS_n = np.sqrt((RMS_s**2)/(10**(SNR/10)))

    #Additive white gausian noise. mean=0
    #Because sample length is large (typically > 40000),
    # we can use the population formula for standard deviation.
    # because mean=0 STD=RMS
    STD_n = RMS_n
    noise = np.random.normal(0, STD_n, signal.shape)
    if np.mean(noise)==0: noise = np.random.normal(0, 1e-6, signal.shape)  #To avoid devide by 0 error if no signal is captured
    return noise

def getSNR(awgn, rx_mic_sig):
    rms_awgn = rms(awgn)
    rms_sig = rms(rx_mic_sig, axis=1)
    SNR = 10*np.log10(rms_sig**2/rms_awgn**2)
    return SNR

def create_interference_noise(signal_speaker, SIR, fs_source, interference_signal):
    """
    Creates the right interference noise value based on a wav file as input noise
    :param signal_speaker: received signal of interesest (e.g. chirp from speaker)
    :param SIR: Signal interference ratio
    :param fs_source: samplefreq of source signal (e.g. chirp)
    :param interference_signal: wav file with interference noise
    :return: noise amplified according to SIR level and signal speaker received at microphone
    """
    interf_sound, fs_interference = lbr.load(interference_signal, sr=None)

    # resample interference sound until same sample frequency as signal
    interf_sound_resampled = resample(interf_sound, int((fs_source/fs_interference)*np.size(interf_sound)))

    # Select samples in interference signal
    n_s = np.size(signal_speaker)
    interf = interf_sound_resampled[5000:5000+n_s]

    #RMS value of signal
    RMS_s = rms(signal_speaker)

    #RMS value of interference
    RMS_i = rms(interf)

    # Needed RMS values of noise for certain SIR value
    RMS_n = np.sqrt((RMS_s ** 2) / (10 ** (SIR / 10)))

    # amplification
    A = RMS_n/RMS_i
    interf_noise = A*interf

    return interf_noise

def resample_source_to_mic(fs_source, fs_mic, chirp_orig):
    """
    Downsample original chirp signal if fs source =/= fs mic
    :param fs_source: sample frequency at speaker
    :param fs_mic: sample frequency at receiver
    :param chirp_orig: original chirp signal
    :return: chirp_orig_resampl: resampled original chirp signal
    """
    if fs_source != fs_mic:
        fs_ratio = float(fs_mic) / float(fs_source)
        new_length = int(fs_ratio * chirp_orig.shape[0])
        re_sampled_orig_chirp_signal = np.zeros((1, new_length))

        # why sinc: http://www.mega-nerd.com/SRC/api_misc.html#ErrorReporting
        re_sampled_orig_chirp_signal = samplerate.resample(chirp_orig, fs_ratio, "sinc_best")
        chirp_orig_resampl = re_sampled_orig_chirp_signal

    else:  # in the sample rates are identical, no up/down sampling is required
        chirp_orig_resampl = chirp_orig

    np.save('chirp_orig_not_or_resampled.npy', chirp_orig_resampl)

    return chirp_orig_resampl

def select_wakeup_part(fs_mic, chirp_orig_resampl, wake_up_duration, rx_mic_sig):
    """
    Select the audio part received in wake up period
    :param fs_mic: sample frequency of microphone
    :param chirp_orig_resampl: original chirp signal with the wright sample frequency
    :param wake_up_duration: duration of the wake up time
    :param rx_mic_sig: the audio fragments
    :return: rx_audio_wake: the selected audio part in wake up mode
    """
    n_wake_up_samples = wake_up_duration * fs_mic
    wake_up_at_sample = int(np.size(chirp_orig_resampl) - n_wake_up_samples)  # The sample where the wake-up signal is in effect

    # Select the wake-up piece of the audio fragment, and give every audio fragment from each mic a row in a matrix
    rx_audio_wake = rx_mic_sig[:, wake_up_at_sample:int(wake_up_at_sample + n_wake_up_samples)]

    return rx_audio_wake

def adjust_gain(chirp_orig_resampl, rx_audio_wake):
    """
    Give received signal an amplification to have similar values as original chirp amplitude and have better correlation values
    :param chirp_orig_resampl: original chirp signal
    :param rx_audio_wake: received chirp fragment
    :return: amplified received chirp fragment
    """
    # Calculate needed amplification factor to have similar values to original signal amplitude
    amp = rms(chirp_orig_resampl) / rms(rx_audio_wake, axis=1)

    # multiplication with ampl
    rx_audio_amp = rx_audio_wake * amp
    return rx_audio_amp

def agc(audio, chirp_orig_resampl):
    """
    Apply AGC to a signal and adjust amplitude to equal gain of original signal
    :param audio: audio fragment
    :param rx_chirp_agc: received audio fragment
    :return:
    """
    # TODO: controleer werking, mogelijks nog een fout in AGC
    # Calculate envelope with hilbert function
    audio_envelope = amplitude_envelope(audio, axis=0)

    # AGC on received signal
    rx_chirp_agc = (1 / audio_envelope) * audio

    # Get amplitude equal to the one of original signal
    ampl = rms(chirp_orig_resampl) / rms(rx_chirp_agc)
    agc_signal = rx_chirp_agc * ampl

    return agc_signal

def pulse_compres(original, received):
    """
    pulse compression for 2 signals and normalisation
    :param original: original theoretical signal
    :param received: received signal
    :return: corr_val: normalised pulse compression of both
    """
    # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
    corr_val = np.abs(np.correlate(original, received, "full"))

    # Normalize y peak values to have smaller values (not 1e9)
    corr_val = (corr_val-np.min(corr_val))/(np.max(corr_val)-np.min(corr_val))
    return corr_val

def pulse_compres_not_abs(original, received):
    """
    pulse compression for 2 signals and normalisation without absolute value operator
    :param original: original theoretical signal
    :param received: received signal
    :return: corr_val: normalised pulse compression of both
    """
    # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
    corr_val = np.correlate(original, received, "full")

    # Normalize y peak values to have smaller values (not 1e9)
    corr_val = (corr_val)/(np.max(corr_val))
    return corr_val

def simple_inter_xyz(X_Tx, Y_Tx, Z_Tx, d_meas):
    """
    Simple intersections algorithm
    :param X_Tx: beacons x_coords
    :param Y_Tx: beacons y_coords
    :param Z_Tx:  beacons z_coords
    :param d_meas:
    :return:
    """
    # determine rows of A
    norm_tool = np.array([X_Tx[0], Y_Tx[0], Z_Tx[0]])
    norm_r = d_meas[0] ** 2
    d_measn = d_meas[1:]
    X_Txn = X_Tx[1:]
    Y_Txn = Y_Tx[1:]
    Z_Txn = Z_Tx[1:]

    a0 = (X_Txn - norm_tool[0])
    a1 = (Y_Txn - norm_tool[1])
    a2 = (Z_Txn - norm_tool[2])
    A = np.stack((a0, a1, a2))
    A = A.transpose()

    B = 0.5 * (norm_r - d_measn ** 2 + calc_distance_3D(X_Txn, Y_Txn, Z_Txn, norm_tool[0], norm_tool[1],
                                                        norm_tool[2]) ** 2)
    B = B[:, np.newaxis]
    Est = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(A), A)), np.transpose(A)), B)
    return Est.ravel() + norm_tool.ravel()

def estimate_xyz_RangeBancroft(S, N, r):
    A = 2 * S
    b = np.sum(np.power(S, 2), axis=1) - np.power(r, 2)
    b = b[:, np.newaxis]
    ones = np.ones((N, 1))
    p = inv(A.T @ A) @ A.T @ ones
    q = inv(A.T @ A) @ A.T @ b
    x, y, z = p.T @ p, 2 * p.T @ q - 1, q.T @ q
    v = np.concatenate((x[0], y[0], z[0]))
    t = np.roots(v)
    result1 = p @ [t[0]] + q.flatten()
    result1b = result1[:, np.newaxis]
    result2 = p @ [t[1]] + q.flatten()
    result2b = result2[:, np.newaxis]

    norm1 = np.linalg.norm(r - np.sqrt(np.sum(np.power(S - ones @ result1b.T, 2), axis=1)))
    norm2 = np.linalg.norm(r - np.sqrt(np.sum(np.power(S - ones @ result2b.T, 2), axis=1)))
    if norm1 < norm2:
        result = result1
    else:
        result = result2
    return np.absolute(result).flatten()


def estimate_xyz_Beck(S, N, r):
    A = np.hstack((2 * S, -np.ones((N, 1))))
    b = np.sum(np.power(S, 2), axis=1) - np.power(r, 2)
    b = b[:, np.newaxis]
    P = np.diag([1, 1, 1, 0])
    q = np.array((0, 0, 0, -0.5))[:, np.newaxis]
    lambda_1 = np.max(np.linalg.eigvals(sp.linalg.sqrtm((A.T @ A)) @ P @ sp.linalg.sqrtm((A.T @ A))))
    tolerance = 1 * 10 ** (-10)

    def functionPhi(x):
        z = inv(A.T @ A + x * P) @ (A.T @ b - x * q)
        phi = (z.T @ P @ z + 2 * q.T @ z).flatten()[0]
        return phi

    # TODO check right boundry, currently set at 1000, doens't work when put on np.INF
    try:
        t = sp.optimize.bisect(f=functionPhi, a=-1 / lambda_1, b=1000, xtol=tolerance)
    except:
        return np.array((0,0,0))
    theta = inv(A.T @ A + t * P) @ (A.T @ b - t * q)
    return np.absolute(theta[0:3]).flatten()


def estimate_xyz_Chueng2(S, N, r):
    eye = np.eye(N)
    A = np.hstack((S, -0.5 * np.ones((N, 1))))
    b = 0.5 * np.sum(np.power(S, 2), axis=1) - 0.5 * np.power(r, 2)
    b = b[:, np.newaxis]
    r_matrix = np.reshape(2*r, (N, 1))
    Psi = r_matrix @ r_matrix.T * eye
    invPsi = inv(Psi)
    P = np.diag([1, 1, 1, 0])
    q = np.array((0, 0, 0, -1))[:, np.newaxis]

    [L, U] = np.linalg.eig((inv(A.T @ invPsi @ A) @ P))
    z = np.sort(L)[::-1]
    si = np.argsort(L)[::-1]
    U2 = U[:, si]
    c = (U2.T @ q).flatten()
    g = (inv(U2) @ inv((A.T @ invPsi @ A)) @ q).flatten()
    e = ((invPsi @ A @ U2).T @ b).flatten()
    f = (inv(U2) @ inv((A.T @ invPsi @ A)) @ A @ invPsi @ b).flatten()

    # solve the five root equation
    # lam = sy.symbols('lam')
    p =  np.zeros(8)
    p[7] = c[0] * f[0] + c[1] * f[1] + c[2] * f[2] + c[3] * f[3] + e[0] * f[0] * z[0] + e[1] * f[1] * z[1] + e[2] * f[2] * z[2]
    p[6] = c[0]*f[0]*z[0]/2 + 2*c[0]*f[0]*z[1] + 2*c[0]*f[0]*z[2] - c[0]*g[0]/2 + 2*c[1]*f[1]*z[0] + c[1]*f[1]*z[1]/2 + 2*c[1]*f[1]*z[2] - c[1]*g[1]/2 + 2*c[2]*f[2]*z[0] + 2*c[2]*f[2]*z[1] + c[2]*f[2]*z[2]/2 - c[2]*g[2]/2 + 2*c[3]*f[3]*z[0] + 2*c[3]*f[3]*z[1] + 2*c[3]*f[3]*z[2] + c[3]*g[3]/2 + 2*e[0]*f[0]*z[0]*z[1] + 2*e[0]*f[0]*z[0]*z[2] - e[0]*g[0]*z[0]/2 + 2*e[1]*f[1]*z[0]*z[1] + 2*e[1]*f[1]*z[1]*z[2] - e[1]*g[1]*z[1]/2 + 2*e[2]*f[2]*z[0]*z[2] + 2*e[2]*f[2]*z[1]*z[2] - e[2]*g[2]*z[2]/2
    p[5] = c[0]*f[0]*z[0]*z[1] + c[0]*f[0]*z[0]*z[2] + c[0]*f[0]*z[1]**2 + 4*c[0]*f[0]*z[1]*z[2] + c[0]*f[0]*z[2]**2 - c[0]*g[0]*z[0]/4 - c[0]*g[0]*z[1] - c[0]*g[0]*z[2] + c[1]*f[1]*z[0]**2 + c[1]*f[1]*z[0]*z[1] + 4*c[1]*f[1]*z[0]*z[2] + c[1]*f[1]*z[1]*z[2] + c[1]*f[1]*z[2]**2 - c[1]*g[1]*z[0] - c[1]*g[1]*z[1]/4 - c[1]*g[1]*z[2] + c[2]*f[2]*z[0]**2 + 4*c[2]*f[2]*z[0]*z[1] + c[2]*f[2]*z[0]*z[2] + c[2]*f[2]*z[1]**2 + c[2]*f[2]*z[1]*z[2] - c[2]*g[2]*z[0] - c[2]*g[2]*z[1] - c[2]*g[2]*z[2]/4 + c[3]*f[3]*z[0]**2 + 4*c[3]*f[3]*z[0]*z[1] + 4*c[3]*f[3]*z[0]*z[2] + c[3]*f[3]*z[1]**2 + 4*c[3]*f[3]*z[1]*z[2] + c[3]*f[3]*z[2]**2 + c[3]*g[3]*z[0] + c[3]*g[3]*z[1] + c[3]*g[3]*z[2] + e[0]*f[0]*z[0]*z[1]**2 + 4*e[0]*f[0]*z[0]*z[1]*z[2] + e[0]*f[0]*z[0]*z[2]**2 - e[0]*g[0]*z[0]*z[1] - e[0]*g[0]*z[0]*z[2] + e[1]*f[1]*z[0]**2*z[1] + 4*e[1]*f[1]*z[0]*z[1]*z[2] + e[1]*f[1]*z[1]*z[2]**2 - e[1]*g[1]*z[0]*z[1] - e[1]*g[1]*z[1]*z[2] + e[2]*f[2]*z[0]**2*z[2] + 4*e[2]*f[2]*z[0]*z[1]*z[2] + e[2]*f[2]*z[1]**2*z[2] - e[2]*g[2]*z[0]*z[2] - e[2]*g[2]*z[1]*z[2]
    p[4] = c[0]*f[0]*z[0]*z[1]**2/2 + 2*c[0]*f[0]*z[0]*z[1]*z[2] + c[0]*f[0]*z[0]*z[2]**2/2 + 2*c[0]*f[0]*z[1]**2*z[2] + 2*c[0]*f[0]*z[1]*z[2]**2 - c[0]*g[0]*z[0]*z[1]/2 - c[0]*g[0]*z[0]*z[2]/2 - c[0]*g[0]*z[1]**2/2 - 2*c[0]*g[0]*z[1]*z[2] - c[0]*g[0]*z[2]**2/2 + c[1]*f[1]*z[0]**2*z[1]/2 + 2*c[1]*f[1]*z[0]**2*z[2] + 2*c[1]*f[1]*z[0]*z[1]*z[2] + 2*c[1]*f[1]*z[0]*z[2]**2 + c[1]*f[1]*z[1]*z[2]**2/2 - c[1]*g[1]*z[0]**2/2 - c[1]*g[1]*z[0]*z[1]/2 - 2*c[1]*g[1]*z[0]*z[2] - c[1]*g[1]*z[1]*z[2]/2 - c[1]*g[1]*z[2]**2/2 + 2*c[2]*f[2]*z[0]**2*z[1] + c[2]*f[2]*z[0]**2*z[2]/2 + 2*c[2]*f[2]*z[0]*z[1]**2 + 2*c[2]*f[2]*z[0]*z[1]*z[2] + c[2]*f[2]*z[1]**2*z[2]/2 - c[2]*g[2]*z[0]**2/2 - 2*c[2]*g[2]*z[0]*z[1] - c[2]*g[2]*z[0]*z[2]/2 - c[2]*g[2]*z[1]**2/2 - c[2]*g[2]*z[1]*z[2]/2 + 2*c[3]*f[3]*z[0]**2*z[1] + 2*c[3]*f[3]*z[0]**2*z[2] + 2*c[3]*f[3]*z[0]*z[1]**2 + 8*c[3]*f[3]*z[0]*z[1]*z[2] + 2*c[3]*f[3]*z[0]*z[2]**2 + 2*c[3]*f[3]*z[1]**2*z[2] + 2*c[3]*f[3]*z[1]*z[2]**2 + c[3]*g[3]*z[0]**2/2 + 2*c[3]*g[3]*z[0]*z[1] + 2*c[3]*g[3]*z[0]*z[2] + c[3]*g[3]*z[1]**2/2 + 2*c[3]*g[3]*z[1]*z[2] + c[3]*g[3]*z[2]**2/2 + 2*e[0]*f[0]*z[0]*z[1]**2*z[2] + 2*e[0]*f[0]*z[0]*z[1]*z[2]**2 - e[0]*g[0]*z[0]*z[1]**2/2 - 2*e[0]*g[0]*z[0]*z[1]*z[2] - e[0]*g[0]*z[0]*z[2]**2/2 + 2*e[1]*f[1]*z[0]**2*z[1]*z[2] + 2*e[1]*f[1]*z[0]*z[1]*z[2]**2 - e[1]*g[1]*z[0]**2*z[1]/2 - 2*e[1]*g[1]*z[0]*z[1]*z[2] - e[1]*g[1]*z[1]*z[2]**2/2 + 2*e[2]*f[2]*z[0]**2*z[1]*z[2] + 2*e[2]*f[2]*z[0]*z[1]**2*z[2] - e[2]*g[2]*z[0]**2*z[2]/2 - 2*e[2]*g[2]*z[0]*z[1]*z[2] - e[2]*g[2]*z[1]**2*z[2]/2
    p[3] = c[0]*f[0]*z[0]*z[1]**2*z[2] + c[0]*f[0]*z[0]*z[1]*z[2]**2 + c[0]*f[0]*z[1]**2*z[2]**2 - c[0]*g[0]*z[0]*z[1]**2/4 - c[0]*g[0]*z[0]*z[1]*z[2] - c[0]*g[0]*z[0]*z[2]**2/4 - c[0]*g[0]*z[1]**2*z[2] - c[0]*g[0]*z[1]*z[2]**2 + c[1]*f[1]*z[0]**2*z[1]*z[2] + c[1]*f[1]*z[0]**2*z[2]**2 + c[1]*f[1]*z[0]*z[1]*z[2]**2 - c[1]*g[1]*z[0]**2*z[1]/4 - c[1]*g[1]*z[0]**2*z[2] - c[1]*g[1]*z[0]*z[1]*z[2] - c[1]*g[1]*z[0]*z[2]**2 - c[1]*g[1]*z[1]*z[2]**2/4 + c[2]*f[2]*z[0]**2*z[1]**2 + c[2]*f[2]*z[0]**2*z[1]*z[2] + c[2]*f[2]*z[0]*z[1]**2*z[2] - c[2]*g[2]*z[0]**2*z[1] - c[2]*g[2]*z[0]**2*z[2]/4 - c[2]*g[2]*z[0]*z[1]**2 - c[2]*g[2]*z[0]*z[1]*z[2] - c[2]*g[2]*z[1]**2*z[2]/4 + c[3]*f[3]*z[0]**2*z[1]**2 + 4*c[3]*f[3]*z[0]**2*z[1]*z[2] + c[3]*f[3]*z[0]**2*z[2]**2 + 4*c[3]*f[3]*z[0]*z[1]**2*z[2] + 4*c[3]*f[3]*z[0]*z[1]*z[2]**2 + c[3]*f[3]*z[1]**2*z[2]**2 + c[3]*g[3]*z[0]**2*z[1] + c[3]*g[3]*z[0]**2*z[2] + c[3]*g[3]*z[0]*z[1]**2 + 4*c[3]*g[3]*z[0]*z[1]*z[2] + c[3]*g[3]*z[0]*z[2]**2 + c[3]*g[3]*z[1]**2*z[2] + c[3]*g[3]*z[1]*z[2]**2 + e[0]*f[0]*z[0]*z[1]**2*z[2]**2 - e[0]*g[0]*z[0]*z[1]**2*z[2] - e[0]*g[0]*z[0]*z[1]*z[2]**2 + e[1]*f[1]*z[0]**2*z[1]*z[2]**2 - e[1]*g[1]*z[0]**2*z[1]*z[2] - e[1]*g[1]*z[0]*z[1]*z[2]**2 + e[2]*f[2]*z[0]**2*z[1]**2*z[2] - e[2]*g[2]*z[0]**2*z[1]*z[2] - e[2]*g[2]*z[0]*z[1]**2*z[2]
    p[2] = c[0]*f[0]*z[0]*z[1]**2*z[2]**2/2 - c[0]*g[0]*z[0]*z[1]**2*z[2]/2 - c[0]*g[0]*z[0]*z[1]*z[2]**2/2 - c[0]*g[0]*z[1]**2*z[2]**2/2 + c[1]*f[1]*z[0]**2*z[1]*z[2]**2/2 - c[1]*g[1]*z[0]**2*z[1]*z[2]/2 - c[1]*g[1]*z[0]**2*z[2]**2/2 - c[1]*g[1]*z[0]*z[1]*z[2]**2/2 + c[2]*f[2]*z[0]**2*z[1]**2*z[2]/2 - c[2]*g[2]*z[0]**2*z[1]**2/2 - c[2]*g[2]*z[0]**2*z[1]*z[2]/2 - c[2]*g[2]*z[0]*z[1]**2*z[2]/2 + 2*c[3]*f[3]*z[0]**2*z[1]**2*z[2] + 2*c[3]*f[3]*z[0]**2*z[1]*z[2]**2 + 2*c[3]*f[3]*z[0]*z[1]**2*z[2]**2 + c[3]*g[3]*z[0]**2*z[1]**2/2 + 2*c[3]*g[3]*z[0]**2*z[1]*z[2] + c[3]*g[3]*z[0]**2*z[2]**2/2 + 2*c[3]*g[3]*z[0]*z[1]**2*z[2] + 2*c[3]*g[3]*z[0]*z[1]*z[2]**2 + c[3]*g[3]*z[1]**2*z[2]**2/2 - e[0]*g[0]*z[0]*z[1]**2*z[2]**2/2 - e[1]*g[1]*z[0]**2*z[1]*z[2]**2/2 - e[2]*g[2]*z[0]**2*z[1]**2*z[2]/2
    p[1] = -c[0]*g[0]*z[0]*z[1]**2*z[2]**2/4 - c[1]*g[1]*z[0]**2*z[1]*z[2]**2/4 - c[2]*g[2]*z[0]**2*z[1]**2*z[2]/4 + c[3]*f[3]*z[0]**2*z[1]**2*z[2]**2 + c[3]*g[3]*z[0]**2*z[1]**2*z[2] + c[3]*g[3]*z[0]**2*z[1]*z[2]**2 + c[3]*g[3]*z[0]*z[1]**2*z[2]**2
    p[0] = c[3]*g[3]*z[0]**2*z[1]**2*z[2]**2/2
    roots = np.roots(p)
    # lambda_star = roots.real
    lambda_star = abs(roots)
    # fval = np.zeros(len(lambda_star))
    # theta = np.zeros((4, len(lambda_star)))
    # for i in range(len(lambda_star)):
    #    try:
    #        theta[:,i] = (inv((A.T @ invPsi @ A + lambda_star[i] * P)) @ (A.T @ invPsi @ b - (lambda_star[i]/2) * q)).flatten()
    #        fval[i] = ((A @ theta[:,i])[:, np.newaxis] - b).T @ invPsi @ ((A @ theta[:,i])[:, np.newaxis] - b)
    #    except:
    #        fval[i] = np.infty
    # minIndex = np.argmin(fval)

    # result = theta[0:3, minIndex].flatten()

    # lambda_star_selected = lambda_star[lambda_star > 0]
    # idx = (np.abs(lambda_star - 0)).argmin()
    # lambda_star_final = lambda_star[idx]
    lambda_star_selected = lambda_star[lambda_star > 0]
    idx = (np.abs(lambda_star - 0)).argmin()
    # lambda_star_final = lambda_star[idx]


    try:
        lambda_star_final = lambda_star_selected[0]
    except:
        idx = (np.abs(lambda_star - 0)).argmin()
        lambda_star_final = np.abs(lambda_star[idx])
        # result = np.zeros(3)
        # return result
    try:
        result = inv((A.T @ invPsi @ A + lambda_star_final * P)) @ (A.T @ invPsi @ b - (lambda_star_final/2) * q)
    except:
        result = np.linalg.pinv((A.T @ invPsi @ A + lambda_star_final * P)) @ (A.T @ invPsi @ b - (lambda_star_final/2) * q)
    # print(result[0:3].flatten())
    return result[0:3].flatten()
    # return theta[0:3, minIndex].flatten()

def estimate_xyz_GaussNewton(S, N, r):
    def residual_toa(x, r, S, eye):
        [n, dim] = S.shape
        R = np.tile(x, (n, 1)) - S
        ranges = np.sqrt(np.sum(np.power(R, 2), axis=1))
        f = ranges - r
        J = R / np.transpose([ranges] * dim)
        W = inv(sp.linalg.sqrtm(eye))
        f = W @ f
        J = W @ J
        return [f, J]

    eye = np.eye(N)
    D = np.hstack((-np.ones((N - 1, 1)), np.eye((N - 1))))
    A = D @ S
    B = D @ (np.sum(np.power(S, 2), axis=1) - r ** 2) / 2
    x0 = np.linalg.lstsq(A, B, rcond=None)[0]
    maxIterations = 8
    tolResult = 0.001
    x = x0

    for i in range(maxIterations):
        [f, J] = residual_toa(x, r, S, eye)
        dx = np.linalg.lstsq(-J, f, rcond=None)[0]
        x[:] = x[:] + dx
        if np.linalg.norm(dx, ord=2) < tolResult:
            break
    return x.flatten()


plt.rcParams['axes.unicode_minus'] = False

def plot_generated_points(vertices, height, all_positions_train, all_positions_test, mic_positions, filename, title):
    """
    Plots the room given its vertices and height an all positions within the room
    :param vertices: coordinates of corners in 2D
    :param height: height of the room
    :param all_positions_train: all the datapoints inside the room for training and validation set
    :param all_positions_test: all the datapoints of the test set
    :param mic_positions: microphone positions
    :param filename: name to save html file
    :param title: title of the plot
    """
    x_train = all_positions_train[:, 0]
    y_train = all_positions_train[:, 1]
    z_train = all_positions_train[:, 2]

    x_test = all_positions_test[:, 0]
    y_test = all_positions_test[:, 1]
    z_test = all_positions_test[:, 2]

    x_mic = []
    y_mic = []
    z_mic = []
    for coord in mic_positions.T:
        x_mic.append(coord[0])
        y_mic.append(coord[1])
        z_mic.append(coord[2])

    traces_vert = []
    for corner in vertices:
        trace = go.Scatter3d(
            x=[corner[0], corner[0]], y=[corner[1], corner[1]], z=[0, height],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_vert.append(trace)

    # Create ground and top plane of room
    edges_top_ground = edges_from_vertices_2D(vertices)

    traces_top_bottom = []
    for edge in edges_top_ground:
        x_tr, y_tr = zip(*edge)
        trace1 = go.Scatter3d(
            x=x_tr, y=y_tr, z=[height, height],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_top_bottom.append(trace1)
        trace2 = go.Scatter3d(
            x=x_tr, y=y_tr, z=[0, 0],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_top_bottom.append(trace2)

    datapoints_training = [(go.Scatter3d(x=x_train, y=y_train, z=z_train, mode='markers', name='Training + Dev set', marker_size=5, marker=dict(color="#e9c46a")))]
    datapoints_test = [(go.Scatter3d(x=x_test, y=y_test, z=z_test, mode='markers', name='Test set', marker_size=5, marker=dict(color="#4D4C38")))]
    mics = [(go.Scatter3d(x=x_mic, y=y_mic, z=z_mic, mode='markers', name='Anchors', marker_size=5, marker=dict(color="#BB4406", symbol='square')))]

    # Create a Scatter trace for the points
    fig = go.Figure(data=datapoints_training + datapoints_test + mics + traces_vert + traces_top_bottom)

    fig.update_layout(title=title, autosize=True)
    fig.write_html(filename + ".html")
    #fig.show()

def plot_CDF(sortIntersection, sortRB, sortBeck, sortChueng, sortGN, title, filename):
    """
    Plot multiple CDF functions in one graph
    :param sortIntersection:
    :param sortRB:
    :param sortBeck:
    :param sortChueng:
    :param sortGN:
    :param title:
    :param filename:
    :return:
    """
    p = 1. * np.arange(len(sortIntersection)) / (len(sortIntersection) - 1)

    fig = go.Figure(data=[
        go.Scatter(x=sortIntersection, y=p, marker=dict(color='#264653'), name='Simple Intersection'),
        #go.Scatter(x=sortRB, y=p, marker=dict(color='#2a9d8f'), name='Bancroft'),
        #go.Scatter(x=sortBeck, y=p, marker=dict(color='#e9c46a'), name='Beck'),
        #go.Scatter(x=sortChueng, y=p, marker=dict(color='#f4a261'), name='Chueng'),
        #go.Scatter(x=sortGN, y=p, marker=dict(color='#e76f51'), name='Gauss-Newton')
    ])
    fig.update_layout(title=title, autosize=True, xaxis_title = 'm', yaxis_title = 'CDF')
    fig.update_xaxes(range=[0, 10])
    fig.write_html(path_fig + filename + ".html")
    fig.write_image(path_fig + filename + ".svg")

    #fig.show()

def plot_CDF_one(sortedData, title, filename, x_max):
    """
    Plot one CDF function in a graph
    :param sortedData:
    :param title:
    :param filename:
    :param x_max:
    :return:
    """
    p = 1. * np.arange(len(sortedData)) / (len(sortedData) - 1)

    fig = go.Figure(data=[
        go.Scatter(x=sortedData, y=p, marker=dict(color='#264653'), name='Simple Intersection'),
    ])
    fig.update_layout(title=title, autosize=True, xaxis_title = 'm', yaxis_title = 'CDF')
    fig.update_xaxes(range=[0, x_max])
    fig.write_html(path_fig + filename + ".html")
    fig.write_image(path_fig + filename + ".svg")

    #fig.show()

def plot_multiple_CDF(sorted_vals_trad, sorted_vals_CNN, title, filename):
    """
    Plot multiple CDF function in one graph
    :param sorted_vals_trad:
    :param sorted_vals_CNN:
    :param title:
    :param filename:
    :return:
    """
    p1 = 1. * np.arange(len(sorted_vals_trad[0])) / (len(sorted_vals_trad[0]) - 1)
    p2 = 1. * np.arange(len(sorted_vals_CNN[0])) / (len(sorted_vals_CNN[0]) - 1)
    p3 = 1. * np.arange(len(sorted_vals_trad[1])) / (len(sorted_vals_trad[1]) - 1)
    p4 = 1. * np.arange(len(sorted_vals_CNN[1])) / (len(sorted_vals_CNN[1]) - 1)
    p5 = 1. * np.arange(len(sorted_vals_trad[2])) / (len(sorted_vals_trad[2]) - 1)
    p6 = 1. * np.arange(len(sorted_vals_CNN[2])) / (len(sorted_vals_CNN[2]) - 1)
    p7 = 1. * np.arange(len(sorted_vals_trad[3])) / (len(sorted_vals_trad[3]) - 1)
    p8 = 1. * np.arange(len(sorted_vals_CNN[3])) / (len(sorted_vals_CNN[3]) - 1)

    fig = go.Figure(data=[
        go.Scatter(x=sorted_vals_trad[0], y=p1, marker=dict(color='#FF030D'), line=dict(dash='dot'), name='Euclidean distance (SI)'),
        go.Scatter(x=sorted_vals_CNN[0], y=p2, marker=dict(color='#FB7E81'), name='Euclidean distance (CNN)'),
        go.Scatter(x=sorted_vals_trad[1], y=p3, marker=dict(color='#000080'), line=dict(dash='dot'), name='x error (SI)'),
        go.Scatter(x=sorted_vals_CNN[1], y=p4, marker=dict(color='#ADADEB'), name='x error (CNN)'),
        go.Scatter(x=sorted_vals_trad[2], y=p5, marker=dict(color='#426352'), line=dict(dash='dot'), name='y error (SI)'),
        go.Scatter(x=sorted_vals_CNN[2], y=p6, marker=dict(color='#B0CABD'), name='y error (CNN)'),
        go.Scatter(x=sorted_vals_trad[3], y=p7, marker=dict(color='#CD950C'), line=dict(dash='dot'), name='z error (SI)'),
        go.Scatter(x=sorted_vals_CNN[3], y=p8, marker=dict(color='#DC8C7C'), name='z error (CNN)')
    ])
    fig.update_layout(title=title, autosize=True, xaxis_title='m', yaxis_title='CDF')
    fig.update_xaxes(range=[0, 3])
    fig.write_html(path_fig + filename + ".html")
    fig.write_image(path_fig + filename + ".svg")

    #fig.show()

def plot_CDF_all(data4mics, data6mics, data8mics, data10mics, data12mics, data14mics, data16mics, title, filename):
    p1 = 1. * np.arange(len(data4mics)) / (len(data4mics) - 1)
    p2 = 1. * np.arange(len(data6mics)) / (len(data6mics) - 1)
    p3 = 1. * np.arange(len(data8mics)) / (len(data8mics) - 1)
    p4 = 1. * np.arange(len(data10mics)) / (len(data10mics) - 1)
    p5 = 1. * np.arange(len(data12mics)) / (len(data12mics) - 1)
    p6 = 1. * np.arange(len(data14mics)) / (len(data14mics) - 1)
    p7 = 1. * np.arange(len(data16mics)) / (len(data16mics) - 1)


    fig = go.Figure(data=[
        go.Scatter(x=data4mics, y=p1, marker=dict(color='#264653'), name='4 mics'),
        go.Scatter(x=data6mics, y=p2, marker=dict(color='#2a9d8f'), name='6 mics'),
        go.Scatter(x=data8mics, y=p3, marker=dict(color='#e9c46a'), name='8 mics'),
        go.Scatter(x=data10mics, y=p4, marker=dict(color='#f4a261'), name='10 mics'),
        go.Scatter(x=data12mics, y=p5, marker=dict(color='#392A50'), name='12 mics'),
        go.Scatter(x=data14mics, y=p6, marker=dict(color='#ADDEC8'), name='14 mics'),
        go.Scatter(x=data16mics, y=p7, marker=dict(color='#e76f51'), name='16 mics')
    ])
    fig.update_layout(title=title, autosize=True, xaxis_title = 'm', yaxis_title = 'CDF')
    fig.update_xaxes(range=[0, 3])
    fig.write_html(path_fig + filename + ".html")
    fig.write_image(path_fig + filename + ".svg")

    #fig.show()

def plot_accuracy_room_ranging(room_dim, mic_positions, sp_positions, error, title, filename):

    # Position speaker
    x_positions, y_positions, z_positions = sp_positions[:, 0], sp_positions[:, 1], sp_positions[:, 2]

    # Positions mics
    mic_pos_x, mic_pos_y, mic_pos_z = mic_positions[0, :], mic_positions[1, :], mic_positions[2, :]

    # Room dimensions
    dx, dy, dz = room_dim[0], room_dim[1], room_dim[2]
    x_grid_lines = [0, dx, dx, 0, 0, 0, 0, 0, 0, dx, dx, dx, dx, dx, dx, 0]
    y_grid_lines = [0, 0, dy, dy, 0, 0, dy, dy, dy,dy, dy, dy, 0, 0, 0, 0]
    z_grid_lines = [0, 0, 0, 0, 0, dz, dz, 0, dz,dz, 0, dz, dz, 0, dz, dz]

    #error adjust
    error_array = np.abs(error)

    fig = go.Figure(data=[
        go.Scatter3d(mode='lines', x=x_grid_lines, y=y_grid_lines, z=z_grid_lines, opacity=1,
                    line=dict(width=2, color='#264653'), name='Room Setup'),    #draw room
        go.Scatter3d(x=mic_pos_x, y=mic_pos_y, z=mic_pos_z, mode='markers',
                    marker=dict(color="#264653", symbol='square', size=5), name="Microphone"),    # draw mic positions
        go.Scatter3d(x=x_positions, y=y_positions, z=z_positions, mode='markers', marker=dict(size=10, color=error_array, colorscale='Viridis', opacity=0.8, showscale=True), name="Speaker Positions")  # draw speaker positions
    ])

    fig.update_layout(title=title, autosize=True, legend=dict(yanchor="top", y=0.99, xanchor='left', x=0.85))
    fig.update_scenes(zaxis_autorange ='reversed')
    # fig.write_html(filename + ".html")
    #fig.show()

plt.rcParams['axes.unicode_minus'] = False

def plot_room_errors(positions, error, mic_positions, filename, title, cmax):
    """
    Plots the room in 3d with corresponding error values at the position as 4th dimension
    :param positions: positions of interest in the room
    :param error: error on specific positions
    :param mic_positions: microphone positions
    :param filename: name of the file to save as html file
    :param title: title of the plot
    :param cmax: max value of colorscale
    """
    if config['shoebox']:
        room_dim = config["room_dim_shoebox"]
        height = room_dim[2]
        vertices = np.array([[0, 0], [0, room_dim[1]], [room_dim[0], room_dim[1]], [room_dim[0], 0]])
    else:
        vertices = config['room_corners_no_shoebox']
        height= config['room_height_no_shoebox']

    x_pos = positions[:, 0]
    y_pos = positions[:, 1]
    z_pos = positions[:, 2]

    x_mic = []
    y_mic = []
    z_mic = []
    for coord in mic_positions.T:
        x_mic.append(coord[0])
        y_mic.append(coord[1])
        z_mic.append(coord[2])

    traces_vert = []
    for corner in vertices:
        trace = go.Scatter3d(
            x=[corner[0], corner[0]], y=[corner[1], corner[1]], z=[0, height],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_vert.append(trace)

    # Create ground and top plane of room
    edges_top_ground = edges_from_vertices_2D(vertices)

    traces_top_bottom = []
    for edge in edges_top_ground:
        x_tr, y_tr = zip(*edge)
        trace1 = go.Scatter3d(
            x=x_tr, y=y_tr, z=[height, height],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_top_bottom.append(trace1)
        trace2 = go.Scatter3d(
            x=x_tr, y=y_tr, z=[0, 0],
            mode='lines',
            line=dict(color='black', width=2),
            hoverinfo='none',
            text=None,
            connectgaps=False,
            showlegend=False
        )
        traces_top_bottom.append(trace2)

    positions = [(go.Scatter3d(x=x_pos, y=y_pos, z=z_pos, mode='markers', name='Error in m', marker_size=10,
                               marker=dict(color=error, colorscale='Viridis', opacity=0.8, showscale=True, cmin=0, cmax=cmax)))]
    mics = [(go.Scatter3d(x=x_mic, y=y_mic, z=z_mic, mode='markers', name='Anchors', marker_size=5,
                              marker=dict(color="#BB4406", symbol='square')))]

    # Create a Scatter trace for the points
    fig = go.Figure(data= positions + mics + traces_vert + traces_top_bottom)

    fig.update_layout(title=title, autosize=True, legend=dict(yanchor="top", y=0.99, xanchor='left', x=0.85))
    fig.write_html(path_fig + filename + ".html")
    #fig.show()