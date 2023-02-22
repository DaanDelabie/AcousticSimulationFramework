import numpy as np
import localFunctions as lf
from scipy.signal import *
import librosa as lbr
import json
from tqdm import tqdm

with open('config.json') as json_file:
    config = json.load(json_file)

path_audio = 'RX_audio\\'
path_rirs = 'RIRs\\'
path_audio_awgn = 'RX_audio_with_AWGN\\'
path_audio_sir = 'RX_audio_with_SIR_and_AWGN\\'
path_corr = 'correlation_functions\\'
path_corr_AGC = 'correlation_functions_with_AGC\\'
path_LPF_curve = 'LPF_curve\\'
path_dist_th = 'dist_th\\'
path_estimation_data = 'estimation_data\\'

AGC = config['AGC']
SNR = config['addSNR']
show_pulse_comp = config['plot_pulse_compr']
peak_prominence = config['peak_prominence']
peak_prominence_factor = config['peak_prominence_factor']   # prominence threshold to later use the index of the first peak in the arry of the
                                                            # promineces larger than the threshold
v_sound = lf.get_speed_of_sound(config['temperature'])
print("Speed of sound: " + str(v_sound) + " m/s")

n_mics = config['n_mics']  # amount of microphones used
n_speakers = config['n_speakers']    #amount of speakers during one position simulation

wake_up_duration = config['wake_up_duration']    # Duration of the wake-up signal in s
chirp_orig_resampl = np.load('chirp_orig_not_or_resampled.npy')

if SNR:
    _, fs_mic = lbr.load(path_audio_awgn + "Received_audio_with_awgn_mic0_speaker_0position0.wav", sr=None)

else:
    _, fs_mic = lbr.load(path_audio + "Received_signal_of_the_0th_mic_from_0th_source_position_0.wav", sr=None)

# Calculate amount of samples within wake-up duration
n_wake_up_samples = wake_up_duration * fs_mic
wake_up_at_sample = int(np.size(chirp_orig_resampl) - n_wake_up_samples)  # The sample where the wake-up signal is in effect

# Read the pulse compression results
if AGC:
    agc_text='AGC'
else:
    agc_text='notAGC'

# Read needed data
# Read positions of mics and speakers
mic_positions = np.load('mic_positions.npy').T
print('mic positions: \n', mic_positions)
print('#mics used: ', n_mics)

# Read speaker positions
sp_loc_all_complete = np.load('speaker_positions.npy')

# Extract only test set data from full dataset
n_grid_points_test_set = config['n_x_test']*config['n_y_test']*config['n_z_test']
sp_loc_all = sp_loc_all_complete[0:n_grid_points_test_set, :]

# calculate amount of positions integrated
n_positions_measured = np.size(sp_loc_all, axis=0)
print('\n Amount of test set speaker positions: ', n_positions_measured)

# calculate lengths for init
LPF_array_length = int(config["sample_rate_RX"]*(config["chirp_duration"]+config["wake_up_duration"])-1)

# loop every speaker position
estimation_data_all = np.array([])
ranging_faults = np.empty(n_mics)
for position_nr in tqdm(range(0, n_positions_measured)):
    if n_positions_measured ==1:
        speaker_loc = sp_loc_all
    else:
        speaker_loc = sp_loc_all[position_nr, :]

    pulse_compr_all = np.empty(LPF_array_length)
    LPF_all = np.empty(LPF_array_length)
    corr_index_array = np.array([])
    for rx in range(n_mics):
        for sp in range(n_speakers):
            pulse_comp = np.load(path_corr + 'corr_val_mic' + str(rx) + 'speaker' + str(sp) + 'position'+ str(position_nr) +'.npy')
            pulse_compr_all = np.vstack((pulse_compr_all, pulse_comp))

            LPF = np.load(path_LPF_curve + 'LPF_mic' + str(rx) + 'speaker' + str(sp) + str(agc_text)+'position'+ str(position_nr) + '.npy')
            LPF_all = np.vstack((LPF_all, LPF))

            if peak_prominence:
                # Peak prominence to determine good peak ------------------------
                # The prominence of a peak measures how much a peak stands out from the surrounding baseline of the signal and is
                # defined as the vertical distance between the peak and its lowest contour line

                # find all peaks and calculate prominences
                peaks, _ = find_peaks(LPF)
                prominences = peak_prominences(LPF, peaks)[0]
                most_prom = prominences[prominences > peak_prominence_factor][-1]
                most_prom_idx = np.where(np.around(prominences, decimals = 5) == np.around(most_prom, decimals=5))    # Select first index from row which > PP Threshold [0][0]
                idx_peak_samples = peaks[most_prom_idx]

                # calculate height of each peak's contour line
                contour_heights = LPF[peaks]-prominences
                index_opt_general = lf.idx_peak_determination_PP(pulse_comp, idx_peak_samples)

                # Save good index in array
                corr_index_array = np.append(corr_index_array, index_opt_general)

                if show_pulse_comp:
                    # Plot correlation function
                    lf.plot_corr_LPF_peaks("Correlation at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position " +str(position_nr),
                                           pulse_comp, LPF, peaks, LPF[peaks], contour_heights, index_opt_general)
                # ---------------------------------------------------------------
            else:
                # peak based on maximum value as easy distance estimation
                # Determine index of the (one) max value on easy way
                index_opt_general = lf.one_peak_determination_LPF(pulse_comp, LPF)

                # Save good index in array
                corr_index_array = np.append(corr_index_array, index_opt_general)

                if show_pulse_comp:
                    # Plot correlation function
                    lf.plot_corr_LPF("Correlation at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position " + str(position_nr),
                                           pulse_comp, LPF, index_opt_general)

    # sample in effective chirp corresponding with start of chirp selection = (corr_index_max+1)-size(chirp-segment)
    eff_start_samp_chirp = (((corr_index_array+1)-n_wake_up_samples)[np.newaxis]).T

    # Calculate difference in amount of samples between synchronisation point(start wake-up) and part of received chirp (start-point)
    delta_sample = wake_up_at_sample-eff_start_samp_chirp

    # Determine distance
    distances_meas = (delta_sample / fs_mic) * v_sound
    #print('\nMeasured Euclidean distance between mics and speaker, position '+ str(position_nr) + ': \n', distances_meas)

    # Read theoretical distances
    distances_th = np.load(path_dist_th+'distances_th_position'+str(position_nr)+'.npy')
    #print('\nTheoretical Euclidean distance between mics and speaker, position '+ str(position_nr) + ': \n', distances_th)

    # Calculate difference between theoretical and measured
    np.set_printoptions(suppress=True)
    diff = (distances_th[0:n_mics,:] - distances_meas)
    #print('\nDifference between theoretical and measured Euclidean distances in m, position '+ str(position_nr) + ': \n', diff)

    # 3D position calculalations
    mic_x_coords = mic_positions[:, 0]
    mic_y_coords = mic_positions[:, 1]
    mic_z_coords = mic_positions[:, 2]
    d_meas = np.squeeze(distances_meas.T, axis=0)

    pos_simple_inter = lf.simple_inter_xyz(mic_x_coords[0:n_mics], mic_y_coords[0:n_mics], mic_z_coords[0:n_mics], d_meas)
    pos_est_total = np.vstack((pos_simple_inter))

    # Euclidean distance error calculation
    euclidian_dist_si = np.linalg.norm(speaker_loc - pos_simple_inter)
    euclidian_dist_total = np.array([euclidian_dist_si])

    # Distance error for each coördinate
    # x_coord_diff = speaker_loc[0]- pos_est_total[:,0]
    # y_coord_diff = speaker_loc[1]- pos_est_total[:,1]
    # z_coord_diff = speaker_loc[2]- pos_est_total[:,2]

    x_coord_diff = speaker_loc[0]- pos_est_total[0]
    y_coord_diff = speaker_loc[1]- pos_est_total[1]
    z_coord_diff = speaker_loc[2]- pos_est_total[2]

    # safe all distance error values in array [position_nr, th_loc, estimated_loc, n_mics, euclidean_distance_error, x_error, y_errror, z_error]
    #estimation_data = [position_nr, n_mics, speaker_loc, pos_simple_inter, euclidian_dist_si, x_coord_diff, y_coord_diff, z_coord_diff]
    estimation_data = dict({'position_nr': position_nr,
                            'n_mics': n_mics,
                            'speaker_loc': speaker_loc,
                            'pos_estimate': pos_simple_inter,
                            'eucl_dist_error': euclidian_dist_si,
                            'x_error': x_coord_diff,
                            'y_error': y_coord_diff,
                            'z_error': z_coord_diff})

    estimation_data_all = np.concatenate((estimation_data_all, np.array([estimation_data])))
    ranging_faults = np.vstack((ranging_faults, np.squeeze(diff, 1)))

ranging_faults = np.delete(ranging_faults, 0, 0)

# save date for analysis
np.save(path_estimation_data+'traditional_estimation_data_all_for_n_mics_'+str(n_mics)+'.npy', estimation_data_all)
np.save(path_estimation_data+'rangingfault_all_data_mics'+str(n_mics)+'.npy', ranging_faults)