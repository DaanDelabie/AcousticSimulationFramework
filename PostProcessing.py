import librosa as lbr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import localFunctions as lf
import soundfile as sf
import samplerate
import json
from tqdm import tqdm

###############################################################
# ----------------- Configuration settings --------------------
###############################################################
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
path_LPF_curve_per_position = 'LPF_curves_per_position\\'
path_LPF_curve_per_position_downsampled = 'LPF_curves_per_position_downsampled\\'

original_chirp = config['original_chirp']
v_sound = lf.get_speed_of_sound(config["temperature"])
print("Speed of sound: " + str(v_sound) + " m/s")

normal_pulse_compression = config['normal_pulse_compression']
plot_first = config['plot_first_pulse_compr']

if config['AGC']: agc_text='AGC'
else: agc_text='notAGC'

if config['xor']: agc_text = 'XOR'

###############################################################
# -------------------------- Setup ----------------------------
###############################################################
# read original Chirp
chirp_orig, fs_source = lbr.load(original_chirp, sr=None)
duration_chirp = lbr.get_duration(y=chirp_orig, sr=fs_source)
print("Sample rate of the source : %.2f Hz" % fs_source)
print("Duration of the chirp: %.2f s" % duration_chirp)

# read speaker positions
sp_loc_all = np.load('speaker_positions.npy')
print('Labled data/speaker positions: \n', sp_loc_all)

# calculate amount of positions integrated
n_positions_measured = np.size(sp_loc_all, axis=0)
print('\n Amount of speaker positions: ', n_positions_measured)

sig, fs_mic = lbr.load(path_audio + "Received_signal_of_the_0th_mic_from_0th_source_position_" + str(0) + ".wav", sr=None)
rx_mic_sig = np.empty(np.size(sig))
print("Sample rate at the microphone : %.2f Hz" % fs_mic)

# 3D matrix: height amount of mics per 1 speaker position, width: amount of samples after LPF, depth: amount of positions
# LPF_array_legth = Length original chirp + length audio wake -1 = fs_mic* chirp duration + fs_mic * wake up duration -1 = fs_mic (chrip duration + wake up duration) -1
LPF_array_length = int(fs_mic*(config["chirp_duration"]+config["wake_up_duration"])-1)

# resample if fs source is not equal to fs receive
chirp_orig_resampl = lf.resample_source_to_mic(fs_source, fs_mic, chirp_orig)

# loop every speaker position
for position_nr in tqdm(range(0, n_positions_measured)):
    if n_positions_measured ==1:
        speaker_loc = sp_loc_all
    else:
        speaker_loc = sp_loc_all[position_nr, :]

    # Read rx audio data and put all rx audio in matrix for the 1 position
    rx_mic_sig = lf.create_rx_audio_matrix(path_audio, position_nr)

    # Add AWGN via SNR value
    if config['addSNR']:
        rx_mic_sig, SNR_val_all = lf.add_AWGN(rx_mic_sig, config['SNR'], position_nr, fs_mic, path_audio_awgn)

    # Add interference noise via SIR value
    if config['addSIR']:
       rx_mic_sig = lf.add_interference(config['interference_signal'], config['SIR'], rx_mic_sig, fs_mic, position_nr, path_audio_sir)

    ###############################################################
    # ----------------------- Analysis ---------------------------
    ###############################################################
    # BPF
    # Here you can add a filter

    # Select wake-up part in received audio
    rx_audio_wake = lf.select_wakeup_part(fs_mic, chirp_orig_resampl, config['wake_up_duration'], rx_mic_sig)

    # Adjust amplitude of received chirp fragment to have better correlation with original chirp signal
    rx_audio_amp = lf.adjust_gain(chirp_orig_resampl, rx_audio_wake)

    index = 0
    corr_index_array = np.array([])
    corr_index_array_fit = np.array([])
    LPF_all_1tr_normal = np.empty(LPF_array_length)
    LPF_all_1tr_down_normal = np.empty(config['n_samples_NN'])
    for rx in range(config['n_mics']):
        for sp in range(config['n_speakers']):
            audio_row = np.array(rx_audio_amp[index, :])
            # Pulse Compression
            if config['AGC'] and not config['xor']:
                agc_text = 'AGC'

                #Apply AGC
                rx_chirp_agc = lf.agc(audio_row, chirp_orig_resampl)

                # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
                corr_val = lf.pulse_compres(chirp_orig_resampl, rx_chirp_agc)

                if config['plot_audio']:
                    # Plot the audio signal received within the wake-up time and with AGC
                    lf.plot_audio_signal(title="RX signal with AGC at the " + str(rx) + "th mic from the " + str(
                        sp) + "th speaker, position " + str(position_nr), signal=rx_chirp_agc, fs=fs_mic, dur_orig_sig=config['wake_up_duration'], delay=0)
                    lf.plot_audio_signal(title="Original RX signal at the " + str(rx) + "th mic from the " + str(
                        sp) + "th speaker, position " + str(position_nr), signal=audio_row, fs=fs_mic, dur_orig_sig=config['wake_up_duration'], delay=0)

                if config['save_corr_val']:
                    # Save values
                    np.save(path_corr_AGC+'corr_val_AGC_mic'+str(rx)+'speaker'+str(sp)+'position'+str(position_nr)+'.npy', corr_val)

            elif not config['AGC'] and not config['xor']:
                agc_text = 'notAGC'

                # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
                corr_val = lf.pulse_compres(chirp_orig_resampl, audio_row)

                if config['save_corr_val']:
                    # Save values
                    np.save(path_corr + 'corr_val_mic' + str(rx) + 'speaker' + str(sp) + 'position'+str(position_nr)+'.npy', corr_val)

            elif config['xor']:
                agc_text= 'XOR'

                chirp_orig_resampl = lf.tobinairy(chirp_orig_resampl)
                audio_row = lf.tobinairy(audio_row)

                # Cross correlation with original chirp signal to determine upper and lower frequency (Pulse compression)
                corr_val = lf.pulse_compres_not_abs(chirp_orig_resampl, audio_row)

                if config['save_corr_val']:
                    # Save values
                    np.save(path_corr + 'corr_val_mic' + str(rx) + 'speaker' + str(sp) + 'position' + str(
                        position_nr) + '.npy', corr_val)

            # Add LPF to determine envelope
            corr_filtered = lf.LPF(corr_val, 'lowpass', 1000, 5000, fs_mic)    #  70, fs_mic/35

            # Downsample LPF curve
            LPF_downsampled = lf.downsample_LPF(corr_filtered, config['n_samples_NN'])

            # Create feature matrix
            LPF_all_1tr_normal = np.vstack((LPF_all_1tr_normal, corr_filtered))
            LPF_all_1tr_down_normal = np.vstack((LPF_all_1tr_down_normal, LPF_downsampled))

            if config['save_seper_lpf_val']:
                # Save LPF values
                np.save(path_LPF_curve + 'LPF_mic' + str(rx) + 'speaker' + str(sp) + str(agc_text)+ 'position'+str(position_nr)+'.npy', corr_filtered)

            if config['plot_pulse_compr'] or plot_first :
                # Plot correlation function
                lf.plot_corr_LPF("Correlation at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position "+str(position_nr), corr_val, corr_filtered)
                lf.plot_corr_easy("Downsampled correlation at the " + str(rx) + "th mic from the " + str(sp) + "th speaker, position " + str(position_nr), LPF_downsampled)
                plot_first = False
            index += 1

    LPF_all_1tr_normal = np.delete(LPF_all_1tr_normal, 0, 0)
    LPF_all_1tr_down_normal = np.delete(LPF_all_1tr_down_normal, 0, 0)

    if config['save_lpf_one_pos']:
        np.save(path_LPF_curve_per_position + 'LPF_dataset_1position_'+ str(agc_text)+'_position_'+str(position_nr)+'.npy', LPF_all_1tr_normal)

    if config['save_lpf_one_pos_downsampled']:
        np.save(path_LPF_curve_per_position_downsampled + 'LPF_dataset_downsampled_1position_'+ str(agc_text)+'_position_' + str(position_nr) + '.npy', LPF_all_1tr_down_normal)

    # RIR (from simulation)
    if config['plot_RIR']:
        lf.loopRIRplot(path_rirs, position_nr, fs_source)
