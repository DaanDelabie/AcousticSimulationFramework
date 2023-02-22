#  ____  ____      _    __  __  ____ ___
# |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
# | | | | |_) |  / _ \ | |\/| | |  | | | |
# | |_| |  _ <  / ___ \| |  | | |__| |_| |
# |____/|_| \_\/_/   \_\_|  |_|\____\___/
#                           research group
#                             dramco.be/
#
#  KU Leuven - Technology Campus Gent,
#  Gebroeders De Smetstraat 1,
#  B-9000 Gent, Belgium
#
#         File: PHY.py
#      Created: 2022-03-25
#       Author: Daan Delabie
#      Version: 1.0
#
#  Description:
#
#  Room simulation with multiple microphones at walls and one mobile speaker at different locations
#
#  License L (optionally)
# --------------------------------------------------------------------------------------------
import pyroomacoustics as pra
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile
import localFunctions as lf
import warnings
import samplerate
import json
import multiprocessing as mp
import logging
import tqdm
from functools import partial

# logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger = logging.getLogger(' ')
warnings.filterwarnings("ignore")  # ignore warning when plotting 3D room
path_fig = 'result_figs\\'

def setup():
    # #############################################################################################
    # ----------------------------------- Configurations ---------------------------------------------
    # #############################################################################################
    with open('config.json') as json_file:
        config = json.load(json_file)
    # --------------------------------
    # MIC AND SPEAKER CHARACTERISTICS
    # --------------------------------
    # Create grid of mic positions
    position_mic_0 = [0.01, 0.01, 0.01]
    position_mic_1 = [7.99, 3.99, 2.39]
    position_mic_2 = [0.01, 3.99, 1.6]
    position_mic_3 = [5.99, 0.01, 0.8]
    position_mic_4 = [3.0, 1.01, 2.0]
    position_mic_5 = [5.0, 3.99, 0.5]
    position_mic_6 = [0.01, 1.1, 2.20]
    position_mic_7 = [7.99, 3.2, 0.1]
    position_mic_8 = [7.0, 1.05, 2.39]
    position_mic_9 = [2.0, 3.0, 2.39]
    position_mic_10 = [1.99, 0.75, 0.01]
    position_mic_11 = [6.3, 1.85, 0.01]


    # Location of all mics
    mic_locs = np.c_[
        position_mic_0,
        position_mic_1,
        position_mic_2,
        position_mic_3,
        position_mic_4,
        position_mic_5,
        position_mic_6,
        position_mic_7,
        position_mic_8,
        position_mic_9,
        position_mic_10,
        position_mic_11,
    ]

    np.save('mic_positions.npy', mic_locs)


    if config["shoebox"]:
        # Create test set
        sp_loc_test_set = lf.create_location_grid(config["distance_boundary"], config["room_dim_shoebox"],
                                                  config['n_x_test'], config['n_y_test'], config["n_z_test"])

        n_test_set_positions = np.size(sp_loc_test_set, axis=0)
        logger.info('Amount of test set speaker positions: {}'.format(n_test_set_positions))

        # Create data and dev set
        n_positions_dev_train = lf.calc_n_dev_train_set(sp_loc_test_set, config['train_set_ratio'], config['dev_set_ratio'], config['test_set_ratio'])
        sp_loc_traindev_set = lf.create_random_positions_shoebox(room_dim, n_positions_dev_train, config['distance_boundary'])

        height = room_dim[2]
        vertices = np.array([[0, 0], [0, room_dim[1]], [room_dim[0], room_dim[1]], [room_dim[0], 0]])
        np.save('outcounted.npy', 0)
    else:
        vertices = config['room_corners_no_shoebox']
        height= config['room_height_no_shoebox']

        # Create test set
        sp_loc_test_set, out_count_test = lf.create_location_grid_non_shoebox(vertices, height, config["distance_boundary"],
                                                              config['n_x_test'], config['n_y_test'], config["n_z_test"])

        n_test_set_positions = np.size(sp_loc_test_set, axis=0)
        logger.info('Amount of test set speaker positions: {}'.format(n_test_set_positions))

        # Create data and dev set
        n_positions_dev_train = lf.calc_n_dev_train_set(sp_loc_test_set, config['train_set_ratio'],
                                                        config['dev_set_ratio'], config['test_set_ratio'])

        sp_loc_traindev_set, out_counts = lf.create_random_positions_in_random_3D_space(vertices, height, config['distance_boundary'], n_positions_dev_train)

        logger.info('Amount of regenerated train and dev set speaker positions that were outside the room: {}'.format(out_counts))
        np.save('outcounted.npy', out_count_test)

    if config["plot_room"]:
        lf.plot_generated_points(vertices, height, sp_loc_traindev_set, sp_loc_test_set, mic_locs, path_fig+'generated_training_points', 'Training and Dev set')

    n_traindev_set_positions = np.size(sp_loc_traindev_set, axis=0)
    logger.info('Amount of train and dev set speaker positions: {}'.format(n_traindev_set_positions))

    sp_loc_all = np.vstack((sp_loc_test_set, sp_loc_traindev_set))

    # calculate amount of positions integrated
    n_positions_measured = np.size(sp_loc_all, axis=0)
    logger.info('Total amount of speaker positions: {}'.format(n_positions_measured))

    # save positions
    np.save('speaker_positions.npy', sp_loc_all)

    # --------------------------------
    #     ROOM CHARACTERISTICS
    # --------------------------------
    # room loggings
    if config["shoebox"]:
        logger.info('Shoebox simulation activated')
    else:
        logger.info('Non Shoebox simulation activated')

    # Use given materials (below) e.g. wood True, if false, pre defined RT60 is used for RIR calculations
    # RT60 Seconds (not needed in case of materials)
    if config['use_given_materials']:
        logger.info('Setup given materials')
    else:
        logger.info('Use predefined RT60')

    if config['use_raytracer']:
        logger.info('Raytracing activated')
        logger.info('No Directivity')
    else:
        # Image Source Model
        logger.info('ISM simulation activated')
        if config["shoebox"]:
            logger.info('Directivity included')
        else:
            logger.info('No Directivity')

    v_sound = lf.get_speed_of_sound(config['temperature'])
    logger.info("Speed of sound: {} m/s".format(v_sound))

    # -------------------------------
    #   SOUND SIGNAL CHARACTERISTICS
    # --------------------------------
    duration = config['chirp_duration']  # Duration of the chirp signal in s
    fs_source = config['sample_rate_TX'] # Sample rate to create the signal, and for simulation
    fs_mic = config['sample_rate_RX']

    # chirp:
    start_freq = config['chirp_f_start']  # Start frequency of the chirp signal
    stop_freq = config['chirp_f_stop']    # Stop frequency of the chirp signal
    chirp_amplitude = np.iinfo(np.int16).max / 10
    chirp_offset = config['chirp_DC']  # DC offset

    # #############################################################################################
    # ---------------------------------------- SETUP ---------------------------------------------
    # #############################################################################################
    # Create sound signal
    chirp = lf.create_chirp(start_freq=start_freq, stop_freq=stop_freq, chirp_duration=duration, fs=fs_source,
                            amplitude=chirp_amplitude, offset=chirp_offset)

    # Write to WAV file
    lf.write_to_WAV(config['original_chirp'], fs_source, chirp)

    # Import a mono wavfile as the source signal, the sampling frequency should match that of the room
    fs, audio = wavfile.read(config['original_chirp'])
    logger.info("Used sample frequency derived from the WAV file: {} Hz".format(fs))

    logger.info("Sample frequency microphone: {} Hz".format(fs_mic))
    logger.info("Sample frequency speaker: {} Hz".format(fs_source))
    duration_signal = len(audio) / fs
    logger.info('Duration of input: {} s\n\n'.format(duration_signal))

    # Plot the used audio signal
    if config["plot_audio"]:
        lf.plot_audio_signal("Signal at speaker", audio, fs, duration_signal, config['delay_sim'])

    return n_positions_measured, sp_loc_all, fs, fs_source, fs_mic, v_sound, mic_locs, audio, duration_signal

def sim_position(position_nr, sp_loc_all, fs, fs_source, fs_mic, v_sound, mic_locs, audio, duration_signal, use_materials,
                 material, corners, shoebox, rt60, room_dim, room_height, air_abs, max_order, use_raytracer, delay_sim,
                 receiver_radius, plot_room, calc_rir, plot_rir, save_rir, plot_audio, save_rx_audio):
    # -----------------------------------------
    #   SETUP ROOM
    # -----------------------------------------
    if use_materials:
        if shoebox:
            room_materials = pra.make_materials(
                ceiling=material,
                floor=material,
                east=material,
                west=material,
                north=material,
                south=material
            )
        else:
            room_materials = pra.Material(material)

    else:
        if shoebox:
            #  # Sabine’s formula to find the wall energy absorption and maximum order of the image source method (ISM)
            e_absorption, _ = pra.inverse_sabine(rt60, room_dim)
            # Calculate equivalent material
            room_materials = pra.Material(e_absorption)

        else:
            # Not fully right assumption to calculate material absorption coefficient since inverse Sabine's formula is used
            # recalculate the volume of a room to a shoebox room, since the sabine code is based on volume
            S = lf.poly_area(corners[0, :], corners[1, :])  # surface of room
            # create rectangular surface
            square_x_y = np.sqrt(S)
            x_recal = 2 * square_x_y / 3  # make it rectangular i.s.o. square
            y_recal = S / x_recal
            # required to achieve a desired reverberation time (RT60, i.e. the time it takes for the room impulse responses (RIR)
            # to decays by 60 dB)
            e_absorption, _ = pra.inverse_sabine(rt60, [x_recal, y_recal, room_height])
            # Calculate equivalent material absorption coefficient
            room_materials = pra.Material(e_absorption)

    from pyroomacoustics.directivities import (
        DirectivityPattern,
        DirectionVector,
        CardioidFamily
    )

    dir_obj = CardioidFamily(
        orientation=DirectionVector(azimuth=90, colatitude=15, degrees=True),
        pattern_enum=DirectivityPattern.SUBCARDIOID
    )

    speaker_loc = sp_loc_all[position_nr, :]
    if not shoebox:
        room = pra.Room.from_corners(corners=corners, fs=fs, materials=room_materials, max_order=max_order,
                                     air_absorption=air_abs, ray_tracing=use_raytracer)
        room.extrude(room_height, materials=room_materials)
        x_max = np.max(corners[0, :])
        y_max = np.max(corners[1, :])
        z_max = room_height

    else:
        room = pra.ShoeBox(room_dim, fs=fs, materials=room_materials, max_order=max_order,
                           air_absorption=air_abs, ray_tracing=use_raytracer)
        x_max = room_dim[0]
        y_max = room_dim[1]
        z_max = room_dim[2]

    # Set sound speed
    room.set_sound_speed(v_sound)

    if use_raytracer:
        # Place mic's in the room
        room.add_microphone_array(mic_locs)    # no Directivity possible with raytracing
        # Place Speaker in the room
        room.add_source(speaker_loc, signal=audio, delay=delay_sim)  # No Directivity possible with raytracing
        room.set_ray_tracing(receiver_radius=receiver_radius)
        # The receiver radius = radius of the sphere around the microphone in which to integrate the energy (default: 0.5 m)

    else:
        # Image Source Model
        if shoebox:
            # Place mic's in the room
            room.add_microphone_array(mic_locs, directivity=dir_obj)  # Directivity list can be created also for different mics: =[dir_1, dir_2]
            # Place Speaker in the room
            room.add_source(speaker_loc, signal=audio, delay=delay_sim, directivity=dir_obj)  # delay = start time in simulation in s
        else:
            # Place mic's in the room
            room.add_microphone_array(mic_locs, directivity=dir_obj)
            # Place Speaker in the room
            room.add_source(speaker_loc, signal=audio, delay=delay_sim)    # delay = start time in simulation in s
        room.image_source_model()

    # if plot_room:
    #     markersize = 150
    #     fig, ax = room.plot(img_order=3, auto_add_to_figure=False)
    #     ax.set_xlim([0, x_max+0.5])
    #     ax.set_ylim([0, y_max+0.5])
    #     ax.set_zlim([0, z_max+0.5])
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')
    #     #ax.scatter(position_mic_0[0], position_mic_0[1], position_mic_0[2], s = markersize, label='Microphone 0', color='red', marker="s")
    #     #ax.scatter(position_mic_1[0], position_mic_1[1], position_mic_1[2], s = markersize, label='Microphone 1', color='red', marker="s")
    #     #ax.scatter(position_speaker[0], position_speaker[1], position_speaker[2], s = markersize, label='Speaker', color='indigo', marker="X")
    #     ax.set_title("Simulation Room")
    #     #plt.legend()
    #     fig.show()

    # #############################################################################################
    # -------------------------------------- SIMULATION--------------------------------------------
    # #############################################################################################
    # Create the Room Impulse Response (RIR)
    #room.compute_rir()

    if calc_rir:
        # Measure the reverberation time
        rt60_meas = room.measure_rt60()

    room.simulate(reference_mic=0)

    # #############################################################################################
    # -------------------------------------- GET DATA --------------------------------------------
    # #############################################################################################
    for m in range(room.n_mics):
        for s in range(room.n_sources):
            if calc_rir:
                # Calculate RT60 between all mics and speakers
                #logger.info("RT60 between the {}th mic and {}th source, position {}: {:.3f} s".format(m, s, position_nr, rt60_meas[m, s]))
                rir = room.rir[m][s]

            # Plot the RIR between all mics and speakers
            if plot_rir:
                lf.plot_RIR(title="RIR between the {}th mic and {}th source, position {}".format(m, s, position_nr), rir=rir, fs=fs)

            if save_rir:
                np.save('RIRs/rir'+str(s)+'source'+str(m)+'mic'+'_position'+str(position_nr)+'.npy', rir)

            # Delete first 40 samples, artifact from simulation (issue #136 GitHub PyroomAcoustics)
            rx_audio = room.mic_array.signals[m, :]
            index_del = np.arange(0, 40, 1)
            new_rx_audio = np.delete(rx_audio, index_del)

            ###############################################################
            # ---------------------- Resample ----------------------------
            ###############################################################
            # if the microphone sample rate differs from the room (=speaker) sample rate,
            if fs_source != fs_mic:
                fs_ratio = float(fs_mic) / float(fs_source)
                mic_signal = samplerate.resample(new_rx_audio, fs_ratio, "sinc_best")

            else:  # in the sample rates are identical, no up/down sampling is required
                mic_signal = new_rx_audio

            # Plot the received audio signals
            if plot_audio:
                lf.plot_audio_signal(title="Received signal at the {}th mic from the {}th source, position".format(m, s, position_nr),
                                     signal=mic_signal, fs=fs_mic, dur_orig_sig=duration_signal, delay=delay_sim)

            # Plot spectogram
            #lf.plot_spectrogram(title="Received signal at the {}th mic from the {}th source, position".format(m, s, position_nr), signal=new_rx_audio, fs=fs, dur_orig_sig=duration_signal, delay=delay)

            if save_rx_audio:
                # Writing the received audio signals to a WAV file
                wavfile.write("RX_audio/Received_signal_of_the_{}th_mic_from_{}th_source_position_{}.wav".format(m, s, position_nr), rate=fs_mic, data=mic_signal.astype(np.int16))

    # #############################################################################################
    # ------------------------ Theoretical Distance Calculation-----------------------------------
    # #############################################################################################
    # Determine real theoretical distances
    distances_th = np.empty(1)
    for coord in mic_locs.T:
        distance = lf.calc_distance_3D(speaker_loc[0], speaker_loc[1], speaker_loc[2], coord[0], coord[1], coord[2])
        distances_th = np.vstack((distances_th, distance))

    distances_th = np.delete(distances_th, 0, 0)
    np.save('dist_th/distances_th_position'+str(position_nr)+'.npy', distances_th)

def sim_position_init(q):
    sim_position.q = q

if __name__ == '__main__':

    # -----------------------------------------
    #   READ CONFIG VARIABLES ONCE
    # -----------------------------------------
    with open('config.json') as json_file:
        config = json.load(json_file)

    use_materials = config['use_given_materials']
    material = config['material']  # wood_1.6cm, panel_fabric_covered_6pcf
    corners = np.array(config['room_corners_no_shoebox']).T
    shoebox = config["shoebox"]
    rt60 = config['rt60']
    room_dim = config["room_dim_shoebox"]
    room_height = config['room_height_no_shoebox']
    air_abs = config['air_abs']
    max_order = config['max_order']
    use_raytracer = config['use_raytracer']
    delay_sim = config['delay_sim']
    receiver_radius = config['receiver_radius']
    plot_room = config['plot_room']
    calc_rir = config['calc_rir']
    plot_rir = config["plot_RIR"]
    save_rir = config['save_rir']
    plot_audio = config["plot_audio"]
    save_rx_audio = config['save_rx_audio']

    # -----------------------------------------
    #   Multiprocessing
    # -----------------------------------------
    n_positions_measured, sp_loc_all, fs, fs_source, fs_mic, v_sound, mic_locs, audio, duration_signal = setup()

    location_idx_all = np.arange(0, n_positions_measured, 1)
    onearg_func = partial(sim_position, sp_loc_all=sp_loc_all, fs=fs, fs_source=fs_source, fs_mic=fs_mic,
                          v_sound=v_sound, mic_locs=mic_locs, audio=audio, duration_signal=duration_signal,
                          use_materials=use_materials, material=material, corners=corners, shoebox=shoebox,
                          rt60=rt60, room_dim=room_dim, room_height=room_height, air_abs=air_abs, max_order=max_order,
                          use_raytracer=use_raytracer, delay_sim=delay_sim, receiver_radius=receiver_radius,
                          plot_room=plot_room, calc_rir=calc_rir, plot_rir=plot_rir, save_rir=save_rir,
                          plot_audio=plot_audio, save_rx_audio=save_rx_audio)

    logger.info('Simulate testset and train/dev set\n')
    q = mp.Queue
    p = mp.Pool(mp.cpu_count()-1, sim_position_init, [q])
    results = list(tqdm.tqdm(p.imap(onearg_func, location_idx_all),
                                    total=len(location_idx_all)))
    p.close()
    p.join()