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
#  Room simulation with multiple speakers microphones (mobile node and anchors can be generated for random rooms)
#
#  License L (optionally)
# --------------------------------------------------------------------------------------------
import pyroomacoustics as pra
import numpy as np
import sys
import csv
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
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily
)

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

    if config["shoebox"]:
        logger.info('Shoebox simulation activated')
        height = room_dim[2]
        vertices = np.array([[0, 0], [0, room_dim[1]], [room_dim[0], room_dim[1]], [room_dim[0], 0]])
    else:
        logger.info('Non Shoebox simulation activated')
        vertices = config['room_corners_no_shoebox']
        height= config['room_height_no_shoebox']

    # -----------------------------------------
    # Read or Generate Mobile Node (MN) AND Anchor Positions
    # -----------------------------------------
    # MOBILE NODE POSITIONS -----------------------------------------------------------------------------------------
    if config['generate_random_MN_pos']:
        # Generate mobile node locations
        # Create test set
        mn_loc_test_set, out_count_test = lf.create_location_grid_non_shoebox(vertices, height, config["distance_boundary"],
                                                              config['n_x_test'], config['n_y_test'], config["n_z_test"])

        n_test_set_positions = np.size(mn_loc_test_set, axis=0)
        logger.info('Amount of test set mobile node positions: {}'.format(n_test_set_positions))

        # Create data and dev set
        n_positions_dev_train = lf.calc_n_dev_train_set(mn_loc_test_set, config['train_set_ratio'],
                                                        config['dev_set_ratio'], config['test_set_ratio'])

        mn_loc_traindev_set, out_counts = lf.create_random_positions_in_random_3D_space(vertices, height, config['distance_boundary'], n_positions_dev_train)

        logger.info('Amount of regenerated train and dev set mobile node positions that were outside the room: {}'.format(out_counts))
        np.save('Sim_data\\outcounted.npy', out_count_test)

        mn_loc_all = np.vstack((mn_loc_test_set, mn_loc_traindev_set))
        np.savetxt('Sim_data\\positions_mobile_node.csv', mn_loc_all, delimiter=',')

    else:
        # Read out from CSV if already exist
        mn_csv = []
        with open('Sim_data\\positions_mobile_node.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                pos_read = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                mn_csv.append(pos_read)

        mn_loc_all = np.array(mn_csv)

        out_counted = np.load('Sim_data\\outcounted.npy')
        # Calculate amount of test set grid points (for non-shoebox not n_x_test * n_y_test * n_z_test)
        n_test_set_positions = (config['n_x_test'] * config['n_y_test'] * config['n_z_test'])-out_counted
        mn_loc_test_set = mn_loc_all[:n_test_set_positions, :]
        mn_loc_traindev_set = mn_loc_all[n_test_set_positions:, :]

    n_traindev_set_positions = np.size(mn_loc_traindev_set, axis=0)
    logger.info('Amount of train and dev set mobile node positions: {}'.format(n_traindev_set_positions))

    n_test_set_positions = np.size(mn_loc_test_set, axis=0)
    logger.info('Amount of test set mobile node positions: {}'.format(n_test_set_positions))

    # calculate amount of positions integrated
    n_positions_measured = np.size(mn_loc_all, axis=0)
    logger.info('Total amount of mobile node positions: {}'.format(n_positions_measured))

    # save positions
    np.save('Sim_data\\positions_mobile_node.npy', mn_loc_all)

    # ANCHOR NODE POSITIONS
    if config['generate_random_ancher_pos']:
        surfaces_array = config['active_anchor_places']  # Floor Ceiling Nord East South West
        offset = config['anchor_wall_offset']
        anchor_locs = lf.generate_random_anchor_positions(vertices, height, config['number_generated_anchors'], offset,
                                                          surfaces_array, config['dir_anchor_direction'])

        np.savetxt('Sim_data\\anchor_positions.csv', anchor_locs, delimiter=',')
        np.save('Sim_data\\anchor_positions.npy', anchor_locs)

    else:  # read the existing csv file with speaker positions
        anchor_pos_csv = []
        with open('Sim_data\\anchor_positions.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                pos_read_anchor = [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])]
                anchor_pos_csv.append(pos_read_anchor)

        anchor_locs = np.array(anchor_pos_csv)

    logger.info('Total amount of anchor node positions: {}'.format(np.size(anchor_locs, axis=0)))
    # Plot room with directivities
    pattern_enum_anchor = DirectivityPattern.CARDIOID
    pattern_enum_mn_test = DirectivityPattern.OMNI
    pattern_enum_mn_train = DirectivityPattern.OMNI

    dirs_mn_test = []
    dirs_mn_train_dev = []
    for mn in mn_loc_test_set:
        dir_obj_mn_test = CardioidFamily(
            orientation=DirectionVector(azimuth=mn[3], colatitude=mn[4], degrees=True),
            pattern_enum=pattern_enum_mn_test
        )
        dirs_mn_test.append(dir_obj_mn_test)

    for mn in mn_loc_traindev_set:
        dir_obj_mn_traindev = CardioidFamily(
            orientation=DirectionVector(azimuth=mn[3], colatitude=mn[4], degrees=True),
            pattern_enum=pattern_enum_mn_train
        )
        dirs_mn_train_dev.append(dir_obj_mn_traindev)

    dirs_mn = dirs_mn_test + dirs_mn_train_dev

    dirs_anchers = []
    for anchor in anchor_locs:
        dir_obj_ancher = CardioidFamily(
            orientation=DirectionVector(azimuth=anchor[3], colatitude=anchor[4], degrees=True),
            pattern_enum=pattern_enum_anchor
        )
        dirs_anchers.append(dir_obj_ancher)

    if config["plot_room"]:
        lf.plot_generated_speaker_pos(vertices, height, anchor_locs, mn_loc_test_set, mn_loc_traindev_set,
                                      True, True, False, False, pattern_enum_anchor, pattern_enum_mn_test,
                                      pattern_enum_mn_train, 'Sim_data\\Simulation_Room_dirs_anchors', 'Simulation Room')

        lf.plot_generated_speaker_pos(vertices, height, anchor_locs, mn_loc_test_set, mn_loc_traindev_set,
                                      True, True, True, True, pattern_enum_anchor, pattern_enum_mn_test,
                                      pattern_enum_mn_train, 'Sim_data\\Simulation_Room_all_dirs', 'Simulation Room')

        lf.plot_generated_speaker_pos(vertices, height, anchor_locs, mn_loc_test_set, mn_loc_traindev_set,
                                      True, False, True, False, pattern_enum_anchor, pattern_enum_mn_test,
                                      pattern_enum_mn_train, 'Sim_data\\Simulation_Room_only_dir_vect', 'Simulation Room')

        lf.plot_generated_speaker_pos(vertices, height, anchor_locs, mn_loc_test_set, mn_loc_traindev_set,
                                      False, False, False, False, pattern_enum_anchor, pattern_enum_mn_test,
                                      pattern_enum_mn_train, 'Sim_data\\Simulation_Room_no_dirs',
                                      'Simulation Room')


    # Map speakers and mics to anchor and mobile node terminology
    if config['anchors'] == 'speakers':
        logger.info('Anchor nodes are speakers, mobile nodes are microphones')
        sp_locs = anchor_locs
        mic_locs = mn_loc_all
        sp_dirs = dirs_anchers
        mic_dirs = dirs_mn
    else:
        logger.info('Anchor nodes are microphones, mobile nodes are speakers')
        sp_locs = mn_loc_all
        mic_locs = anchor_locs
        sp_dirs = dirs_mn
        mic_dirs = dirs_anchers

    # --------------------------------
    #     ROOM CHARACTERISTICS
    # --------------------------------
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

    n_simulations_needed = np.size(sp_locs, axis=0)

    return n_simulations_needed, mic_locs, sp_locs, mic_dirs, sp_dirs, fs, fs_source, fs_mic, v_sound, audio, duration_signal

def sim_position(sim_nr, mic_locs, sp_locs, mic_dirs, sp_dirs, fs, fs_source, fs_mic, v_sound, audio, duration_signal, use_materials,
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
                east='panel_fabric_covered_6pcf',
                west='panel_fabric_covered_6pcf',
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

    # Select one speaker (to not interference with others and do it via TDM
    one_speaker_loc = sp_locs[sim_nr, :3]
    dir_obj_one_speaker = sp_dirs[sim_nr]

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

    ##########################################
    # --Add mics and speakers to the room ---
    ##########################################
    mic_loc_coord = mic_locs[:, 0:3]
    mic_locs = mic_loc_coord.T

    if use_raytracer:
        # Place mic's in the room
        room.add_microphone_array(mic_locs)    # no Directivity possible with raytracing
        # Place Speaker in the room
        room.add_source(one_speaker_loc, signal=audio, delay=delay_sim)  # No Directivity possible with raytracing
        room.set_ray_tracing(receiver_radius=receiver_radius)
        # The receiver radius = radius of the sphere around the microphone in which to integrate the energy (default: 0.5 m)

    else:
        # Image Source Model
        if shoebox:
            # Place mic's in the room
            room.add_microphone_array(mic_locs, directivity=mic_dirs)  # Directivity list can be created also for different mics: =[dir_1, dir_2]
            # Place Speaker in the room
            room.add_source(one_speaker_loc, signal=audio, delay=delay_sim, directivity=dir_obj_one_speaker)  # delay = start time in simulation in s
        else:
            # Place mic's in the room
            room.add_microphone_array(mic_locs, directivity=mic_dirs)
            # Place Speaker in the room
            room.add_source(one_speaker_loc, signal=audio, delay=delay_sim)    # delay = start time in simulation in s
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
                #logger.info("RT60 between the {}th mic and {}th source, simulation {}: {:.3f} s".format(m, s, sim_nr, rt60_meas[m, s]))
                rir = room.rir[m][s]

            # Plot the RIR between all mics and speakers
            if plot_rir:
                lf.plot_RIR(title="RIR between the {}th mic and {}th source, simulation {}".format(m, s, sim_nr), rir=rir, fs=fs)

            if save_rir:
                np.save('RIRs/rir'+str(s)+'source'+str(m)+'mic'+'_simulation'+str(sim_nr)+'.npy', rir)

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
                lf.plot_audio_signal(title="Received signal at the {}th mic from the {}th source, simulation".format(m, s, sim_nr),
                                     signal=mic_signal, fs=fs_mic, dur_orig_sig=duration_signal, delay=delay_sim)

            # Plot spectogram
            #lf.plot_spectrogram(title="Received signal at the {}th mic from the {}th source, simulation".format(m, s, sim_nr), signal=new_rx_audio, fs=fs, dur_orig_sig=duration_signal, delay=delay)

            if save_rx_audio:
                # Writing the received audio signals to a WAV file
                wavfile.write("RX_audio/Received_signal_of_the_{}th_mic_from_{}th_source_simulation_{}.wav".format(m, s, sim_nr), rate=fs_mic, data=mic_signal.astype(np.int16))

    # #############################################################################################
    # ------------------------ Theoretical Distance Calculation-----------------------------------
    # #############################################################################################
    # Determine real theoretical distances
    distances_th = np.empty(1)
    for coord in mic_locs.T:
        distance = lf.calc_distance_3D(one_speaker_loc[0], one_speaker_loc[1], one_speaker_loc[2], coord[0], coord[1], coord[2])
        distances_th = np.vstack((distances_th, distance))

    distances_th = np.delete(distances_th, 0, 0)
    np.save('dist_th/distances_th_simulation_'+str(sim_nr)+'.npy', distances_th)

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
    n_simulations_needed, mic_locs, sp_locs, mic_dirs, sp_dirs, fs, fs_source, fs_mic, v_sound, audio, duration_signal = setup()

    # def sim_position(sim_nr, mic_locs, sp_locs, mic_dirs, sp_dirs, fs, fs_source, fs_mic, v_sound, audio, duration_signal, use_materials,
    #                  material, corners, shoebox, rt60, room_dim, room_height, air_abs, max_order, use_raytracer, delay_sim,
    #                  receiver_radius, plot_room, calc_rir, plot_rir, save_rir, plot_audio, save_rx_audio):

    location_idx_all = np.arange(0, n_simulations_needed, 1)
    onearg_func = partial(sim_position, mic_locs=mic_locs, sp_locs=sp_locs, mic_dirs=mic_dirs, sp_dirs=sp_dirs, fs=fs,
                          fs_source=fs_source, fs_mic=fs_mic, v_sound=v_sound, audio=audio, duration_signal=duration_signal,
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