import numpy as np
import localFunctions as lf
import json

with open('config.json') as json_file:
    config = json.load(json_file)

# Analysis variables
plot_traditional_analysis = True
plot_NN_analysis = True
plot_CDF_ranging_traditional = False
plot_CDF_eucl_dist = False
plot_CDF_x_error = False
plot_CDF_y_error = False
plot_CDF_z_error = False
plot_heatmap_eucl = True
plot_heatmap_x_error = False
plot_heatmap_y_error = False
plot_heatmap_z_error = False
plot_all_CDF_one = False

n_mics = config['n_mics']
print(n_mics, ' microphones were used')
mic_locs = np.load('mic_positions.npy')

path_estimation_data = 'estimation_data\\'

ranging_faults_all = np.load(path_estimation_data+'rangingfault_all_data_mics'+str(n_mics)+'.npy')
tradition_estimation_data = np.load(path_estimation_data+'traditional_estimation_data_all_for_n_mics_'+str(n_mics)+'.npy', allow_pickle=True)
NN_estimation_data = np.load(path_estimation_data+'NN_estimation_data_all_for_n_mics_'+str(n_mics)+'.npy', allow_pickle=True)

sp_loc_all = np.load('speaker_positions.npy')
n_positions_measured = np.size(sp_loc_all, axis=0)
print('\nAmount of speaker positions: ', n_positions_measured)

def analyse(estimation_data):
    """
    Analyses the data
    :param estimation_data: all data given by dict
    :param method: SI (simple intersections) or CNN based
    :return: sorted_vals, error_vals, speaker_loc_all
    """
    speaker_loc_all = [estimation_data[pos_nr]['speaker_loc'] for pos_nr in range(len(estimation_data)-1)]
    pos_estimate_all = [estimation_data[pos_nr]['pos_estimate'] for pos_nr in range(len(estimation_data)-1)]
    euclid_dist_error_all = [estimation_data[pos_nr]['eucl_dist_error'] for pos_nr in range(len(estimation_data)-1)]
    x_error_all = [estimation_data[pos_nr]['x_error'] for pos_nr in range(len(estimation_data)-1)]
    y_error_all = [estimation_data[pos_nr]['y_error'] for pos_nr in range(len(estimation_data)-1)]
    z_error_all = [estimation_data[pos_nr]['z_error'] for pos_nr in range(len(estimation_data)-1)]

    x_error_all = np.asarray(x_error_all).ravel()
    y_error_all = np.asarray(y_error_all).ravel()
    z_error_all = np.asarray(z_error_all).ravel()
    error_vals = [euclid_dist_error_all, x_error_all, y_error_all, z_error_all]

    sorted_eucl_dist = np.sort(np.abs(euclid_dist_error_all))
    sorted_x_error = np.sort(np.abs(x_error_all))
    sorted_y_error = np.sort(np.abs(y_error_all))
    sorted_z_error = np.sort(np.abs(z_error_all))
    sorted_vals = [sorted_eucl_dist, sorted_x_error, sorted_y_error, sorted_z_error]

    return sorted_vals, error_vals, speaker_loc_all
def plot_figs(sorted_vals, error_vals, speaker_loc_all ,method, cmax):
    if plot_CDF_eucl_dist:
        lf.plot_CDF_one(sorted_vals[0], 'CDF of Euclidean distance errors ('+method+')', 'CDF_plot_eucl_dist_'+method, 8)

    if plot_CDF_x_error:
        lf.plot_CDF_one(sorted_vals[1], 'CDF of x errors ('+method+')', 'CDF_plot_x_error'+method, 8)

    if plot_CDF_y_error:
         lf.plot_CDF_one(sorted_vals[2], 'CDF of y errors ('+method+')', 'CDF_plot_y_error'+method, 8)

    if plot_CDF_z_error:
        lf.plot_CDF_one(sorted_vals[3], 'CDF of z errors ('+method+')', 'CDF_plot_z_error'+method, 8)

    if plot_heatmap_eucl:
        lf.plot_room_errors(np.array(speaker_loc_all), error_vals[0], mic_locs, 'heatmap_Eucl_'+method,
                            'Euclidean distance error on position estimation ('+method+')', cmax)

    if plot_heatmap_x_error:
        lf.plot_room_errors(np.array(speaker_loc_all), np.abs(error_vals[1]), mic_locs, 'heatmap_x_error_'+method,
                            'X distance error on position estimation ('+method+')', cmax)

    if plot_heatmap_y_error:
        lf.plot_room_errors(np.array(speaker_loc_all), np.abs(error_vals[2]), mic_locs, 'heatmap_y_error_' + method,
                        'Y distance error on position estimation ('+method+')', cmax)

    if plot_heatmap_z_error:
        lf.plot_room_errors(np.array(speaker_loc_all), np.abs(error_vals[3]), mic_locs, 'heatmap_z_error_' + method,
                        'Z distance error on position estimation ('+method+')', cmax)


if __name__ == '__main__':
    if plot_traditional_analysis:
        if plot_CDF_ranging_traditional:
            # plot ranging faults
            sorted_ranging_trad = np.sort(np.abs(ranging_faults_all.ravel()))
            lf.plot_CDF_one(sorted_ranging_trad, 'CDF of ranging errors', 'CDF_plot_ranging_traditional', 8)

        sorted_vals_trad, error_vals_trad, speaker_loc_all_trad = analyse(tradition_estimation_data)
        plot_figs(sorted_vals_trad, error_vals_trad, speaker_loc_all_trad, 'SI', cmax=10)

    if plot_NN_analysis:
        sorted_vals_CNN, error_vals_CNN, speaker_loc_all_CNN = analyse(NN_estimation_data)
        plot_figs(sorted_vals_CNN, error_vals_CNN, speaker_loc_all_CNN, 'CNN', cmax=10)

    if plot_all_CDF_one:
        sorted_vals_trad, error_vals_trad, speaker_loc_all_trad = analyse(tradition_estimation_data)
        sorted_vals_CNN, error_vals_CNN, speaker_loc_all_CNN = analyse(NN_estimation_data)

        lf.plot_multiple_CDF(sorted_vals_trad, sorted_vals_CNN, 'CDF of position predictions ', 'CDF_all')
