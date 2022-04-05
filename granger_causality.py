import warnings
from pathlib import Path

from copy import deepcopy
import numpy as np
import pandas as pd
import math
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import json

from itertools import permutations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import combinations

# import logging
# logger = logging.getLogger('my_module_name')
warnings.filterwarnings("ignore")

MAXLAG = 15
TEST = 'ssr_chi2test'


def load_file(path):
    df = pd.read_csv(path, usecols=['frame_no', 'cell_id', 'heading', 'x', 'y'])
    sim_param_path = str(path).replace("frames_dfs", "simulation_params").replace("csv", "json")
    with open(sim_param_path, 'r') as f:
        sim_params_dict = json.load(f)
    return df, sim_params_dict


def kpss_test(feature_series):
    statistic, p_value, n_lags, critical_values = kpss(feature_series.values, nlags="legacy")

    # print(f'KPSS Statistic: {statistic}')
    # print(f'p-value: {p_value}')
    # print(f'num lags: {n_lags}')
    # print('Critial Values:')
    # for key, value in critical_values.items():
    #     print(f'   {key} : {value}')
    return p_value


def adf_test(feature_series):
    result = adfuller(feature_series.values)
    # print('ADF Statistics: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    return result[1]


def stationary_test(df, col, cell_id, warm_up_window):
    # print(f"** start test stationary - cell_id: {cell_id} **")
    d_col = f'd{col}'
    cell_trajectory = df[df['cell_id'] == cell_id].iloc[warm_up_window:].copy().reset_index(drop=True)
    cell_trajectory[d_col] = cell_trajectory[col].diff()
    cell_trajectory = cell_trajectory.drop(columns=[col])
    cell_trajectory = cell_trajectory.dropna().reset_index(drop=True)
    is_stat_per_feature = []
    p_value_adf = adf_test(cell_trajectory[d_col])
    p_value_kpss = kpss_test(cell_trajectory[d_col])
    is_stat_per_feature.append(True if p_value_adf <= 0.05 or p_value_kpss >= 0.05 else False)
    # cell_trajectory.to_csv(f'./stat_files/{all(is_stat_per_feature)}_stat_cell_{cell_id}.csv')
    return all(is_stat_per_feature)


def granger_causality_matrix(data, features, neigh_df, res_df, res_col, test='ssr_chi2test', maxlag=MAXLAG,
                             verbose=False, warm_up_window=0, neigh_radius=10, neigh_radius_ratio=0.95):
    """
        The row are the response (y) and the columns are the predictors (x)
        If a given p-value is < significance level (0.05), we can reject the null hypothesis and conclude that walmart_x Granger causes apple_y.
    """
    for col in features:
        print(f"** Start calculate GC on {col} **")
        # res_df[col].loc[:, int(res_col)] = np.nan
        time_series_per_cell = {}
        feature_data = data[['frame_no', 'cell_id', col]].copy()
        for cell_id in feature_data['cell_id'].unique():
            is_stat = stationary_test(data, col, cell_id, warm_up_window)
            if not is_stat:
                continue
            time_series_per_cell[cell_id] = feature_data[feature_data['cell_id'] == cell_id].iloc[
                                            warm_up_window:].set_index('frame_no')[col].diff()[1:]
        neigh_pairs = check_neighbors(neigh_df, time_series_per_cell.keys(), neigh_radius, neigh_radius_ratio)
        time_frame_df = pd.DataFrame(time_series_per_cell)
        for p, r in res_df[col].index:
            if p not in time_frame_df or r not in time_frame_df or \
                    ((p, r) not in neigh_pairs and (r, p) not in neigh_pairs):
                continue
            pair_cells = time_frame_df[[r, p]]
            model = VAR(pair_cells)
            adjusted_lag = maxlag
            while True:
                try:
                    lags_results = model.select_order(adjusted_lag)
                    break
                except np.linalg.LinAlgError as err:
                    adjusted_lag -= 1
            lags = [lags_results.aic, lags_results.bic]
            opt_lag = np.min(lags)
            ## if the minimum is 0, the maximum will be taken. if it also 0, 1 will be taken.
            if opt_lag == 0:
                print(f"at least one if the metrics yield lag 0")
                opt_lag = np.max([np.max(lags), 1])  ## TODO change the 1
                if np.max(lags) == 0:
                    print(f"both lags are 0; 1 will be taken")
            gc_result = grangercausalitytests(time_frame_df[[r, p]], maxlag=opt_lag, verbose=verbose)
            p_value = gc_result[opt_lag][0][test][1]
            res_df[col].loc[(p, r), int(res_col)] = p_value
    return res_df


def check_neighbors(neigh_df, cells_ids, neigh_radius, neigh_dist_ratio):
    neigh_list = []
    all_pairs = list(combinations(list(cells_ids), 2))
    for c1, c2 in all_pairs:
        c1_pos_df = neigh_df[neigh_df['cell_id'] == c1]
        c2_pos_df = neigh_df[neigh_df['cell_id'] == c2]
        dist_df = c1_pos_df.apply(lambda c1_pos: math.dist(c1_pos[['x', 'y']].values,
                                                 c2_pos_df[c2_pos_df['frame_no'] == c1_pos['frame_no']].iloc[0][
                                                     ['x', 'y']].values), axis=1)
        is_neigh = dist_df < neigh_radius
        if np.mean(is_neigh) >= neigh_dist_ratio:
            neigh_list.append((c1, c2))
        else:
            print(f'neighbors test: {c1} and {c2} are neighbors only on {np.mean(is_neigh)}% of the frames')
    return neigh_list


def expand_weights_vector(weights_vector, num_of_cells):
    expanded_weights = np.full(num_of_cells - len(weights_vector), fill_value=weights_vector[-1], dtype=np.float64)
    expanded_weights = weights_vector + list(expanded_weights)
    cell_ids = []
    for i, val in enumerate(expanded_weights):
        if val > 0:
            cell_ids.append(str(num_of_cells - 1 - i))
    return cell_ids


def extract_interacted_cells(sim_params_dict):
    num_of_cells = int((sim_params_dict['length']**2)*sim_params_dict['density'])

    leaders_weights = sim_params_dict['leader_weights']
    leaders_ids = expand_weights_vector(leaders_weights, num_of_cells)

    followers_weights = sim_params_dict['follower_weights']
    followers_ids = expand_weights_vector(followers_weights, num_of_cells)

    return leaders_ids, followers_ids


def init_gc_df(sim_params_dict, cell_ids, columns):
    all_simulation_gc_df = {}
    leaders_ids, followers_ids = extract_interacted_cells(sim_params_dict)
    gc_df = pd.DataFrame(np.zeros((len(cell_ids) * len(cell_ids) - len(cell_ids))), columns=['temp'])
    gc_df['leader_cell'], gc_df['follower_cell'] = zip(*permutations(cell_ids, 2))
    gc_df['temp'] = gc_df.apply(
        lambda x: 0 if x['leader_cell'] in leaders_ids and x['follower_cell'] in followers_ids else
                    1 if x['follower_cell'] in leaders_ids and x['leader_cell'] in followers_ids else 2, axis=1)
    gc_df.set_index(['leader_cell', 'follower_cell'], inplace=True)
    for col in set(columns) - set(['frame_no', 'cell_id']):
        all_simulation_gc_df[col] = gc_df.copy()
    return all_simulation_gc_df


def run(paths_list=None, top_folder='', warm_up_window=0, separate_outputs=False, neigh_radius_ratio=0.95):
    if separate_outputs:
        separate_output_dict = {0: 'leader', 1: 'follower', 2: 'control'}
    else:
        separate_output_dict = {0: 'all'}
    if paths_list is None:
        root_path = 'C:/Users/hilon/OneDrive - post.bgu.ac.il/תואר שני/master/vicsek-v2/examples'
        files_path = Path(root_path) / top_folder
        paths_list = files_path.rglob('frames_dfs*.csv')
    for simulation_no, path in enumerate(paths_list):
        filename = path.name
        folder = path.parent.parts[-1]
        print(f'start handling file no. {simulation_no}: {folder}/{filename}')
        weight_param = folder[folder.rindex('_') + 1:]
        df, sim_params_dict = load_file(path)
        df['cell_id'] = df['cell_id'].astype(str)
        neigh_df = df[['cell_id', 'frame_no', 'x', 'y']].copy()
        df = df.drop(columns=['x', 'y'])
        cell_ids = df['cell_id'].unique()
        if simulation_no == 0:
            all_simulation_gc_df = init_gc_df(sim_params_dict, cell_ids, df.columns)
        granger_causality_matrix(df, features=set(df.columns) - set(['frame_no', 'cell_id']), neigh_df=neigh_df, maxlag=15,
                                 res_df=all_simulation_gc_df, res_col=weight_param, warm_up_window=warm_up_window,
                                 neigh_radius=sim_params_dict['radius'], neigh_radius_ratio=neigh_radius_ratio)
    for col, df in all_simulation_gc_df.items():
        for output_id, output_name in separate_output_dict.items():
            if output_name != 'all':
                sub_df = df[df['temp'] == output_id].copy()
            else:
                sub_df = df
            sub_df.drop(columns=['temp'], inplace=True)
            output_path = f'{files_path}/gc_{output_name}_{col}'
            sub_df.to_csv(f'{output_path}.csv')
            create_gc_heatmap(sub_df, output_path)


def create_gc_heatmap(df, output_path):
    fig, ax = plt.subplots(figsize=(70, 90))  # Sample figsize in inches
    plt.rc('font', size=25)  # controls default text sizes
    plt.rc('legend', fontsize=30)  # legend fontsize
    plt.ylabel('cells pairs', fontsize=30)
    plt.xlabel('follower weight', fontsize=30)
    sns.heatmap(df, annot=True, ax=ax, vmin=0, vmax=1)
    ax.tick_params(axis='y', size=15, rotation=0, labelsize=30)
    ax.tick_params(axis='x', size=15, labelsize=30)
    plt.savefig(f'{output_path}.jpg')


if __name__ == '__main__':
    # path = 'C:\\Users\\hilon\\OneDrive - post.bgu.ac.il\\תואר שני\\master\\vicsek-v2\\examples\\111221'
    # name = 'gc_y'
    # df = pd.read_csv(f'{path}/{name}.csv', index_col=[0, 1])
    # create_gc_heatmap(df, path, name)

    # run(folder='011221/04-12-21_1302')
    # run_only_gc(folder='011221/04-12-21_1304')
    # run_only_gc(folder='301121/30-11-21_1718')
    # run(top_folder='temp')
    run(top_folder='04042022_von_mise_noise/04-04-22_1348_150', warm_up_window=0,
        separate_outputs=True, neigh_radius_ratio=0.9)
