import warnings
from pathlib import Path

from copy import deepcopy
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

from itertools import permutations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# import logging
# logger = logging.getLogger('my_module_name')
warnings.filterwarnings("ignore")

MAXLAG = 15
TEST = 'ssr_chi2test'


def load_file(path):
    df = pd.read_csv(path)
    return df


def var_test(df):
    opt_lag_list = []
    for cell_id in df['cell_id'].unique():
        # print(f"** start test stationary - cell_id: {cell_id} **")
        cell_trajectory = df[df['cell_id'] == cell_id].copy().set_index('frame_no').drop(columns=['cell_id'])
        cell_trajectory['dx'] = cell_trajectory['x'].diff()
        cell_trajectory['dy'] = cell_trajectory['y'].diff()
        cell_trajectory = cell_trajectory.drop(columns=['x', 'y'])
        cell_trajectory = cell_trajectory.dropna()
        model = VAR(cell_trajectory)
        lags_results = model.select_order(MAXLAG)
        lags = [lag_results.aic, lag_results.bic]
        opt_lag_list.append(lags[np.argmin(lags)])
        return
        # # print(order_results.ics)
        # print(order_results.selected_orders)
        # results = model.fit(MAXLAGs=order_results.selected_orders['aic'], ic='aic')
        # print(results.summary())
        # print('*******************************')

        aic_list = []
        for i in range(0, MAXLAG + 1):
            result = model.fit(i)
            aic_list.append(result.aic)
            # print('Lag Order =', i)
            # print('AIC : ', result.aic, '\n')
            # print('BIC : ', result.bic)
        # print(f"The optimal lag: {np.argmin(aic_list)} with minimum aic: {np.min(aic_list)}")
        opt_lag = np.argmin(aic_list)
        min_aic = np.min(aic_list)
        opt_lag_dict['opt_lag'].append(opt_lag)
        opt_lag_dict['min_aic'].append(min_aic)
    optimal_lag = max(opt_lag_dict['opt_lag'], key=opt_lag_dict['opt_lag'].count)
    print(f"The optimal lag that appears the most: {optimal_lag}")
    print(
        f"The final optimal lag: {opt_lag_dict['opt_lag'][np.argmin(opt_lag_dict['min_aic'])]} with final minimum aic: {np.min(opt_lag_dict['min_aic'])}")
    return optimal_lag


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


def stationary_test(df, cell_id, warm_up_window):
    # print(f"** start test stationary - cell_id: {cell_id} **")
    cell_trajectory = df[df['cell_id'] == cell_id].iloc[warm_up_window:].copy().reset_index(drop=True)
    cell_trajectory['dx'] = cell_trajectory['x'].diff()
    cell_trajectory['dy'] = cell_trajectory['y'].diff()
    cell_trajectory = cell_trajectory.drop(columns=['x', 'y'])
    cell_trajectory = cell_trajectory.dropna().reset_index(drop=True)
    is_stat_per_feature = []
    for col in ['dx', 'dy']:
        p_value_adf = adf_test(cell_trajectory[col])
        p_value_kpss = kpss_test(cell_trajectory[col])
        is_stat_per_feature.append(True if p_value_adf <= 0.05 or p_value_kpss >= 0.05 else False)
    # cell_trajectory.to_csv(f'./stat_files/{all(is_stat_per_feature)}_stat_cell_{cell_id}.csv')
    return all(is_stat_per_feature)


def granger_causality_matrix(data, features, res_df, res_col, test='ssr_chi2test', maxlag=MAXLAG, verbose=False,
                             warm_up_window=0):
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
            is_stat = stationary_test(data, cell_id, warm_up_window)
            if not is_stat:
                continue
            time_series_per_cell[cell_id] = feature_data[feature_data['cell_id'] == cell_id].iloc[
                                            warm_up_window:].set_index('frame_no')[col].diff()[1:]
        time_frame_df = pd.DataFrame(time_series_per_cell)
        for p, r in res_df[col].index:
            if p not in time_frame_df or r not in time_frame_df:
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
                opt_lag = np.max([np.max(lags), 1])  ## TODO change the 1
                if np.max(lags) == 0:
                    print(f"both lags are 0; 1 will be taken")
            gc_result = grangercausalitytests(time_frame_df[[r, p]], maxlag=opt_lag, verbose=verbose)
            p_value = gc_result[opt_lag][0][test][1]
            res_df[col].loc[(p, r), int(res_col)] = p_value
    return res_df


def run_pre_tests(df):
    col_test_list = ['dx', 'dy']
    stationary_results_combo = {}
    for col in col_test_list:
        stationary_results_combo[col] = []
    stationary_results_combo['total'] = len(df['cell_id'].unique()) * [True]
    for cell_id in df['cell_id'].unique():
        is_stat = stationary_test(df, cell_id)
        stationary_results_combo['total'] = is_stat
    stat_indexes = np.where(stationary_results_combo['total'])[0]
    return stat_indexes


def run(paths_list=None, top_folder='', warm_up_window=0):
    all_simulation_gc_df = {}
    if paths_list is None:
        root_path = 'C:/Users/hilon/OneDrive - post.bgu.ac.il/תואר שני/master/vicsek-v2/examples'
        files_path = Path(root_path) / top_folder
        paths_list = files_path.rglob('frames_dfs*.csv')
    for simulation_no, path in enumerate(paths_list):
        filename = path.name
        folder = path.parent.parts[-1]
        print(f'start handling file no. {simulation_no}: {folder}/{filename}')
        weight_param = folder[folder.rindex('_') + 1:]
        df = load_file(path)
        df = df.drop(columns=['heading'])
        df['cell_id'] = df['cell_id'].astype(str)
        cell_ids = df['cell_id'].unique()
        if simulation_no == 0:
            gc_df = pd.DataFrame(np.zeros((len(cell_ids) * len(cell_ids) - len(cell_ids))), columns=['temp'])
            gc_df['cell_predict'], gc_df['cell_response'] = zip(*permutations(cell_ids, 2))
            gc_df.set_index(['cell_predict', 'cell_response'], inplace=True)
            gc_df.drop(columns=['temp'], inplace=True)
            for col in set(df.columns) - set(['frame_no', 'cell_id']):
                all_simulation_gc_df[col] = gc_df.copy()
        granger_causality_matrix(df, features=set(df.columns) - set(['frame_no', 'cell_id']), maxlag=15,
                                   res_df=all_simulation_gc_df, res_col=weight_param, warm_up_window=warm_up_window)
    for col, df in all_simulation_gc_df.items():
        df.to_csv(f'{files_path}/gc_{col}.csv')
        create_gc_heatmap(df, files_path, col)


def create_gc_heatmap(df, files_path, col):
    fig, ax = plt.subplots(figsize=(70, 90))  # Sample figsize in inches
    plt.rc('font', size=25)  # controls default text sizes
    plt.rc('legend', fontsize=30)  # legend fontsize
    plt.ylabel('cells pairs', fontsize=30)
    plt.xlabel('follower weight', fontsize=30)
    sns.heatmap(df, annot=True, ax=ax)
    ax.tick_params(axis='y', size=15, rotation=0, labelsize=30)
    ax.tick_params(axis='x', size=15, labelsize=30)
    plt.savefig(f'{files_path}/gc_{col}.jpg')


if __name__ == '__main__':
    # path = 'C:\\Users\\hilon\\OneDrive - post.bgu.ac.il\\תואר שני\\master\\vicsek-v2\\examples\\111221'
    # name = 'gc_y'
    # df = pd.read_csv(f'{path}/{name}.csv', index_col=[0, 1])
    # create_gc_heatmap(df, path, name)

    # run(folder='011221/04-12-21_1302')
    # run_only_gc(folder='011221/04-12-21_1304')
    # run_only_gc(folder='301121/30-11-21_1718')
    # run(top_folder='temp')
    run(top_folder='01032022/01-03-22_1240_random_avg_270_4', warm_up_window=0)
