import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

# import logging
# logger = logging.getLogger('my_module_name')
warnings.filterwarnings("ignore")

MAXLAG = 15
TEST = 'ssr_chi2test'


def load_file(path):
    df = pd.read_csv(path)
    return df


def var_test(df):
    opt_lag_dict = {'opt_lag': [], 'min_aic': []}
    for cell_id in df['cell_id'].unique():
        # print(f"** start test stationary - cell_id: {cell_id} **")
        cell_trajectory = df[df['cell_id'] == cell_id].copy().set_index('frame_no').drop(columns=['cell_id'])
        cell_trajectory['dx'] = cell_trajectory['x'].diff()
        cell_trajectory['dy'] = cell_trajectory['y'].diff()
        cell_trajectory = cell_trajectory.drop(columns=['x', 'y'])
        cell_trajectory = cell_trajectory.dropna()
        model = VAR(cell_trajectory)
        # order_results = model.select_order(MAXLAG)
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


def stationary_test(df, features):
    stationary_results_adf = {}
    for col in features:
        stationary_results_adf[col] = []
    stationary_results_kpss = stationary_results_adf.copy()
    for cell_id in df['cell_id'].unique():
        # print(f"** start test stationary - cell_id: {cell_id} **")
        cell_trajectory = df[df['cell_id'] == cell_id].copy().reset_index(drop=True)
        cell_trajectory['dx'] = cell_trajectory['x'].diff()
        cell_trajectory['dy'] = cell_trajectory['y'].diff()
        cell_trajectory = cell_trajectory.drop(columns=['x', 'y'])
        cell_trajectory = cell_trajectory.dropna()
        for col in features:
            p_value_adf = adf_test(cell_trajectory[col])
            p_value_kpss = kpss_test(cell_trajectory[col])
            stationary_results_adf[col].append(True if p_value_adf < 0.1 else False)
            stationary_results_kpss[col].append(False if p_value_kpss < 0.1 else True)

    avg_stat_dict = {'adf': {}, 'kpss': {}}
    for col in features:
        adf_avg_stat = np.mean(stationary_results_adf[col])
        kpss_avg_stat = np.mean(stationary_results_kpss[col])
        avg_stat_dict['adf'][col] = adf_avg_stat
        avg_stat_dict['kpss'][col] = kpss_avg_stat
        print(f"ADF stationary results: {adf_avg_stat} of the cells are stat in their {col}")
        print(f"KPSS stationary results: {kpss_avg_stat} of the cells are stat in their {col}")
    return avg_stat_dict


def granger_causality_matrix(data, features, variables, res_df, test='ssr_chi2test', maxlag=MAXLAG, verbose=False, path=None):
    """
        The row are the response (y) and the columns are the predictors (x)
        If a given p-value is < significance level (0.05), we can reject the null hypothesis and conclude that walmart_x Granger causes apple_y.
    """
    gc_per_feature_list = []
    for col in features:
        print(f"** Start calculate GC on {col} **")
        time_series_per_cell = {}
        feature_data = data[['frame_no', 'cell_id', col]].copy()
        for cell_id in feature_data['cell_id'].unique():
            time_series_per_cell[cell_id] = feature_data[feature_data['cell_id'] == cell_id].set_index('frame_no')[
                col].diff()[1:]
        time_frame_df = pd.DataFrame(time_series_per_cell)
        gc_df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in gc_df.columns:
            for r in gc_df.index:
                test_result = grangercausalitytests(time_frame_df[[r, c]], maxlag=maxlag, verbose=verbose)
                p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                min_p_value_lag = np.argmin(p_values)
                gc_df.loc[r, c] = min_p_value
        gc_df.columns = [str(var) + '_x' for var in variables]
        gc_df.index = [str(var) + '_y' for var in variables]
        gc_per_feature_list.append(gc_df)
        if path:
            gc_df.to_csv(f'{path}/gc_{str(maxlag)}_{col}.csv')
            gc_df = gc_df[gc_df < 0.05]
            gc_df.to_csv(f'{path}/significant_gc_{str(maxlag)}_{col}.csv')
    return gc_per_feature_list


def run_all_process(paths_list=None, folder=''):
    global all_simulations
    all_simulations = {'dist': [], 'dist_diff': [], 'ma_dist_diff': [], 'theta_dist': [], 'theta_dist_diff': [],
                       'direction_diff': []}
    if paths_list is None:
        root_path = 'C:/Users/hilon/OneDrive - post.bgu.ac.il/תואר שני/master/vicsek-v2/examples'
        folder = Path(root_path) / folder
        paths_list = folder.rglob('frames_dfs*.csv')
    for simulation_no, path in enumerate(paths_list):
        file_ts = path.name
        print(f'start handling file no. {simulation_no}: {file_ts}')
        df = load_file(path)
        df = df.drop(columns=['heading'])
        stationary_results = stationary_test(df, features=['dx', 'dy'])
        col_test_list = ['dx', 'dy']
        for col_test in col_test_list:
            if stationary_results['adf'][col_test] > 0.7:
                print(f"passed ADF stationary test on {col_test} feature")
            if stationary_results['kpss'][col_test] > 0.7:
                print(f"passed KPSS stationary test on {col_test} feature")
        opt_lag = var_test(df)
        granger_causality_matrix(df, features=set(df.columns) - set(['frame_no', 'cell_id']),
                                 variables=df['cell_id'].unique(), path=path.parent)


def run_only_gc(paths_list=None, folder=''):
    all_simulation_gc_df = pd.DataFrame()
    if paths_list is None:
        root_path = 'C:/Users/hilon/OneDrive - post.bgu.ac.il/תואר שני/master/vicsek-v2/examples'
        folder = Path(root_path) / folder
        paths_list = folder.rglob('frames_dfs*.csv')
    for simulation_no, path in enumerate(paths_list):
        file_ts = path.name
        print(f'start handling file no. {simulation_no}: {folder}/{file_ts}')
        weight_param = folder[folder.rindex('_') + 1:]
        all_simulation_gc_df[weight_param] = np.nan
        df = load_file(path)
        df = df.drop(columns=['heading'])
        granger_causality_matrix(df, features=set(df.columns) - set(['frame_no', 'cell_id']),
                                 variables=df['cell_id'].unique(), path=path.parent, maxlag=10, res_df=all_simulation_gc_df)

def run(paths_list=None, folder='', all_process=True):
    all_simulation_gc_df = pd.DataFrame()
    if paths_list is None:
        root_path = 'C:/Users/hilon/OneDrive - post.bgu.ac.il/תואר שני/master/vicsek-v2/examples'
        folder = Path(root_path) / folder
        paths_list = folder.rglob('frames_dfs*.csv')
    for simulation_no, path in enumerate(paths_list):
        file_ts = path.name
        print(f'start handling file no. {simulation_no}: {folder}/{file_ts}')
        weight_param = folder[folder.rindex('_') + 1:]
        all_simulation_gc_df[weight_param] = np.nan
        df = load_file(path)
        df = df.drop(columns=['heading'])
        if all_process:
            run_all_process(paths_list=None, folder='')
        else:
            granger_causality_matrix(df, features=set(df.columns) - set(['frame_no', 'cell_id']),
                                     variables=df['cell_id'].unique(), path=path.parent, maxlag=10,
                                     res_df=all_simulation_gc_df)


if __name__ == '__main__':
    # run(folder='011221/04-12-21_1302')
    # run_only_gc(folder='011221/04-12-21_1304')
    # run_only_gc(folder='301121/30-11-21_1718')
    run_only_gc(folder='081221')
