import os
from datetime import datetime
from vicsek.model import VicsekModel
from vicsek.visualize import ParticlesAnimation
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

FNAME = "animation"
ts = datetime.now().strftime('%d-%m-%y_%H%M')


def run_vic_ani(outpath, model_params: {}):
    # ts = datetime.now().strftime('%d-%m-%y_%H%M')
    try:
        os.mkdir(outpath)
    except FileExistsError:
        pass

    model = VicsekModel(**model_params)

    animator = ParticlesAnimation(model)
    animation = animator.animate(interval=100)
    animation.save(f'{outpath}/{FNAME}_{ts}.gif')


def run_vic_snap(outpath, model_params: {}, steps=1, frames=100, suffix_folder='', run_gc=False, separate_outputs=False,
                 neigh_radius_ratio=0.95):
    last_folder = f"{ts}{suffix_folder}"
    folders_list = outpath.split('/') + [last_folder]
    nested_path = folders_list[0] + '/'
    for p in folders_list[1:]:
        nested_path += p + '/'
        try:
            os.mkdir(nested_path)
        except FileExistsError:
            pass
    model_params_copy = model_params.copy()
    model_params_copy.update({'frames': frames})
    save_configuration(nested_path, model_params_copy)

    model = VicsekModel(**model_params)
    model.calculate_plot_size(frames)
    n = len(str(steps * frames))

    # Save initial config
    fig = model.view()
    fig.savefig(f"{nested_path}/snap_{ts}.png")
    plt.close(fig)

    pbar = tqdm(range(frames))
    # pbar = range(frames)
    for i in pbar:
        model.evolve(steps=steps)
        fig = model.view(point_annotate=False)
        fig.savefig(f"{nested_path}/snap_{str(model.current_step).zfill(n)}.jpg")
        plt.close(fig)

    pbar.close()
    create_video(nested_path)

    simulation_positions_df = pd.concat(model.frames_dfs())
    simulation_positions_df.to_csv(f"{nested_path}/frames_dfs_{ts}.csv")
    if run_gc:
        from granger_causality import run
        run(top_folder=nested_path, separate_outputs=separate_outputs, neigh_radius_ratio=neigh_radius_ratio)


def save_configuration(outpath, model_params):
    import json
    with open(f'{outpath}/simulation_params_{ts}.json', 'w') as f:
        json.dump(model_params, f, indent=2)


def create_video(file_path):
    import glob
    import matplotlib.image as mpimg
    import cv2

    img_array = []
    for filename in glob.glob(f'{file_path}/*.jpg'):
        img = mpimg.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        try:
            os.remove(filename)
        except:
            pass
    if len(img_array)>0:
        out = cv2.VideoWriter(f'{file_path}/video_plots_{ts}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, size)
    else:
        print(f"no images")
        return
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def create_outputs(particles_coords_list, folder_path):
    coords_dfs_list = []
    for cell_idx, cell_coords in particles_coords_list.items():
        coords_dfs_list.append(pd.DataFrame(cell_coords, columns=[f'x_{cell_idx}',
                                                                  f'y_{cell_idx}',
                                                                  f't_{cell_idx}']))
    full_coords_df = pd.concat(coords_dfs_list, axis=1)
    csv_path = f'{folder_path}\\all_cells_coords_df.csv'
    full_coords_df.to_csv(csv_path)
    create_video(folder_path)
    return csv_path


if __name__ == "__main__":
    path = os.getcwd()
    output_path = f'{path}\\examples\\22062022_experiments'
    # run_vic_snap(output_path,
    #             {"length": 10,
    #              "density": 0.1,
    #              "speed": 0.2,
    #              "noise": [0.3, 0.1, 0.2],
    #              "radius": 20,
    #              "leader_weights": [1, 0],
    #              "follower_weights": [0, 0.8, 0.3],
    #              "memory_weights": [0.7, 0.1, 0.5],
    #              "rw_type": 'CRW',
    #              "seed": 16555,
    #              "center_start": True}, frames=150, suffix_folder=f'_10_cells_center_all_followers_r20_GC_0', run_gc=True, separate_outputs=True,
    #              neigh_radius_ratio=0.9)

    # run_vic_snap(output_path,
    #              {"length": 10,
    #               "density": 0.5,
    #               "speed": 0.2,
    #               "noise": [0.3, 0.1, 0.1, 0.1, 0.2],
    #               "radius": 20,
    #               "leader_weights": [1, 0],
    #               "follower_weights": [0, 0.8, 0.7, 0.6, 0.3],
    #               "memory_weights": [0.7, 0.1, 0.2, 0.3, 0.5],
    #               "rw_type": 'CRW',
    #               "seed": 16555,
    #               "center_start": False}, frames=150, suffix_folder=f'_50_cells_all_followers_r20_0', run_gc=True,
    #              separate_outputs=True,
    #              neigh_radius_ratio=0.9)
    #
    # run_vic_snap(output_path,
    #             {"length": 10,
    #              "density": 0.5,
    #              "speed": 0.2,
    #              "noise": [0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2],
    #              "radius": 20,
    #              "leader_weights": [1, 1, 1, 0],
    #              "follower_weights": [0, 0.1, 0.1, 0.8, 0.7, 0.6, 0.3],
    #              "memory_weights": [0.7, 0.6, 0.6, 0.1, 0.2, 0.3, 0.5],
    #              "rw_type": 'CRW',
    #              "seed": 16555,
    #              "center_start": True}, frames=150, suffix_folder=f'_50_cells_center_3_leaders_r20_0', run_gc=True, separate_outputs=True,
    #              neigh_radius_ratio=0.9)
    #
    # run_vic_snap(output_path,
    #              {"length": 10,
    #               "density": 0.5,
    #               "speed": 0.2,
    #               "noise": [0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2],
    #               "radius": 20,
    #               "leader_weights": [1, 0],
    #               "follower_weights": [0, 0.1, 0.1, 0.8, 0.7, 0.6, 0.3],
    #               "memory_weights": [0.7, 0.6, 0.6, 0.1, 0.2, 0.3, 0.5],
    #               "rw_type": 'CRW',
    #               "seed": 16555,
    #               "center_start": False}, frames=150, suffix_folder=f'_50_cells_3_leaders_r20_0', run_gc=True,
    #              separate_outputs=True,
    #              neigh_radius_ratio=0.9)
    #
    # run_vic_snap(output_path,
    #             {"length": 10,
    #              "density": 0.5,
    #              "speed": 0.2,
    #              "noise": [0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2],
    #              "radius": 10,
    #              "leader_weights": [1, 1, 1, 0],
    #              "follower_weights": [0, 0.1, 0.1, 0.8, 0.7, 0.6, 0.3],
    #              "memory_weights": [0.7, 0.6, 0.6, 0.1, 0.2, 0.3, 0.5],
    #              "rw_type": 'CRW',
    #              "seed": 16555,
    #              "center_start": True}, frames=150, suffix_folder=f'_50_cells_center_3_leaders_r10_0', run_gc=True, separate_outputs=True,
    #              neigh_radius_ratio=0.9)
    #
    # run_vic_snap(output_path,
    #              {"length": 10,
    #               "density": 0.5,
    #               "speed": 0.2,
    #               "noise": [0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.2],
    #               "radius": 10,
    #               "leader_weights": [1, 0],
    #               "follower_weights": [0, 0.1, 0.1, 0.8, 0.7, 0.6, 0.3],
    #               "memory_weights": [0.7, 0.6, 0.6, 0.1, 0.2, 0.3, 0.5],
    #               "rw_type": 'CRW',
    #               "seed": 16555,
    #               "center_start": False}, frames=150, suffix_folder=f'_50_cells_3_leaders_r10_0', run_gc=True,
    #              separate_outputs=True,
    #              neigh_radius_ratio=0.9)

    output_path = f'{path}/examples/22062022_experiments/10_cells_center_one_followers_r20_v0'
    for weight in [3, 6, 9]:
        weight = np.round(weight/10, 1)
        params = {"length": 10,
                  "density": 0.1,
                  "speed": 0.3,
                  "noise": [0.3, 0.1, 0.6],
                  "radius": 20,
                  "leader_weights": [1, 0],
                  "follower_weights": [0, weight, 0],
                  "memory_weights": [0.7, np.round(0.9-weight, 1), 0.4],
                  "rw_type": 'CRW',
                  "seed": 1655,
                  "center_start": True}
        run_vic_snap(output_path, params, frames=150,
                     suffix_folder=f'_Follower-Weight-{weight}',
                     run_gc=False, separate_outputs=True,
                     neigh_radius_ratio=0.9)

    # for weight in range(1, 10):
    #     leader_weight = weight / 10
    #     leader_noise = (1-leader_weight)/3
    #     leader_memory = (1-leader_weight)*2/3
    #     params = {"length": 10,
    #                  "density": 0.1,  # how many cells in one unit
    #                  "speed": 0.2,  # pixels per frame
    #                  "noise": [leader_noise, 0],
    #                  "radius": 10,  # the radius in pixels to determine neighbors
    #                  "leader_weights": [leader_weight, 0],  # fill the gap with the right value
    #                  "follower_weights": [0, 0.5, 0],  # from right to left
    #                  "memory_weights": [leader_memory, 0.5, 1],
    #                  "seed": 12236}
    #     run_vic_snap(output_path, params, suffix_folder=f'_leader_weight_{weight}', frames=200)


    # for weight in range(1, 10):
    #     follower_weight = weight / 10
    #     follower_noise = np.round((1-follower_weight)/3, 2)
    #     follower_memory = np.round((1-follower_weight)*2/3, 2)
    #     params = {"length": 10,
    #                  "density": 0.1,  # how many cells in one unit
    #                  "speed": 0.2,  # pixels per frame
    #                  "noise": [0.2, 0],
    #                  "radius": 30,  # the radius in pixels to determine neighbors
    #                  "leader_weights": [0.6, 0],  # fill the gap with the right value
    #                  "follower_weights": [0, follower_weight, 0.3],  # from right to left
    #                  "memory_weights": [0.2, 1-follower_weight, 0.7],
    #                  "seed": 12236}
    #     run_vic_snap(output_path, params, suffix_folder=f'_follower_weight_{weight}', frames=200)