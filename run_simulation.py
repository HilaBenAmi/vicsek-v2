import os
from datetime import datetime
from vicsek.model import VicsekModel
from vicsek.visualize import ParticlesAnimation
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

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


def run_vic_snap(outpath, model_params: {}, steps=1, frames=100, suffix_folder=''):
    nested_path = f"{outpath}/{ts}{suffix_folder}/"
    for p in [outpath, nested_path]:
        try:
            os.mkdir(p)
        except FileExistsError:
            pass
    model_params_copy = model_params.copy()
    model_params_copy.update({'frames': frames})
    save_configuration(nested_path, model_params_copy)

    model = VicsekModel(**model_params)
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
    output_path = f'{path}\\examples\\26022022_leader_no_boundaries'
    # run_vic_snap(output_path,
    #             {"length": 10,
    #              "density": 0.02,
    #              "speed": 0.2,
    #              "noise": 1,
    #              "radius": 10,
    #              "leader_weights": [100, 1],
    #              "follower_weights": [1, 100],
    #              "memory_weights": [1],
    #              "seed": 1236}, frames=20)

    # for follower_weight in range(1,12):
    #     params = {"length": 10,
    #                  "density": 0.1,  # how many cells in one unit
    #                  "speed": 0.2,  # pixels per frame
    #                  "noise": 0,
    #                  "radius": 100,  # the radius in pixels to determine neighbors
    #                  "leader_weights": [15, 0],  # fill the gap with the right value
    #                  "follower_weights": [0,0,0,0,0,0,0,0,0,follower_weight], # from right to left
    #                  "memory_weights": [1],
    #                  "seed": 12236}
    #     run_vic_snap(output_path, params, suffix_folder=f'_follower_{follower_weight}')

    for leader_weight in range(1,2):
        params = {"length": 10,
                     "density": 0.1,  # how many cells in one unit
                     "speed": 0.2,  # pixels per frame
                     "noise": [0.3, 0],
                     "radius": 100,  # the radius in pixels to determine neighbors
                     "leader_weights": [leader_weight, 0],  # fill the gap with the right value
                     "follower_weights": [0,0,0,0,0,0,0,0,0,15], # fro right to left
                     "memory_weights": [1],
                     "seed": 12236}
        run_vic_snap(output_path, params, suffix_folder=f'_leader_{leader_weight}_leader_noise')