import os
from datetime import datetime
from vicsek.model import VicsekModel
from vicsek.visualize import ParticlesAnimation
from matplotlib import pyplot as plt
from tqdm import tqdm

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


def run_vic_snap(outpath, model_params: {}, steps=1, frames=100):
    try:
        os.mkdir(outpath)
    except FileExistsError:
        pass

    model = VicsekModel(**model_params)
    n = len(str(steps * frames))

    # Save initial config
    fig = model.view()
    fig.savefig(f"{outpath}/snap_{ts}.png")
    plt.close(fig)

    # pbar = tqdm(range(frames))
    pbar = range(frames)
    for i in pbar:
        model.evolve(steps=steps)
        fig = model.view(point_annotate=False)
        fig.savefig(f"{outpath}/snap_{str(model.current_step).zfill(n)}.jpg")
        plt.close(fig)

    # pbar.close()
    create_video(outpath)


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


if __name__ == "__main__":
    path = os.getcwd()
    output_path = f'{path}\\examples\\021121'
    # run_vic_ani(output_path,
    #             {"length": 10,
    #              "density": 0.05,
    #              "speed": 0.2,
    #              "noise": 1,
    #              "radius": 5,
    #              "leader_weights": [100, 1],
    #              "follower_weights": [1, 100],
    #              "memory_weights": [1],
    #              "seed": 1234})

    run_vic_snap(output_path,
                {"length": 10,
                 "density": 0.5,  # how many cells in one unit
                 "speed": 0.2,  # pixels per frame
                 "noise": 2,
                 "radius": 2,  # the radius in pixels to determine neighbors
                 "leader_weights": [8, 10, 5, 0],
                 "follower_weights": [0, 1, 2, 8],
                 "memory_weights": [4],
                 "seed": 12235}, frames=100)
