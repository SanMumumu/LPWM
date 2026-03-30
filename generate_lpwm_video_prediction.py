"""
Script to generate conditional video prediction from a pre-trained DDLP
"""
# imports
import os
import argparse
import json
from tqdm import tqdm
from utils.model_builder import get_model_tag, load_model_from_config
from eval.eval_model import animate_trajectory_lpwm
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def load_dlp_from_config(conf_path, ckpt_path=None):
    model, _ = load_model_from_config(conf_path, ckpt_path)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LPWM Video Generation")
    parser.add_argument("-d", "--dataset", type=str, default='sketchy',
                        help="dataset to use: ['sketchy', 'bridge', 'obj3d128', ...]")
    parser.add_argument("-p", "--path", type=str,
                        help="path to model directory, e.g. ./checkpoints/sketchy")
    parser.add_argument("--checkpoint", type=str,
                        help="direct path to model checkpoint, e.g. ./checkpoints/sketchy/sketchy.pth",
                        default="")
    parser.add_argument("--use_last", action='store_true',
                        help="use the last checkpoint instead of best")
    parser.add_argument("--use_train", action='store_true',
                        help="use the train set for the predictions")
    parser.add_argument("--sample", action='store_true',
                        help="use stochastic (non-deterministic) predictions")
    parser.add_argument("--cpu", action='store_true',
                        help="use cpu for inference")
    parser.add_argument("-c", "--cond_steps", type=int, help="the initial number of frames for predictions", default=-1)
    parser.add_argument("-n", "--num_predictions", type=int, help="number of animations to generate", default=5)
    parser.add_argument("--horizon", type=int, help="timestep horizon for prediction", default=50)
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix used for model saving")
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    checkpoint_path = args.checkpoint
    # ds = args.dataset
    use_train = args.use_train
    cond_steps = args.cond_steps
    timestep_horizon = args.horizon
    num_predictions = args.num_predictions
    use_cpu = args.cpu
    deterministic = not args.sample
    prefix = args.prefix
    # load model config
    conf_path = os.path.join(dir_path, 'hparams.json')
    with open(conf_path, 'r') as f:
        config = json.load(f)
    pref = get_model_tag(config)
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ds = config['ds']

    model_ckpt_name = f'{ds}_{pref}{prefix}.pth'
    # model_best_ckpt_name = f'{ds}_ddlp{prefix}_best.pth'
    model_best_ckpt_name = f'{ds}_{pref}{prefix}_best_lpips.pth'
    use_last = args.use_last if os.path.exists(os.path.join(dir_path, f'saves/{model_best_ckpt_name}')) else True

    if checkpoint_path.endswith('.pth'):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(dir_path, f'saves/{model_ckpt_name if use_last else model_best_ckpt_name}')

    print(f'checkpoint path: {ckpt_path}')

    model = load_dlp_from_config(conf_path, ckpt_path)
    model = model.to(device)
    model.eval()
    # create dir for videos
    pred_dir = os.path.join(dir_path, 'videos')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames
    cond_steps = cond_steps if cond_steps > 0 else config['timestep_horizon']
    print(f'conditional input frames: {cond_steps}')
    print(f'deterministic predictions (use only mu): {deterministic}')
    # generate
    print('generating animations...')
    animate_trajectory_lpwm(model, config, epoch=0, device=device, fig_dir=pred_dir,
                            timestep_horizon=timestep_horizon,
                            num_trajetories=num_predictions, accelerator=None, train=use_train, prefix=prefix,
                            cond_steps=cond_steps, deterministic=deterministic)
