"""
Evaluate video generation metric - FVD
FVD tools from https://github.com/JunyaoHu/common_metrics_on_video_quality
"""

# set workdir
import os
import sys

sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from tqdm import tqdm
from utils.model_builder import get_model_tag, load_model_from_config
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fvd.calculate_fvd import calculate_fvd


def load_dlp_from_config(conf_path, ckpt_path=None):
    model, _ = load_model_from_config(conf_path, ckpt_path)
    return model


def eval_lpwm_fvd(model, device, config, timestep_horizon=16, val_mode='test', eval_dir='./',
                  cond_steps=1, use_all_ctx=False, batch_size=10, accelerator=None,
                  deterministic=False, n_videos_per_clip=1):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    n_views = config.get('n_views', 1)
    root = config['root']  # dataset root
    dataset_kwargs = config.get('dataset_kwargs', {})
    action_condition = config.get('action_condition', False)
    language_condition = config.get('language_condition', False)
    img_goal_condition = config.get('image_goal_condition', False)

    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon, mode=val_mode, image_size=image_size,
                                dataset_kwargs=dataset_kwargs)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)
    model_timestep_horizon = model.timestep_horizon
    cond_steps = model_timestep_horizon if cond_steps is None else cond_steps

    if accelerator is not None and not accelerator.is_main_process:
        disable_pbar = True
    else:
        disable_pbar = False

    # pixel value should be in [0, 1]!
    # real videos
    real_videos = []
    for i, batch in enumerate(tqdm(dataloader, disable=disable_pbar)):
        x = batch[0][:, :timestep_horizon]
        real_videos.append(x)
    real_videos = torch.cat(real_videos, dim=0)  # [n_real_videos, ts, ch, h, w]
    real_videos = real_videos.repeat(n_videos_per_clip, 1, 1, 1, 1)
    print(f'real videos: {real_videos.shape}')

    # generate videos
    generated_videos = []
    for k in range(n_videos_per_clip):
        for i, batch in enumerate(tqdm(dataloader, disable=disable_pbar)):
            x = batch[0][:, :timestep_horizon].to(device)
            actions = None if not action_condition else batch[1][:, :timestep_horizon].to(device)
            lang_str = None if not language_condition else batch[2]
            lang_embed = None if not language_condition else batch[3].to(device).to(device)
            x_goal = None if not img_goal_condition else batch[3].to(device)
            if n_views > 1:
                # expect: [bs, T, n_views, ...]
                x = x.permute(0, 2, 1, 3, 4, 5)
                x = x.reshape(-1, *x.shape[2:])  # [bs * n_views, T, ...]
                if x_goal is not None:
                    x_goal = x_goal.reshape(-1, *x_goal.shape[2:])  # [bs * n_views, ...]
                if actions is not None:
                    actions = actions.permute(0, 2, 1, 3)
                    actions = actions.reshape(-1, *actions.shape[2:])
            with torch.no_grad():
                generated = model.sample_from_x(x, cond_steps=cond_steps, num_steps=timestep_horizon - cond_steps,
                                                use_all_ctx=use_all_ctx, actions=actions, lang_embed=lang_embed,
                                                x_goal=x_goal, deterministic=deterministic)
                generated = generated.clamp(0, 1)
                assert x.shape[1] == generated.shape[1], "prediction and gt frames shape don't match"
            generated_videos.append(generated.cpu())
    generated_videos = torch.cat(generated_videos, dim=0)
    print(f'generated videos: {generated_videos.shape}')

    fvd = calculate_fvd(real_videos, generated_videos, device, method='styleganv', only_final=True)
    results = {'fvd': fvd}

    # save results
    if accelerator is not None:
        save_metric = accelerator.is_main_process
    else:
        save_metric = True
    if save_metric:
        path_to_conf = os.path.join(eval_dir, f'last_{val_mode}_fvd.json')
        with open(path_to_conf, "w") as outfile:
            json.dump(results, outfile, indent=2)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LPWM Video Prediction Evaluation")
    parser.add_argument("-d", "--dataset", type=str, default='balls',
                        help="dataset to use: ['balls', 'traffic', 'clevrer', 'obj3d128', ...]")
    parser.add_argument("-p", "--path", type=str,
                        help="path to model directory, e.g. ./310822_141959_balls_ddlp")
    parser.add_argument("--checkpoint", type=str,
                        help="direct path to model checkpoint, e.g. ./checkpoints/ddlp-obj3d128/obj3d_ddlp.pth",
                        default="")
    parser.add_argument("--use_last", action='store_true',
                        help="use the last checkpoint instead of best")
    parser.add_argument("--use_train", action='store_true',
                        help="use the train set for the predictions")
    parser.add_argument("--sample", action='store_true',
                        help="use stochastic (non-deterministic) predictions")
    parser.add_argument("--cpu", action='store_true',
                        help="use cpu for inference")
    parser.add_argument("--ctx", action='store_true',
                        help="use context for inference")
    parser.add_argument("-c", "--cond_steps", type=int, help="the initial number of frames for predictions", default=-1)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=10)
    parser.add_argument("--horizon", type=int, help="timestep horizon for prediction", default=50)
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix used for model saving")
    parser.add_argument("--n_videos_per_clip", type=int, help="n_videos to generate per data sample", default=1)
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    checkpoint_path = args.checkpoint
    # ds = args.dataset
    use_train = args.use_train
    cond_steps = args.cond_steps
    timestep_horizon = args.horizon
    batch_size = args.batch_size
    use_cpu = args.cpu
    use_ctx = args.ctx
    deterministic = not args.sample
    prefix = args.prefix
    n_videos_per_clip = args.n_videos_per_clip
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

    # create dir for results
    pred_dir = os.path.join(dir_path, 'eval')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames
    cond_steps = cond_steps if cond_steps > 0 else config['timestep_horizon']
    val_mode = 'train' if use_train else 'test'
    results = eval_lpwm_fvd(model, device, config=config, timestep_horizon=timestep_horizon, val_mode=val_mode,
                            eval_dir=pred_dir,
                            cond_steps=cond_steps, use_all_ctx=use_ctx, batch_size=batch_size, accelerator=None,
                            deterministic=deterministic, n_videos_per_clip=n_videos_per_clip)
    print(f'results: {results}')
