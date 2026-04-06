"""
Evaluate image metrics such as LPIPS, PSNR and SSIM using PIQA,
"""

# set workdir
import os
import sys

sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from tqdm import tqdm
from utils.model_builder import build_model_from_config, get_model_tag, load_model_from_config
# datasets
from datasets.get_dataset import get_video_dataset, get_image_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from piqa import PSNR, LPIPS, SSIM
except ImportError:
    print("piqa library required to compute image metrics")
    raise SystemExit


def disable_sparse_router(model):
    if hasattr(model, "particle_router"):
        model.particle_router = None
    dyn_module = getattr(model, "dyn_module", None)
    if dyn_module is not None and hasattr(dyn_module, "particle_router"):
        dyn_module.particle_router = None
    ctx_module = getattr(model, "ctx_module", None)
    if ctx_module is not None:
        backbone = getattr(ctx_module, "backbone", None)
        if backbone is not None and hasattr(backbone, "particle_router"):
            backbone.particle_router = None
        if hasattr(ctx_module, "particle_router"):
            ctx_module.particle_router = None
    encoder_module = getattr(model, "encoder_module", None)
    if encoder_module is not None:
        ctx_enc = getattr(encoder_module, "ctx_enc", None)
        backbone = getattr(ctx_enc, "backbone", None) if ctx_enc is not None else None
        if backbone is not None and hasattr(backbone, "particle_router"):
            backbone.particle_router = None
    return model


class ImageMetrics(nn.Module):
    """
    A class to calculate visual metrics between generated and ground-truth images
    """

    def __init__(self, metrics=('ssim', 'psnr', 'lpips')):
        super().__init__()
        self.metrics = metrics
        self.ssim = SSIM(reduction='none') if 'ssim' in self.metrics else None
        self.psnr = PSNR(reduction='none') if 'psnr' in self.metrics else None
        self.lpips = LPIPS(network='vgg', reduction='none') if 'lpips' in self.metrics else None

    @torch.no_grad()
    def forward(self, x, y):
        # x, y: [batch_size, 3, im_size, im_size] in [0,1]
        results = {}
        if self.ssim is not None:
            results['ssim'] = self.ssim(x, y)
        if self.psnr is not None:
            results['psnr'] = self.psnr(x, y)
        if self.lpips is not None:
            results['lpips'] = self.lpips(x, y)
        return results


def eval_lpwm_im_metric(model, device, config, timestep_horizon=50, val_mode='val', eval_dir='./',
                        cond_steps=10, use_all_ctx=False,
                        metrics=('ssim', 'psnr', 'lpips'), batch_size=32, verbose=False, accelerator=None,
                        deterministic=True):
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

    # image metric instance
    evaluator = ImageMetrics(metrics=metrics).to(device)

    results = {}
    ssims = []
    psnrs = []
    lpipss = []

    if accelerator is not None and not accelerator.is_main_process:
        disable_pbar = True
    else:
        disable_pbar = False

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
            results = evaluator(x[:, cond_steps:].reshape(-1, *x.shape[2:]),
                                generated[:, cond_steps:].reshape(-1, *generated.shape[2:]))
        # [batch_size * T]
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        ssims = torch.cat(ssims, dim=0)
        mean_ssim = ssims.mean().data.cpu().item()
        std_ssim = ssims.std().data.cpu().item()
        results['ssim'] = mean_ssim
        results['ssim_std'] = std_ssim
    if 'psnr' in metrics:
        psnrs = torch.cat(psnrs, dim=0)
        mean_psnr = psnrs.mean().data.cpu().item()
        std_psnr = psnrs.std().data.cpu().item()
        results['psnr'] = mean_psnr
        results['psnr_std'] = std_psnr
    if 'lpips' in metrics:
        lpipss = torch.cat(lpipss, dim=0)
        mean_lpips = lpipss.mean().data.cpu().item()
        std_lpips = lpipss.std().data.cpu().item()
        results['lpips'] = mean_lpips
        results['lpips_std'] = std_lpips

    # save results
    if accelerator is not None:
        save_metric = accelerator.is_main_process
    else:
        save_metric = True
    if save_metric:
        path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
        with open(path_to_conf, "w") as outfile:
            json.dump(results, outfile, indent=2)

    del evaluator  # clear memory

    return results


def eval_dlp_im_metric(model, device, config, val_mode='val', eval_dir='./',
                       metrics=('ssim', 'psnr', 'lpips'), batch_size=32, verbose=False, accelerator=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    ds = config['ds']
    ch = config['ch']  # image channels
    image_size = config['image_size']
    root = config['root']  # dataset root
    dataset_kwargs = config.get('dataset_kwargs', {})
    # use_tracking = config['use_tracking']
    dataset = get_image_dataset(ds, root, mode=val_mode, image_size=image_size, dataset_kwargs=dataset_kwargs)

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False)

    # image metric instance
    evaluator = ImageMetrics(metrics=metrics).to(device)

    results = {}
    ssims = []
    psnrs = []
    lpipss = []
    if accelerator is not None and not accelerator.is_main_process:
        disable_pbar = True
    else:
        disable_pbar = False

    for i, batch in enumerate(tqdm(dataloader, disable=disable_pbar)):
        x = batch[0].to(device)
        if len(x.shape) == 4:
            # [bs, ch, h, w]
            x = x.unsqueeze(1)
        with torch.no_grad():
            output = model(x, deterministic=True)
            generated = output['rec_rgb'].clamp(0, 1)
            if len(x.shape) == 5:
                # [bs, T, ch, h, w]
                x = x.view(-1, *x.shape[2:])
            results = evaluator(x, generated)
        # [batch_size * T]
        if 'ssim' in metrics:
            ssims.append(results['ssim'])
        if 'psnr' in metrics:
            psnrs.append(results['psnr'])
        if 'lpips' in metrics:
            lpipss.append(results['lpips'])

    if 'ssim' in metrics:
        ssims = torch.cat(ssims, dim=0)
        mean_ssim = ssims.mean().data.cpu().item()
        std_ssim = ssims.std().data.cpu().item()
        results['ssim'] = mean_ssim
        results['ssim_std'] = std_ssim
    if 'psnr' in metrics:
        psnrs = torch.cat(psnrs, dim=0)
        mean_psnr = psnrs.mean().data.cpu().item()
        std_psnr = psnrs.std().data.cpu().item()
        results['psnr'] = mean_psnr
        results['psnr_std'] = std_psnr
    if 'lpips' in metrics:
        lpipss = torch.cat(lpipss, dim=0)
        mean_lpips = lpipss.mean().data.cpu().item()
        std_lpips = lpipss.std().data.cpu().item()
        results['lpips'] = mean_lpips
        results['lpips_std'] = std_lpips

    # save results
    if accelerator is not None:
        save_metric = accelerator.is_main_process
    else:
        save_metric = True
    if save_metric:
        path_to_conf = os.path.join(eval_dir, 'last_val_image_metrics.json')
        with open(path_to_conf, "w") as outfile:
            json.dump(results, outfile, indent=2)

    del evaluator  # clear memory

    return results


def load_dlp_from_config(conf_path, ckpt_path=None, model_type='gddlp'):
    if model_type == 'gdlp':
        model, config = load_model_from_config(conf_path, None, model_name_override='dlp')
        config = dict(config)
        config['timestep_horizon'] = 1
        model = build_model_from_config(config)
        if ckpt_path is not None:
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False))
                print("loaded dlp model from checkpoint")
            except Exception:
                print("dlp model checkpoint not found")
        return model

    model, _ = load_model_from_config(conf_path, ckpt_path)
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DLPWM Video Prediction Evaluation")
    parser.add_argument("-d", "--dataset", type=str, default='balls',
                        help="dataset to use: ['balls', 'traffic', 'clevrer', 'obj3d128', ...]")
    parser.add_argument("--model_type", type=str, default='ddlp',
                        help="dataset to use: ['dlp', 'ddlp']")
    parser.add_argument("-p", "--path", type=str,
                        help="path to model directory, e.g. ./310822_141959_balls_ddlp")
    parser.add_argument("--checkpoint", type=str,
                        help="direct path to model checkpoint, e.g. ./checkpoints/ddlp-obj3d128/obj3d_ddlp.pth",
                        default="")
    parser.add_argument("--use_last", action='store_true',
                        help="use the last checkpoint instead of best")
    parser.add_argument("--use_best_elbo", action='store_true',
                        help="use the best checkpoint according to elbo value")
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
    parser.add_argument("--dense_eval", action='store_true',
                        help="disable sparse routing during evaluation and use dense rollout")
    args = parser.parse_args()
    # parse input
    dir_path = args.path
    checkpoint_path = args.checkpoint
    ds = args.dataset
    model_type = args.model_type
    use_train = args.use_train
    cond_steps = args.cond_steps
    timestep_horizon = args.horizon
    batch_size = args.batch_size
    use_cpu = args.cpu
    use_ctx = args.ctx
    use_best_lpips = not args.use_best_elbo
    deterministic = not args.sample
    prefix = args.prefix
    dense_eval = args.dense_eval
    # load model config
    is_static_model = model_type in ('dlp', 'gdlp')
    conf_path = os.path.join(dir_path, 'hparams.json')
    with open(conf_path, 'r') as f:
        config = json.load(f)
    ds = config.get('ds', ds)
    if is_static_model:
        pref = model_type = 'gdlp'
    else:
        pref = model_type = get_model_tag(config)
    model_ckpt_name = f'{ds}_{pref}{prefix}.pth'
    if use_best_lpips:
        model_best_ckpt_name = f'{ds}_{pref}{prefix}_best_lpips.pth'
    else:
        model_best_ckpt_name = f'{ds}_{pref}{prefix}_best.pth'
    use_last = args.use_last if os.path.exists(os.path.join(dir_path, f'saves/{model_best_ckpt_name}')) else True
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if checkpoint_path.endswith('.pth'):
        ckpt_path = checkpoint_path
    else:
        ckpt_path = os.path.join(dir_path, f'saves/{model_ckpt_name if use_last else model_best_ckpt_name}')

    print(f'ckpt path: {ckpt_path}')
    model = load_dlp_from_config(conf_path, ckpt_path, pref)
    if dense_eval:
        model = disable_sparse_router(model)
        print("dense_eval: sparse routing disabled for evaluation")
    model = model.to(device)
    model.eval()

    # create dir for results
    pred_dir = os.path.join(dir_path, 'eval')
    os.makedirs(pred_dir, exist_ok=True)

    # conditional frames
    cond_steps = cond_steps if cond_steps > 0 else config['timestep_horizon']
    val_mode = 'train' if use_train else 'test'
    if model_type == 'gdlp':
        results = eval_dlp_im_metric(model, device, val_mode=val_mode,
                                     config=config,
                                     eval_dir=pred_dir, metrics=('ssim', 'psnr', 'lpips'),
                                     batch_size=batch_size)
    else:
        results = eval_lpwm_im_metric(model, device, timestep_horizon=timestep_horizon, val_mode=val_mode,
                                      config=config,
                                      eval_dir=pred_dir, cond_steps=cond_steps, metrics=('ssim', 'psnr', 'lpips'),
                                      batch_size=batch_size, use_all_ctx=use_ctx, deterministic=deterministic)
    print(f'results: {results}')
